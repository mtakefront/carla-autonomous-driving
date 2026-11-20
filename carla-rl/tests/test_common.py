"""Tests shared by v16 and v17 (this file is identical in both versions).

The version-specific agent, wrapper and sizes come from version_hooks.py.
Nothing here imports carla; FakeCarlaEnv stands in for CarlaEnv.
"""

import argparse
import ast
import copy
import importlib
import math
import os
import signal
import sys
import textwrap
import types

import gymnasium as gym
import numpy as np
import pytest
import torch

import version_hooks as vh
from d2drive_common import (
    COAST_LONGITUDINAL,
    SIDE_RANGE_M,
    D2IntegrationConfig,
    ObsNormalizer,
    RewardScaler,
    SpeedLimitShield,
    map_v15_to_d2,
    read_checkpoint,
    speed_limit_kmh,
)
from d2drive_ppo import AdaptiveSafetyShield, D2DriveConfig
from d2drive_training import (
    COMMON_CLI_FIELDS,
    TerminationStats,
    add_common_train_args,
    resolve_dataclass,
    run_evaluation,
    run_training,
)
from fake_carla_env import make_v15_obs, v15_observation_bounds

CARLA_RL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARLA_ENV_PATH = os.path.join(CARLA_RL_DIR, "carla_env.py")


def quiet(*_args, **_kwargs):
    pass


def mask_for(obs, limit_kmh=None):
    shield = SpeedLimitShield(D2IntegrationConfig())
    shield.set_speed_limit_kmh(limit_kmh or speed_limit_kmh(obs))
    return shield.unsafe_action_mask(map_v15_to_d2(obs))


def read_carla_env_source():
    with open(CARLA_ENV_PATH, encoding="utf-8") as f:
        return f.read().replace("\r\n", "\n")


# ---------------------------------------------------------------- B1 mapping

def test_mapping_units_and_ego_fields():
    obs = make_v15_obs({18: 0.4, 19: 30.0, 15: 1.0, 16: 12.0, 17: 3.0})
    d2 = map_v15_to_d2(obs)
    assert d2.shape == (15,) and d2.dtype == np.float32
    assert d2[0] == pytest.approx(10.0)              # 36 km/h
    assert d2[1] == pytest.approx(0.4)               # + = right, as v15
    assert d2[2] == pytest.approx(math.pi / 6)       # 30 deg
    assert d2[3] == pytest.approx(12.0)
    assert d2[4] == pytest.approx(3.0)


def test_mapping_nothing_ahead_gives_front_range():
    d2 = map_v15_to_d2(make_v15_obs())
    assert d2[3] == pytest.approx(40.0) and d2[4] == 0.0


def test_mapping_left_vehicle_ahead():
    # 10 m ahead in the left lane, 2 m/s slower than the ego (rel = other - ego)
    d2 = map_v15_to_d2(make_v15_obs({20: 1.0, 21: 10.0, 22: -2.0}))
    assert list(d2[5:9]) == pytest.approx([10.0, SIDE_RANGE_M, 2.0, 0.0])
    assert list(d2[9:13]) == pytest.approx([SIDE_RANGE_M, SIDE_RANGE_M, 0.0, 0.0])


def test_mapping_left_vehicle_behind():
    # 8 m behind in the left lane, 3 m/s faster: it closes on the ego at +3
    d2 = map_v15_to_d2(make_v15_obs({20: 1.0, 21: -8.0, 22: 3.0}))
    assert list(d2[5:9]) == pytest.approx([SIDE_RANGE_M, 8.0, 0.0, 3.0])


def test_mapping_right_vehicle_ahead_and_behind():
    ahead = map_v15_to_d2(make_v15_obs({23: 1.0, 24: 6.0, 25: 1.5}))
    assert list(ahead[9:13]) == pytest.approx([6.0, SIDE_RANGE_M, -1.5, 0.0])
    behind = map_v15_to_d2(make_v15_obs({23: 1.0, 24: -12.0, 25: -1.0}))
    assert list(behind[9:13]) == pytest.approx([SIDE_RANGE_M, 12.0, 0.0, -1.0])


def test_mapping_vehicle_alongside_counts_as_ahead_and_blocks():
    obs = make_v15_obs({23: 1.0, 24: 0.0, 25: 1.0})
    d2 = map_v15_to_d2(obs)
    assert list(d2[9:13]) == pytest.approx([0.0, SIDE_RANGE_M, -1.0, 0.0])
    assert mask_for(obs)[4] == 1.0


def test_mapping_lane_without_vehicle_allows_lane_change():
    obs = make_v15_obs()
    d2 = map_v15_to_d2(obs)
    assert list(d2[5:13]) == pytest.approx([SIDE_RANGE_M, SIDE_RANGE_M, 0, 0] * 2)
    mask = mask_for(obs)
    assert mask[3] == 0.0 and mask[4] == 0.0


def test_mapping_no_lane_gives_zero_gaps_and_shield_blocks():
    obs = make_v15_obs({7: 0.0})
    d2 = map_v15_to_d2(obs)
    assert list(d2[5:9]) == [0.0, 0.0, 0.0, 0.0]
    assert list(d2[9:13]) == pytest.approx([SIDE_RANGE_M, SIDE_RANGE_M, 0.0, 0.0])
    mask = mask_for(obs)
    assert mask[3] == 1.0 and mask[4] == 0.0


def test_mapping_inside_junction():
    # v15 in a junction: no lane counts, no side radar, offset/heading 0;
    # the front sensor still works (it follows the path through the junction).
    obs = make_v15_obs({7: 0.0, 8: 0.0, 15: 1.0, 16: 15.0, 17: 2.0})
    d2 = map_v15_to_d2(obs)
    assert list(d2[5:13]) == [0.0] * 8
    assert d2[1] == 0.0 and d2[2] == 0.0
    assert d2[3] == pytest.approx(15.0) and d2[4] == pytest.approx(2.0)
    mask = mask_for(obs)
    assert mask[3] == 1.0 and mask[4] == 1.0


def test_mapping_density_and_speed_spread():
    obs = make_v15_obs({15: 1.0, 16: 20.0, 17: 3.0,       # front, closing at 3
                        20: 1.0, 21: 5.0, 22: -2.0,       # left, 2 m/s slower
                        26: 1.0, 27: -10.0, 28: 3.0})     # behind, 3 m/s faster
    d2 = map_v15_to_d2(obs)
    assert d2[13] == pytest.approx(0.75)
    # ego 10, left 8, rear 13, front 10 - 3 = 7
    assert d2[14] == pytest.approx(np.std([10.0, 8.0, 13.0, 7.0]))
    alone = map_v15_to_d2(make_v15_obs())
    assert alone[13] == 0.0 and alone[14] == 0.0


def test_mapping_rejects_other_sizes():
    with pytest.raises(ValueError):
        map_v15_to_d2(np.zeros(15))


# ---------------------------------------------------------- B4 speed limit

def test_speed_limit_shield_uses_the_current_limit():
    cfg = D2IntegrationConfig()
    shield = SpeedLimitShield(cfg)
    base = map_v15_to_d2(make_v15_obs({0: 45.0}))
    shield.set_speed_limit_kmh(50.0)
    assert shield.unsafe_action_mask(base)[1] == 0.0
    shield.set_speed_limit_kmh(40.0)
    assert shield.unsafe_action_mask(base)[1] == 1.0
    # Arnav's config is untouched; his shield alone (60 km/h) would allow 45.
    assert shield.cfg is cfg and cfg.max_speed_mps == pytest.approx(30.0 / 3.6)
    arnav = AdaptiveSafetyShield(D2DriveConfig(device="cpu"))
    assert arnav.unsafe_action_mask(base)[1] == 0.0


def test_unknown_speed_limit_is_30():
    assert speed_limit_kmh(make_v15_obs({1: 0.0})) == 30.0
    assert speed_limit_kmh(make_v15_obs({1: 50.0})) == 50.0


# ------------------------------------------------------- B2/B3 normalization

def test_normalizers_match_sb3_vecnormalize():
    vec_env = pytest.importorskip("stable_baselines3.common.vec_env")

    class Toy(gym.Env):
        observation_space = gym.spaces.Box(-1e3, 1e3, (3,), np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)

        def __init__(self):
            super().__init__()
            self.rng = np.random.default_rng(7)
            self.t = 0

        def _obs(self):
            return self.rng.normal([5.0, -2.0, 40.0], [3.0, 1.0, 10.0]).astype(np.float32)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.t = 0
            return self._obs(), {}

        def step(self, action):
            self.t += 1
            return self._obs(), float(self.rng.normal(1.0, 4.0)), self.t % 7 == 0, False, {}

    venv = vec_env.VecNormalize(vec_env.DummyVecEnv([Toy]), gamma=0.99,
                                clip_obs=10.0, clip_reward=10.0)
    obs_norm, rewards = ObsNormalizer(3, clip=10.0), RewardScaler(0.99, clip=10.0)
    out = venv.reset()
    np.testing.assert_allclose(
        obs_norm.normalize(venv.get_original_obs()[0], update=True), out[0], atol=1e-5)
    for _ in range(60):
        out, r, dones, _ = venv.step(np.zeros((1, 1), dtype=np.float32))
        mine = rewards.scale(float(venv.get_original_reward()[0]), bool(dones[0]))
        assert mine == pytest.approx(float(r[0]), rel=1e-5, abs=1e-6)
        np.testing.assert_allclose(
            obs_norm.normalize(venv.get_original_obs()[0], update=True), out[0], atol=1e-5)


def test_normalization_updates_in_training_and_freezes_in_eval():
    agent, env = vh.make_agent(), vh.make_env()
    obs, info = env.reset()
    rms = agent.obs_normalizer.rms
    count0 = rms.count
    agent.act(obs, info)
    agent.act(obs, info)
    assert rms.count == pytest.approx(count0 + 2)
    agent.peek_input(obs, info)                   # bootstrap peek: no update
    assert rms.count == pytest.approx(count0 + 2)
    agent.set_training(False)
    mean, var = rms.mean.copy(), rms.var.copy()
    agent.act(obs, info, deterministic=True)
    assert rms.count == pytest.approx(count0 + 2)
    np.testing.assert_array_equal(rms.mean, mean)
    np.testing.assert_array_equal(rms.var, var)


def test_policy_input_layout():
    agent, env = vh.make_agent(), vh.make_env()
    obs, info = env.reset()
    for _ in range(5):
        agent.act(obs, info)
    estimator = copy.deepcopy(agent.estimator)
    x = agent.act(obs, info)["augmented_obs"]
    assert x.shape == (vh.POLICY_INPUT_DIM,) and x.dtype == np.float32
    assert np.all(np.abs(x[:90]) <= 10.0)                       # normalized, clipped
    expected_disturbance = estimator.transform(map_v15_to_d2(obs))[15:]
    np.testing.assert_allclose(x[90:94], expected_disturbance)  # raw


def test_shield_and_estimator_see_raw_base(monkeypatch):
    agent, env = vh.make_agent(), vh.make_env(seed=3)
    seen_shield, seen_estimator = [], []
    mask_fn, transform_fn = agent.shield.unsafe_action_mask, agent.estimator.transform

    def spy_mask(raw):
        seen_shield.append(np.array(raw, copy=True))
        return mask_fn(raw)

    def spy_transform(raw):
        seen_estimator.append(np.array(raw, copy=True))
        return transform_fn(raw)

    monkeypatch.setattr(agent.shield, "unsafe_action_mask", spy_mask)
    monkeypatch.setattr(agent.estimator, "transform", spy_transform)
    obs, info = env.reset()
    for _ in range(4):
        agent.act(obs, info)
    expected = map_v15_to_d2(obs)
    assert seen_shield and all(np.array_equal(s, expected) for s in seen_shield)
    assert seen_estimator and all(np.array_equal(s, expected) for s in seen_estimator)


def test_peek_does_not_advance_the_estimator():
    agent, env = vh.make_agent(), vh.make_env(seed=4)
    obs, info = env.reset()
    agent.act(obs, info)
    n = len(agent.estimator.history)
    obs2, _, _, _, info2 = env.step(agent.act(obs, info)["executed_action"])
    n = len(agent.estimator.history)
    peeked = agent.peek_input(obs2, info2)
    assert len(agent.estimator.history) == n
    acted = agent.act(obs2, info2)["augmented_obs"]
    assert len(agent.estimator.history) == n + 1
    np.testing.assert_allclose(peeked[90:94], acted[90:94])     # same disturbance


def test_policy_input_dimension():
    agent = vh.make_agent()
    first = next(m for m in agent.policy.modules() if isinstance(m, torch.nn.Linear))
    assert agent.policy_input_dim == vh.POLICY_INPUT_DIM
    assert first.in_features == vh.POLICY_INPUT_DIM


def test_agent_uses_cpu_and_one_thread():
    torch.set_num_threads(4)
    agent = vh.make_agent()
    assert torch.get_num_threads() == 1
    assert agent.device.type == "cpu"
    assert all(p.device.type == "cpu" for p in agent.policy.parameters())


# ------------------------------------------------------------ B6 training

def test_two_ppo_updates_run(tmp_path):
    agent, env = vh.make_agent(), vh.make_env(seed=1)
    csv_path = tmp_path / "metrics.csv"
    logger = run_training(env, agent, 2 * agent.cfg.rollout_steps, model_name="test",
                          decision_stats=vh.make_decision_stats(),
                          metrics_path=str(csv_path),
                          checkpoint_dir=str(tmp_path / "checkpoints"), print_fn=quiet)
    assert len(logger.rows) == 2
    assert agent.num_timesteps == 2 * agent.cfg.rollout_steps
    assert agent.buffer.data == []
    for row in logger.rows:
        for key in ("policy_loss", "value_loss", "entropy", "shield_loss",
                    "approx_kl_post", "clip_fraction_post"):
            assert math.isfinite(row[f"train/{key}"]), key
        for reason in ("goal", "collision", "red_light", "stuck", "time_limit"):
            assert f"episodes/{reason}" in row
        for m in ("LEFT", "RIGHT", "STRAIGHT"):
            assert f"junctions/{m}" in row and f"junctions_total/{m}_pass_rate" in row
        for part in ("progress", "speed", "step", "illegal_crossing", "lane", "terminal"):
            assert f"reward_parts/{part}" in row
        for key in ("rollout/ep_rew_mean", "rollout/ep_len_mean", "shield/veto_rate",
                    "time/total_timesteps", "time/fps"):
            assert key in row
    lines = csv_path.read_text().strip().splitlines()
    assert len(lines) == 3
    assert lines[0].split(",") == list(logger.rows[0])


def test_checkpoints_every_n_steps(tmp_path):
    agent, env = vh.make_agent(), vh.make_env(seed=2)
    run_training(env, agent, 64, model_name="test", metrics_path=str(tmp_path / "m.csv"),
                 checkpoint_dir=str(tmp_path / "checkpoints"), checkpoint_freq=32,
                 print_fn=quiet)
    names = sorted(os.listdir(tmp_path / "checkpoints"))
    assert names == ["test_32_steps.pt", "test_64_steps.pt"]


def test_truncation_bootstraps_and_termination_does_not(monkeypatch, tmp_path):
    for end_probability, max_steps in ((0.0, 5), (1.0, 1000)):
        agent = vh.make_agent(normalize_reward=False)
        captured = []

        def fake_update(_last_input, agent=agent, captured=captured):
            captured.extend(agent.buffer.data)
            agent.buffer.clear()
            return {}

        monkeypatch.setattr(agent, "value_of", lambda _x: 2.0)
        monkeypatch.setattr(agent, "update", fake_update)
        env = vh.make_env(seed=3, max_steps=max_steps, end_probability=end_probability)
        run_training(env, agent, agent.cfg.rollout_steps, model_name="test",
                     metrics_path=str(tmp_path / f"m{end_probability}.csv"),
                     checkpoint_dir=str(tmp_path / "checkpoints"), print_fn=quiet)
        returns = vh.inner_env(env).step_returns
        assert len(captured) == len(returns) == agent.cfg.rollout_steps
        truncations = 0
        for transition, (_obs, reward, terminated, truncated, _info) in zip(captured, returns):
            assert transition.done == (terminated or truncated)
            expected = reward + agent.cfg.gamma * 2.0 if truncated else reward
            assert transition.reward == pytest.approx(expected)
            truncations += truncated
        assert truncations > 0 if end_probability == 0.0 else truncations == 0


def test_save_load_resume_keeps_statistics(tmp_path):
    agent, env = vh.make_agent(), vh.make_env(seed=5)
    run_training(env, agent, 64, model_name="test", metrics_path=str(tmp_path / "a.csv"),
                 checkpoint_dir=str(tmp_path / "checkpoints"), print_fn=quiet)
    path = str(tmp_path / "model.pt")
    agent.save(path, extra={"note": "kept"})

    other = vh.make_agent()
    checkpoint = other.load(path)
    assert checkpoint["extra"] == {"note": "kept"}
    for (name, a), (_, b) in zip(agent.policy.state_dict().items(),
                                 other.policy.state_dict().items()):
        assert torch.equal(a, b), name
    assert agent.optimizer.state_dict()["state"].keys() == other.optimizer.state_dict()["state"].keys()
    for key in ("mean", "var", "count"):
        np.testing.assert_array_equal(agent.obs_normalizer.rms.state_dict()[key],
                                      other.obs_normalizer.rms.state_dict()[key])
        np.testing.assert_array_equal(agent.reward_scaler.ret_rms.state_dict()[key],
                                      other.reward_scaler.ret_rms.state_dict()[key])
    assert other.reward_scaler.ret == agent.reward_scaler.ret
    assert other.num_timesteps == agent.num_timesteps == 64

    count = other.obs_normalizer.rms.count
    run_training(env, other, 64, model_name="test", metrics_path=str(tmp_path / "b.csv"),
                 checkpoint_dir=str(tmp_path / "checkpoints"), print_fn=quiet)
    assert other.num_timesteps == 128
    assert other.obs_normalizer.rms.count == pytest.approx(count + 64)


def test_resume_takes_explicit_options_from_the_command_line(tmp_path):
    agent = vh.make_agent()
    path = str(tmp_path / "model.pt")
    agent.save(path)
    parser = argparse.ArgumentParser()
    add_common_train_args(parser, "test")
    args = parser.parse_args(["--lr", "1e-4", "--no-reward-norm"])
    cfg, sources = resolve_dataclass(vh.CONFIG_CLASS, read_checkpoint(path)["config"],
                                     args, COMMON_CLI_FIELDS, fixed={"device": "cpu"})
    assert cfg.learning_rate == 1e-4 and sources["learning_rate"] == "command line"
    assert cfg.normalize_reward is False and sources["normalize_reward"] == "command line"
    assert cfg.rollout_steps == 64 and sources["rollout_steps"] == "checkpoint"
    assert cfg.device == "cpu"
    resumed = vh.AGENT_CLASS(cfg)
    resumed.load(path)
    assert all(group["lr"] == 1e-4 for group in resumed.optimizer.param_groups)


def test_fresh_config_uses_arnav_ppo_defaults():
    parser = argparse.ArgumentParser()
    add_common_train_args(parser, "test")
    cfg, sources = resolve_dataclass(vh.CONFIG_CLASS, None, parser.parse_args([]),
                                     COMMON_CLI_FIELDS, fixed={"device": "cpu"})
    assert (cfg.rollout_steps, cfg.minibatch_size, cfg.update_epochs, cfg.learning_rate,
            cfg.gamma, cfg.gae_lambda, cfg.clip_eps, cfg.entropy_coef, cfg.value_coef,
            cfg.shield_coef, cfg.max_grad_norm, cfg.hidden_dim) == (
        2048, 256, 10, 3e-4, 0.99, 0.95, 0.2, 0.01, 0.5, 0.15, 0.5, 256)
    assert cfg.device == "cpu" and cfg.normalize_reward and not cfg.store_executed_action
    assert sources["rollout_steps"] == "default"


def test_cli_hyperparameters_reach_the_config():
    parser = argparse.ArgumentParser()
    add_common_train_args(parser, "test")
    args = parser.parse_args(["--rollout-steps", "512", "--minibatch-size", "64", "--epochs",
                              "4", "--lr", "1e-4", "--ent-coef", "0.02", "--shield-coef",
                              "0.3", "--store-executed-action"])
    cfg, _ = resolve_dataclass(vh.CONFIG_CLASS, None, args, COMMON_CLI_FIELDS)
    assert (cfg.rollout_steps, cfg.minibatch_size, cfg.update_epochs, cfg.learning_rate,
            cfg.entropy_coef, cfg.shield_coef, cfg.store_executed_action) == (
        512, 64, 4, 1e-4, 0.02, 0.3, True)


# ------------------------------------------------------------ B9 wrapper

def test_wrapper_returns_env_values_unchanged():
    env, agent = vh.make_env(seed=6, end_probability=0.1), vh.make_agent()
    inner = vh.inner_env(env)
    obs, info = env.reset()
    reset_obs, reset_info = inner.reset_returns[-1]
    np.testing.assert_array_equal(obs, reset_obs)
    assert all(info[k] == v for k, v in reset_info.items())
    episodes = 0
    for _ in range(60):
        decision = agent.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(
            decision["executed_action"], decision=decision)
        e_obs, e_reward, e_terminated, e_truncated, e_info = inner.step_returns[-1]
        np.testing.assert_array_equal(obs, e_obs)
        assert reward == e_reward
        assert (terminated, truncated) == (e_terminated, e_truncated)
        for key, value in e_info.items():
            assert info[key] == value, key
        added = set(info) - set(e_info)
        assert {"d2_proposed_action", "d2_executed_action", "d2_vetoed"} <= added
        assert all(key.startswith("d2_") for key in added)
        if terminated or truncated:
            episodes += 1
            obs, info = env.reset()
            agent.reset_episode()
    assert episodes > 0


# ------------------------------------------------------------ B8 evaluation

def test_evaluation_report_and_frozen_statistics():
    agent, env = vh.make_agent(), vh.make_env(seed=7, max_steps=30)
    count = agent.obs_normalizer.rms.count
    summary = run_evaluation(env, agent, episodes=3, seed=0,
                             decision_stats=vh.make_decision_stats(), print_fn=quiet)
    assert summary["episodes"] == 3 and sum(summary["reasons"].values()) == 3
    assert 0.0 <= summary["goal_rate"] <= 1.0
    assert set(summary["maneuver_pass_rate"]) == {"LEFT", "RIGHT", "STRAIGHT"}
    assert summary["shield/veto_rate"] is not None
    assert math.isfinite(summary["mean_speed_kmh"])
    assert agent.obs_normalizer.rms.count == count


def test_mean_lane_offset_skips_junction_steps(monkeypatch):
    agent, env = vh.make_agent(), vh.make_env(seed=8, max_steps=20, end_probability=0.0)
    inner = vh.inner_env(env)
    original = inner._observation
    calls = []

    def observation():
        obs = original()
        calls.append(1)
        # every other observation looks like a junction (offset and heading 0)
        obs[18], obs[19] = (0.0, 0.0) if len(calls) % 2 else (-0.5, 3.0)
        return obs

    monkeypatch.setattr(inner, "_observation", observation)
    summary = run_evaluation(env, agent, episodes=1, print_fn=quiet)
    assert summary["mean_abs_lane_offset_m"] == pytest.approx(0.5)


# ------------------------------------------------- v15 is used, not copied

def test_v15_action_mapping_lines_unchanged():
    text = read_carla_env_source()
    for line in ("steer = float(np.clip(action[0], -1.0, 1.0))",
                 "longitudinal = float(np.clip(action[1], -1.0, 1.0))",
                 "throttle = float(np.clip(0.25 + 0.75 * longitudinal, 0.0, 1.0))",
                 "brake = float(np.clip((-longitudinal - 0.6) / 0.4, 0.0, 1.0))"):
        assert line in text, line


def test_coast_value_is_in_v15_coast_band():
    throttle = float(np.clip(0.25 + 0.75 * COAST_LONGITUDINAL, 0.0, 1.0))
    brake = float(np.clip((-COAST_LONGITUDINAL - 0.6) / 0.4, 0.0, 1.0))
    assert throttle == 0.0 and brake == 0.0


def test_fake_env_bounds_match_carla_env_source():
    """Runs CarlaEnv's own bound-building code (from its source) and compares."""
    source = read_carla_env_source()
    constants = {}
    for node in ast.parse(source).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    start = source.index("        low = ([0.0")
    end = source.index("        self.observation_space = spaces.Box(")
    namespace = dict(constants)
    exec(textwrap.dedent(source[start:end]), {}, namespace)
    low, high = v15_observation_bounds()
    assert len(low) == 75
    np.testing.assert_array_equal(np.asarray(namespace["low"], dtype=np.float32), low)
    np.testing.assert_array_equal(np.asarray(namespace["high"], dtype=np.float32), high)


def test_termination_stats_match_v15_callback():
    stats = TerminationStats()
    stats.rollout_start()
    stats.on_step({"maneuver_results": [("LEFT", True)]}, False)
    stats.on_step({"maneuver_results": [("LEFT", False)], "termination_reason": "red_light",
                   "reward_parts": {"terminal": -30.0}}, True)
    stats.on_step({"termination_reason": "goal",
                   "reward_parts": {"progress": 100.0, "terminal": 50.0}}, True)
    out = stats.record()
    assert out["episodes/red_light"] == 1 and out["episodes/goal"] == 1
    assert out["episodes/collision"] == 0
    assert out["junctions/LEFT"] == "1/2" and out["junctions/RIGHT"] == "0/0"
    assert out["junctions_total/LEFT_pass_rate"] == 0.5
    assert out["reward_parts/terminal"] == 10.0 and out["reward_parts/progress"] == 50.0


def test_entry_points_import_without_carla():
    for name in (f"train_{vh.VERSION}", f"evaluate_{vh.VERSION}"):
        importlib.import_module(name)
    assert "carla" not in sys.modules
    assert "carla_env" not in sys.modules


def test_train_and_evaluate_main(monkeypatch, tmp_path, capsys):
    """train_vXX.main() / evaluate_vXX.main() end to end, with a stand-in
    carla_env module: CLI, CarlaEnv arguments, saving, --resume, saving
    when stopped early, env.close()."""
    instances = []

    class CarlaEnvStub(vh.FAKE_CARLA_ENV):
        def __init__(self, min_route_length=120.0, traffic_vehicles=0, pedestrians=0,
                     no_rendering=False):
            super().__init__(seed=len(instances))
            self.kwargs = dict(min_route_length=min_route_length,
                               traffic_vehicles=traffic_vehicles,
                               pedestrians=pedestrians, no_rendering=no_rendering)
            instances.append(self)

    stub = types.ModuleType("carla_env")
    stub.CarlaEnv = CarlaEnvStub
    monkeypatch.setitem(sys.modules, "carla_env", stub)
    monkeypatch.setattr(signal, "signal", lambda *_args: None)
    monkeypatch.chdir(tmp_path)
    train = importlib.import_module(f"train_{vh.VERSION}")
    evaluate = importlib.import_module(f"evaluate_{vh.VERSION}")
    model = f"d2drive_{vh.VERSION}.pt"
    small = ["--rollout-steps", "32", "--minibatch-size", "16", "--epochs", "1"]

    train.main(["--steps", "64", "--traffic", "3", "--pedestrians", "4", "--no-render", *small])
    assert instances[0].kwargs == dict(min_route_length=60.0, traffic_vehicles=3,
                                       pedestrians=4, no_rendering=True)
    assert instances[0].closed
    assert os.path.isfile(f"train_{vh.VERSION}_metrics.csv")
    assert read_checkpoint(model)["num_timesteps"] == 64

    train.main(["--resume", f"d2drive_{vh.VERSION}", "--steps", "32", "--lr", "1e-4"])
    out = capsys.readouterr().out
    assert "learning_rate = 0.0001  (command line)" in out
    assert "rollout_steps = 32  (checkpoint)" in out
    assert "device = 'cpu'" in out
    checkpoint = read_checkpoint(model)
    assert checkpoint["num_timesteps"] == 96
    assert checkpoint["config"]["learning_rate"] == 1e-4

    def interrupted(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(train, "run_training", interrupted)
    with pytest.raises(KeyboardInterrupt):
        train.main(["--resume", model, "--steps", "32"])
    assert instances[-1].closed
    assert "stopped early, saved" in capsys.readouterr().out
    assert read_checkpoint(model)["num_timesteps"] == 96

    evaluate.main(["--model", model, "--episodes", "2", "--traffic", "0"])
    assert instances[-1].closed and instances[-1].kwargs["traffic_vehicles"] == 0
    assert "Goal reached:" in capsys.readouterr().out
