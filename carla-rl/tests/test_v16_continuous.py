"""v16-only tests: continuous shield, continuous shield penalty, Gaussian
policy, minimum std, policy input size, wrapper pass-through."""

import math

import numpy as np
import pytest
import torch
from torch.distributions import Normal

import version_hooks as vh
from d2drive_common import (
    COAST_LONGITUDINAL,
    SpeedLimitShield,
    map_v15_to_d2,
    speed_limit_kmh,
)
from d2drive_continuous import (
    ContinuousShield,
    GaussianActorCritic,
    V16Config,
    continuous_shield_loss,
)
from d2drive_training import run_training
from fake_carla_env import make_v15_obs


def quiet(*_args, **_kwargs):
    pass


def shield_filter(obs, action, offset=0.6):
    shield = ContinuousShield(SpeedLimitShield(V16Config()), offset)
    shield.mask_shield.set_speed_limit_kmh(speed_limit_kmh(obs))
    return shield.filter(np.asarray(action, dtype=np.float32), map_v15_to_d2(obs))


# ------------------------------------------------------------ C3 shield

def test_accelerate_unsafe_cuts_throttle_but_keeps_brake():
    obs = make_v15_obs({15: 1.0, 16: 8.0, 17: 2.0})    # car 8 m ahead, closing
    executed, flags, vetoed = shield_filter(obs, [0.1, 0.5])
    assert executed[1] == pytest.approx(COAST_LONGITUDINAL)
    assert executed[0] == pytest.approx(0.1)
    assert flags[0] == 1.0 and vetoed
    executed, flags, vetoed = shield_filter(obs, [0.1, -0.9])      # agent brakes harder
    assert executed[1] == pytest.approx(-0.9) and flags[0] == 1.0 and not vetoed
    executed, _, vetoed = shield_filter(obs, [0.0, COAST_LONGITUDINAL])
    assert executed[1] == pytest.approx(COAST_LONGITUDINAL) and not vetoed


def test_accelerate_unsafe_above_the_speed_limit():
    over = make_v15_obs({0: 52.0, 1: 50.0})
    under = make_v15_obs({0: 45.0, 1: 50.0})
    assert shield_filter(over, [0.0, 0.8])[0][1] == pytest.approx(COAST_LONGITUDINAL)
    executed, flags, vetoed = shield_filter(under, [0.0, 0.8])
    assert executed[1] == pytest.approx(0.8) and flags[0] == 0.0 and not vetoed


def test_left_rule_when_drifting_left_without_a_safe_left_lane():
    obs = make_v15_obs({7: 0.0, 18: -0.8, 19: -5.0})
    executed, flags, vetoed = shield_filter(obs, [-0.3, 0.0])
    assert executed[0] == 0.0 and flags[1] == 1.0 and vetoed
    executed, flags, vetoed = shield_filter(obs, [0.2, 0.0])       # steering back: allowed
    assert executed[0] == pytest.approx(0.2) and flags[1] == 1.0 and not vetoed


def test_left_rule_off_when_not_drifting_or_lane_is_safe():
    cases = [
        {7: 0.0, 18: -0.8, 19: 2.0},    # offset left but heading back right
        {7: 0.0, 18: -0.3, 19: -5.0},   # offset within D
        {7: 1.0, 18: -0.8, 19: -5.0},   # empty left lane: a lane change is safe
    ]
    for overrides in cases:
        executed, flags, vetoed = shield_filter(make_v15_obs(overrides), [-0.3, 0.0])
        assert executed[0] == pytest.approx(-0.3) and flags[1] == 0.0 and not vetoed, overrides


def test_right_rule_when_drifting_right_without_a_safe_right_lane():
    obs = make_v15_obs({8: 0.0, 18: 0.8, 19: 5.0})
    executed, flags, vetoed = shield_filter(obs, [0.4, 0.0])
    assert executed[0] == 0.0 and flags[2] == 1.0 and vetoed
    executed, _, vetoed = shield_filter(obs, [-0.4, 0.0])
    assert executed[0] == pytest.approx(-0.4) and not vetoed


def test_right_rule_uses_the_offset_option():
    obs = make_v15_obs({8: 0.0, 18: 0.8, 19: 5.0})
    executed, flags, _ = shield_filter(obs, [0.4, 0.0], offset=1.0)
    assert executed[0] == pytest.approx(0.4) and flags[2] == 0.0


def test_lateral_rules_are_off_in_junctions():
    # In junctions v15 reports no lanes (lane changes masked) but offset and
    # heading 0, so full turns pass.
    obs = make_v15_obs({7: 0.0, 8: 0.0})
    mask = SpeedLimitShield(V16Config()).unsafe_action_mask(map_v15_to_d2(obs))
    assert mask[3] == 1.0 and mask[4] == 1.0
    for steer in (-1.0, 1.0):
        executed, flags, vetoed = shield_filter(obs, [steer, 0.0])
        assert executed[0] == steer and flags[1] == 0.0 and flags[2] == 0.0 and not vetoed


def test_tiny_changes_are_not_vetoes():
    obs = make_v15_obs({15: 1.0, 16: 8.0, 17: 2.0})
    _, flags, vetoed = shield_filter(obs, [0.0, COAST_LONGITUDINAL + 5e-7])
    assert flags[0] == 1.0 and not vetoed


# ----------------------------------------------------- C4 shield penalty

def test_shield_loss_gradient_on_the_means():
    mean = torch.zeros(4, 2, requires_grad=True)
    dist = Normal(mean, torch.full((4, 2), 0.3))
    flags = torch.tensor([[1.0, 0.0, 0.0]] * 4)
    continuous_shield_loss(dist, flags).backward()
    assert torch.all(mean.grad[:, 1] > 0)      # lower longitudinal mean -> less throttle
    for flag, sign in ((1, -1.0), (2, 1.0)):   # left: push steer right, right: push left
        mean = torch.zeros(4, 2, requires_grad=True)
        dist = Normal(mean, torch.full((4, 2), 0.3))
        flags = torch.zeros(4, 3)
        flags[:, flag] = 1.0
        continuous_shield_loss(dist, flags).backward()
        assert torch.all(torch.sign(mean.grad[:, 0]) == sign)


def test_shield_loss_step_lowers_unsafe_probability():
    torch.manual_seed(0)
    x = torch.randn(64, 8)

    def p_accelerate(d):
        return 1.0 - Normal(d.mean[:, 1], d.stddev[:, 1]).cdf(
            torch.full((64,), COAST_LONGITUDINAL))

    def p_left(d):
        return Normal(d.mean[:, 0], d.stddev[:, 0]).cdf(torch.zeros(64))

    def p_right(d):
        return 1.0 - p_left(d)

    for column, probability in ((0, p_accelerate), (1, p_left), (2, p_right)):
        policy = GaussianActorCritic(8, 16)
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
        flags = torch.zeros(64, 3)
        flags[:, column] = 1.0
        dist, _ = policy.distribution(x)
        before = float(probability(dist).mean().detach())
        loss = continuous_shield_loss(dist, flags)
        assert float(loss.detach()) == pytest.approx(before, rel=1e-5)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        dist, _ = policy.distribution(x)
        assert float(probability(dist).mean().detach()) < before


def test_shield_loss_is_zero_without_flags():
    dist = Normal(torch.zeros(3, 2), torch.ones(3, 2))
    assert float(continuous_shield_loss(dist, torch.zeros(3, 3))) == 0.0


# ------------------------------------------------------ C2 Gaussian policy

def test_gaussian_log_prob_and_entropy_shapes():
    policy = GaussianActorCritic(94, 32)
    dist, value = policy.distribution(torch.randn(5, 94))
    action = dist.sample()
    log_prob = dist.log_prob(action).sum(-1)
    entropy = dist.entropy().sum(-1)
    assert dist.mean.shape == (5, 2) and value.shape == (5,)
    assert log_prob.shape == (5,) and entropy.shape == (5,)
    std = policy.std()
    assert torch.allclose(std, torch.full((2,), math.exp(-1.0)))
    expected = sum(0.5 * math.log(2 * math.pi * math.e * float(s) ** 2) for s in std)
    assert torch.allclose(entropy, torch.full((5,), expected), atol=1e-5)
    manual = sum(Normal(dist.mean[:, i], dist.stddev[:, i]).log_prob(action[:, i])
                 for i in range(2))
    assert torch.allclose(log_prob, manual, atol=1e-6)


def test_min_std_is_respected_and_std_can_grow_back():
    policy = GaussianActorCritic(4, 8, log_std_init=-1.0, min_std=0.05)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.5)
    for sign in (1.0, -1.0):          # first push the std down, then up again
        for _ in range(40):
            dist, _ = policy.distribution(torch.randn(16, 4))
            loss = sign * dist.stddev.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            policy.project_log_std()
            assert torch.all(policy.std() >= 0.05 - 1e-7)
            assert torch.all(policy.log_std >= math.log(0.05) - 1e-6)
        if sign > 0:
            assert torch.allclose(policy.std(), torch.full((2,), 0.05), atol=1e-6)
    assert torch.all(policy.std() > 0.06)


def test_agent_std_floor_comes_from_the_config():
    agent = vh.make_agent(log_std_init=-5.0, min_std=0.05)
    assert torch.allclose(agent.policy.std(), torch.full((2,), 0.05))


def test_policy_input_is_94_and_actor_outputs_two_means():
    agent = vh.make_agent()
    assert agent.policy_input_dim == 94
    assert agent.policy.net.shared[0].in_features == 94
    assert agent.policy.net.actor[-1].out_features == 2


def test_act_clips_after_sampling_and_logs_the_unclipped_sample():
    agent, env = vh.make_agent(), vh.make_env(seed=11)
    with torch.no_grad():
        agent.policy.log_std.fill_(math.log(3.0))     # wide: samples leave [-1, 1]
    obs, info = env.reset()
    outside = 0
    for _ in range(20):
        decision = agent.act(obs, info)
        raw, proposed = decision["raw_action"], decision["proposed_action"]
        np.testing.assert_array_equal(proposed, np.clip(raw, -1.0, 1.0))
        assert np.all(np.abs(decision["executed_action"]) <= 1.0)
        outside += int(np.any(np.abs(raw) > 1.0))
        x = torch.as_tensor(decision["augmented_obs"]).unsqueeze(0)
        with torch.no_grad():
            dist, _ = agent.policy.distribution(x)
            expected = float(dist.log_prob(torch.as_tensor(raw).unsqueeze(0)).sum())
        assert decision["log_prob"] == pytest.approx(expected, rel=1e-5)
    assert outside > 0


def test_deterministic_action_is_the_mean():
    agent, env = vh.make_agent(), vh.make_env(seed=12)
    obs, info = env.reset()
    agent.set_training(False)
    decision = agent.act(obs, info, deterministic=True)
    mean, _, _ = agent.policy(torch.as_tensor(decision["augmented_obs"]).unsqueeze(0))
    np.testing.assert_allclose(decision["raw_action"], mean.squeeze(0).detach().numpy())


def test_buffer_stores_proposal_by_default_and_executed_with_the_flag():
    obs = make_v15_obs({15: 1.0, 16: 8.0, 17: 2.0})    # accelerate unsafe -> veto likely
    for flag in (False, True):
        agent = vh.make_agent(store_executed_action=flag)
        with torch.no_grad():
            agent.policy.net.actor[-1].bias.copy_(torch.tensor([0.0, 0.9]))
        decision = agent.act(obs, {})
        assert decision["vetoed"]
        agent.store(obs, decision, 1.0, False)
        stored = agent.buffer.data[-1]
        key, log_key = (("executed_action", "log_prob_executed") if flag
                        else ("raw_action", "log_prob"))
        np.testing.assert_array_equal(stored.action, decision[key])
        assert stored.log_prob == pytest.approx(decision[log_key])
        np.testing.assert_array_equal(stored.unsafe_mask, decision["shield_flags"])


def test_two_updates_log_the_v16_metrics(tmp_path):
    agent, env = vh.make_agent(), vh.make_env(seed=13)
    logger = run_training(env, agent, 128, model_name="test",
                          decision_stats=vh.make_decision_stats(),
                          metrics_path=str(tmp_path / "m.csv"),
                          checkpoint_dir=str(tmp_path / "checkpoints"), print_fn=quiet)
    for row in logger.rows:
        for key in ("train/std_steer", "train/std_longitudinal", "train/approx_kl",
                    "train/clip_fraction", "train/approx_kl_post", "train/clip_fraction_post",
                    "shield/acc_unsafe_rate", "shield/left_active_rate",
                    "shield/right_active_rate", "actions/mean_abs_steer"):
            assert row[key] is not None and math.isfinite(row[key]), key
        assert row["train/std_steer"] >= agent.cfg.min_std - 1e-7


# ------------------------------------------------------- C1 wrapper

def test_wrapper_gives_the_action_to_carla_env_unchanged():
    env, agent = vh.make_env(seed=14), vh.make_agent()
    inner = vh.inner_env(env)
    obs, info = env.reset()
    action = np.array([0.3, -0.7], dtype=np.float32)
    _, _, _, _, info = env.step(action)
    np.testing.assert_array_equal(inner.actions[-1], action)
    assert info["d2_vetoed"] is False
    assert info["d2_shield_flags"] == {"acc_unsafe": False, "left_active": False,
                                       "right_active": False}
    decision = agent.act(obs, info)
    _, _, _, _, info = env.step(decision["executed_action"], decision=decision)
    np.testing.assert_array_equal(inner.actions[-1], decision["executed_action"])
    assert info["d2_executed_action"] == pytest.approx(list(decision["executed_action"]))
    assert info["d2_proposed_action"] == pytest.approx(list(decision["proposed_action"]))
