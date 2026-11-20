"""Shared pieces of the D2-Drive integration on the v15 environment.

This file is identical in v16 (continuous actions) and v17 (discrete
commands). It must not import carla or carla_env, so the tests can run on a
machine without a CARLA install.

Contents:
  map_v15_to_d2()       v15's 75-value observation -> D2-Drive's 15-value base vector
  RunningMeanStd,       replacements for SB3's VecNormalize (observation
  ObsNormalizer,        normalization and reward scaling), same formulas
  RewardScaler
  SpeedLimitShield      Arnav's shield with the current speed limit as max speed
  D2IntegrationConfig   Arnav's config with CPU and the integration options
  D2IntegrationMixin    policy-input assembly, bootstrap peeking, save/load
"""

from __future__ import annotations

import copy
import dataclasses
import math
import os
from dataclasses import dataclass

import numpy as np
import torch

from d2drive_ppo import AdaptiveSafetyShield, D2DriveConfig


# ---------------------------------------------------------------------------
# v15 observation layout (CarlaEnv docstring and CarlaEnv._get_state())

V15_OBS_DIM = 75
I_SPEED_KMH = 0
I_SPEED_LIMIT_KMH = 1      # 0 = unknown
I_LANES_LEFT = 7           # same-direction lanes to the left; 0 in junctions
I_LANES_RIGHT = 8
I_FRONT_PRESENT = 15
I_FRONT_DISTANCE_M = 16    # 40 = nothing ahead (center-to-center distance)
I_FRONT_CLOSING_MPS = 17   # + = approaching
I_LANE_OFFSET_M = 18       # + = right of the lane center; 0 in junctions / off the lane
I_LANE_HEADING_DEG = 19    # + = nose pointing right; 0 in junctions / off the lane
# [present, longitudinal x (+ ahead, - behind), relative speed (other - ego)]
I_LEFT = 20
I_RIGHT = 23
I_REAR = 26                # own lane, behind

# Copied from carla_env.py (importing it would import carla).
SIDE_RANGE_M = 25.0
FRONT_RANGE_M = 40.0

UNKNOWN_SPEED_LIMIT_KMH = 30.0
D2_BASE_DIM = 15
D2_DISTURBANCE_DIM = 4
# CarlaEnv.step() maps longitudinal l to throttle clip(0.25 + 0.75 l) and
# brake clip((-l - 0.6) / 0.4); -0.45 is the middle of the band with neither.
COAST_LONGITUDINAL = -0.45


def speed_limit_kmh(obs, unknown_kmh: float = UNKNOWN_SPEED_LIMIT_KMH) -> float:
    """Current speed limit from a v15 observation; `unknown_kmh` if v15 reports 0."""
    limit = float(obs[I_SPEED_LIMIT_KMH])
    return limit if limit > 0.0 else float(unknown_kmh)


def _side_gaps(lanes, present, x, rel):
    """(front gap, rear gap, front closing, rear closing) for one side."""
    if lanes < 0.5:
        # No same-direction lane on that side (always so inside junctions):
        # zero gaps make Arnav's shield veto a lane change to this side.
        return 0.0, 0.0, 0.0, 0.0
    if present < 0.5:
        # Lane but no vehicle within the 25 m radar range.
        return SIDE_RANGE_M, SIDE_RANGE_M, 0.0, 0.0
    if x >= 0.0:
        # Ahead, or exactly alongside (zero front gap). rel = other - ego, so
        # the ego closes on it at -rel.
        return float(x), SIDE_RANGE_M, float(-rel), 0.0
    # Behind: it closes on the ego at +rel.
    return SIDE_RANGE_M, float(-x), 0.0, float(rel)


def map_v15_to_d2(obs) -> np.ndarray:
    """D2-Drive's 15-value base observation from v15's 75-value observation.

    See INTEGRATION_NOTES.md for the mapping table and its limitations.
    """
    o = np.asarray(obs, dtype=np.float64).reshape(-1)
    if o.size != V15_OBS_DIM:
        raise ValueError(f"Expected the {V15_OBS_DIM}-value v15 observation, got {o.size}")
    ego = o[I_SPEED_KMH] / 3.6
    left = _side_gaps(o[I_LANES_LEFT], o[I_LEFT], o[I_LEFT + 1], o[I_LEFT + 2])
    right = _side_gaps(o[I_LANES_RIGHT], o[I_RIGHT], o[I_RIGHT + 1], o[I_RIGHT + 2])
    density = (o[I_FRONT_PRESENT] + o[I_LEFT] + o[I_RIGHT] + o[I_REAR]) / 4.0
    speeds = [ego]
    for start in (I_LEFT, I_RIGHT, I_REAR):
        if o[start] >= 0.5:
            speeds.append(ego + o[start + 2])
    if o[I_FRONT_PRESENT] >= 0.5:
        speeds.append(ego - o[I_FRONT_CLOSING_MPS])
    return np.array([
        ego,
        o[I_LANE_OFFSET_M],
        math.radians(o[I_LANE_HEADING_DEG]),
        o[I_FRONT_DISTANCE_M],
        o[I_FRONT_CLOSING_MPS],
        *left,
        *right,
        density,
        float(np.std(speeds)),
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# VecNormalize replacements (formulas of stable_baselines3 2.x)

class RunningMeanStd:
    """Running mean and variance, as stable_baselines3.common.running_mean_std."""

    def __init__(self, shape=(), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, batch):
        batch = np.asarray(batch, dtype=np.float64)
        self.update_from_moments(np.mean(batch, axis=0), np.var(batch, axis=0), batch.shape[0])

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / total
        self.mean, self.var, self.count = new_mean, m_2 / total, float(total)

    def state_dict(self):
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, state):
        self.mean = np.array(state["mean"], dtype=np.float64)
        self.var = np.array(state["var"], dtype=np.float64)
        self.count = float(state["count"])


class ObsNormalizer:
    """Running mean/std observation normalization, clipped to +-clip (VecNormalize)."""

    def __init__(self, dim: int, clip: float = 10.0, epsilon: float = 1e-8):
        self.rms = RunningMeanStd(shape=(dim,))
        self.clip = float(clip)
        self.epsilon = float(epsilon)

    def normalize(self, x, update: bool) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if update:
            self.rms.update(x[None, :])
        z = (x - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon)
        return np.clip(z, -self.clip, self.clip).astype(np.float32)

    def state_dict(self):
        return {"rms": self.rms.state_dict(), "clip": self.clip, "epsilon": self.epsilon}

    def load_state_dict(self, state):
        self.rms.load_state_dict(state["rms"])


class RewardScaler:
    """VecNormalize's reward scaling: divide by the running std of the
    discounted return (no mean subtraction), clip to +-clip."""

    def __init__(self, gamma: float, clip: float = 10.0, epsilon: float = 1e-8,
                 enabled: bool = True):
        self.gamma = float(gamma)
        self.clip = float(clip)
        self.epsilon = float(epsilon)
        self.enabled = bool(enabled)
        self.ret_rms = RunningMeanStd(shape=())
        self.ret = 0.0

    def scale(self, reward: float, done: bool, update: bool = True) -> float:
        if not self.enabled:
            return float(reward)
        if update:
            self.ret = self.ret * self.gamma + float(reward)
            self.ret_rms.update(np.array([self.ret]))
        scaled = float(np.clip(reward / np.sqrt(self.ret_rms.var + self.epsilon),
                               -self.clip, self.clip))
        if done:
            self.ret = 0.0
        return scaled

    def state_dict(self):
        return {"ret_rms": self.ret_rms.state_dict(), "ret": self.ret,
                "enabled": self.enabled}

    def load_state_dict(self, state):
        self.ret_rms.load_state_dict(state["ret_rms"])
        self.ret = float(state.get("ret", 0.0))


# ---------------------------------------------------------------------------
# Config and shield

@dataclass
class D2IntegrationConfig(D2DriveConfig):
    """Arnav's D2DriveConfig; PPO defaults unchanged (rollout 2048,
    minibatch 256, 10 epochs, lr 3e-4, ...), plus the integration options."""

    # Not used by d2drive_ppo.py; set to v15's values for reference.
    target_speed_mps: float = UNKNOWN_SPEED_LIMIT_KMH / 3.6
    min_speed_mps: float = 10.0 / 3.6            # v15's MIN_SPEED_KMH
    # Only the fallback when the speed limit is unknown: SpeedLimitShield
    # puts the current limit here for every mask.
    max_speed_mps: float = UNKNOWN_SPEED_LIMIT_KMH / 3.6
    # Arnav's default is cuda when available; the MLP is tiny and CPU is
    # faster next to CARLA (same reason as train_v15.py).
    device: str = "cpu"
    torch_threads: int = 1
    clip_obs: float = 10.0
    normalize_reward: bool = True
    clip_reward: float = 10.0
    store_executed_action: bool = False
    unknown_speed_limit_kmh: float = UNKNOWN_SPEED_LIMIT_KMH


def configure_torch(cfg: D2IntegrationConfig):
    """One thread: the network is tiny, and extra threads only compete with
    CARLA for the job's few CPUs (as train_v15.py)."""
    if cfg.torch_threads and cfg.torch_threads > 0:
        torch.set_num_threads(int(cfg.torch_threads))


class SpeedLimitShield(AdaptiveSafetyShield):
    """Arnav's AdaptiveSafetyShield with max_speed_mps = the current speed limit.

    The mask logic is not copied: unsafe_action_mask() calls Arnav's method
    with a config whose max_speed_mps is the limit, only for that call.
    """

    def __init__(self, cfg: D2DriveConfig):
        super().__init__(cfg)
        self.speed_limit_mps = float(cfg.max_speed_mps)

    def set_speed_limit_kmh(self, limit_kmh: float):
        self.speed_limit_mps = float(limit_kmh) / 3.6

    def unsafe_action_mask(self, raw_obs):
        base_cfg = self.cfg
        self.cfg = dataclasses.replace(base_cfg, max_speed_mps=self.speed_limit_mps)
        try:
            return super().unsafe_action_mask(raw_obs)
        finally:
            self.cfg = base_cfg


def ratio_stats(new_log_probs: torch.Tensor, old_log_probs: torch.Tensor,
                clip_eps: float):
    """(approx_kl, clip_fraction) with SB3's estimators."""
    with torch.no_grad():
        log_ratio = new_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        approx_kl = torch.mean((ratio - 1.0) - log_ratio)
        clip_fraction = torch.mean((torch.abs(ratio - 1.0) > clip_eps).float())
    return float(approx_kl.item()), float(clip_fraction.item())


# ---------------------------------------------------------------------------
# Checkpoints

def resolve_checkpoint_path(path: str) -> str:
    """Accept the model name with or without the .pt suffix."""
    if os.path.isfile(path):
        return path
    if os.path.isfile(path + ".pt"):
        return path + ".pt"
    raise FileNotFoundError(f"No checkpoint {path} or {path}.pt")


def read_checkpoint(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Agent mixin

class D2IntegrationMixin:
    """Shared by the v16 and v17 agents (mixed into D2DrivePPO subclasses).

    Policy input: [normalized 75 v15 values, normalized 15 base values,
    4 raw disturbance values, normalized controller values, raw controller
    values]. The controller parts are empty in v16. The shield and the
    DisturbanceEstimator always get the raw 15 base values.

    Call _init_integration() after D2DrivePPO.__init__().
    """

    n_normalized_extra = 0
    n_raw_extra = 0

    def _init_integration(self):
        cfg = self.cfg
        self.shield = SpeedLimitShield(cfg)
        self.obs_normalizer = ObsNormalizer(
            V15_OBS_DIM + D2_BASE_DIM + self.n_normalized_extra, clip=cfg.clip_obs)
        self.reward_scaler = RewardScaler(cfg.gamma, clip=cfg.clip_reward,
                                          enabled=cfg.normalize_reward)
        self.training = True
        self.num_timesteps = 0
        self.feature_dim = self.policy_input_dim

    @property
    def policy_input_dim(self) -> int:
        return (V15_OBS_DIM + D2_BASE_DIM + D2_DISTURBANCE_DIM
                + self.n_normalized_extra + self.n_raw_extra)

    def controller_state(self, info):
        """(values to normalize, raw values) taken from the wrapper's info."""
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    def set_training(self, training: bool):
        """Training: normalization statistics update. Evaluation: frozen."""
        self.training = bool(training)

    def _features(self, obs, info, estimator, update_stats: bool):
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        base = map_v15_to_d2(obs)
        disturbance = estimator.transform(base)[D2_BASE_DIM:]
        norm_extra, raw_extra = self.controller_state(info)
        normalized = self.obs_normalizer.normalize(
            np.concatenate([obs, base, norm_extra]), update=update_stats)
        split = V15_OBS_DIM + D2_BASE_DIM
        policy_input = np.concatenate([
            normalized[:split], disturbance, normalized[split:], raw_extra,
        ]).astype(np.float32)
        return obs, base, policy_input

    def prepare_step(self, obs, info):
        """For act(): advances the estimator history once and (in training)
        the normalization statistics; points the shield at the current limit."""
        obs, base, policy_input = self._features(
            obs, info, self.estimator, update_stats=self.training)
        self.shield.set_speed_limit_kmh(
            speed_limit_kmh(obs, self.cfg.unknown_speed_limit_kmh))
        return obs, base, policy_input

    def peek_input(self, obs, info=None) -> np.ndarray:
        """Policy input for a bootstrap value. Works on a copy of the
        estimator and does not update the statistics, so the next act() on
        the same observation sees the history advanced exactly once."""
        _, _, policy_input = self._features(
            obs, info, copy.deepcopy(self.estimator), update_stats=False)
        return policy_input

    @torch.no_grad()
    def value_of(self, policy_input) -> float:
        x = torch.as_tensor(np.asarray(policy_input, dtype=np.float32),
                            device=self.device).unsqueeze(0)
        return float(self.policy(x)[-1].item())

    def scale_reward(self, reward: float, done: bool) -> float:
        return self.reward_scaler.scale(reward, done, update=self.training)

    def save(self, path: str, extra: dict | None = None):
        """Arnav's checkpoint keys plus normalization, reward scaling,
        num_timesteps and `extra` (e.g. v17's controller settings)."""
        state = {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": dict(self.cfg.__dict__),
            "agent_class": type(self).__name__,
            "obs_normalizer": self.obs_normalizer.state_dict(),
            "reward_scaler": self.reward_scaler.state_dict(),
            "num_timesteps": int(self.num_timesteps),
            "extra": dict(extra or {}),
        }
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        tmp = path + ".tmp"
        torch.save(state, tmp)
        # Replace in one step, so a kill during the save keeps the old file.
        os.replace(tmp, path)

    def load(self, path: str, load_optimizer: bool = True) -> dict:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        saved_class = checkpoint.get("agent_class")
        if saved_class is not None and saved_class != type(self).__name__:
            raise ValueError(f"{path} was saved by {saved_class}, not {type(self).__name__}")
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # The resolved config decides the learning rate (it may have
            # been changed on the command line for this run).
            for group in self.optimizer.param_groups:
                group["lr"] = self.cfg.learning_rate
        if "obs_normalizer" in checkpoint:
            self.obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])
        if "reward_scaler" in checkpoint:
            self.reward_scaler.load_state_dict(checkpoint["reward_scaler"])
        self.num_timesteps = int(checkpoint.get("num_timesteps", 0))
        self._after_load()
        return checkpoint

    def _after_load(self):
        pass
