"""v16: D2-Drive with v15's continuous action interface.

The policy outputs [steer, longitudinal] in [-1, 1] exactly like the SB3
agent of v15; CarlaEnv's own pedal mapping (0 = light throttle, coasting
band, braking) is used unchanged. What comes from Arnav's d2drive_ppo.py:
the ActorCritic network (2 Gaussian means instead of 6 logits), the
DisturbanceEstimator, the shield's unsafe-action mask (turned into limits on
the continuous action), the shield-distillation penalty (in a continuous
form), PPO clipping and his GAE. See INTEGRATION_NOTES.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from d2drive_common import (
    COAST_LONGITUDINAL,
    D2IntegrationConfig,
    D2IntegrationMixin,
    SpeedLimitShield,
    configure_torch,
    ratio_stats,
)
from d2drive_ppo import ActorCritic, D2DrivePPO, Transition
from d2drive_training import DecisionStats

STEER, LONGITUDINAL = 0, 1
ACTION_DIM = 2
SHIELD_RULES = ("acc_unsafe", "left_active", "right_active")


@dataclass
class V16Config(D2IntegrationConfig):
    n_actions: int = ACTION_DIM       # the actor head gives the 2 Gaussian means
    log_std_init: float = -1.0        # as v15 (std ~0.37)
    min_std: float = 0.05             # v15's std fell to 0.029 and the policy collapsed
    lateral_shield_offset_m: float = 0.6


# ---------------------------------------------------------------------------
# Policy

class GaussianActorCritic(nn.Module):
    """Arnav's ActorCritic with n_actions=2: the actor output is the mean of
    a diagonal Gaussian over [steer, longitudinal]. The log std is a separate
    state-independent parameter, kept at or above log(min_std)."""

    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int = ACTION_DIM,
                 log_std_init: float = -1.0, min_std: float = 0.05):
        super().__init__()
        self.net = ActorCritic(input_dim, hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))
        self.min_log_std = math.log(float(min_std))

    def effective_log_std(self) -> torch.Tensor:
        clamped = torch.clamp(self.log_std, min=self.min_log_std)
        # Straight-through: the value is clamped, but the gradient always
        # reaches the parameter (a plain clamp has zero gradient at/below the
        # floor, so a std that hit the floor could never grow again).
        return self.log_std + (clamped - self.log_std).detach()

    def forward(self, x):
        mean, value = self.net(x)
        return mean, self.effective_log_std().expand_as(mean), value

    def distribution(self, x):
        mean, log_std, value = self(x)
        return Normal(mean, log_std.exp()), value

    def std(self) -> torch.Tensor:
        return torch.clamp(self.log_std, min=self.min_log_std).exp().detach()

    @torch.no_grad()
    def project_log_std(self):
        """Pull the parameter back to the floor after an optimizer step, so
        it never drifts below it (with Adam's momentum it otherwise could)."""
        self.log_std.clamp_(min=self.min_log_std)


def continuous_shield_loss(dist: Normal, flags: torch.Tensor) -> torch.Tensor:
    """Continuous form of Arnav's shield-distillation penalty ("probability
    given to unsafe actions"), with the current policy's Normal:

      mean[ acc_unsafe  * P(longitudinal > -0.45)
          + left_active * P(steer < 0)
          + right_active * P(steer > 0) ]

    The thresholds are inside [-1, 1], so the probabilities are the same for
    the clipped action that reaches the environment.
    """
    mean, std = dist.mean, dist.stddev
    steer = Normal(mean[:, STEER], std[:, STEER])
    longitudinal = Normal(mean[:, LONGITUDINAL], std[:, LONGITUDINAL])
    zero = torch.zeros_like(mean[:, STEER])
    p_accelerate = 1.0 - longitudinal.cdf(torch.full_like(zero, COAST_LONGITUDINAL))
    p_left = steer.cdf(zero)
    p_right = 1.0 - steer.cdf(zero)
    penalty = flags[:, 0] * p_accelerate + flags[:, 1] * p_left + flags[:, 2] * p_right
    return penalty.mean()


# ---------------------------------------------------------------------------
# Shield

class ContinuousShield:
    """Arnav's unsafe-action mask as limits on [steer, longitudinal].

    It only restricts the action, never adds a behaviour:
      mask[1] (accelerate unsafe): longitudinal = min(l, -0.45), i.e. no
          throttle; the agent's own stronger braking is kept.
      mask[3] (left change unsafe) and already drifting left
          (offset < -D and heading < 0): steer = max(steer, 0).
      mask[4] (right change unsafe) and already drifting right
          (offset > D and heading > 0): steer = min(steer, 0).
    v15 gives offset and heading 0 inside junctions and off the lane, so the
    lateral rules are off there and turns are never blocked.
    """

    def __init__(self, mask_shield: SpeedLimitShield, lateral_offset_m: float = 0.6):
        self.mask_shield = mask_shield
        self.lateral_offset_m = float(lateral_offset_m)

    def filter(self, proposed, base):
        """proposed: clipped [steer, longitudinal]; base: raw D2 15-vector.
        Returns (executed, flags[acc_unsafe, left_active, right_active], vetoed)."""
        mask = self.mask_shield.unsafe_action_mask(base)
        offset, heading = float(base[1]), float(base[2])
        acc_unsafe = mask[1] >= 0.5
        left_active = (mask[3] >= 0.5 and offset < -self.lateral_offset_m
                       and heading < 0.0)
        right_active = (mask[4] >= 0.5 and offset > self.lateral_offset_m
                        and heading > 0.0)
        steer, longitudinal = float(proposed[STEER]), float(proposed[LONGITUDINAL])
        if acc_unsafe:
            longitudinal = min(longitudinal, COAST_LONGITUDINAL)
        if left_active:
            steer = max(steer, 0.0)
        if right_active:
            steer = min(steer, 0.0)
        executed = np.array([steer, longitudinal], dtype=np.float32)
        vetoed = bool(np.max(np.abs(executed - np.asarray(proposed, dtype=np.float32))) > 1e-6)
        flags = np.array([acc_unsafe, left_active, right_active], dtype=np.float32)
        return executed, flags, vetoed


# ---------------------------------------------------------------------------
# Agent

class D2DriveContinuousAgent(D2IntegrationMixin, D2DrivePPO):
    """D2DrivePPO with a Gaussian policy over v15's [steer, longitudinal].

    Policy input (94): 75 normalized v15 values, 15 normalized base values,
    4 raw disturbance values. act(), store() and update() are overridden
    because Arnav's versions are written for a Categorical policy.
    """

    def __init__(self, cfg: V16Config | None = None):
        cfg = cfg or V16Config()
        if cfg.n_actions != ACTION_DIM:
            raise ValueError("v16 needs n_actions=2 (steer, longitudinal)")
        configure_torch(cfg)
        super().__init__(cfg)   # Arnav's setup; the network is replaced below
        self._init_integration()
        self.policy = GaussianActorCritic(
            self.policy_input_dim, cfg.hidden_dim, ACTION_DIM,
            cfg.log_std_init, cfg.min_std).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.learning_rate)
        self.continuous_shield = ContinuousShield(self.shield, cfg.lateral_shield_offset_m)

    def _after_load(self):
        # --min-std may have been raised for this run.
        self.policy.project_log_std()

    @torch.no_grad()
    def act(self, obs, info=None, deterministic=False):
        _, base, policy_input = self.prepare_step(obs, info)
        x = torch.as_tensor(policy_input, device=self.device).unsqueeze(0)
        dist, value = self.policy.distribution(x)
        sample = dist.mean if deterministic else dist.sample()
        # SB3's method: the log-probability of the unclipped sample.
        log_prob = dist.log_prob(sample).sum(-1)
        raw_action = sample.squeeze(0).cpu().numpy().astype(np.float32)
        proposed = np.clip(raw_action, -1.0, 1.0)
        executed, flags, vetoed = self.continuous_shield.filter(proposed, base)
        log_prob_executed = dist.log_prob(
            torch.as_tensor(executed, device=self.device).unsqueeze(0)).sum(-1)
        return {
            "raw_action": raw_action,
            "proposed_action": proposed,
            "executed_action": executed,
            "log_prob": float(log_prob.item()),
            "log_prob_executed": float(log_prob_executed.item()),
            "value": float(value.item()),
            "augmented_obs": policy_input,
            "base_obs": base,
            "unsafe_mask": flags,
            "shield_flags": flags,
            "vetoed": vetoed,
        }

    def store(self, obs, decision, reward, done):
        if self.cfg.store_executed_action:
            action, log_prob = decision["executed_action"], decision["log_prob_executed"]
        else:
            # Default, as Arnav: the policy's own (unclipped) proposal.
            action, log_prob = decision["raw_action"], decision["log_prob"]
        self.buffer.add(Transition(
            obs=np.asarray(decision["augmented_obs"], dtype=np.float32),
            action=np.asarray(action, dtype=np.float32),
            log_prob=float(log_prob),
            reward=float(reward),
            done=bool(done),
            value=float(decision["value"]),
            unsafe_mask=np.asarray(decision["shield_flags"], dtype=np.float32),
        ))

    def update(self, last_augmented_obs):
        """Arnav's update() with a Gaussian instead of a Categorical; same
        terms: clipped surrogate, MSE value loss, entropy bonus, shield
        penalty, gradient clipping, advantage normalization, his GAE."""
        if not self.buffer.data:
            return {}
        cfg = self.cfg
        advantages, returns = self.buffer.compute_gae(self.value_of(last_augmented_obs), cfg)

        data = self.buffer.data
        device = self.device
        obs = torch.as_tensor(np.stack([t.obs for t in data]), dtype=torch.float32, device=device)
        actions = torch.as_tensor(np.stack([t.action for t in data]),
                                  dtype=torch.float32, device=device)
        old_log_probs = torch.as_tensor(np.array([t.log_prob for t in data], dtype=np.float32),
                                        device=device)
        adv = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        ret = torch.as_tensor(returns, dtype=torch.float32, device=device)
        flags = torch.as_tensor(np.stack([t.unsafe_mask for t in data]),
                                dtype=torch.float32, device=device)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        n = obs.shape[0]
        idx = np.arange(n)
        sums = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "shield_loss": 0.0}
        updates = 0
        clip_fractions = []
        approx_kls = []
        for _ in range(cfg.update_epochs):
            approx_kls = []  # SB3 logs the last epoch's mean
            np.random.shuffle(idx)
            for start in range(0, n, cfg.minibatch_size):
                b = torch.as_tensor(idx[start:start + cfg.minibatch_size],
                                    dtype=torch.long, device=device)
                dist, values = self.policy.distribution(obs[b])
                new_log_probs = dist.log_prob(actions[b]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                log_ratio = new_log_probs - old_log_probs[b]
                ratio = torch.exp(log_ratio)
                surr1 = ratio * adv[b]
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv[b]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values, ret[b])
                shield_loss = continuous_shield_loss(dist, flags[b])

                loss = (policy_loss + cfg.value_coef * value_loss
                        - cfg.entropy_coef * entropy + cfg.shield_coef * shield_loss)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                self.policy.project_log_std()

                with torch.no_grad():
                    approx_kls.append(float(torch.mean((ratio - 1.0) - log_ratio)))
                    clip_fractions.append(float(torch.mean(
                        (torch.abs(ratio - 1.0) > cfg.clip_eps).float())))
                sums["policy_loss"] += float(policy_loss.item())
                sums["value_loss"] += float(value_loss.item())
                sums["entropy"] += float(entropy.item())
                sums["shield_loss"] += float(shield_loss.item())
                updates += 1

        metrics = {k: v / max(1, updates) for k, v in sums.items()}
        metrics["approx_kl"] = float(np.mean(approx_kls)) if approx_kls else 0.0
        metrics["clip_fraction"] = float(np.mean(clip_fractions)) if clip_fractions else 0.0
        # The same estimators measured once after the update over the whole
        # rollout (comparable with v17).
        with torch.no_grad():
            dist, _ = self.policy.distribution(obs)
            new_log_probs = dist.log_prob(actions).sum(-1)
        metrics["approx_kl_post"], metrics["clip_fraction_post"] = ratio_stats(
            new_log_probs, old_log_probs, cfg.clip_eps)
        std = self.policy.std().cpu().numpy()
        metrics["std_steer"] = float(std[STEER])
        metrics["std_longitudinal"] = float(std[LONGITUDINAL])
        self.buffer.clear()
        return metrics


# ---------------------------------------------------------------------------
# Wrapper and statistics

class D2ContinuousWrapper(gym.Wrapper):
    """Gives the shielded [steer, longitudinal] to CarlaEnv.step() unchanged.

    Reward, terminated, truncated and CarlaEnv's info keys come back
    unchanged; only d2_* keys are added. Observation and action spaces are
    CarlaEnv's. The wrapper never touches CarlaEnv's internals.
    """

    def step(self, action, decision=None):
        executed = np.asarray(action, dtype=np.float32).reshape(-1)
        obs, reward, terminated, truncated, info = self.env.step(executed)
        info = dict(info)
        if decision is None:
            proposed, vetoed = executed, False
            flags = np.zeros(len(SHIELD_RULES), dtype=np.float32)
        else:
            proposed = decision["proposed_action"]
            vetoed = decision["vetoed"]
            flags = decision["shield_flags"]
        info["d2_proposed_action"] = [float(v) for v in proposed]
        info["d2_executed_action"] = [float(v) for v in executed]
        info["d2_vetoed"] = bool(vetoed)
        info["d2_shield_flags"] = {name: bool(flag >= 0.5)
                                   for name, flag in zip(SHIELD_RULES, flags)}
        return obs, reward, terminated, truncated, info


class ContinuousDecisionStats(DecisionStats):
    """v16 logs: veto rate, how often each shield rule acted, mean |steer|."""

    def reset(self):
        super().reset()
        self.rule_counts = np.zeros(len(SHIELD_RULES))
        self.abs_steer = 0.0
        self.longitudinal = 0.0

    def on_step(self, decision, info):
        super().on_step(decision, info)
        self.rule_counts += np.asarray(decision["shield_flags"], dtype=np.float64)
        executed = decision["executed_action"]
        self.abs_steer += abs(float(executed[STEER]))
        self.longitudinal += float(executed[LONGITUDINAL])

    def record(self):
        out = super().record()
        for name, count in zip(SHIELD_RULES, self.rule_counts):
            out[f"shield/{name}_rate"] = self._rate(count)
        out["actions/mean_abs_steer"] = self._rate(self.abs_steer)
        out["actions/mean_longitudinal"] = self._rate(self.longitudinal)
        return out
