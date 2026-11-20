"""
D2-Drive: Disturbance-Guided Decision Reinforcement Learning for Autonomous Driving
Standalone PPO implementation with:
  - disturbance / temporal-stability feature augmentation
  - discrete high-level driving actions
  - adaptive rule-based safety shield
  - shield-distillation penalty
  - PPO + GAE

Expected base observation (float vector, 15 values):
  0  ego_speed_mps
  1  lane_offset_m                 (positive = right of lane center)
  2  heading_error_rad
  3  front_gap_m
  4  front_closing_speed_mps       (positive = ego closing on front vehicle)
  5  left_front_gap_m
  6  left_rear_gap_m
  7  left_front_closing_mps
  8  left_rear_closing_mps        (positive = rear vehicle closing on ego)
  9  right_front_gap_m
  10 right_rear_gap_m
  11 right_front_closing_mps
  12 right_rear_closing_mps
  13 traffic_density               (normalized 0..1)
  14 speed_spread_mps

High-level actions:
  0 KEEP_LANE
  1 ACCELERATE
  2 DECELERATE
  3 CHANGE_LEFT
  4 CHANGE_RIGHT
  5 ABORT

Environment action mapping is the responsibility of the CARLA Gymnasium wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ----------------------------- Configuration -----------------------------

@dataclass
class D2DriveConfig:
    obs_dim: int = 15
    hidden_dim: int = 256
    n_actions: int = 6

    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    shield_coef: float = 0.15
    max_grad_norm: float = 0.5

    rollout_steps: int = 2048
    minibatch_size: int = 256
    update_epochs: int = 10
    total_timesteps: int = 500_000

    disturbance_window: int = 20
    target_speed_mps: float = 11.11  # 40 km/h
    max_speed_mps: float = 16.67     # 60 km/h
    min_speed_mps: float = 2.0

    base_safe_gap_m: float = 5.0
    min_headway_s: float = 1.0
    max_headway_s: float = 2.0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


ACTION_NAMES = (
    "KEEP_LANE",
    "ACCELERATE",
    "DECELERATE",
    "CHANGE_LEFT",
    "CHANGE_RIGHT",
    "ABORT",
)


# ------------------------- Disturbance Estimator -------------------------

class DisturbanceEstimator:
    """
    Adds four temporal/context features to the 15-D observation:
      disturbance magnitude A_t
      disturbance frequency f_t (normalized)
      stability S_t
      risk trend Delta R_t

    These are engineering estimates from the observation stream, not learned
    sensor uncertainty. Replace/augment them with perception uncertainty if
    your CARLA pipeline exposes it.
    """

    def __init__(self, window: int = 20):
        self.window = max(3, int(window))
        self.history = deque(maxlen=self.window)
        self.risk_history = deque(maxlen=self.window)

    def reset(self):
        self.history.clear()
        self.risk_history.clear()

    @staticmethod
    def _clip01(x):
        return float(np.clip(x, 0.0, 1.0))

    def _risk(self, obs: np.ndarray) -> float:
        speed = max(0.0, float(obs[0]))
        lane_offset = abs(float(obs[1]))
        front_gap = max(0.0, float(obs[3]))
        front_closing = max(0.0, float(obs[4]))
        density = self._clip01(obs[13])
        spread = max(0.0, float(obs[14]))

        lane_risk = self._clip01(lane_offset / 1.5)
        ttc = front_gap / max(front_closing, 0.1)
        ttc_risk = self._clip01((3.0 - ttc) / 3.0)
        proximity_risk = self._clip01((20.0 - front_gap) / 20.0)
        speed_risk = self._clip01(speed / 20.0)
        traffic_risk = self._clip01(0.6 * density + 0.4 * spread / 10.0)

        return float(np.clip(
            0.30 * lane_risk + 0.30 * ttc_risk +
            0.15 * proximity_risk + 0.10 * speed_risk +
            0.15 * traffic_risk, 0.0, 1.0
        ))

    def transform(self, raw_obs: Sequence[float]) -> np.ndarray:
        obs = np.asarray(raw_obs, dtype=np.float32).flatten()
        if obs.size != 15:
            raise ValueError(f"D2-Drive expects 15 base observation values; got {obs.size}")

        # Normalize a compact disturbance proxy from traffic density, speed
        # variation, relative closing speeds, and changing gaps.
        rel_speeds = np.abs(obs[[4, 7, 8, 11, 12]])
        closing_component = self._clip01(float(np.mean(rel_speeds)) / 8.0)
        density_component = self._clip01(float(obs[13]))
        spread_component = self._clip01(float(obs[14]) / 10.0)
        gap_component = self._clip01((25.0 - max(0.0, float(obs[3]))) / 25.0)
        disturbance = float(np.clip(
            0.30 * density_component + 0.25 * spread_component +
            0.25 * closing_component + 0.20 * gap_component, 0.0, 1.0
        ))

        current = np.array([
            disturbance,
            density_component,
            spread_component,
            closing_component,
            float(obs[1]),
            float(obs[2]),
            float(obs[3]),
        ], dtype=np.float32)
        self.history.append(current)

        if len(self.history) >= 2:
            hist = np.stack(self.history)
            # Frequency proxy: average normalized variation over the window.
            diffs = np.abs(np.diff(hist[:, 0]))
            disturbance_frequency = self._clip01(float(np.mean(diffs)) * 10.0)
            # Stability: low temporal variance in the traffic/context vector.
            std = np.std(hist[:, :4], axis=0)
            stability = self._clip01(1.0 - float(np.mean(std)) * 3.0)
        else:
            disturbance_frequency = 0.0
            stability = 1.0

        risk = self._risk(obs)
        self.risk_history.append(risk)
        risk_trend = 0.0
        if len(self.risk_history) >= 2:
            risk_trend = float(np.clip(
                self.risk_history[-1] - self.risk_history[-2], -1.0, 1.0
            ))

        augmented = np.concatenate([
            obs,
            np.array([disturbance, disturbance_frequency, stability, risk_trend],
                     dtype=np.float32)
        ])
        return augmented.astype(np.float32)


# ------------------------------ Actor-Critic ----------------------------

class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_actions: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        h = self.shared(x)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value


# ------------------------------ Safety Shield ----------------------------

class AdaptiveSafetyShield:
    """
    Deterministic action filter. Unsafe policy proposals are replaced with
    DECELERATE or KEEP_LANE.

    This is a baseline shield; validate and calibrate it in CARLA before
    treating it as a safety guarantee.
    """

    def __init__(self, cfg: D2DriveConfig):
        self.cfg = cfg

    def safe_gap(self, obs: np.ndarray) -> float:
        speed = max(0.0, float(obs[0]))
        density = float(np.clip(obs[13], 0.0, 1.0))
        spread_norm = float(np.clip(float(obs[14]) / 10.0, 0.0, 1.0))
        tau = float(np.clip(
            1.0 + 0.5 * density + 0.5 * spread_norm,
            self.cfg.min_headway_s,
            self.cfg.max_headway_s
        ))
        return self.cfg.base_safe_gap_m + speed * tau

    @staticmethod
    def _lane_change_safe(front_gap, rear_gap, front_closing, rear_closing,
                         safe_gap):
        # Require space both ahead and behind. TTC checks protect against
        # rapidly closing vehicles even if the current gap is large.
        front_ttc = front_gap / max(front_closing, 0.1)
        rear_ttc = rear_gap / max(rear_closing, 0.1)
        return (
            front_gap >= safe_gap and rear_gap >= safe_gap and
            front_ttc >= 2.0 and rear_ttc >= 2.0
        )

    def unsafe_action_mask(self, raw_obs: Sequence[float]) -> np.ndarray:
        obs = np.asarray(raw_obs, dtype=np.float32).flatten()
        mask = np.zeros(6, dtype=np.float32)
        safe_gap = self.safe_gap(obs)

        # Accelerating is discouraged if the front gap is short or TTC low.
        front_gap = max(0.0, float(obs[3]))
        front_closing = max(0.0, float(obs[4]))
        front_ttc = front_gap / max(front_closing, 0.1)
        if front_gap < safe_gap or front_ttc < 2.0 or obs[0] >= self.cfg.max_speed_mps:
            mask[1] = 1.0

        # Decelerating is not unsafe in this simplified shield.
        # Left lane change: indexes 5-8
        mask[3] = 0.0 if self._lane_change_safe(
            float(obs[5]), float(obs[6]),
            max(0.0, float(obs[7])), max(0.0, float(obs[8])),
            safe_gap
        ) else 1.0

        # Right lane change: indexes 9-12
        mask[4] = 0.0 if self._lane_change_safe(
            float(obs[9]), float(obs[10]),
            max(0.0, float(obs[11])), max(0.0, float(obs[12])),
            safe_gap
        ) else 1.0

        # ABORT is allowed; KEEP_LANE is the fallback.
        return mask

    def filter_action(self, proposed_action: int, raw_obs: Sequence[float]):
        unsafe_mask = self.unsafe_action_mask(raw_obs)
        if unsafe_mask[proposed_action] < 0.5:
            return int(proposed_action), unsafe_mask, False

        # If lane change / acceleration is vetoed, use the conservative action.
        fallback = 2 if proposed_action in (1, 3, 4) else 0
        return fallback, unsafe_mask, True


# ------------------------------- Rollout ---------------------------------

@dataclass
class Transition:
    obs: np.ndarray
    action: int
    log_prob: float
    reward: float
    done: bool
    value: float
    unsafe_mask: np.ndarray


class RolloutBuffer:
    def __init__(self):
        self.data = []

    def add(self, transition: Transition):
        self.data.append(transition)

    def clear(self):
        self.data.clear()

    def compute_gae(self, last_value: float, cfg: D2DriveConfig):
        rewards = np.array([t.reward for t in self.data], dtype=np.float32)
        dones = np.array([t.done for t in self.data], dtype=np.float32)
        values = np.array([t.value for t in self.data] + [last_value], dtype=np.float32)

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + cfg.gamma * values[t + 1] * nonterminal - values[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * nonterminal * gae
            advantages[t] = gae
        returns = advantages + values[:-1]
        return advantages, returns


# ------------------------------- PPO Agent -------------------------------

class D2DrivePPO:
    def __init__(self, cfg: Optional[D2DriveConfig] = None):
        self.cfg = cfg or D2DriveConfig()
        self.device = torch.device(self.cfg.device)
        self.feature_dim = self.cfg.obs_dim + 4
        self.policy = ActorCritic(
            self.feature_dim, self.cfg.hidden_dim, self.cfg.n_actions
        ).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(),
                                   lr=self.cfg.learning_rate)
        self.estimator = DisturbanceEstimator(self.cfg.disturbance_window)
        self.shield = AdaptiveSafetyShield(self.cfg)
        self.buffer = RolloutBuffer()

    def reset_episode(self):
        self.estimator.reset()

    def get_augmented_obs(self, raw_obs):
        return self.estimator.transform(raw_obs)

    @torch.no_grad()
    def act(self, raw_obs, deterministic=False):
        raw = np.asarray(raw_obs, dtype=np.float32)
        augmented = self.get_augmented_obs(raw)
        x = torch.as_tensor(augmented, dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        logits, value = self.policy(x)
        dist = Categorical(logits=logits)
        proposed = int(torch.argmax(logits, dim=-1).item()) if deterministic \
            else int(dist.sample().item())
        log_prob = float(dist.log_prob(torch.tensor([proposed], device=self.device)).item())

        executed, unsafe_mask, vetoed = self.shield.filter_action(proposed, raw)
        return {
            "proposed_action": proposed,
            "executed_action": executed,
            "log_prob": log_prob,
            "value": float(value.item()),
            "augmented_obs": augmented,
            "unsafe_mask": unsafe_mask,
            "vetoed": vetoed,
        }

    def store(self, obs, decision, reward, done):
        self.buffer.add(Transition(
            obs=np.asarray(decision["augmented_obs"], dtype=np.float32),
            action=int(decision["proposed_action"]),
            log_prob=float(decision["log_prob"]),
            reward=float(reward),
            done=bool(done),
            value=float(decision["value"]),
            unsafe_mask=np.asarray(decision["unsafe_mask"], dtype=np.float32),
        ))

    def update(self, last_augmented_obs):
        if not self.buffer.data:
            return {}

        with torch.no_grad():
            x_last = torch.as_tensor(last_augmented_obs, dtype=torch.float32,
                                     device=self.device).unsqueeze(0)
            _, last_value_t = self.policy(x_last)
            last_value = float(last_value_t.item())

        advantages, returns = self.buffer.compute_gae(last_value, self.cfg)

        obs = torch.tensor(np.stack([t.obs for t in self.buffer.data],
                                    axis=0), dtype=torch.float32, device=self.device)
        actions = torch.tensor([t.action for t in self.buffer.data],
                               dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor([t.log_prob for t in self.buffer.data],
                                      dtype=torch.float32, device=self.device)
        adv = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        ret = torch.tensor(returns, dtype=torch.float32, device=self.device)
        unsafe = torch.tensor(np.stack([t.unsafe_mask for t in self.buffer.data]),
                              dtype=torch.float32, device=self.device)

        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        n = obs.shape[0]
        idx = np.arange(n)
        metrics = {"policy_loss": 0.0, "value_loss": 0.0,
                   "entropy": 0.0, "shield_loss": 0.0}
        updates = 0

        for _ in range(self.cfg.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.cfg.minibatch_size):
                batch_idx = idx[start:start + self.cfg.minibatch_size]
                b = torch.as_tensor(batch_idx, dtype=torch.long, device=self.device)

                logits, values = self.policy(obs[b])
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions[b])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs[b])
                surr1 = ratio * adv[b]
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps,
                                    1.0 + self.cfg.clip_eps) * adv[b]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values, ret[b])

                probs = torch.softmax(logits, dim=-1)
                # Penalize probability assigned to actions that the shield
                # identified as unsafe at collection time.
                shield_loss = (probs * unsafe[b]).sum(dim=-1).mean()

                loss = (
                    policy_loss +
                    self.cfg.value_coef * value_loss -
                    self.cfg.entropy_coef * entropy +
                    self.cfg.shield_coef * shield_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(),
                                         self.cfg.max_grad_norm)
                self.optimizer.step()

                metrics["policy_loss"] += float(policy_loss.item())
                metrics["value_loss"] += float(value_loss.item())
                metrics["entropy"] += float(entropy.item())
                metrics["shield_loss"] += float(shield_loss.item())
                updates += 1

        for k in metrics:
            metrics[k] /= max(1, updates)
        self.buffer.clear()
        return metrics

    def save(self, path):
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.cfg.__dict__,
        }, path)

    def load(self, path, load_optimizer=True):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


# --------------------------- Training Interface ---------------------------

def train_d2drive(env, cfg: Optional[D2DriveConfig] = None,
                  save_path: str = "d2drive_ppo.pt"):
    """
    Train against a Gymnasium environment whose observation is the 15-D vector
    described above and whose action is the executed high-level action index.

    The environment should interpret:
      0 keep lane, 1 accelerate, 2 decelerate, 3 change left,
      4 change right, 5 abort.

    env.step(action) must return Gymnasium:
      obs, reward, terminated, truncated, info

    IMPORTANT: the shield runs inside this agent. The environment receives
    decision["executed_action"], not the proposed action.
    """
    cfg = cfg or D2DriveConfig()
    agent = D2DrivePPO(cfg)
    obs, info = env.reset()
    agent.reset_episode()

    total_steps = 0
    episode_reward = 0.0
    episode_count = 0
    recent_rewards = deque(maxlen=20)

    while total_steps < cfg.total_timesteps:
        for _ in range(cfg.rollout_steps):
            decision = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(
                decision["executed_action"]
            )
            done = bool(terminated or truncated)

            agent.store(obs, decision, reward, done)
            obs = next_obs
            episode_reward += float(reward)
            total_steps += 1

            if done:
                episode_count += 1
                recent_rewards.append(episode_reward)
                print(
                    f"steps={total_steps} episode={episode_count} "
                    f"reward={episode_reward:.2f} "
                    f"recent_mean={np.mean(recent_rewards):.2f}"
                )
                obs, info = env.reset()
                agent.reset_episode()
                episode_reward = 0.0

            if total_steps >= cfg.total_timesteps:
                break

        # Transform the current observation to provide bootstrap value.
        # This advances estimator history; use it as the next state feature.
        last_aug = agent.get_augmented_obs(obs)
        metrics = agent.update(last_aug)
        print("PPO update:", metrics)

        agent.save(save_path)

    env.close()
    return agent


if __name__ == "__main__":
    print("D2-Drive module loaded.")
    print("Import train_d2drive(env, cfg) from your CARLA Gymnasium wrapper.")
