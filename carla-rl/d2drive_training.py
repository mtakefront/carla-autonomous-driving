"""Shared training and evaluation loops for D2-Drive on the v15 environment.

This file is identical in v16 and v17. It does not import carla or carla_env:
train_vXX.py / evaluate_vXX.py create the CarlaEnv and pass it in, so the
tests can run these loops on a fake environment.

The agent interface used here (D2IntegrationMixin + a version subclass):
  act(obs, info, deterministic) -> decision dict ("executed_action", "vetoed", ...)
  store(obs, decision, reward, done), update(last_policy_input) -> metrics
  peek_input(obs, info), value_of(policy_input), scale_reward(reward, done)
  reset_episode(), set_training(flag), save(path, extra), num_timesteps
The wrapper's step() takes the executed action plus decision=... (it only
adds d2_* keys to info).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import os
import time
from collections import Counter, deque

import numpy as np

# Same names as train_v15.py's TerminationStatsCallback.
REASONS = ("goal", "collision", "off_road", "wrong_road", "wrong_way", "red_light",
           "stop_sign", "stuck", "time_limit")
REWARD_PARTS = ("progress", "speed", "step", "illegal_crossing", "lane", "terminal")
MANEUVERS = ("LEFT", "RIGHT", "STRAIGHT")
CHECKPOINT_FREQ = 10_000


# ---------------------------------------------------------------------------
# Statistics

class TerminationStats:
    """train_v15.py's TerminationStatsCallback, computed the same way: how
    episodes ended in this rollout, junction passes per maneuver (this
    rollout and the whole run) and the average per-episode reward terms."""

    def __init__(self):
        self.counts = Counter()
        self.parts = Counter()
        self.junctions = Counter()        # this rollout
        self.junctions_total = Counter()  # whole run

    def rollout_start(self):
        self.counts.clear()
        self.parts.clear()
        self.junctions.clear()

    def on_step(self, info, done):
        for maneuver, passed in info.get("maneuver_results", ()):
            self.junctions[(maneuver, passed)] += 1
            self.junctions_total[(maneuver, passed)] += 1
        if done:
            self.counts[info.get("termination_reason") or "unknown"] += 1
            for name, value in (info.get("reward_parts") or {}).items():
                self.parts[name] += value

    def record(self) -> dict:
        out = {}
        for reason in REASONS:
            out[f"episodes/{reason}"] = self.counts[reason]
        episodes = sum(self.counts.values())
        for name in REWARD_PARTS:
            out[f"reward_parts/{name}"] = (
                round(self.parts[name] / episodes, 1) if episodes else None)
        for m in MANEUVERS:
            ok, fail = self.junctions[(m, True)], self.junctions[(m, False)]
            out[f"junctions/{m}"] = f"{ok}/{ok + fail}"
            ok_t, fail_t = self.junctions_total[(m, True)], self.junctions_total[(m, False)]
            rate = ok_t / (ok_t + fail_t) if ok_t + fail_t else 0.0
            out[f"junctions_total/{m}_pass_rate"] = round(rate, 3)
        return out


class DecisionStats:
    """Per-rollout statistics of the agent's decisions. The versions extend
    it (v16: steering and shield rules, v17: commands and lane changes)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.steps = 0
        self.vetoes = 0

    def on_step(self, decision: dict, info: dict):
        self.steps += 1
        self.vetoes += int(bool(decision.get("vetoed")))

    def _rate(self, count):
        return round(count / self.steps, 4) if self.steps else None

    def record(self) -> dict:
        return {"shield/veto_rate": self._rate(self.vetoes)}


# ---------------------------------------------------------------------------
# Output

def _format_value(value):
    if isinstance(value, float):
        return f"{value:<8.3g}"
    return str(value)


def format_table(metrics: dict) -> str:
    """SB3-style table, keys grouped by their prefix."""
    rows, group = [], None
    for key in sorted(metrics):
        value = metrics[key]
        if value is None:
            continue
        prefix, _, name = key.rpartition("/")
        if prefix != group:
            rows.append((prefix + "/", ""))
            group = prefix
        rows.append(("    " + name, _format_value(value)))
    if not rows:
        return ""
    key_w = max(len(k) for k, _ in rows)
    val_w = max(len(v) for _, v in rows)
    line = "-" * (key_w + val_w + 7)
    body = [f"| {k:<{key_w}} | {v:<{val_w}} |" for k, v in rows]
    return "\n".join([line, *body, line])


class MetricsLogger:
    """Prints each row as a table and appends it to a CSV file."""

    def __init__(self, csv_path: str, print_fn=print):
        self.csv_path = csv_path
        self.print_fn = print_fn
        self.fieldnames = None
        self.rows = []

    def _open_csv(self, metrics):
        self.fieldnames = list(metrics)
        if os.path.exists(self.csv_path):
            with open(self.csv_path, newline="") as f:
                header = next(csv.reader(f), None)
            if header != self.fieldnames:
                # A file from another version/config: don't mix columns.
                base, ext = os.path.splitext(self.csv_path)
                n = 1
                while os.path.exists(f"{base}_{n}{ext}"):
                    n += 1
                old, self.csv_path = self.csv_path, f"{base}_{n}{ext}"
                self.print_fn(f"{old} has different columns; writing {self.csv_path}")
        directory = os.path.dirname(os.path.abspath(self.csv_path))
        os.makedirs(directory, exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(self.fieldnames)

    def write(self, metrics: dict):
        self.rows.append(dict(metrics))
        self.print_fn(format_table(metrics))
        if self.fieldnames is None:
            self._open_csv(metrics)
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                ["" if metrics.get(k) is None else metrics.get(k) for k in self.fieldnames])


# ---------------------------------------------------------------------------
# Command line and config

COMMON_CLI_FIELDS = {
    # argparse dest -> config field
    "rollout_steps": "rollout_steps",
    "minibatch_size": "minibatch_size",
    "epochs": "update_epochs",
    "lr": "learning_rate",
    "ent_coef": "entropy_coef",
    "shield_coef": "shield_coef",
    "reward_norm": "normalize_reward",
    "store_executed_action": "store_executed_action",
}


def add_common_train_args(parser: argparse.ArgumentParser, model_name: str):
    """train_v15.py's options plus the D2-Drive ones. The hyperparameter
    options default to None: on --resume only the ones given here override
    the checkpoint."""
    parser.add_argument("--steps", type=int, default=50_000,
                        help="timesteps to train in this run")
    parser.add_argument("--traffic", type=int, default=15, help="autopilot cars")
    parser.add_argument("--pedestrians", type=int, default=20)
    parser.add_argument("--no-render", action="store_true",
                        help="CARLA draws nothing: faster, but no Live View")
    parser.add_argument("--resume", default=None,
                        help=f"continue from this saved model, e.g. {model_name}")
    parser.add_argument("--rollout-steps", type=int, default=None,
                        help="steps per PPO update (default 2048)")
    parser.add_argument("--minibatch-size", type=int, default=None, help="default 256")
    parser.add_argument("--epochs", type=int, default=None,
                        help="PPO epochs per update (default 10)")
    parser.add_argument("--lr", type=float, default=None, help="learning rate (default 3e-4)")
    parser.add_argument("--ent-coef", type=float, default=None,
                        help="entropy coefficient (default 0.01)")
    parser.add_argument("--shield-coef", type=float, default=None,
                        help="shield-distillation coefficient (default 0.15)")
    parser.add_argument("--reward-norm", action=argparse.BooleanOptionalAction, default=None,
                        help="scale rewards like VecNormalize (default on; "
                             "--no-reward-norm turns it off)")
    parser.add_argument("--store-executed-action", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="store the shield's executed action in the PPO buffer instead "
                             "of the policy's proposal (default off, as Arnav's code)")


def resolve_dataclass(cls, saved: dict | None, args, cli_fields: dict,
                      fixed: dict | None = None):
    """Build `cls` from defaults, then saved values (checkpoint), then
    options given on the command line, then `fixed`. Returns the instance
    and {field: source}."""
    defaults = cls()
    saved = saved or {}
    values, sources = {}, {}
    for f in dataclasses.fields(cls):
        if not f.init:
            continue
        if f.name in saved:
            values[f.name], sources[f.name] = saved[f.name], "checkpoint"
        else:
            values[f.name], sources[f.name] = getattr(defaults, f.name), "default"
    for dest, name in cli_fields.items():
        value = getattr(args, dest, None)
        if value is not None:
            values[name], sources[name] = value, "command line"
    for name, value in (fixed or {}).items():
        if values.get(name) != value:
            values[name], sources[name] = value, "fixed"
    return cls(**values), sources


def format_sources(title: str, obj, sources: dict) -> str:
    lines = [f"{title}:"]
    for name in sorted(sources):
        lines.append(f"  {name} = {getattr(obj, name)!r}  ({sources[name]})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Training

def run_training(env, agent, steps: int, *, model_name: str,
                 decision_stats: DecisionStats | None = None,
                 metrics_path: str | None = None,
                 checkpoint_dir: str = "checkpoints",
                 checkpoint_freq: int = CHECKPOINT_FREQ,
                 checkpoint_extra: dict | None = None,
                 print_fn=print) -> MetricsLogger:
    """Train for `steps` more timesteps; one PPO update per rollout.

    Differences from Arnav's train_d2drive() (see INTEGRATION_NOTES.md):
    the bootstrap input comes from peek_input() (the estimator history is
    not advanced an extra time), rewards are scaled like VecNormalize, and a
    time-limit truncation stores r + gamma * V(last observation) with
    done=True (SB3's method), so Arnav's GAE stays unchanged.
    """
    cfg = agent.cfg
    agent.set_training(True)
    stats = TerminationStats()
    decisions = decision_stats or DecisionStats()
    logger = MetricsLogger(metrics_path or f"train_{model_name}_metrics.csv", print_fn)
    episode_rewards, episode_lengths = deque(maxlen=100), deque(maxlen=100)

    target = agent.num_timesteps + int(steps)
    start_steps, start_time = agent.num_timesteps, time.time()
    iteration = 0
    obs, info = env.reset()
    agent.reset_episode()
    episode_reward, episode_length = 0.0, 0

    while agent.num_timesteps < target:
        stats.rollout_start()
        decisions.reset()
        for _ in range(cfg.rollout_steps):
            decision = agent.act(obs, info)
            next_obs, reward, terminated, truncated, next_info = env.step(
                decision["executed_action"], decision=decision)
            done = bool(terminated or truncated)
            buffer_reward = agent.scale_reward(reward, done)
            if truncated and not terminated:
                buffer_reward += cfg.gamma * agent.value_of(
                    agent.peek_input(next_obs, next_info))
            agent.store(obs, decision, buffer_reward, done)
            agent.num_timesteps += 1
            stats.on_step(next_info, done)
            decisions.on_step(decision, next_info)
            episode_reward += float(reward)
            episode_length += 1

            if agent.num_timesteps % checkpoint_freq == 0:
                path = os.path.join(checkpoint_dir,
                                    f"{model_name}_{agent.num_timesteps}_steps.pt")
                agent.save(path, extra=checkpoint_extra)

            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_reward, episode_length = 0.0, 0
                obs, info = env.reset()
                agent.reset_episode()
            else:
                obs, info = next_obs, next_info
            if agent.num_timesteps >= target:
                break

        train_metrics = agent.update(agent.peek_input(obs, info)) or {}
        iteration += 1
        elapsed = max(time.time() - start_time, 1e-9)
        row = {
            "time/iterations": iteration,
            "time/total_timesteps": agent.num_timesteps,
            "time/fps": int((agent.num_timesteps - start_steps) / elapsed),
            "time/time_elapsed": int(elapsed),
            "rollout/ep_rew_mean": (float(np.mean(episode_rewards))
                                    if episode_rewards else None),
            "rollout/ep_len_mean": (float(np.mean(episode_lengths))
                                    if episode_lengths else None),
        }
        row.update(stats.record())
        row.update(decisions.record())
        row.update({f"train/{k}": v for k, v in train_metrics.items()})
        logger.write(row)
    return logger


# ---------------------------------------------------------------------------
# Evaluation

def run_evaluation(env, agent, episodes: int = 20, seed: int | None = None,
                   decision_stats: DecisionStats | None = None,
                   print_fn=print) -> dict:
    """Deterministic policy on `episodes` random routes; statistics frozen."""
    agent.set_training(False)
    decisions = decision_stats or DecisionStats()
    decisions.reset()
    reasons, junctions = Counter(), Counter()
    speeds, offsets, rewards, lengths = [], [], [], []
    for episode in range(episodes):
        obs, info = env.reset(seed=seed if episode == 0 else None)
        agent.reset_episode()
        episode_reward, length = 0.0, 0
        while True:
            decision = agent.act(obs, info, deterministic=True)
            speeds.append(float(obs[0]))
            # Lane offset and heading are both exactly 0 in junctions and off
            # the lane; those steps are left out of the mean offset.
            if not (obs[18] == 0.0 and obs[19] == 0.0):
                offsets.append(abs(float(obs[18])))
            obs, reward, terminated, truncated, info = env.step(
                decision["executed_action"], decision=decision)
            decisions.on_step(decision, info)
            for maneuver, passed in info.get("maneuver_results", ()):
                junctions[(maneuver, passed)] += 1
            episode_reward += float(reward)
            length += 1
            if terminated or truncated:
                reason = info.get("termination_reason") or "unknown"
                reasons[reason] += 1
                print_fn(f"episode {episode + 1}/{episodes}: {reason}, "
                         f"{length} steps, reward {episode_reward:.1f}, "
                         f"progress {info.get('route_progress_m', 0.0):.0f} m")
                rewards.append(episode_reward)
                lengths.append(length)
                break

    summary = {
        "episodes": episodes,
        "goal_rate": reasons["goal"] / episodes if episodes else 0.0,
        "reasons": {r: reasons[r] for r in sorted(reasons)},
        "maneuver_pass_rate": {},
        "mean_speed_kmh": float(np.mean(speeds)) if speeds else 0.0,
        "mean_abs_lane_offset_m": float(np.mean(offsets)) if offsets else 0.0,
        "mean_episode_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_episode_length": float(np.mean(lengths)) if lengths else 0.0,
    }
    for m in MANEUVERS:
        ok, fail = junctions[(m, True)], junctions[(m, False)]
        summary["maneuver_pass_rate"][m] = (ok / (ok + fail) if ok + fail else math.nan,
                                            f"{ok}/{ok + fail}")
    summary.update(decisions.record())
    return summary


def format_evaluation(summary: dict) -> str:
    lines = [f"Episodes: {summary['episodes']}",
             f"Goal reached: {summary['goal_rate']:.1%}",
             "End reasons: " + ", ".join(f"{k}={v}" for k, v in summary["reasons"].items())]
    for m, (rate, frac) in summary["maneuver_pass_rate"].items():
        shown = "n/a" if math.isnan(rate) else f"{rate:.1%}"
        lines.append(f"Junction {m}: {shown} ({frac})")
    lines += [f"Mean speed: {summary['mean_speed_kmh']:.1f} km/h",
              f"Mean |lane offset| (in-lane steps): {summary['mean_abs_lane_offset_m']:.3f} m",
              f"Mean episode reward: {summary['mean_episode_reward']:.1f}, "
              f"length: {summary['mean_episode_length']:.0f} steps"]
    for key in sorted(summary):
        if "/" in key and summary[key] is not None:
            lines.append(f"{key}: {summary[key]}")
    return "\n".join(lines)
