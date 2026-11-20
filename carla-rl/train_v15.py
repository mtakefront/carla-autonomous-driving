import argparse
import signal
import sys
import traceback
from collections import Counter

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from carla_env import CarlaEnv

REASONS = ("goal", "collision", "off_road", "wrong_road", "wrong_way", "red_light",
           "stop_sign", "stuck", "time_limit")
REWARD_PARTS = ("progress", "speed", "step", "illegal_crossing", "lane", "terminal")
MANEUVERS = ("LEFT", "RIGHT", "STRAIGHT")


class TerminationStatsCallback(BaseCallback):
    """Logs how many episodes ended for each reason during every rollout,
    how often each junction maneuver was passed or failed, and the average
    per-episode sum of each reward term."""

    def __init__(self):
        super().__init__()
        self.counts = Counter()
        self.parts = Counter()
        self.junctions = Counter()        # this rollout
        self.junctions_total = Counter()  # whole run

    def _on_rollout_start(self):
        self.counts.clear()
        self.parts.clear()
        self.junctions.clear()

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            for maneuver, passed in info.get("maneuver_results", ()):
                self.junctions[(maneuver, passed)] += 1
                self.junctions_total[(maneuver, passed)] += 1
            if done:
                self.counts[info.get("termination_reason") or "unknown"] += 1
                for name, value in (info.get("reward_parts") or {}).items():
                    self.parts[name] += value
        return True

    def _on_rollout_end(self):
        for reason in REASONS:
            self.logger.record(f"episodes/{reason}", self.counts[reason])
        episodes = sum(self.counts.values())
        if episodes:
            for name in REWARD_PARTS:
                self.logger.record(f"reward_parts/{name}",
                                   round(self.parts[name] / episodes, 1))
        for m in MANEUVERS:
            ok, fail = self.junctions[(m, True)], self.junctions[(m, False)]
            self.logger.record(f"junctions/{m}", f"{ok}/{ok + fail}")
            ok_t, fail_t = self.junctions_total[(m, True)], self.junctions_total[(m, False)]
            rate = ok_t / (ok_t + fail_t) if ok_t + fail_t else 0.0
            self.logger.record(f"junctions_total/{m}_pass_rate", round(rate, 3))


# One thread: the MLP is tiny, and extra threads only compete with CARLA
# for the job's few CPUs.
torch.set_num_threads(1)

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=50_000, help="timesteps to train in this run")
parser.add_argument("--traffic", type=int, default=15, help="autopilot cars")
parser.add_argument("--pedestrians", type=int, default=20)
parser.add_argument("--no-render", action="store_true",
                    help="CARLA draws nothing: faster, but no Live View")
parser.add_argument("--resume", default=None,
                    help="continue from this saved model, e.g. ppo_lane_keeping_v15")
parser.add_argument("--resume-norm", default="vecnormalize_v15.pkl",
                    help="VecNormalize stats saved with that model")
args = parser.parse_args()

# SIGTERM (kill, Slurm time limit) -> normal exit so finally/close() runs
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

# routes 60-600 m, with autopilot traffic and pedestrians
raw_env = CarlaEnv(min_route_length=60.0, traffic_vehicles=args.traffic,
                   pedestrians=args.pedestrians, no_rendering=args.no_render)
venv = DummyVecEnv([lambda: Monitor(raw_env)])
if args.resume:
    env = VecNormalize.load(args.resume_norm, venv)
    env.training, env.norm_reward = True, True
else:
    env = VecNormalize(venv, norm_obs=True, norm_reward=True)

checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./checkpoints/",
    name_prefix="ppo_lane_keeping_v15",
    save_vecnormalize=True,
)

model = None
completed = False
try:
    if args.resume:
        model = PPO.load(args.resume, env=env, device="cpu")
        print(f"Resuming {args.resume} at {model.num_timesteps} timesteps", flush=True)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            n_steps=1024,
            batch_size=64,
            learning_rate=3e-4,
            # Start with std ~0.37 instead of 1 so steering/throttle don't flail;
            # 256x256 (default 64x64) so the policy can act differently per maneuver
            policy_kwargs=dict(log_std_init=-1.0, net_arch=[256, 256]),
            # Small MLP policy: CPU is faster than GPU here
            device="cpu",
        )

    model.learn(
        total_timesteps=args.steps,
        callback=CallbackList([checkpoint_callback, TerminationStatsCallback()]),
        reset_num_timesteps=not args.resume,
    )
    completed = True

except BaseException:
    # Show the cause now; cleanup below can take a while.
    traceback.print_exc()
    sys.stdout.flush()
    raise

finally:
    # Also save when stopped early (Ctrl+C, Slurm time limit)
    if model is not None:
        model.save("ppo_lane_keeping_v15")
        env.save("vecnormalize_v15.pkl")
        status = "completed" if completed else "stopped early, saved"
        print(f"Training {status}: ppo_lane_keeping_v15")
    raw_env.close()
