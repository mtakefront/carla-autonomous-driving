import signal
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from carla_env import CarlaEnv

# SIGTERM (kill, Slurm time limit) -> normal exit so finally/close() runs
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

raw_env = CarlaEnv()
env = VecNormalize(
    DummyVecEnv([lambda: Monitor(raw_env)]),
    norm_obs=True,
    norm_reward=True,
)

checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./checkpoints/",
    name_prefix="ppo_lane_keeping_v14",
    save_vecnormalize=True,
)

try:
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=1024,
        batch_size=64,
        learning_rate=3e-4,
    )

    model.learn(
        total_timesteps=100000,
        callback=checkpoint_callback,
    )

    model.save("ppo_lane_keeping_v14")
    env.save("vecnormalize_v14.pkl")
    print("Training completed: ppo_lane_keeping_v14")

finally:
    raw_env.close()
