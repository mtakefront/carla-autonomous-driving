import sys

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from carla_env import CarlaEnv

# Usage: python evaluate_v14.py [model_path] [vecnormalize_path]
model_path = sys.argv[1] if len(sys.argv) > 1 else "ppo_lane_keeping_v14"
norm_path = sys.argv[2] if len(sys.argv) > 2 else "vecnormalize_v14.pkl"

raw_env = CarlaEnv()
env = VecNormalize.load(norm_path, DummyVecEnv([lambda: raw_env]))
env.training = False
env.norm_reward = False
model = PPO.load(model_path)

obs = env.reset()

total_reward = 0
episode_reward = 0
episode_steps = 0
episode_count = 1

try:
    for step in range(3000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = env.step(action)
        info = infos[0]

        total_reward += reward[0]
        episode_reward += reward[0]
        episode_steps += 1

        # Raw (unnormalized) observation for printing
        speed, offset, heading = env.get_original_obs()[0][:3]

        print(
            f"Episode={episode_count} Step={step + 1} "
            f"Speed={speed:.1f} km/h "
            f"Offset={offset:.2f} m "
            f"Heading={heading:.1f}° "
            f"Throttle={action[0][1]:.2f} "
            f"Steer={action[0][0]:.2f} "
            f"Reward={reward[0]:.2f}",
            flush=True
        )

        if done[0]:
            reasons = [
                k for k in ("off_road", "lost_route", "stuck", "collision",
                            "route_end")
                if info.get(k)
            ] or ["time limit"]
            print(
                f"\nEpisode {episode_count} finished: {', '.join(reasons)} "
                f"| Steps={episode_steps} "
                f"| Reward={episode_reward:.2f}\n",
                flush=True
            )

            episode_count += 1
            episode_steps = 0
            episode_reward = 0

finally:
    raw_env.close()

print(f"\nTotal steps: {step + 1}")
print(f"Completed episodes: {episode_count - 1}")
print(f"Total reward: {total_reward:.2f}")
