from carla_env import CarlaEnv

env = CarlaEnv(vehicle_id=24)
obs, info = env.reset()

print("Initial observation:", obs)

try:
    for step in range(30):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Step {step + 1}: obs={obs}, reward={reward:.3f}")

        if terminated or truncated:
            print("Episode finished")
            break
finally:
    env.close()
