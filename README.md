# carla-autonomous-driving

Lane keeping with reinforcement learning (PPO, Stable-Baselines3) in the CARLA 0.9.16
simulator, running headless on the Northeastern Explorer HPC cluster.

## Repository layout

```
carla/                      CARLA server scripts (expected at ~/carla)
  start_carla.sh            starts the CARLA server (Apptainer) with the GPU fixes
  start_all.sh              CARLA + vehicle spawn + live view, stops on Ctrl+C
  client_test.py            quick check that the server renders on the GPU

carla-rl/                   RL project (expected at ~/carla-rl)
  carla_env.py              Gymnasium environment (observation, reward, termination)
  train_v14.py              PPO training, 100k steps
  evaluate_v14.py           runs the trained model deterministically
  train_v14.sbatch          unattended training job (CARLA + spawn + train)
  ppo_lane_keeping_v14.zip  trained model
  vecnormalize_v14.pkl      observation normalization stats (needed with the model)
  spawn_vehicle.py          spawns a Tesla Model 3
  find_straight.py          moves the vehicle to the start road (road 12, lane 1)
  live_view.py              chase camera stream on port 8080
  control_vehicle.py, read_state.py, reset_vehicle.py, test_env.py   small helpers
```

The scripts use absolute paths `~/carla` and `~/carla-rl`, so copy (or symlink) the two
folders there.

## Setup

1. CARLA server image (not in the repo, 8 GB):
   ```bash
   cd ~/carla
   apptainer pull carla-0.9.16.sif docker://carlasim/carla:0.9.16
   ```
2. Python client environment:
   ```bash
   cd ~/carla-rl
   module load anaconda3/2024.06      # Python 3.12 on Explorer
   python3 -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```

CARLA must run on a GPU node, and the server and the client must run in the same Slurm job.
`start_carla.sh` contains the fix for the UE4 render thread crash on Explorer
(the host's EGL library is injected into the container and Vulkan falls back to the CPU).

## Running

In an interactive GPU job:

```bash
# terminal 1: CARLA + vehicle + live view (http://<node>:8080)
~/carla/start_all.sh

# terminal 2
cd ~/carla-rl && source env/bin/activate
python find_straight.py                              # move car to the start road
python -u evaluate_v14.py 2>&1 | tee eval_v14.log    # or train_v14.py
```

`find_straight.py` must be run before training or evaluation; the first spawn point is
inside a junction. Stop with Ctrl+C so the environment turns synchronous mode off.

Unattended training: `sbatch ~/carla-rl/train_v14.sbatch` (output in `train_v14.log`).

## Environment (v14)

- **Route:** a fixed 2000 m route is built from the start lane, taking the straightest
  branch at junctions. Offset and heading are measured against this route.
- **Observation (8):** speed, lateral offset, heading error, route heading 5/10/20/40 m
  ahead, previous steering.
- **Action (2):** steering, throttle/brake (-0.3..0 coasts).
- **Reward:** forward speed toward 40 km/h, minus lane offset, heading error and
  steering change (anti-zigzag) penalties.
- **Termination (-30):** off-road, collision, stuck, or more than 6 m from the route.
  Episodes are truncated at 3000 steps (150 s at 20 Hz) or at the end of the route.

## Results

After 100k training steps, the deterministic policy drives the full 3000-step episode
on Town10HD_Opt without leaving the road: ~40.8 km/h, ~0.02 m lateral offset,
episode reward 2859. Tested only on the training start point and route.
