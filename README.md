# carla-autonomous-driving

Route navigation with reinforcement learning (PPO, Stable-Baselines3) in the CARLA 0.9.16
simulator, running headless on the Northeastern Explorer HPC cluster. The current
environment (v15) drives random routes through Town10HD_Opt with turns, traffic lights,
stop signs, autopilot traffic and pedestrians. v14 was single-route lane keeping.

## Repository layout

```
carla/                      CARLA server scripts (+ carla-0.9.16.sif, not in git)
  start_carla.sh            starts the CARLA server (Apptainer) with the GPU fixes
  start_all.sh              CARLA + vehicle spawn + live view, stops on Ctrl+C
  client_test.py            quick check that the server renders on the GPU

carla-rl/                   RL project (+ env/ virtualenv, not in git)
  carla_env.py              v15 Gymnasium environment (observation, reward, termination)
  train_v15.py              PPO training; options below
  train_v15.sbatch          unattended 3 h training job (CARLA + spawn + train, no rendering)
  agents/                   CARLA 0.9.16 PythonAPI agents (route planner), see Setup
  spawn_vehicle.py          spawns a Tesla Model 3 (role name "hero")
  live_view.py              chase camera stream on port 8080
  train_v14.py, evaluate_v14.py, train_v14.sbatch, ppo_lane_keeping_v14.zip,
  vecnormalize_v14.pkl      v14 (legacy, needs the v14 carla_env.py, see below)
  find_straight.py          moves the vehicle to road 12 (only used by v14)
  control_vehicle.py, read_state.py, reset_vehicle.py, test_env.py   small helpers
```

Scripts find their paths relative to the repository, so it can be cloned anywhere.
The commands below assume the clone is at `~/carla-autonomous-driving`.

## Setup

1. CARLA server image (not in the repo, 8 GB):
   ```bash
   cd ~/carla-autonomous-driving/carla
   apptainer pull carla-0.9.16.sif docker://carlasim/carla:0.9.16
   ```
2. Python client environment:
   ```bash
   cd ~/carla-autonomous-driving/carla-rl
   module load anaconda3/2024.06      # Python 3.12 on Explorer
   python3 -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```

`carla-rl/agents/` is CARLA's own `PythonAPI/carla/agents` package (MIT license),
copied out of the 0.9.16 image because the `carla` pip package does not include it.
The environment uses its `GlobalRoutePlanner`; it needs `networkx` (in requirements.txt).

CARLA must run on a GPU node, and the server and the client must run in the same Slurm job.
`start_carla.sh` contains the fix for the UE4 render thread crash on Explorer
(the host's EGL library is injected into the container and Vulkan falls back to the CPU).

## Running (v15)

In an interactive GPU job:

```bash
# terminal 1: CARLA + vehicle + live view (http://<node>:8080)
~/carla-autonomous-driving/carla/start_all.sh

# terminal 2
cd ~/carla-autonomous-driving/carla-rl && source env/bin/activate
python -u train_v15.py 2>&1 | tee train_v15.log
```

`train_v15.py` options:

| Option | Meaning | Default |
|---|---|---|
| `--steps N` | timesteps to train in this run | 50000 |
| `--traffic N` | autopilot cars (Traffic Manager) | 15 |
| `--pedestrians N` | AI pedestrians | 20 |
| `--no-render` | CARLA draws nothing: faster, but no live view | off |
| `--resume MODEL` | continue from a saved model (with `--resume-norm`, default `vecnormalize_v15.pkl`) | none |

The model is saved as `ppo_lane_keeping_v15.zip` + `vecnormalize_v15.pkl` at the end, also
when stopped with Ctrl+C or by the Slurm time limit; checkpoints every 10k steps go to
`checkpoints/`. Stop with Ctrl+C so the environment turns synchronous mode off and
removes its traffic.

Unattended training (3 h, no rendering, output in `train_v15.log`):

```bash
cd ~/carla-autonomous-driving/carla-rl
sbatch train_v15.sbatch                                 # from scratch
sbatch train_v15.sbatch --resume ppo_lane_keeping_v15   # continue the last model
```

Do not run an interactive training at the same time: both write the same files.

## Environment (v15)

- **Routes:** each episode starts at a random spawn point (not in or near a junction,
  no traffic close by) and drives to a random driving-lane point 60-600 m away along
  the shortest route.
- **Observation (75):** speed, speed limit, previous steering; next maneuver
  (left/right/straight) and its distance; lanes to the left/right; traffic light
  (red/yellow/green) or stop sign ahead, stop-line distance, stopped-at-sign flag;
  front sensor (nearest car/pedestrian on the path: presence, distance, closing speed);
  lateral offset and heading in the own lane; left lane / right lane / behind radar
  (nearest car: presence, distance, relative speed); lane lines at 0-50 m ahead.
  Ahead of a junction the lines show the path through it from the car's own lane
  (or the nearest lane that can make the turn), marked as virtual lines inside it.
- **Action (2):** steering, throttle/brake (0 = light throttle, coasting from -0.33,
  braking below -0.6). Control at 10 Hz.
- **Reward:** progress along the route; penalties for speeding, crawling below 10 km/h
  (not while waiting at a red light, a stop sign, or behind a car/pedestrian), lane
  offset and heading, touching solid lines, and time.
- **Termination (-30):** off-road, collision, wrong road after a junction, oncoming
  lanes, running a red light, not stopping at a stop sign, stuck. +50 at the goal.
- **Training log:** besides the SB3 table, `episodes/*` (how episodes ended),
  `junctions/*` (pass rate per maneuver), `reward_parts/*` (reward per term) and a
  `[fail]` line per failed episode.

## v14 (legacy)

v14 trained on a single fixed 2000 m route (straightest branch at junctions) with an
8-value observation measured against that route. Its scripts and model are kept, but
`carla_env.py` is now v15, so to run v14 restore its environment from git history:

```bash
git show 6a1da41:carla-rl/carla_env.py > carla-rl/carla_env.py
```

After 100k training steps, the deterministic v14 policy drove the full 3000-step episode
on Town10HD_Opt without leaving the road: 40.8 km/h, mean lateral offset 0.09 m,
episode reward 2859. Tested only on the training start point and route.

Training curves, PPO diagnostics and evaluation plots:
[docs/ppo_lane_keeping_v14_report.pdf](docs/ppo_lane_keeping_v14_report.pdf)
