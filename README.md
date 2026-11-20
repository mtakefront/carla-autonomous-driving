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

## v14 (earlier version)

v14 was single-route lane keeping (fixed 2000 m route, 8-value observation). Its
environment, scripts, model and report are no longer in the working tree; they are in
commit `6a1da41` and can be restored with:

```bash
git checkout 6a1da41 -- carla-rl/carla_env.py carla-rl/train_v14.py \
    carla-rl/evaluate_v14.py carla-rl/train_v14.sbatch carla-rl/find_straight.py \
    carla-rl/ppo_lane_keeping_v14.zip carla-rl/vecnormalize_v14.pkl \
    docs/ppo_lane_keeping_v14_report.pdf
```

Result: after 100k steps the deterministic policy drove the full 3000-step episode on its
training route at 40.8 km/h with a mean lateral offset of 0.09 m (episode reward 2859).

## v16 (D2-Drive, continuous actions)

v16 trains on the unchanged v15 environment with D2-Drive, Arnav's PPO variant
(`carla-rl/d2drive_ppo.py`), instead of Stable-Baselines3's PPO. The car is driven as
in v15: the policy outputs steering and combined throttle/brake, each in [-1, 1].
v16 has not been run against a CARLA server yet; its first runs are the ones below.

What changed compared with v15:

- **Algorithm:** D2-Drive's PPO (own training loop, clipped surrogate, GAE) with a Gaussian
  policy over [steer, longitudinal] on Arnav's network. The input is 94 values: the 75 v15
  values, D2-Drive's 15 traffic values derived from them, and Arnav's 4 disturbance
  features.
- **Safety shield:** Arnav's rule-based shield, turned into limits on the continuous action.
  - No throttle when accelerating is unsafe (short gap, closing fast, at or above the speed
    limit). The agent's own braking is kept.
  - No further steering toward a side where a lane change is unsafe, once the car already
  drifts that way by more than 0.6 m.
  - A shield-distillation penalty lowers the policy's probability of those actions.
- **Normalization:** a VecNormalize replacement for observations and rewards, saved in the
  checkpoint.
- **Policy std:** never below 0.05 (v15's std fell to 0.029 and the policy collapsed).
- **Unchanged:** environment, observation, reward, termination and `[fail]` lines.
  `train_v15.py` (the SB3 baseline) still runs.

Design decisions, deviations from Arnav's code and risks: `carla-rl/INTEGRATION_NOTES.md`.

| File (in `carla-rl/`) | |
|---|---|
| `d2drive_ppo.py` | Arnav's D2-Drive (unmodified) |
| `d2drive_common.py`, `d2drive_training.py` | observation mapping, normalization, shield, training/evaluation loops (same files in v17) |
| `d2drive_continuous.py` | v16 policy, continuous shield, wrapper |
| `train_v16.py`, `train_v16.sbatch`, `evaluate_v16.py` | training, 3 h job, evaluation |
| `tests/` | pytest tests (no CARLA needed) |

### Running (v16)

On the cluster, in this order:

1. Short trial in an interactive GPU job:
   ```bash
   # terminal 1: CARLA + vehicle + live view
   ~/carla-autonomous-driving/carla/start_all.sh

   # terminal 2
   cd ~/carla-autonomous-driving/carla-rl && source env/bin/activate
   python -u train_v16.py --steps 20000 2>&1 | tee train_v16.log
   ```
   One table is printed per PPO update (every 2048 steps, so about 10 tables). Check:
   - **Watch the std columns** `train/std_steer` and `train/std_longitudinal`: they should
     not sink to the 0.05 floor early.
   - `train/approx_kl` and `train/clip_fraction` stay small (no jumps).
   - `episodes/*`: how the episodes end.
2. If that looks fine, the 3 h job (no rendering, output in `train_v16.log`):
   ```bash
   cd ~/carla-autonomous-driving/carla-rl
   sbatch train_v16.sbatch                          # from scratch
   sbatch train_v16.sbatch --resume d2drive_v16     # continue the last model
   ```
   A run from scratch overwrites `d2drive_v16.pt` and appends to `train_v16_metrics.csv`.
   Keep the trial's files (rename them) or continue from them with `--resume`.

`train_v16.py` options:

| Option | Meaning | Default |
|---|---|---|
| `--steps N` | timesteps to train in this run | 50000 |
| `--traffic N` | autopilot cars (Traffic Manager) | 15 |
| `--pedestrians N` | AI pedestrians | 20 |
| `--no-render` | CARLA draws nothing: faster, but no live view | off |
| `--resume MODEL` | continue from a saved model (normalization statistics and timestep count included) | none |
| `--rollout-steps`, `--minibatch-size`, `--epochs` | PPO rollout and update sizes | 2048, 256, 10 |
| `--lr`, `--ent-coef`, `--shield-coef` | learning rate, entropy and shield-penalty coefficients | 3e-4, 0.01, 0.15 |
| `--no-reward-norm` | turn reward scaling off | on |
| `--store-executed-action` | store the shield's executed action in the PPO buffer instead of the proposal | off |
| `--min-std` | lower bound of the policy's std | 0.05 |
| `--lateral-shield-offset` | lane offset (m) from which the lateral shield rules apply | 0.6 |

With `--resume`, options given on the command line override the checkpoint's values and
the others come from the checkpoint. Each value and its source are printed at the start.

Saved files:
- `d2drive_v16.pt` at the end, also when stopped with Ctrl+C or by the Slurm time limit;
- `checkpoints/d2drive_v16_<N>_steps.pt` every 10k steps;
- `train_v16_metrics.csv` with one row per update: the table's values, i.e. v15's
  `episodes/*`, `junctions/*` and `reward_parts/*` plus losses, KL, std, veto and
  shield-rule rates.

Do not train v15 and v16 at the same time on the same CARLA server.

### Evaluating (v16)

```bash
python -u evaluate_v16.py --model d2drive_v16 --episodes 20
```

The deterministic policy (Gaussian mean, then the shield) drives 20 random routes with the
training traffic (`--traffic`, `--pedestrians`, `--no-render`, `--seed` as above). It
reports:
- goal rate and end reasons;
- pass rate per maneuver;
- shield veto rate and how often each rule acted;
- mean speed;
- mean |lane offset| (junction steps excluded).

### Tests (v16)

```bash
cd carla-rl
pip install pytest        # not in requirements.txt; only needed for the tests
python -m pytest tests -q
```

The tests use a fake environment and run without CARLA.
