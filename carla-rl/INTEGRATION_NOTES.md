# D2-Drive integration notes: v16 (continuous actions)

v16 replaces v15's Stable-Baselines3 PPO with Arnav's D2-Drive (`d2drive_ppo.py`) and
keeps v15's environment and its driving interface: the policy still outputs
`[steer, longitudinal]` in [-1, 1] and steers the car itself. v17 (a separate copy of the
repository) uses Arnav's six discrete commands with a low-level controller instead.

Nothing in v16 has been run against a CARLA server yet. Everything below that is about
behaviour in CARLA is design intent, not a measured result. See "What was tested".

## Files

| File | Role |
|---|---|
| `d2drive_ppo.py` | Arnav's file, byte-for-byte copy (not modified) |
| `d2drive_common.py` | shared with v17: observation mapping, normalization, reward scaling, speed-limit shield, agent mixin |
| `d2drive_training.py` | shared with v17: training loop, v15's termination metrics, logging, CLI options, evaluation loop |
| `d2drive_continuous.py` | v16 only: Gaussian policy, continuous shield, continuous shield penalty, agent, wrapper, v16 logs |
| `train_v16.py`, `train_v16.sbatch`, `evaluate_v16.py` | entry points |
| `tests/` | pytest, runs without CARLA (`FakeCarlaEnv`) |

Unchanged from v15 (SHA256-checked): `carla_env.py`, `agents/`, `../carla/`, `hero.py`,
`spawn_vehicle.py`, `live_view.py`, `control_vehicle.py`, `read_state.py`,
`reset_vehicle.py`, `test_env.py`, `train_v15.py`, `train_v15.sbatch`. The environment's
observation (75 values), reward, termination rules, traffic, sensors, route generation
and `[fail]` lines are v15's. The wrapper only reads, and only CarlaEnv's return values:
it calls `env.step()` / `env.reset()` and nothing else (no private fields, no
`world.tick()`).

## Observation mapping (v15 75 values -> D2-Drive 15 values)

`o` = v15 observation (verified against `CarlaEnv._get_state()`).

| # | D2 field | Formula |
|---|---|---|
| 0 | ego_speed_mps | `o[0] / 3.6` |
| 1 | lane_offset_m | `o[18]` (+ = right, same sign as D2; 0 in junctions) |
| 2 | heading_error_rad | `radians(o[19])` (+ = nose pointing right) |
| 3 | front_gap_m | `o[16]` (40 = nothing ahead) |
| 4 | front_closing_mps | `o[17]` (+ = approaching) |
| 5-8 | left: front gap, rear gap, front closing, rear closing | `side(o[7], o[20], o[21], o[22])` |
| 9-12 | right: same | `side(o[8], o[23], o[24], o[25])` |
| 13 | traffic_density | `(o[15] + o[20] + o[23] + o[26]) / 4` |
| 14 | speed_spread_mps | `np.std` of: ego speed, `ego + rel` for each visible side/rear vehicle, `ego - o[17]` if a front object is present |

`side(lanes, present, x, rel)`:
- `lanes == 0` (also always inside junctions): `(0, 0, 0, 0)`. The zero gaps make Arnav's
  shield mark that lane change unsafe.
- lane, no vehicle: `(25, 25, 0, 0)` (25 m = `SIDE_RANGE_M`, the radar range).
- `x >= 0` (ahead, or exactly alongside): `(x, 25, -rel, 0)`. `rel` is other minus ego, so the
  ego closes on it at `-rel`.
- `x < 0` (behind): `(25, |x|, 0, +rel)`.

Known limitations:
- **Center-to-center distances.** v15 measures the front distance and the side `x` from
  vehicle center to vehicle center. The bumper-to-bumper gap is about 4-5 m less, so
  Arnav's safe gap (5 m + speed x headway) is effectively 4-5 m shorter than it reads.
- **Same road only.** The side radar counts only vehicles on the same OpenDRIVE `road_id`.
  A car in the neighbouring lane just past a road boundary is missed.
- **One vehicle per side.** v15 gives only the nearest vehicle per side. A second car
  behind the first one is invisible to D2's lane-change check.
- **Own-lane rear vehicle.** It (`o[26..28]`) is not in D2's 15 values; it only counts in the
  density and the speed spread.
- **25 m side range.** An empty lane reads as a 25 m gap. Arnav's safe gap is
  `5 + v * tau` with `tau` in [1, 2] (it grows with density and speed spread), so with
  `tau = 2` an empty lane already counts as unsafe above (25 - 5) / 2 = 10 m/s = 36 km/h
  (72 km/h with `tau = 1`).
- **Approximate speeds.** The front object's absolute speed (`ego - closing`) is an
  approximation: the closing speed is along the line of sight, and the object may be a
  parked car or a pedestrian.

## Policy input (94) and normalization

`[75 normalized v15 values, 15 normalized base values, 4 raw disturbance values]`.

- **Normalization.** `ObsNormalizer` replaces VecNormalize: running mean/std with SB3's
  formulas (count starts at 1e-4, epsilon 1e-8), clipped to +-10. The statistics update
  once per observation the agent acts on in training, are frozen in evaluation
  (`set_training(False)`), and are saved in the checkpoint. A test compares the outputs
  with SB3's `VecNormalize` step by step.
- **Raw inputs.** The 4 disturbance values (Arnav's `DisturbanceEstimator`) stay raw; they
  are already bounded.
- **Shield and estimator.** Both always get the raw 15 base values; only the network sees
  normalized values (tested).
- **Reward scaling** (VecNormalize's method): rewards are divided by the running std of the
  discounted return (no mean subtraction) and clipped to +-10. `--no-reward-norm` turns it
  off. The statistics are in the checkpoint. Logs (`rollout/ep_rew_mean`,
  `reward_parts/*`) always show raw rewards.

## Action interface

Same as v15: the policy gives `[steer, longitudinal]` in [-1, 1]. The wrapper passes the
shielded action to `CarlaEnv.step()` unchanged. v15's pedal mapping (0 = light throttle,
coasting band, braking below -0.6) stays inside the environment. There is no low-level
controller: the agent follows lanes and turns with its own steering, as in v15. v15's
exploration property is kept: a zero-mean action is light throttle, so a random policy
still rolls forward.

## Policy (Gaussian)

- **Network.** Arnav's `ActorCritic(94, 256, n_actions=2)`; the actor's two outputs are the
  Gaussian means. A separate, state-independent `log_std` parameter starts at -1.0 (std
  0.37, as v15).
- **Minimum std** (`--min-std`, default 0.05). This is a deviation: v15's training ended
  with std 0.029 and a collapsed policy.
  - The forward pass uses `max(log_std, log(min_std))` with a straight-through gradient.
  - The parameter is pulled back to the floor after every optimizer step.
  - A plain clamp is not enough: below the floor its gradient is 0, and the parameter
    lands just below the floor after float32 rounding. A test showed the std could then
    never grow again.
- **Log-probability and entropy.** The log-probability is the sum over both dimensions,
  and the entropy is the Gaussian entropy summed over both dimensions.
  `entropy_coef = 0.01` (Arnav's; v15's SB3 used 0).
- **Clipping.** The sample is clipped to [-1, 1] before the shield and the environment. The
  log-probability is that of the unclipped sample (SB3's method).
- **Evaluation.** The deterministic action is the mean (then clipped and shielded).

## Shield

**Speed limit.** `SpeedLimitShield` calls Arnav's `unsafe_action_mask()` unchanged.
Only for that call, `max_speed_mps` is the current speed limit (`o[1]`, 30 km/h if 0)
instead of his fixed 60 km/h. v15 penalizes speeding, so 60 km/h was too high.

**Continuous rules.** Arnav's mask is for discrete commands. In v16 it only restricts the
continuous action and never adds a behaviour:

| Mask | Condition | Executed action |
|---|---|---|
| `mask[1]` accelerate unsafe (short gap, TTC < 2 s or speed >= limit) | always | `longitudinal = min(proposed, -0.45)` (no throttle, no brake; a stronger brake from the agent is kept) |
| `mask[3]` left change unsafe | and `o[18] < -D` and `o[19] < 0` (already drifting left) | `steer = max(proposed, 0)` |
| `mask[4]` right change unsafe | and `o[18] > D` and `o[19] > 0` | `steer = min(proposed, 0)` |

`D = 0.6 m` (`--lateral-shield-offset`). v15 reports offset and heading as 0 inside
junctions and off the lane, so the lateral rules are off there and turns are not blocked
(tested). `d2_vetoed` is true if a rule changed the action by more than 1e-6. Each
transition stores the three flags `acc_unsafe`, `left_active`, `right_active` (instead of
Arnav's 6-value mask); `info["d2_shield_flags"]` has them too.

**Shield-distillation penalty, continuous form.** Arnav's penalty is the probability the
policy gives to unsafe actions. Here it is computed with the current policy's Normal and
`Normal.cdf`:

    shield_loss = mean[ acc_unsafe  * P(longitudinal > -0.45)
                      + left_active * P(steer < 0)
                      + right_active * P(steer > 0) ]

The coefficient is `shield_coef` (0.15). The thresholds lie inside [-1, 1], so the
probabilities are the same for the clipped action. Tests check that the gradient on the
means points away from the unsafe region and that one step lowers that probability.

## Update

Arnav's `update()` assumes a Categorical, so v16 overrides it. Same terms and formulas:
clipped surrogate (`clip_eps` 0.2), MSE value loss (`value_coef` 0.5), entropy bonus,
shield penalty, gradient clipping (0.5), full-batch advantage normalization, his
`RolloutBuffer.compute_gae()` and `Transition`. Differences:

- Gaussian log-probabilities and entropy instead of Categorical ones; the continuous
  shield penalty above.
- `log_std` is projected to the floor after each optimizer step.
- **Two KL measurements.**
  - SB3 style (comparable with v15): `train/approx_kl` is the mean over the last epoch's
    minibatches, and `train/clip_fraction` the mean over all minibatches. Both are measured
    before each minibatch's step.
  - Post-update (comparable with v17): `train/approx_kl_post` and
    `train/clip_fraction_post`, measured once after the update over the whole rollout.
- The returned metrics also include `std_steer` and `std_longitudinal`.

## Training loop (`d2drive_training.run_training`)

Arnav's `train_d2drive()` is not used. Differences:

- **Bootstrap.** At the end of a rollout, `peek_input()` builds the policy input on a copy of
  the estimator, without updating the statistics. Arnav's loop calls
  `get_augmented_obs()`, which advances the estimator history, and then `act()` on the same
  observation advances it again.
- **Truncation (time limit).** The buffer gets `r + gamma * V(last observation)` and
  `done = True` (SB3's method). Arnav's GAE (`done` = terminal) stays unchanged.
  Terminations get no bootstrap (tested).
- **Reward scaling** (above).
- **SIGTERM and Ctrl+C.** On either, the `finally` block saves the model, then calls
  `env.close()`.
- **Checkpoints.** Every 10k steps: `checkpoints/d2drive_v16_<N>_steps.pt`. Final:
  `d2drive_v16.pt`. A checkpoint holds Arnav's keys (`policy_state_dict`,
  `optimizer_state_dict`, `config`) plus the normalizer, the reward scaler,
  `num_timesteps` and the agent class. It is written to a temporary file first and then
  renamed, so a kill during the save keeps the old file.
- **`--resume`.** Options given on the command line override the checkpoint; the others
  come from the checkpoint, and fields not in the checkpoint from the defaults. The device
  is always CPU. The learning rate in the optimizer is set from the resolved config. The
  value and source of every config field are printed at start.
- **Logging.** Each update writes an SB3-style table to stdout and the same row to
  `train_v16_metrics.csv`:
  - `episodes/*`, `junctions/*`, `junctions_total/*_pass_rate` and `reward_parts/*`,
    computed exactly as in `train_v15.py`'s `TerminationStatsCallback` (tested).
  - `rollout/ep_rew_mean` and `ep_len_mean`: raw rewards, last 100 episodes.
  - `train/policy_loss`, `value_loss`, `entropy`, `shield_loss`, `approx_kl`,
    `clip_fraction`, `approx_kl_post`, `clip_fraction_post`, `std_steer`,
    `std_longitudinal`.
  - `shield/veto_rate`, plus `shield/acc_unsafe_rate`, `left_active_rate` and
    `right_active_rate` (how often each rule was active).
  - `actions/mean_abs_steer` and `actions/mean_longitudinal` (executed actions).
  - `time/*`.

  The environment's `[fail]` lines are printed by CarlaEnv as before.

## Configuration

| Value | v16 | Arnav default | v15 (SB3) |
|---|---|---|---|
| rollout_steps | 2048 | 2048 | 1024 |
| minibatch_size | 256 | 256 | 64 |
| update_epochs | 10 | 10 | 10 |
| learning_rate | 3e-4 | 3e-4 | 3e-4 |
| gamma / gae_lambda | 0.99 / 0.95 | same | same |
| clip_eps | 0.2 | 0.2 | 0.2 |
| entropy_coef | 0.01 | 0.01 | 0.0 |
| value_coef / max_grad_norm | 0.5 / 0.5 | same | same |
| shield_coef | 0.15 | 0.15 | - |
| hidden_dim | 256 (shared trunk) | 256 | [256, 256] separate |
| initial std / floor | 0.37 / 0.05 | - | 0.37 / none |
| device, threads | CPU, 1 | cuda if available | CPU, 1 |

Command line: `--rollout-steps --minibatch-size --epochs --lr --ent-coef --shield-coef
--no-reward-norm --store-executed-action --min-std --lateral-shield-offset`.

`target_speed_mps` (30 km/h) and `min_speed_mps` (10 km/h, v15's `MIN_SPEED_KMH`) are set to
v15 values but are unused by Arnav's code. `max_speed_mps` is only the fallback for an
unknown limit; `SpeedLimitShield` overrides it for every mask.

## Deviations from Arnav's code (his file is not modified)

1. **Policy input of 94 values** instead of his 19 (15 base + 4 disturbance). The
   policy needs v15's full observation: the next maneuver, traffic lights, lane lines.
2. **Gaussian over 2 continuous actions** instead of a Categorical over 6 commands: v16
   keeps v15's driving interface. His `ActorCritic` is used with `n_actions=2`, plus a
   separate `log_std` and a std floor.
3. **`act()`, `store()`, `update()` overridden** for the Gaussian. The formulas are kept;
   the update was rewritten in `d2drive_continuous.py`, with the differences listed above.
4. **Shield as action limits.** His mask is turned into limits on the continuous action
   (table above). The mask itself is his method, called unchanged.
5. **`max_speed_mps` = the current speed limit** (`SpeedLimitShield`).
6. **Continuous shield penalty** (CDF form) in place of his `sum(probs * unsafe_mask)`.
7. **Observation normalization and reward scaling** added (VecNormalize's methods; v15
   used both).
8. **Truncation bootstrap** and the **non-advancing bootstrap peek** (training loop).
9. **Device CPU and one torch thread** instead of his cuda default.
10. **Checkpoint extended** with the normalization statistics, `num_timesteps`, the agent
    class and a learning-rate update on resume. His keys are kept.
11. **`--store-executed-action`** (default off) stores the executed, clipped and shielded
    action and its log-probability under the old Gaussian.

## Known issues in Arnav's code (not changed; noted)

- **Proposed vs. executed action.** When the shield changes an action, the buffer and the
  log-probability get the proposed action, but the reward comes from the executed one.
  PPO then credits the proposal with the outcome of a different action. In v16 the
  proposal is the unclipped Gaussian sample and the executed action is clipped and
  shielded, so the same mismatch exists.
  - `--store-executed-action` (default off) stores the executed action instead. This
    is not exact either: clipping and the shield put probability mass on single points
    (for example -0.45 and steer 0), which a Gaussian density does not describe, so its
    importance ratios are biased.
- **ABORT is undefined.** Its meaning in the environment is not defined in his code. It
  does not exist in v16; v17 defines it.
- **No red-light or junction rule** in the shield. In the last v15 training, all
  failures were red-light violations. Not added here (see suggestions).
- **`train_d2drive()` double-advances the estimator.** Its bootstrap
  `get_augmented_obs()` advances the estimator history one extra step, so the next
  `act()` sees that observation twice.
- **His GAE treats time-limit truncation as termination** (no bootstrap).
- **No normalization.** No observation or reward normalization. With a shared trunk,
  large unnormalized returns make the value loss dominate the gradient (and gradient
  clipping then shrinks the policy part).
- **Defaults.** Default device is cuda when available. `max_speed_mps` is fixed at 60 km/h.
  `target_speed_mps` and `min_speed_mps` are unused.
- **Shield fallback.** Only commands 1, 3 and 4 can be masked, so his fallback is always
  DECELERATE; the KEEP_LANE fallback branch is unreachable. DECELERATE and ABORT are never
  masked.
- **No KL metrics.** `update()` reports no approx_kl or clip_fraction.
- **`train_d2drive()` save handling.** It overwrites its checkpoint after every update,
  saves no normalization state, and does not handle SIGTERM.

## Risks

- **Not run in CARLA.** Nothing has been checked against the real simulator: the speed at
  which the policy learns, whether the shield rules behave as intended, the throughput.
- **Shield penalty pushes the std down.** When the mean is already on the safe side, a
  smaller std lowers `P(unsafe)`, so the penalty pulls the std toward the floor. This can
  feed the std collapse seen in v15. The floor (0.05) is the guard. **Watch
  `train/std_steer` and `train/std_longitudinal` in the short trial.**
- **25 m side range.** See the limitations above. In v16 it only matters when the car is
  already drifting more than 0.6 m toward a lane that reads as unsafe.
- **The lateral rules do not bring the car back.** `max(steer, 0)` stops further steering
  to the left, but the car keeps its current heading and can still cross the line.
  Restricting only, without adding behaviour, has this cost. On the outermost lanes
  (oncoming traffic on the left, curb on the right) the rules work as a soft barrier.
- **The acceleration rule is a speed limiter.** At or above the limit the throttle is
  cut, so near the limit the speed oscillates slightly. The rule also cuts throttle for
  parked cars and pedestrians on the path (v15's front sensor includes them).
- **Center-to-center distances** make the shield less conservative than its numbers
  suggest.
- **Fewer updates per hour.** Rollout 2048 (Arnav) vs 1024 (v15): half as many PPO updates
  per hour (about 98 vs 195 in 200k steps).
- **Shared trunk.** Arnav's network shares the first layers between actor and critic (v15's
  SB3 policy had separate 256x256 networks). Reward scaling keeps the value loss in range,
  but a large value loss can still dominate the shared layers.

## Suggestions for later (not implemented)

- **Shield rule for red and yellow lights**, the main failure in v15. When `o[9]` or `o[10]`
  is set and the stop line (`o[13]`) is within the braking distance for the current speed,
  treat accelerating as unsafe (cap at coast). Optionally require braking once the line is
  close; that adds a behaviour, so it needs a decision.
- **Junction rule.** Limit the speed while `o[6]` (distance to a LEFT/RIGHT maneuver) is
  small.
- **Solid lines.** Use the lane marking codes at d=0 (`o[30]`, `o[32]`) to treat crossing a
  solid line as unsafe (v15 charges -5 per illegal crossing).
- **Bumper gaps.** Subtract half vehicle lengths from v15's distances, or raise
  `base_safe_gap_m`.
- **Own-lane rear vehicle.** Pass it to the shield (for braking hard with a car close
  behind).

## What was tested (and what was not)

Run `python -m pytest tests -q` from `carla-rl/`. The tests use `FakeCarlaEnv` and never
import carla.

**Tested without CARLA:**
- the observation mapping on hand-built vectors: left ahead, left behind, right
  ahead/behind, alongside, no vehicle, no lane, junction, density and speed spread;
- that the speed-limit shield uses the current limit;
- that the normalization matches SB3's VecNormalize, updates in training, freezes in
  evaluation, and that the shield and estimator see raw values;
- policy input 94;
- two PPO updates;
- checkpoints every N steps, and save / load / resume with the normalizer, reward scaler and
  `num_timesteps`;
- `--resume` option sources and the learning-rate change;
- the truncation bootstrap, and that the bootstrap peek does not advance the estimator;
- that the wrapper returns reward, terminated, truncated and the environment's info
  unchanged;
- the continuous shield rules (throttle cut with the brake kept, left/right rules, off in
  junctions);
- the shield-penalty gradient;
- Gaussian log-probability and entropy, and the std floor (including that the std can grow
  again);
- clipping, the log-probability of the unclipped sample, and that the deterministic action
  is the mean;
- the `--store-executed-action` flag;
- `train_v16.main()` and `evaluate_v16.main()` end to end with a stand-in `carla_env` module:
  CarlaEnv arguments, saving, `--resume`, saving when stopped early, `env.close()`;
- that the FakeCarlaEnv bounds equal CarlaEnv's (by running CarlaEnv's own bound code from
  its source);
- that v15's action mapping lines in `carla_env.py` are still there.

**Not tested:** anything with a CARLA server. That includes whether the agent learns, real
shield behaviour, speed and lane behaviour, throughput (fps), the Slurm job and SIGTERM
under Slurm.
