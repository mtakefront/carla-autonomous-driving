"""FakeCarlaEnv: a stand-in for carla_env.CarlaEnv without the carla package.

Same observation space (75 values, same bounds), same action space
([steer, longitudinal] in [-1, 1]) and the same info keys as CarlaEnv.step().
Observations are random within the bounds (categorical fields 0/1), episodes
end at random with one of v15's reasons, and the time limit truncates.

The bounds are copied from CarlaEnv.__init__; test_common.py checks them
against carla_env.py's source, so they cannot drift apart silently.
"""

import copy

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# carla_env.py constants
SIDE_RANGE_M = 25.0
FRONT_RANGE_M = 40.0
MAX_LANES = 6
VIRTUAL_MARKING = 11
LOOKAHEAD = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0)

FAILURES = ("collision", "off_road", "wrong_road", "wrong_way", "red_light",
            "stop_sign", "stuck")
ENDINGS = FAILURES + ("goal",)
MANEUVERS = ("LEFT", "RIGHT", "STRAIGHT")
BINARY_FIELDS = (3, 4, 5, 9, 10, 11, 12, 14, 15, 20, 23, 26)


def v15_observation_bounds():
    """(low, high) exactly as built in CarlaEnv.__init__."""
    low = ([0.0, 0.0, -1.0] + [0.0] * 3 + [0.0] + [0.0, 0.0]
           + [0.0] * 4 + [0.0, 0.0] + [0.0, 0.0, -30.0]
           + [-3.0, -90.0] + [0.0, -SIDE_RANGE_M, -30.0] * 3)
    high = ([200.0, 200.0, 1.0] + [1.0] * 3 + [100.0] + [float(MAX_LANES)] * 2
            + [1.0] * 4 + [100.0, 1.0] + [1.0, FRONT_RANGE_M, 30.0]
            + [3.0, 90.0] + [1.0, SIDE_RANGE_M, 30.0] * 3)
    for distance in LOOKAHEAD:
        for _side in ("left", "right"):
            if distance == 0.0:
                low.extend([-60.0, 0.0])
                high.extend([60.0, float(VIRTUAL_MARKING)])
            else:
                low.extend([-60.0, -60.0, 0.0])
                high.extend([60.0, 60.0, float(VIRTUAL_MARKING)])
    return np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)


def make_v15_obs(overrides=None):
    """A hand-built v15 observation: 36 km/h, limit 50 km/h, one empty
    same-direction lane on each side, nothing ahead, centered in the lane.
    `overrides` maps observation index -> value."""
    obs = np.zeros(75, dtype=np.float32)
    obs[0] = 36.0     # speed km/h
    obs[1] = 50.0     # speed limit km/h
    obs[6] = 100.0    # no maneuver close
    obs[7] = 1.0      # lanes to the left
    obs[8] = 1.0      # lanes to the right
    obs[13] = 100.0   # no stop line
    obs[16] = FRONT_RANGE_M
    for index, value in (overrides or {}).items():
        obs[index] = value
    return obs


class FakeCarlaEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed=0, max_steps=200, end_probability=0.02,
                 maneuver_probability=0.05):
        super().__init__()
        low, high = v15_observation_bounds()
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32),
                                       high=np.array([1.0, 1.0], dtype=np.float32),
                                       dtype=np.float32)
        self.max_steps = int(max_steps)
        self.end_probability = float(end_probability)
        self.maneuver_probability = float(maneuver_probability)
        self.rng = np.random.default_rng(seed)
        self.step_scale = 2.0          # CarlaEnv: fixed_delta_seconds / 0.05
        self.steps = 0
        self.actions = []              # every action passed to step()
        self.step_returns = []         # deep copies of what step() returned
        self.reset_returns = []
        self.closed = False
        self.route_length_m = 100.0
        self._parts = {}

    # Hooks for subclasses (v17's fake road adds vehicle motion); no-ops here.
    def _on_reset(self):
        pass

    def _advance(self, action):
        pass

    def _observation(self):
        low, high = self.observation_space.low, self.observation_space.high
        obs = self.rng.uniform(low, high).astype(np.float32)
        obs[list(BINARY_FIELDS)] = self.rng.integers(0, 2, size=len(BINARY_FIELDS))
        obs[[7, 8]] = self.rng.integers(0, 3, size=2)
        obs[1] = self.rng.choice([0.0, 30.0, 50.0])
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.steps = 0
        self._parts = {}
        self._on_reset()
        result = (self._observation(),
                  {"route_length_m": self.route_length_m, "start_road_id": 1, "goal_road_id": 2})
        self.reset_returns.append(copy.deepcopy(result))
        return result

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        assert action.shape == (2,), action
        assert np.all(np.abs(action) <= 1.0 + 1e-6), action
        self.actions.append(action.copy())
        self.steps += 1
        self._advance(action)

        reason = None
        if self.rng.random() < self.end_probability:
            reason = ENDINGS[int(self.rng.integers(len(ENDINGS)))]
        terminated = reason is not None
        truncated = bool(self.steps >= self.max_steps and not terminated)
        if truncated:
            reason = "time_limit"
        parts = {
            "progress": float(self.rng.normal(1.0, 0.3)),
            "speed": float(-abs(self.rng.normal(0.0, 0.1))),
            "step": -0.04,
            "illegal_crossing": 0.0,
            "lane": float(-abs(self.rng.normal(0.0, 0.05))),
        }
        if reason in FAILURES:
            parts = {"terminal": -30.0}
        elif reason == "goal":
            parts["terminal"] = 50.0
        reward = sum(parts.values())
        for name, value in parts.items():
            self._parts[name] = self._parts.get(name, 0.0) + value

        maneuver_results = []
        if self.rng.random() < self.maneuver_probability:
            maneuver_results.append((MANEUVERS[int(self.rng.integers(3))],
                                     bool(self.rng.integers(2))))
        info = {
            "termination_reason": reason,
            "maneuver_results": maneuver_results,
            "reward_parts": dict(self._parts) if reason else None,
            "route_progress_m": float(self.steps),
            "route_remaining_m": float(max(0, self.max_steps - self.steps)),
        }
        result = (self._observation(), float(reward), bool(terminated), bool(truncated), info)
        self.step_returns.append(copy.deepcopy(result))
        return result

    def close(self):
        self.closed = True
