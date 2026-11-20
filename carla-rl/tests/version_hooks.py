"""What the shared tests (test_common.py) need from this version: v16,
continuous [steer, longitudinal] actions."""

from d2drive_continuous import (
    ContinuousDecisionStats,
    D2ContinuousWrapper,
    D2DriveContinuousAgent,
    V16Config,
)
from fake_carla_env import FakeCarlaEnv

VERSION = "v16"
POLICY_INPUT_DIM = 94
CONFIG_CLASS = V16Config
AGENT_CLASS = D2DriveContinuousAgent
FAKE_CARLA_ENV = FakeCarlaEnv      # stands in for carla_env.CarlaEnv in main() tests


def small_config(**overrides):
    """A config small enough for fast tests."""
    values = dict(rollout_steps=64, minibatch_size=32, update_epochs=2, hidden_dim=32)
    values.update(overrides)
    return V16Config(**values)


def make_agent(**overrides):
    return D2DriveContinuousAgent(small_config(**overrides))


def make_env(seed=0, **fake_kwargs):
    return D2ContinuousWrapper(FakeCarlaEnv(seed=seed, **fake_kwargs))


def inner_env(env):
    """The FakeCarlaEnv inside make_env()'s wrapper."""
    return env.env


def make_decision_stats():
    return ContinuousDecisionStats()
