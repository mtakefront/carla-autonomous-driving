"""Puts carla-rl/ and tests/ first on sys.path, so the tests import the
project modules and carla-rl/d2drive_ppo.py (not the copy at the repo root).

The tests never import carla: they use FakeCarlaEnv (fake_carla_env.py).
Run from carla-rl/:  python -m pytest tests -q
"""

import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
CARLA_RL_DIR = os.path.dirname(TESTS_DIR)
for _path in (TESTS_DIR, CARLA_RL_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)
