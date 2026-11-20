"""Evaluate a v16 D2-Drive model (continuous actions) with the deterministic
policy (the Gaussian mean, then the shield) on random routes.

Reports: goal rate, end reasons, pass rate per maneuver, shield veto rate and
rule rates, mean speed, mean |lane offset|. Normalization statistics are
frozen (taken from the checkpoint).
"""

import argparse
import signal
import sys

import torch

from d2drive_common import read_checkpoint, resolve_checkpoint_path
from d2drive_continuous import (
    ContinuousDecisionStats,
    D2ContinuousWrapper,
    D2DriveContinuousAgent,
    V16Config,
)
from d2drive_training import format_evaluation, resolve_dataclass, run_evaluation


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="d2drive_v16", help="checkpoint (.pt optional)")
    parser.add_argument("--episodes", type=int, default=20, help="random routes to drive")
    parser.add_argument("--traffic", type=int, default=15, help="autopilot cars")
    parser.add_argument("--pedestrians", type=int, default=20)
    parser.add_argument("--no-render", action="store_true",
                        help="CARLA draws nothing: faster, but no Live View")
    parser.add_argument("--seed", type=int, default=0, help="route sampling seed")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    torch.set_num_threads(1)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    path = resolve_checkpoint_path(args.model)
    cfg, _ = resolve_dataclass(V16Config, read_checkpoint(path).get("config"), args, {},
                               fixed={"device": "cpu"})
    agent = D2DriveContinuousAgent(cfg)
    agent.load(path, load_optimizer=False)
    agent.set_training(False)
    print(f"Evaluating {path} ({agent.num_timesteps} training timesteps)", flush=True)

    from carla_env import CarlaEnv

    raw_env = CarlaEnv(min_route_length=60.0, traffic_vehicles=args.traffic,
                       pedestrians=args.pedestrians, no_rendering=args.no_render)
    try:
        summary = run_evaluation(D2ContinuousWrapper(raw_env), agent, args.episodes,
                                 seed=args.seed, decision_stats=ContinuousDecisionStats(),
                                 print_fn=lambda text: print(text, flush=True))
        print(format_evaluation(summary), flush=True)
    finally:
        raw_env.close()


if __name__ == "__main__":
    main()
