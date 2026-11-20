"""Train D2-Drive (Arnav's PPO variant) on the v15 environment with v15's
continuous [steer, longitudinal] actions (v16). See INTEGRATION_NOTES.md.

Same basic options as train_v15.py (--steps, --traffic, --pedestrians,
--no-render, --resume) plus the D2-Drive ones (--help).
"""

import argparse
import signal
import sys
import traceback

import torch

from d2drive_common import read_checkpoint, resolve_checkpoint_path
from d2drive_continuous import (
    ContinuousDecisionStats,
    D2ContinuousWrapper,
    D2DriveContinuousAgent,
    V16Config,
)
from d2drive_training import (
    COMMON_CLI_FIELDS,
    add_common_train_args,
    format_sources,
    resolve_dataclass,
    run_training,
)

MODEL_NAME = "d2drive_v16"
METRICS_PATH = "train_v16_metrics.csv"
CLI_FIELDS = dict(COMMON_CLI_FIELDS, min_std="min_std",
                  lateral_shield_offset="lateral_shield_offset_m")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_train_args(parser, MODEL_NAME)
    parser.add_argument("--min-std", type=float, default=None,
                        help="lower bound of the policy's std (default 0.05)")
    parser.add_argument("--lateral-shield-offset", type=float, default=None,
                        help="lane offset (m) beyond which the lateral shield rules "
                             "apply (default 0.6)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    # One thread: the MLP is tiny, and extra threads only compete with CARLA
    # for the job's few CPUs.
    torch.set_num_threads(1)
    # SIGTERM (kill, Slurm time limit) -> normal exit so finally/close() runs
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    resume_path, checkpoint = None, None
    if args.resume:
        resume_path = resolve_checkpoint_path(args.resume)
        checkpoint = read_checkpoint(resume_path)
    cfg, sources = resolve_dataclass(
        V16Config, (checkpoint or {}).get("config"), args, CLI_FIELDS,
        fixed={"device": "cpu"})
    print(format_sources("D2-Drive v16 config (value, source)", cfg, sources), flush=True)

    # Imported here so the tests can import this module without carla.
    from carla_env import CarlaEnv

    # routes 60-600 m, with autopilot traffic and pedestrians (as train_v15.py)
    raw_env = CarlaEnv(min_route_length=60.0, traffic_vehicles=args.traffic,
                       pedestrians=args.pedestrians, no_rendering=args.no_render)
    env = D2ContinuousWrapper(raw_env)

    agent = None
    completed = False
    try:
        agent = D2DriveContinuousAgent(cfg)
        if resume_path:
            agent.load(resume_path)
            print(f"Resuming {resume_path} at {agent.num_timesteps} timesteps", flush=True)
        run_training(env, agent, args.steps, model_name=MODEL_NAME,
                     decision_stats=ContinuousDecisionStats(),
                     metrics_path=METRICS_PATH,
                     print_fn=lambda text: print(text, flush=True))
        completed = True

    except BaseException:
        # Show the cause now; cleanup below can take a while.
        traceback.print_exc()
        sys.stdout.flush()
        raise

    finally:
        # Also save when stopped early (Ctrl+C, Slurm time limit)
        if agent is not None:
            agent.save(f"{MODEL_NAME}.pt")
            status = "completed" if completed else "stopped early, saved"
            print(f"Training {status}: {MODEL_NAME}.pt "
                  f"({agent.num_timesteps} timesteps)", flush=True)
        raw_env.close()


if __name__ == "__main__":
    main()
