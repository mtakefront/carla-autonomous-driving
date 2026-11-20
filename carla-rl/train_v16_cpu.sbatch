#!/bin/bash
#SBATCH --job-name=carla-train-v16-cpu
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
# SIGTERM 3 min before the limit, so the model is saved before Slurm kills it
#SBATCH --signal=B:TERM@180
#SBATCH --output=slurm_train_v16_cpu_%j.out

# train_v16.sbatch without a GPU. CARLA runs with UE4's null renderer
# (-nullrhi): nothing is drawn, so the GPU render-thread crash
# ("GameThread timed out waiting for RenderThread", signal 11) cannot happen.
# v16 needs no rendering: it uses no camera (the observation comes from the
# map and actor queries, plus the collision and lane-invasion sensors) and
# trains with --no-render. Live View (camera) does not work in this mode.
#
# Submit from the carla-rl folder; extra arguments go to train_v16.py:
#   cd carla-rl && sbatch train_v16_cpu.sbatch                          # fresh
#   cd carla-rl && sbatch train_v16_cpu.sbatch --resume d2drive_v16     # continue
# Short test on the 1 h "sharing" partition (no queue in practice):
#   sbatch -p sharing --time=00:59:00 train_v16_cpu.sbatch --steps 10000 --traffic 0 --pedestrians 0
# Python: $PROJECT_DIR/env/bin/python unless PYTHON is set when submitting:
#   PYTHON=/path/to/env/bin/python sbatch train_v16_cpu.sbatch ...

# sbatch runs a copy of this file, so paths come from the submit directory
PROJECT_DIR="$SLURM_SUBMIT_DIR"
CARLA_DIR="$PROJECT_DIR/../carla"
PYTHON="${PYTHON:-$PROJECT_DIR/env/bin/python}"

CARLA_PID=""

cleanup() {
    echo "Stopping CARLA..."
    [[ -n "$CARLA_PID" ]] && kill "$CARLA_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

cd "$PROJECT_DIR"
echo "Node: $(hostname), CPUs: ${SLURM_CPUS_PER_TASK:-?}, Python: $PYTHON"

# Fail fast if the environment lacks a package (before CARLA's start-up).
if ! "$PYTHON" -c 'import carla, gymnasium, networkx, numpy, torch' ; then
    echo "Python environment $PYTHON is missing packages (see requirements.txt)."
    exit 1
fi

echo "Starting CARLA (no GPU, -nullrhi)..."
# As carla/start_carla.sh, without --nv and the NVIDIA EGL/Vulkan binds.
XDG_DIR="/tmp/xdg-runtime-$USER"
mkdir -p "$XDG_DIR" && chmod 700 "$XDG_DIR"
apptainer exec \
  --env XDG_RUNTIME_DIR="$XDG_DIR" \
  "$CARLA_DIR/carla-0.9.16.sif" \
  /workspace/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping CarlaUE4 \
  -RenderOffScreen -nullrhi -nosound -carla-rpc-port=2000 &
CARLA_PID=$!

ready=0
for i in {1..90}; do
    if ! kill -0 "$CARLA_PID" 2>/dev/null; then
        echo "CARLA exited unexpectedly."
        exit 1
    fi
    if "$PYTHON" -c '
import carla
client = carla.Client("127.0.0.1", 2000)
client.set_timeout(2.0)
client.get_world()
' >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 2
done

if [[ "$ready" -ne 1 ]]; then
    echo "CARLA did not become ready."
    exit 1
fi
echo "CARLA ready after ~$((i * 4)) s"

echo "Spawning vehicle..."
"$PYTHON" spawn_vehicle.py || exit 1

echo "Training..."
# Run in background + wait so SIGTERM at the time limit reaches the trap.
"$PYTHON" -u train_v16.py --no-render --steps 200000 "$@" > train_v16.log 2>&1 &
TRAIN_PID=$!
trap 'kill -TERM "$TRAIN_PID" 2>/dev/null; wait "$TRAIN_PID"; cleanup' TERM
wait "$TRAIN_PID"
echo "Training exited with code $?"
