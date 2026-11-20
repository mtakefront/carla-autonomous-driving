#!/bin/bash
#SBATCH --job-name=carla-train-v16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=03:00:00
# SIGTERM 3 min before the limit, so the model is saved before Slurm kills it
#SBATCH --signal=B:TERM@180
#SBATCH --output=slurm_train_v16_%j.out

# Starts CARLA, spawns the vehicle and trains for up to 3 h without rendering
# (no Live View), saving the model at the end. Training output: train_v16.log
# Submit from the carla-rl folder; extra arguments go to train_v16.py:
#   cd carla-rl && sbatch train_v16.sbatch                          # fresh
#   cd carla-rl && sbatch train_v16.sbatch --resume d2drive_v16     # continue

# sbatch runs a copy of this file, so paths come from the submit directory
PROJECT_DIR="$SLURM_SUBMIT_DIR"
CARLA_DIR="$PROJECT_DIR/../carla"
PYTHON="$PROJECT_DIR/env/bin/python"

CARLA_PID=""
LIVE_PID=""

cleanup() {
    echo "Stopping Live View and CARLA..."
    [[ -n "$LIVE_PID" ]] && kill "$LIVE_PID" 2>/dev/null || true
    [[ -n "$CARLA_PID" ]] && kill "$CARLA_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

cd "$PROJECT_DIR"
echo "Node: $(hostname)"

echo "Starting CARLA..."
"$CARLA_DIR/start_carla.sh" 2000 &
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

echo "Spawning vehicle..."
"$PYTHON" spawn_vehicle.py || exit 1

echo "Training..."
# Run in background + wait so SIGTERM at the time limit reaches the trap.
# 200k steps is ~2.7 h at 21 fps (the rendered rate); without rendering it is
# faster and finishes early. If it doesn't, the time limit stops it and it saves.
"$PYTHON" -u train_v16.py --no-render --steps 200000 "$@" > train_v16.log 2>&1 &
TRAIN_PID=$!
trap 'kill -TERM "$TRAIN_PID" 2>/dev/null; wait "$TRAIN_PID"; cleanup' TERM
wait "$TRAIN_PID"
echo "Training exited with code $?"
