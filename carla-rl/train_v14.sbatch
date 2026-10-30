#!/bin/bash
#SBATCH --job-name=carla-train-v14
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/sahin.o/carla-rl/slurm_train_v14_%j.out

# Starts CARLA, spawns the vehicle, runs train_v14.py and exits when
# training finishes (or at the 1 h limit). Training output: train_v14.log

CARLA_DIR="$HOME/carla"
PROJECT_DIR="$HOME/carla-rl"
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
SPAWN_OUT=$("$PYTHON" spawn_vehicle.py) || exit 1
echo "$SPAWN_OUT"
VEHICLE_ID=$(echo "$SPAWN_OUT" | awk '/Vehicle ID:/ {print $3}')
if [[ -z "$VEHICLE_ID" ]]; then
    echo "Could not read vehicle id."
    exit 1
fi

# Spawn point 0 is inside a junction; move the car to the same straight
# road (road 12) the earlier runs started on
echo "Moving vehicle to straight road..."
"$PYTHON" find_straight.py || exit 1

echo "Starting Live View (port 8080)..."
"$PYTHON" -u live_view.py > live_view_train_v14.log 2>&1 &
LIVE_PID=$!

echo "Training with VEHICLE_ID=$VEHICLE_ID..."
# Run in background + wait so SIGTERM at the time limit reaches the trap
VEHICLE_ID="$VEHICLE_ID" "$PYTHON" -u train_v14.py > train_v14.log 2>&1 &
TRAIN_PID=$!
trap 'kill -TERM "$TRAIN_PID" 2>/dev/null; wait "$TRAIN_PID"; cleanup' TERM
wait "$TRAIN_PID"
echo "Training exited with code $?"
