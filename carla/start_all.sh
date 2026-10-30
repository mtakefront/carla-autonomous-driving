#!/bin/bash
set -e

# Paths relative to this script, so the repo works from any location
CARLA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$CARLA_DIR/../carla-rl" && pwd)"
PYTHON="$PROJECT_DIR/env/bin/python"

CARLA_PID=""
LIVE_PID=""

cleanup() {
    echo "Stopping CARLA and Live View..."
    [[ -n "$LIVE_PID" ]] && kill "$LIVE_PID" 2>/dev/null || true
    [[ -n "$CARLA_PID" ]] && kill "$CARLA_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Starting CARLA..."
"$CARLA_DIR/start_carla.sh" 2000 &
CARLA_PID=$!

echo "Waiting for CARLA to become ready..."

ready=0
for i in {1..60}; do
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
"$PYTHON" "$PROJECT_DIR/spawn_vehicle.py"
echo "CARLA is ready!"
echo "Starting Live View..."

"$PYTHON" -u "$PROJECT_DIR/live_view.py" &
LIVE_PID=$!

echo "Live View: port 8080"
echo "Press Ctrl+C to stop both services."

# Blocks until Ctrl+C or CARLA exits; Live View crashing does not stop CARLA
wait "$CARLA_PID"
