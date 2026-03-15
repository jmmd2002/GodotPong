#!/bin/bash

# Activate virtual environment
source python_files/.venv/bin/activate

# Read num_workers and base port from Python config
BASE_PORT=5000
NUM_WORKERS=$(python -c "
import yaml
with open('python_files/config/PolicyGradientDNN_student.yaml') as f:
    c = yaml.safe_load(f)
print(c.get('model', {}).get('num_workers', 1))
")

echo "Starting $NUM_WORKERS Godot instance(s) on ports $BASE_PORT to $((BASE_PORT + NUM_WORKERS - 1))..."

# Kill any lingering processes still holding the target ports
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    PORT=$((BASE_PORT + i))
    PIDS=$(fuser ${PORT}/tcp 2>/dev/null)
    if [ -n "$PIDS" ]; then
        echo "Killing leftover process(es) on port $PORT: $PIDS"
        fuser -k ${PORT}/tcp 2>/dev/null
    fi
done

# Start Python server in a new process group
setsid python -u python_files/main.py &
PYTHON_PID=$!
echo "Python started (PID: $PYTHON_PID)"

# Give Python a moment to bind its sockets before Godot connects
sleep 0.1

# Launch one Godot instance per worker, each on its own port.
# A 1s pause between launches lets you identify which window is worker 0.
GODOT_PIDS=()
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    PORT=$((BASE_PORT + i))
    godot --headless --path godot_files/ -- --port $PORT &
    GODOT_PIDS+=("$!")
    echo "Godot instance $i started on port $PORT (PID: ${GODOT_PIDS[-1]})"
    sleep 0.1
done

# Cleanup function (called on exit)
cleanup() {
    echo "Stopping Godot instances..."
    for pid in "${GODOT_PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    echo "Stopping Python..."
    kill -- -$PYTHON_PID 2>/dev/null
    wait $PYTHON_PID 2>/dev/null
}

# Ensure cleanup runs when script exits (Ctrl+C or all Godot windows closed)
trap cleanup EXIT

# Wait for all Godot instances to exit
for pid in "${GODOT_PIDS[@]}"; do
    wait "$pid"
done
