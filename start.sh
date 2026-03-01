#!/bin/bash

# Activate virtual environment
source python_files/.venv/bin/activate

# Start Python in a new process group
setsid python -u python_files/main.py &
PYTHON_PID=$!

echo "Python started (PID: $PYTHON_PID)"

# Cleanup function (called on exit)
cleanup() {
    echo "Stopping Python..."
    kill -- -$PYTHON_PID 2>/dev/null
    wait $PYTHON_PID 2>/dev/null
}

# Ensure cleanup runs when script exits
trap cleanup EXIT

# Launch Godot (foreground)
godot --path godot_files/
