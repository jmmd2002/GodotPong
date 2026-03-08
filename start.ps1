# Activate virtual environment
& "python_files\.venv\Scripts\Activate.ps1"

# Read num_workers and base port from Python config
$BASE_PORT = 5000
$NUM_WORKERS = [int](python -c @"
import yaml
with open('python_files/config/QAgent_coach.yaml') as f:
    c = yaml.safe_load(f)
print(c.get('model', {}).get('num_workers', 1))
"@)

Write-Host "Starting $NUM_WORKERS Godot instance(s) on ports $BASE_PORT to $($BASE_PORT + $NUM_WORKERS - 1)..."

# Kill any lingering processes still holding the target ports
for ($i = 0; $i -lt $NUM_WORKERS; $i++) {
    $PORT = $BASE_PORT + $i
    $conn = Get-NetTCPConnection -LocalPort $PORT -ErrorAction SilentlyContinue
    if ($conn) {
        $conn | ForEach-Object {
            $pid_ = $_.OwningProcess
            Write-Host "Killing leftover process on port $PORT (PID: $pid_)"
            Stop-Process -Id $pid_ -Force -ErrorAction SilentlyContinue
        }
    }
}

# Start Python server
$pythonProc = Start-Process -FilePath "python" -ArgumentList "-u", "python_files/main.py" -PassThru -NoNewWindow
Write-Host "Python started (PID: $($pythonProc.Id))"

# Give Python a moment to bind its sockets before Godot connects
Start-Sleep -Milliseconds 100

# Launch one Godot instance per worker, each on its own port
$godotProcs = @()
for ($i = 0; $i -lt $NUM_WORKERS; $i++) {
    $PORT = $BASE_PORT + $i
    $proc = Start-Process -FilePath "godot" -ArgumentList "--headless", "--path", "godot_files/", "--", "--port", $PORT -PassThru -NoNewWindow
    $godotProcs += $proc
    Write-Host "Godot instance $i started on port $PORT (PID: $($proc.Id))"
    Start-Sleep -Milliseconds 100
}

# Cleanup function
function Cleanup {
    Write-Host "Stopping Godot instances..."
    foreach ($proc in $godotProcs) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
    Write-Host "Stopping Python..."
    Stop-Process -Id $pythonProc.Id -Force -ErrorAction SilentlyContinue
}

# Register cleanup on Ctrl+C
try {
    # Wait for all Godot instances to exit
    foreach ($proc in $godotProcs) {
        $proc.WaitForExit()
    }
} finally {
    Cleanup
}
