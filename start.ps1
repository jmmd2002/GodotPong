# Activate virtual environment
& "python_files\.venv\Scripts\Activate.ps1"

# Read num_workers and base port from Python config
$BASE_PORT = 5000
$NUM_WORKERS = [int](python -c @"
import yaml
with open('python_files/config/PolicyGradientDNN_coach.yaml') as f:
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
$pythonProc = $null
$pythonProc = Start-Process -FilePath "python" -ArgumentList "-u", "python_files/main.py" -PassThru -NoNewWindow
Write-Host "Python started (PID: $($pythonProc.Id))"

# Give Python a moment to bind its sockets before Godot connects
Start-Sleep -Milliseconds 100

# Resolve Godot executable (env override, PATH, local, common Windows locations)
function Resolve-GodotExecutable {
    # 1) Explicit env var
    if ($env:GODOT_EXE -and (Test-Path $env:GODOT_EXE)) {
        return (Resolve-Path $env:GODOT_EXE).Path
    }

    # 2) PATH commands
    foreach ($candidate in @("godot", "godot4", "godot4.exe", "godot.exe")) {
        $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($cmd -and $cmd.Source -and (Test-Path $cmd.Source)) {
            return $cmd.Source
        }
    }

    # 3) Local project/script folder
    $localPatterns = @(
        (Join-Path $PSScriptRoot "godot.exe"),
        (Join-Path $PSScriptRoot "Godot*.exe"),
        (Join-Path (Get-Location).Path "godot.exe"),
        (Join-Path (Get-Location).Path "Godot*.exe")
    )

    foreach ($pattern in $localPatterns) {
        $found = Get-ChildItem -Path $pattern -File -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if ($found) {
            return $found.FullName
        }
    }

    # 4) Common Windows install/download locations
    $searchPatterns = @(
        (Join-Path $env:ProgramFiles "Godot\Godot*.exe"),
        (Join-Path $env:ProgramFiles "Godot Engine\Godot*.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Godot\Godot*.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Godot Engine\Godot*.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Godot\Godot*.exe"),
        (Join-Path $env:USERPROFILE "Downloads\Godot*.exe"),
        (Join-Path $env:USERPROFILE "Downloads\Godot*\Godot*.exe")
    )

    foreach ($pattern in $searchPatterns) {
        $found = Get-ChildItem -Path $pattern -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -notmatch "_mono" } |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if ($found) {
            return $found.FullName
        }
    }

    return $null
}

$godotExecutable = Resolve-GodotExecutable

if (-not $godotExecutable) {
    Write-Host "Godot executable not auto-detected."
    $manualPath = Read-Host "Enter full path to Godot executable (or press Enter to cancel)"
    if ($manualPath -and (Test-Path $manualPath)) {
        $godotExecutable = (Resolve-Path $manualPath).Path
    }
}

if (-not $godotExecutable) {
    throw "Godot executable not found. Set GODOT_EXE to full path (e.g. C:\\Tools\\Godot\\Godot_v4.x-stable_win64_console.exe), or install/add Godot to PATH."
}

Write-Host "Using Godot executable: $godotExecutable"

# Import project if .godot cache is missing (required for headless runs)
if (-not (Test-Path (Join-Path (Get-Location).Path "godot_files\.godot"))) {
    Write-Host "Importing Godot project (first run)..."
    & $godotExecutable --headless --path "godot_files/" --import
    Write-Host "Import complete."
}

# Launch one Godot instance per worker, each on its own port
$godotProcs = @()
for ($i = 0; $i -lt $NUM_WORKERS; $i++) {
    $PORT = $BASE_PORT + $i
    $proc = Start-Process -FilePath $godotExecutable -ArgumentList "--headless", "--path", "godot_files/", "--", "--port", $PORT -PassThru -NoNewWindow
    $godotProcs += $proc
    Write-Host "Godot instance $i started on port $PORT (PID: $($proc.Id))"
    Start-Sleep -Milliseconds 100
}

# Cleanup function
function Cleanup {
    Write-Host "Stopping Godot instances..."
    foreach ($proc in $godotProcs) {
        if ($proc -and $proc.Id) {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        }
    }
    Write-Host "Stopping Python..."
    if ($pythonProc -and $pythonProc.Id) {
        Stop-Process -Id $pythonProc.Id -Force -ErrorAction SilentlyContinue
    } else {
        Write-Host "Python process was not started (nothing to stop)."
    }
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
