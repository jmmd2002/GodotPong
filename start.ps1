# Activate virtual environment
& "python_files\.venv\Scripts\Activate.ps1"

# Start Python process
$python = Start-Process python -ArgumentList "python_files/main.py" -PassThru

Write-Host "Python started (PID: $($python.Id))"

# Start Godot and wait for it to exit
$godot = Start-Process godot -ArgumentList "--headless --path godot_files/" -PassThru -Wait

# When Godot closes, stop Python
Write-Host "Stopping Python..."
Stop-Process -Id $python.Id -Force
