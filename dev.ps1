# Voicebox Dev Launcher
# Starts the Python backend and Tauri desktop app in separate windows.

$Root = $PSScriptRoot

# Ensure cargo is on the PATH
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    $env:PATH += ";$env:USERPROFILE\.cargo\bin"
}

# Free port 17493 if already occupied
$portPid = (Get-NetTCPConnection -LocalPort 17493 -ErrorAction SilentlyContinue).OwningProcess
if ($portPid) {
    Write-Host "Killing process $portPid occupying port 17493..."
    Stop-Process -Id $portPid -Force
    Start-Sleep -Milliseconds 500
}

# Start backend in a new PowerShell window
Write-Host "Starting backend on http://localhost:17493 ..."
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$Root'; backend\venv\Scripts\uvicorn backend.main:app --reload --port 17493"
)

# Give the backend a moment to start before launching Tauri
Start-Sleep -Seconds 2

# Start Tauri in a new PowerShell window
Write-Host "Starting Tauri desktop app..."
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "`$env:PATH += ';`$env:USERPROFILE\.cargo\bin'; Set-Location '$Root\tauri'; bun run tauri dev"
)

Write-Host ""
Write-Host "Both processes launched. Close their windows to stop them."
