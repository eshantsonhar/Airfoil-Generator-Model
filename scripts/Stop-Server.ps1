# Stop-Server.ps1
# Hard-kill the uvicorn platform server using the persisted PID file.
# Runs in under 2 s. Never waits for the process.

[CmdletBinding()]
param()

$PSStyle.OutputRendering = "PlainText"
$ErrorActionPreference   = "SilentlyContinue"

$PROJECT_ROOT = Split-Path -Parent $PSScriptRoot
$PID_FILE     = Join-Path $PROJECT_ROOT ".runtime" "ui.pid"

if (-not (Test-Path $PID_FILE)) {
    Write-Host "[stop] No PID file at $PID_FILE — nothing to stop"
    exit 0
}

$pidTxt = Get-Content $PID_FILE -Raw
$pid    = [int]($pidTxt.Trim())

$proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
if (-not $proc) {
    Write-Host "[stop] PID $pid not running — cleaning PID file"
    Remove-Item $PID_FILE -Force
    exit 0
}

Write-Host "[stop] Killing PID $pid ($($proc.ProcessName))..."
Stop-Process -Id $pid -Force

# Also kill child processes (Python/uwsgi/uvicorn sub-workers)
Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $pid } | ForEach-Object {
    Write-Host "[stop] Killing child PID $($_.ProcessId) ($($_.Name))"
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}

Remove-Item $PID_FILE -Force
Write-Host "[stop] Done."
