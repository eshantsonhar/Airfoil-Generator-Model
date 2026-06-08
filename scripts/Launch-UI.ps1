# Launch-UI.ps1
# Properly detached server launcher with readiness probe, PID tracking, and timeout.
# Does NOT tail stdout/stderr after launch. Returns in ~2-8 s.

[CmdletBinding()]
param(
    [string]$ListenHost = "127.0.0.1",
    [int]$ListenPort = 8000,
    [switch]$OpenBrowser
)

$ErrorActionPreference = "Stop"
$PSStyle.OutputRendering = "PlainText"

$PROJECT_ROOT = Split-Path -Parent $PSScriptRoot
$SRC_ROOT     = Join-Path $PROJECT_ROOT "src"
$LOG_DIR      = Join-Path $PROJECT_ROOT "logs"
$PID_FILE     = Join-Path $PROJECT_ROOT ".runtime" "ui.pid"

# ── 1. Create log and runtime dirs ──────────────────────────────────────────
New-Item -ItemType Directory -Path $LOG_DIR  -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path $PID_FILE) -Force | Out-Null

# ── 2. Start uvicorn detached, output to log files ──────────────────────────
$pythonExe = Join-Path $PROJECT_ROOT ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Error "python.exe not found at $pythonExe"
    exit 1
}

$scriptToRun = Join-Path $PROJECT_ROOT "scripts\run_ui.py"
$stdoutLog   = Join-Path $LOG_DIR "ui_stdout.log"
$stderrLog   = Join-Path $LOG_DIR "ui_stderr.log"

# Fresh log files
"" | Set-Content -Path $stdoutLog -Encoding utf8
"" | Set-Content -Path $stderrLog -Encoding utf8

$psi = @{
    FilePath               = $pythonExe
    ArgumentList           = "`"$scriptToRun`" --host $ListenHost --port $ListenPort"
    WorkingDirectory        = $PROJECT_ROOT
    RedirectStandardOutput  = $stdoutLog
    RedirectStandardError   = $stderrLog
    NoNewWindow            = $true
    PassThru               = $true
}

$proc = Start-Process @psi

# Write PID immediately so Stop-Server.ps1 can find it
$proc.Id | Set-Content -Path $PID_FILE -Encoding ascii

Write-Host "[launch] uvicorn PID=$($proc.Id) -> $ListenHost : $ListenPort"
Write-Host "[launch] stdout → $stdoutLog"
Write-Host "[launch] stderr → $stderrLog"
Write-Host "[launch] PID file → $PID_FILE"

# ── 3. Readiness probe (max 20 s) ───────────────────────────────────────────
$url  = "http://${ListenHost}:${ListenPort}/platform/"
$max  = 20
$ok   = $false
$attemptsLog = @()

for ($i = 0; $i -lt $max; $i++) {
    try {
        $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        if ($r.StatusCode -eq 200) {
            Write-Host "[ready] HTTP 200 — backend healthy (attempt $($i+1)/$max)"
            $ok = $true
            break
        }
    }
    catch {
        # Wait 1 s and retry
    }
    Start-Sleep -Milliseconds 1000
}

if (-not $ok) {
    Write-Error "[fatal] Backend did not respond within ${max}s"
    Write-Host "--- stdout ---"
    Get-Content $stdoutLog -ErrorAction SilentlyContinue | Select-Object -Last 40
    Write-Host "--- stderr ---"
    Get-Content $stderrLog -ErrorAction SilentlyContinue | Select-Object -Last 40
    exit 1
}

# ── 4. PID file is stable – delete the stale starter handle ─────────────────
if ($null -ne $proc) { try { $proc.Dispose() } catch {} }

# ── 5. Optionally open browser ──────────────────────────────────────────────
if ($OpenBrowser) {
    Start-Process $url
    Write-Host "[launch] Browser opened → $url"
}

Write-Host "[done] Backend running at $url  PID=$(Get-Content $PID_FILE)"
