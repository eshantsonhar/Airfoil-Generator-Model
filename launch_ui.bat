@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "PROJECT_ROOT=%~dp0"
pushd "%PROJECT_ROOT%"

set "HOST=127.0.0.1"
set "PORT=8000"
set "VENV_PY=%PROJECT_ROOT%.venv\Scripts\python.exe"
set "UI_URL=http://%HOST%:%PORT%/platform/"
set "EXIT_CODE=0"

title Airfoil Discovery Platform — Launching...

echo ============================================================
echo   Airfoil Discovery Platform — Launch
echo ============================================================
echo.

rem ── 1. Create venv if missing ──────────────────────────────────
if not exist "%VENV_PY%" (
    echo [1/5] Creating local virtual environment...
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 -m venv ".venv"
    ) else (
        python -m venv ".venv"
    )
    if errorlevel 1 (
        echo [FAIL] Could not create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [1/5] Virtual environment found.
)

rem ── 2. Install dependencies ─────────────────────────────────────
echo [2/5] Installing Python dependencies...
"%VENV_PY%" -m pip install --quiet --upgrade pip setuptools wheel >nul 2>&1
"%VENV_PY%" -m pip install --quiet -e ".[ui]" >nul 2>&1
if errorlevel 1 (
    echo [FAIL] pip install failed. Run manually:
    echo   "%VENV_PY%" -m pip install -e ".[ui]"
    pause
    exit /b 1
)
echo [OK] Python dependencies installed.

rem ── 3. Build frontend ───────────────────────────────────────────
echo [3/5] Checking frontend build...
set "DIST_HTML=%PROJECT_ROOT%frontend\dist\index.html"
if exist "%DIST_HTML%" (
    echo [OK] Frontend build exists.
) else (
    echo [3/5] Frontend build missing. Building...
    where npm >nul 2>nul
    if errorlevel 1 (
        echo [WARN] npm not found — frontend won't render.
        echo        Install Node.js from https://nodejs.org
    ) else (
        pushd "%PROJECT_ROOT%frontend"
        call npm install >nul 2>&1
        if errorlevel 1 (
            echo [FAIL] npm install failed.
            popd
            pause
            exit /b 1
        )
        call npm run build >nul 2>&1
        if errorlevel 1 (
            echo [FAIL] npm build failed.
            popd
            pause
            exit /b 1
        )
        popd
        echo [OK] Frontend built successfully.
    )
)

rem ── 4. Kill stale processes on port 8000 ────────────────────────
echo [4/5] Cleaning up previous instances...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo [INFO] Killing stale PID %%a on port 8000...
    taskkill /F /PID %%a >nul 2>&1
    if errorlevel 1 (
        echo [WARN] Could not kill PID %%a
    )
)
timeout /t 1 /nobreak >nul

rem Also remove stale PID file
if exist "%PROJECT_ROOT%.runtime\ui.pid" del "%PROJECT_ROOT%.runtime\ui.pid" >nul 2>&1

rem ── 5. Launch backend ───────────────────────────────────────────
echo [5/5] Starting backend...
echo.
"%VENV_PY%" "%PROJECT_ROOT%scripts\run_ui.py" --host %HOST% --port %PORT%
if errorlevel 1 (
    echo [FAIL] Backend failed to start.
    pause
    exit /b 1
)

rem ── If we get here, the backend launched and opened browser ─────
echo.
echo ============================================================
echo   Platform is running!
echo.
echo   %UI_URL%
echo.
echo   Backend persists after this window closes.
echo   To stop: run scripts\Stop-Server.ps1
echo   Restart: run this batch file again.
echo ============================================================
echo.

popd
endlocal
exit /b 0