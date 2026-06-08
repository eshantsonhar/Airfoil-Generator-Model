@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "PROJECT_ROOT=%~dp0"
pushd "%PROJECT_ROOT%"

set "BACKEND_HOST=127.0.0.1"
set "BACKEND_PORT=8000"
set "FRONTEND_PORT=5173"
set "VENV_PY=%PROJECT_ROOT%.venv\Scripts\python.exe"
set "BACKEND_URL=http://%BACKEND_HOST%:%BACKEND_PORT%"
set "FRONTEND_URL=http://127.0.0.1:%FRONTEND_PORT%"
set "EXIT_CODE=0"

rem ── venv ─────────────────────────────────────────────────────────────────
if not exist "%VENV_PY%" (
    echo [setup] Creating virtual environment...
    py -3 -m venv ".venv" 2>nul || python -m venv ".venv"
    if errorlevel 1 goto :fail
)

echo [setup] Installing Python dependencies...
"%VENV_PY%" -m pip install -q --upgrade pip setuptools wheel
if errorlevel 1 goto :fail
"%VENV_PY%" -m pip install -q -e ".[ui,meshing]"
if errorlevel 1 goto :fail

rem ── React frontend build (skip — Launch-UI.ps1 handles this) ─────────────────
rem Frontend build (including npm install + vite build) is performed by
rem scripts\Launch-UI.ps1 which checks for an existing dist/index.html first
rem and rebuilds only when needed.  Removing this step cuts redundant work.
rem ── Backend via PowerShell harness ───────────────────────────────────────────
:backend_only

rem call is required so errorlevel propagates back to this batch file
call powershell -NoProfile -ExecutionPolicy Bypass ^
    -File "%PROJECT_ROOT%scripts\Launch-UI.ps1" ^
    -Host %BACKEND_HOST% -Port %BACKEND_PORT%
if errorlevel 1 goto :fail

echo.
echo Backend:  %BACKEND_URL%
echo Platform: %BACKEND_URL%/platform
goto :end

:fail
echo.
echo [error] Platform launch failed.
set "EXIT_CODE=1"

:end
popd
endlocal
exit /b %EXIT_CODE%
