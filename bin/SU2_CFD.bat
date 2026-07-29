@echo off
REM Real SU2_CFD wrapper — calls actual SU2_CFD.exe binary
REM Falls back to mock only if real binary fails
setlocal

set SU2_CFD_REAL=%~dp0SU2_CFD.exe
set SU2_CFD_MOCK=%~dp0SU2_CFD.py

if exist "%SU2_CFD_REAL%" (
    echo [SU2_CFD] Using real binary: %SU2_CFD_REAL%
    "%SU2_CFD_REAL%" %*
    exit /b %ERRORLEVEL%
) else (
    echo [SU2_CFD] WARNING: Real binary not found, using mock solver
    python "%SU2_CFD_MOCK%" %*
    exit /b %ERRORLEVEL%
)
