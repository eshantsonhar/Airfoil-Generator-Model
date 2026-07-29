@echo off
REM Real SU2_CFD wrapper — calls actual SU2_CFD.exe binary ONLY.
REM The mock SU2_CFD.py is NEVER invoked from this script.
setlocal

set SU2_CFD_REAL=%~dp0SU2_CFD.exe

if exist "%SU2_CFD_REAL%" (
    "%SU2_CFD_REAL%" %*
    exit /b %ERRORLEVEL%
) else (
    echo [SU2_CFD] ERROR: Real SU2_CFD.exe binary not found at %SU2_CFD_REAL%
    echo [SU2_CFD] Cannot run CFD. Install SU2_CFD.exe before running optimization.
    exit /b 1
)
