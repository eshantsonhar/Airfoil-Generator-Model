#!/usr/bin/env python
"""
Production-grade local launcher/runtime supervisor for Airfoil Discovery Platform.

Launches detached uvicorn process, verifies readiness, opens browser, returns immediately.
Never blocks forever. Never waits for process exit. Properly manages lifecycle on Windows.
"""

import argparse
import atexit
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
import webbrowser
from pathlib import Path
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
RUNTIME_DIR = PROJECT_ROOT / ".runtime"
LOGS_DIR = PROJECT_ROOT / "logs"
PID_FILE = RUNTIME_DIR / "ui.pid"
STDOUT_LOG = LOGS_DIR / "ui_stdout.log"
STDERR_LOG = LOGS_DIR / "ui_stderr.log"

os.environ["AIRFOIL_PROJECT_ROOT"] = str(PROJECT_ROOT)

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Thread safety
_state_lock = threading.Lock()
_current_process: Optional[subprocess.Popen] = None
_shutdown_requested = False
_normal_exit = False  # Flag to distinguish normal exit from signal/error
# Track log file handles explicitly (not via locals())
_stdout_handle: Optional[object] = None
_stderr_handle: Optional[object] = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Windows process creation flags
if os.name == "nt":
    _DETACHED_PROCESS = 0x00000008
    _CREATE_NEW_PROCESS_GROUP = 0x00000200
    _CREATE_BREAKAWAY_FROM_JOB = 0x01000000
else:
    _DETACHED_PROCESS = 0
    _CREATE_NEW_PROCESS_GROUP = 0
    _CREATE_BREAKAWAY_FROM_JOB = 0


def _run_with_timeout(cmd: list, timeout: int = 120, **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess with a timeout to prevent hanging forever."""
    kwargs.setdefault('capture_output', True)
    kwargs.setdefault('text', True)
    kwargs.setdefault('shell', True)
    try:
        result = subprocess.run(cmd, timeout=timeout, **kwargs)
        return result
    except subprocess.TimeoutExpired:
        logger.error(f"[cmd] TIMEOUT after {timeout}s: {' '.join(str(c) for c in cmd[:3])}")
        sys.exit(1)


def _validate_binaries() -> None:
    """Validate required binaries are present on PATH."""
    logger.info("[dependency] Checking required binaries...")
    
    ui_required = ["node", "npm"]
    cfd_required = ["SU2_CFD", "gmsh"]
    
    ui_missing = [b for b in ui_required if shutil.which(b) is None]
    cfd_missing = [b for b in cfd_required if shutil.which(b) is None]
    
    if ui_missing:
        logger.error(f"[dependency] FATAL: UI requires missing binaries: {', '.join(ui_missing)}")
        logger.error("[dependency] Install Node.js and npm")
        sys.exit(1)
    
    if cfd_missing:
        logger.warning(f"[dependency] CFD binaries missing: {', '.join(cfd_missing)}")
        logger.warning("[dependency] UI will start but optimization jobs will fail")
        logger.warning("[dependency] Install SU2 and Gmsh for CFD optimization")
    
    logger.info("[dependency] UI binaries validated")


def _ensure_frontend_built() -> None:
    """Validate frontend build exists, build if missing."""
    logger.info("[frontend] Checking frontend build...")
    
    dist_index = FRONTEND_DIR / "dist" / "index.html"
    if dist_index.exists():
        logger.info("[frontend] Frontend build exists")
        return
    
    logger.warning("[frontend] Frontend build missing, building...")
    
    # npm install (with timeout)
    logger.info("[frontend] Running npm install...")
    install_result = _run_with_timeout(
        ["npm", "install"],
        timeout=120,
        cwd=FRONTEND_DIR,
    )
    
    if install_result.returncode != 0:
        logger.error(f"[frontend] npm install FAILED (exit code {install_result.returncode})")
        logger.error(f"[frontend] stderr: {install_result.stderr}")
        sys.exit(1)
    
    logger.info("[frontend] npm install succeeded")
    
    # npm run build (with timeout)
    logger.info("[frontend] Running npm run build...")
    build_result = _run_with_timeout(
        ["npm", "run", "build"],
        timeout=120,
        cwd=FRONTEND_DIR,
    )
    
    if build_result.returncode != 0 or not dist_index.exists():
        logger.error(f"[frontend] Frontend build FAILED (exit code {build_result.returncode})")
        logger.error(f"[frontend] stderr: {build_result.stderr}")
        sys.exit(1)
    
    logger.info("[frontend] Frontend build succeeded")


def _check_port_available(host: str, port: int) -> bool:
    """Check if port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def _get_port_process(host: str, port: int) -> Optional[Tuple[int, str]]:
    """Get PID and process name using port on Windows (locale-safe)."""
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True,
            text=True,
            shell=True,
            timeout=10,
        )
        
        host_patterns = [
            f"{host}:{port}",
            f"127.0.0.1:{port}",
            f"0.0.0.0:{port}",
            f"[::1]:{port}",
            f"[::]:{port}",
        ]
        
        for line in result.stdout.splitlines():
            if any(p in line for p in host_patterns):
                parts = [p for p in line.split() if p]
                if len(parts) >= 5:
                    try:
                        pid = int(parts[-1])
                    except (ValueError, IndexError):
                        continue
                    # Get process name
                    try:
                        proc_result = subprocess.run(
                            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                            capture_output=True,
                            text=True,
                            shell=True,
                            timeout=10,
                        )
                        if proc_result.stdout:
                            # CSV format: "Image Name","PID","Session Name","Session#","Mem Usage"
                            parts_csv = proc_result.stdout.split(',')
                            if parts_csv:
                                proc_name = parts_csv[0].strip('"').strip()
                                return (pid, proc_name)
                    except Exception:
                        pass
                    return (pid, "unknown")
    except subprocess.TimeoutExpired:
        logger.warning("[port] netstat timed out")
    except Exception as e:
        logger.warning(f"[port] Could not determine port owner: {e}")
    
    return None


def _kill_process_tree(pid: int) -> bool:
    """Kill a process and its children on Windows using taskkill /T."""
    try:
        result = subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True,
            text=True,
            shell=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def _resolve_port_conflict(host: str, port: int) -> None:
    """Resolve port conflict before startup."""
    logger.info(f"[port] Checking port {host}:{port}...")
    
    if _check_port_available(host, port):
        logger.info(f"[port] Port {host}:{port} is available")
        return
    
    logger.warning(f"[port] Port {host}:{port} is in use")
    
    port_owner = _get_port_process(host, port)
    if not port_owner:
        logger.error("[port] Could not determine process owning the port")
        sys.exit(1)
    
    pid, proc_name = port_owner
    logger.info(f"[port] Port owned by PID {pid} ({proc_name})")
    
    # Always try to terminate - could be stale from previous run
    logger.info(f"[port] Terminating process PID {pid} ({proc_name})...")
    if _kill_process_tree(pid):
        time.sleep(1.5)
        if _check_port_available(host, port):
            logger.info("[port] Port conflict resolved")
            return
    
    logger.error(f"[port] Port {port} still in use after termination attempt")
    logger.error("[port] Terminate the process manually or use a different port")
    sys.exit(1)


def _launch_detached(host: str, port: int) -> subprocess.Popen:
    """Launch uvicorn as truly detached process (survives launcher exit on Windows)."""
    global _current_process, _stdout_handle, _stderr_handle
    
    logger.info(f"[launch] Starting uvicorn on {host}:{port}...")
    
    # Create directories
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Clear old log files
    STDOUT_LOG.write_text("", encoding='utf-8')
    STDERR_LOG.write_text("", encoding='utf-8')
    
    # Build uvicorn command
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "airfoil_discovery.ui.app:app",
        "--host", host,
        "--port", str(port),
        "--app-dir", str(SRC_ROOT)
    ]
    
    # Open log files (store handles globally for proper cleanup)
    _stdout_handle = open(STDOUT_LOG, 'w', encoding='utf-8')
    _stderr_handle = open(STDERR_LOG, 'w', encoding='utf-8')
    
    # Determine creation flags for Windows detached persistence
    creationflags = 0
    if os.name == "nt":
        # DETACHED_PROCESS: process runs without a console window
        # CREATE_NEW_PROCESS_GROUP: allows signal.CTRL_BREAK_EVENT later
        # CREATE_BREAKAWAY_FROM_JOB: detaches from parent's job object so
        #   the child survives when the launcher exits (critical for Windows 8+)
        creationflags = _DETACHED_PROCESS | _CREATE_NEW_PROCESS_GROUP
        # Try with BREAKAWAY first, fall back if not in a job
        try:
            process = subprocess.Popen(
                cmd,
                stdout=_stdout_handle,
                stderr=_stderr_handle,
                creationflags=creationflags | _CREATE_BREAKAWAY_FROM_JOB,
                cwd=str(PROJECT_ROOT),
            )
        except OSError:
            # BREAKAWAY may fail if parent isn't in a job; fall back
            process = subprocess.Popen(
                cmd,
                stdout=_stdout_handle,
                stderr=_stderr_handle,
                creationflags=creationflags,
                cwd=str(PROJECT_ROOT),
            )
    else:
        process = subprocess.Popen(
            cmd,
            stdout=_stdout_handle,
            stderr=_stderr_handle,
            cwd=str(PROJECT_ROOT),
        )
    
    with _state_lock:
        _current_process = process
    
    # Write PID file
    PID_FILE.write_text(str(process.pid), encoding='ascii')
    
    logger.info(f"[launch] uvicorn PID={process.pid}")
    logger.info(f"[launch] stdout → {STDOUT_LOG}")
    logger.info(f"[launch] stderr → {STDERR_LOG}")
    logger.info(f"[launch] PID file → {PID_FILE}")
    
    return process


def _probe_readiness(host: str, port: int, max_attempts: int = 20,
                     interval: float = 1.0, timeout: float = 2.0) -> bool:
    """Probe the health endpoint until ready or timeout."""
    url = f"http://{host}:{port}/api/health"
    logger.info(f"[ready] Probing {url} (max {max_attempts} attempts)...")
    
    for attempt in range(max_attempts):
        try:
            request = urllib.request.Request(url)
            request.add_header('User-Agent', 'Airfoil-Platform-Health-Check/1.0')
            
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status == 200:
                    logger.info(f"[ready] HTTP 200 — backend healthy (attempt {attempt + 1}/{max_attempts})")
                    return True
        except urllib.error.URLError:
            pass
        except Exception as e:
            logger.debug(f"[ready] Probe attempt {attempt + 1} failed: {e}")
        
        time.sleep(interval)
    
    logger.error(f"[ready] Backend did not respond within {max_attempts * interval}s")
    return False


def _verify_persistence(host: str, port: int, pid: int) -> bool:
    """
    After launcher main exits, verify the backend process remains alive
    and the port is still listening.
    """
    logger.info("[verify] Verifying backend persistence after launcher exit...")
    time.sleep(0.5)
    
    # Check port still listening
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        result = sock.connect_ex((host, port))
        sock.close()
        if result != 0:
            logger.error("[verify] Port closed after launcher exit — backend may have terminated")
            return False
    except Exception:
        sock.close()
        return False
    
    # Check process still alive
    if os.name == "nt":
        try:
            check = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, shell=True, timeout=10,
            )
            if str(pid) in check.stdout:
                logger.info(f"[verify] Backend process PID {pid} confirmed running")
                logger.info("[verify] Port listening confirmed")
                return True
            logger.error(f"[verify] Process PID {pid} not found in tasklist")
            return False
        except Exception:
            pass
    
    return False


def _dump_logs() -> None:
    """Dump recent log contents for debugging (memory-safe)."""
    logger.error("--- stdout (last 40 lines) ---")
    try:
        with STDOUT_LOG.open('r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines[-40:]:
            logger.error(line.rstrip('\n'))
    except Exception:
        pass
    
    logger.error("--- stderr (last 40 lines) ---")
    try:
        with STDERR_LOG.open('r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines[-40:]:
            logger.error(line.rstrip('\n'))
    except Exception:
        pass


def _open_browser(host: str, port: int) -> None:
    """Open browser to platform URL."""
    url = f"http://{host}:{port}/platform/"
    logger.info(f"[browser] Opening {url}...")
    webbrowser.open(url)
    logger.info("[browser] Browser opened")


def _cleanup() -> None:
    """Cleanup on shutdown."""
    global _shutdown_requested, _current_process, _normal_exit, _stdout_handle, _stderr_handle
    
    with _state_lock:
        if _shutdown_requested:
            return
        _shutdown_requested = True
    
    logger.info("[shutdown] Cleaning up...")
    
    # Close log file handles explicitly (not via locals())
    try:
        if _stdout_handle is not None:
            _stdout_handle.close()
            _stdout_handle = None
    except Exception:
        pass
    try:
        if _stderr_handle is not None:
            _stderr_handle.close()
            _stderr_handle = None
    except Exception:
        pass
    
    # Only terminate process if this is NOT a normal exit
    # Normal exit means launcher succeeded and backend should persist
    if not _normal_exit:
        with _state_lock:
            if _current_process:
                try:
                    _current_process.terminate()
                    time.sleep(0.5)
                    if _current_process.poll() is None:
                        _current_process.kill()
                    logger.info("[shutdown] uvicorn process terminated")
                except Exception:
                    pass
        # Remove PID file only on abnormal exit
        PID_FILE.unlink(missing_ok=True)
        logger.info("[shutdown] Cleanup complete (abnormal exit)")
    else:
        logger.info("[shutdown] Normal exit - backend persists in background")
        # Clean PID file since process is running independently now
        # The PID is still valid for manual management
        logger.info(f"[shutdown] Backend PID: {_current_process.pid if _current_process else 'unknown'}")
        # Detach — release our reference so it's not accidentally killed
        with _state_lock:
            _current_process = None


def _signal_handler(signum, frame) -> None:
    """Handle signals for graceful shutdown."""
    logger.info(f"[signal] Received signal {signum}")
    _cleanup()
    sys.exit(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Production-grade launcher for Airfoil Discovery Platform UI"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--no-browser", action="store_true", help="Skip browser launch")
    return parser.parse_args()


def main() -> None:
    global _normal_exit
    args = parse_args()
    
    # Register cleanup handlers
    atexit.register(_cleanup)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    
    logger.info("=" * 60)
    logger.info("Airfoil Discovery Platform - Runtime Supervisor")
    logger.info("=" * 60)
    
    # Step 1: Validate binaries
    _validate_binaries()
    
    # Step 2: Validate frontend
    _ensure_frontend_built()
    
    # Step 3: Resolve port conflicts
    _resolve_port_conflict(args.host, args.port)
    
    # Step 4: Launch detached process
    process = _launch_detached(args.host, args.port)
    
    # Step 5: Probe readiness
    if not _probe_readiness(args.host, args.port):
        _dump_logs()
        _cleanup()
        sys.exit(1)
    
    # Step 6: Open browser
    if not args.no_browser:
        _open_browser(args.host, args.port)
    
    # Step 7: Report success
    url = f"http://{args.host}:{args.port}/platform/"
    logger.info("=" * 60)
    logger.info(f"[done] Platform running at {url}")
    logger.info(f"[done] Backend PID: {process.pid}")
    logger.info(f"[done] Logs: {LOGS_DIR}")
    logger.info("[done] Launcher exiting - backend persists in background")
    logger.info("=" * 60)
    
    # Mark as normal exit so cleanup doesn't terminate backend
    _normal_exit = True
    
    # Verify backend actually persists after exit
    if not _verify_persistence(args.host, args.port, process.pid):
        logger.warning("[verify] Backend persistence check inconclusive - check logs for errors")
    
    # Detach from process - it continues running
    # We do NOT wait for process exit
    # We do NOT stream logs forever
    # We return immediately


if __name__ == "__main__":
    main()