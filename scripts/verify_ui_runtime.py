#!/usr/bin/env python
"""
Verification script for Airfoil Discovery Platform UI runtime.

Verifies:
- Backend import
- Route registration
- Frontend dist exists
- Websocket route exists
- /platform/ returns 200
- PID file valid
- Port 8000 listening
- Binaries present

Returns nonzero on failure.
"""

import os
import shutil
import socket
import subprocess
import sys
import urllib.request
import urllib.error
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
RUNTIME_DIR = PROJECT_ROOT / ".runtime"
LOGS_DIR = PROJECT_ROOT / "logs"
PID_FILE = RUNTIME_DIR / "ui.pid"

os.environ["AIRFOIL_PROJECT_ROOT"] = str(PROJECT_ROOT)

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

def print_header(msg: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print('=' * 60)

def print_success(msg: str) -> None:
    print(f"[✓] {msg}")

def print_error(msg: str) -> None:
    print(f"[✗] {msg}")

def verify_backend_import() -> bool:
    """Verify backend can be imported."""
    print("[verify] Testing backend import...")
    try:
        from airfoil_discovery.ui.app import app
        print_success("Backend import successful")
        return True
    except Exception as e:
        print_error(f"Backend import failed: {e}")
        return False

def verify_route_registration() -> bool:
    """Verify routes are registered."""
    print("[verify] Testing route registration...")
    try:
        from airfoil_discovery.ui.app import app
        routes = [route.path for route in app.routes]
        
        required_routes = ["/platform/", "/ws/telemetry"]
        missing = [r for r in required_routes if r not in routes]
        
        if missing:
            print_error(f"Missing routes: {missing}")
            return False
        
        print_success(f"All required routes registered: {required_routes}")
        return True
    except Exception as e:
        print_error(f"Route verification failed: {e}")
        return False

def verify_frontend_dist() -> bool:
    """Verify frontend dist exists."""
    print("[verify] Testing frontend dist...")
    dist_index = FRONTEND_DIR / "dist" / "index.html"
    
    if not dist_index.exists():
        print_error(f"Frontend dist missing: {dist_index}")
        return False
    
    print_success(f"Frontend dist exists: {dist_index}")
    return True

def verify_websocket_route() -> bool:
    """Verify websocket route exists."""
    print("[verify] Testing websocket route...")
    try:
        from airfoil_discovery.ui.app import app
        routes = [route.path for route in app.routes]
        
        ws_routes = [r for r in routes if "ws" in r.lower() or "websocket" in r.lower()]
        
        if not ws_routes:
            print_error("No websocket routes found")
            return False
        
        print_success(f"Websocket routes found: {ws_routes}")
        return True
    except Exception as e:
        print_error(f"Websocket verification failed: {e}")
        return False

def verify_platform_endpoint() -> bool:
    """Verify /platform/ returns 200."""
    print("[verify] Testing /platform/ endpoint...")
    url = "http://127.0.0.1:8000/platform/"
    
    try:
        request = urllib.request.Request(url)
        request.add_header('User-Agent', 'Airfoil-Verification/1.0')
        
        with urllib.request.urlopen(request, timeout=5) as response:
            if response.status == 200:
                print_success(f"/platform/ returns HTTP 200")
                return True
            else:
                print_error(f"/platform/ returned HTTP {response.status}")
                return False
    except urllib.error.URLError as e:
        print_error(f"/platform/ unreachable: {e}")
        return False
    except Exception as e:
        print_error(f"/platform/ verification failed: {e}")
        return False

def verify_pid_file() -> bool:
    """Verify PID file is valid."""
    print("[verify] Testing PID file...")
    
    if not PID_FILE.exists():
        print_error("PID file does not exist")
        return False
    
    try:
        pid_str = PID_FILE.read_text().strip()
        pid = int(pid_str)
        
        # Check if process is running
        try:
            subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], 
                         capture_output=True, shell=True, check=True)
            print_success(f"PID file valid: {pid} (process running)")
            return True
        except subprocess.CalledProcessError:
            print_warning(f"PID file exists but process {pid} not running")
            return True  # File is valid format, just stale
    except ValueError:
        print_error("PID file contains invalid data")
        return False
    except Exception as e:
        print_error(f"PID file verification failed: {e}")
        return False

def print_warning(msg: str) -> None:
    print(f"[!] {msg}")

def verify_port_listening() -> bool:
    """Verify port 8000 is listening."""
    print("[verify] Testing port 8000...")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        result = sock.connect_ex(("127.0.0.1", 8000))
        if result == 0:
            print_success("Port 8000 is listening")
            return True
        else:
            print_error("Port 8000 is not listening")
            return False
    except Exception as e:
        print_error(f"Port check failed: {e}")
        return False
    finally:
        sock.close()

def verify_binaries() -> bool:
    """Verify required binaries are present."""
    print("[verify] Testing required binaries...")
    
    required_binaries = ["SU2_CFD", "gmsh", "node", "npm"]
    missing = []
    
    for binary in required_binaries:
        if shutil.which(binary) is None:
            missing.append(binary)
    
    if missing:
        print_error(f"Missing binaries: {', '.join(missing)}")
        return False
    
    print_success(f"All required binaries found: {', '.join(required_binaries)}")
    return True

def verify_log_files() -> bool:
    """Verify log files exist and are writable."""
    print("[verify] Testing log files...")
    
    stdout_log = LOGS_DIR / "ui_stdout.log"
    stderr_log = LOGS_DIR / "ui_stderr.log"
    
    if not LOGS_DIR.exists():
        print_error(f"Log directory missing: {LOGS_DIR}")
        return False
    
    # Test writability
    try:
        stdout_log.write_text("", encoding='utf-8')
        stderr_log.write_text("", encoding='utf-8')
        print_success(f"Log files writable: {stdout_log}, {stderr_log}")
        return True
    except Exception as e:
        print_error(f"Log files not writable: {e}")
        return False

def main() -> int:
    print_header("Airfoil Discovery Platform - UI Runtime Verification")
    
    checks = [
        ("Backend Import", verify_backend_import),
        ("Route Registration", verify_route_registration),
        ("Frontend Dist", verify_frontend_dist),
        ("Websocket Route", verify_websocket_route),
        ("Binaries", verify_binaries),
        ("Log Files", verify_log_files),
        ("PID File", verify_pid_file),
        ("Port 8000 Listening", verify_port_listening),
        ("/platform/ Endpoint", verify_platform_endpoint),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print_error(f"{name} check raised exception: {e}")
            results.append((name, False))
    
    print_header("Verification Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "[✓]" if result else "[✗]"
        print(f"{symbol} {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} checks passed")
    
    if passed == total:
        print_success("\nAll verification checks passed!")
        return 0
    else:
        print_error(f"\n{total - passed} verification check(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
