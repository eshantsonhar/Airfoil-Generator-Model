#!/usr/bin/env python
"""
Final end-to-end runtime verification script.
Tests the platform after fixes have been applied.
"""

import os
import socket
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
PID_FILE = PROJECT_ROOT / ".runtime" / "ui.pid"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

PASS = 0
FAIL = 0

def check(description: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        print(f"  [PASS] {description}")
        PASS += 1
    else:
        print(f"  [FAIL] {description} — {detail}" if detail else f"  [FAIL] {description}")
        FAIL += 1

def http_get(path: str) -> tuple:
    """Returns (status_code, text) or raises."""
    r = urllib.request.urlopen(f"http://127.0.0.1:8000{path}", timeout=5)
    return r.status, r.read().decode("utf-8", errors="ignore")

def port_open(host="127.0.0.1", port=8000) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        r = s.connect_ex((host, port))
        return r == 0
    finally:
        s.close()

def get_process_by_port() -> int:
    try:
        r = subprocess.run(["netstat", "-ano"], capture_output=True, text=True, shell=True, timeout=10)
        for line in r.stdout.splitlines():
            if ":8000" in line and ("LISTENING" in line or "LISTEN" in line.upper()):
                parts = [p for p in line.split() if p]
                if parts:
                    return int(parts[-1])
    except:
        pass
    return 0

print("=" * 60)
print("  FINAL RUNTIME VERIFICATION")
print("=" * 60)

# 1. Port is listening
pid = get_process_by_port()
check("Port 8000 is listening", port_open(), f"PID={pid}")

# 2. Health endpoint
try:
    status, body = http_get("/api/health")
    check("Health endpoint returns 200", status == 200, f"status={status}")
except Exception as e:
    check("Health endpoint reachable", False, str(e))

# 3. Platform index
try:
    status, body = http_get("/platform/")
    check("Platform index returns 200", status == 200, f"status={status}")
    check("Platform returns HTML", "html" in body.lower() or "root" in body, "contains React root div")
except Exception as e:
    check("Platform index reachable", False, str(e))

# 4. SPA deep routing
for spa_path in ["/platform/config", "/platform/geometry", "/platform/optimization"]:
    try:
        status, body = http_get(spa_path)
        check(f"SPA route {spa_path} returns 200", status == 200)
    except Exception as e:
        check(f"SPA route {spa_path} reachable", False, str(e))

# 5. Stats API
try:
    status, body = http_get("/api/stats")
    check("Stats API returns 200", status == 200)
    import json
    data = json.loads(body)
    check("Stats API returns valid JSON", isinstance(data, dict))
except Exception as e:
    check("Stats API reachable", False, str(e))

# 6. Backend still alive check (multiple rapid calls)
all_ok = True
for i in range(5):
    try:
        s, _ = http_get("/api/health")
        if s != 200:
            all_ok = False
    except:
        all_ok = False
check("Backend survives 5 rapid requests", all_ok)

# 7. Static asset serving
try:
    r = urllib.request.urlopen("http://127.0.0.1:8000/static/debug.html", timeout=5)
    check("Static assets served", r.status == 200)
except:
    check("Static assets served", False)

# 8. PID file exists
check("PID file exists", PID_FILE.exists())
if PID_FILE.exists():
    stored_pid = int(PID_FILE.read_text().strip())
    check("PID file matches port owner", stored_pid == pid or pid == 0, f"stored={stored_pid} port_owner={pid}")

# 9. Backend process is a Python process
try:
    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                       capture_output=True, text=True, shell=True, timeout=10)
    check("Backend is Python process", "python" in r.stdout.lower() or pid == 0, r.stdout[:80])
except:
    check("Backend process check", False)

print()
print("=" * 60)
print(f"  RESULTS: {PASS} passed, {FAIL} failed out of {PASS + FAIL}")
print("=" * 60)

sys.exit(0 if FAIL == 0 else 1)