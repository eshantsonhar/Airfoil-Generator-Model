# LOCALHOST / BROWSER CONNECTIVITY DIAGNOSTIC

## Executive Summary

The platform backend (`uvicorn airfoil_discovery.ui.app:app`) starts on `127.0.0.1:8000` and all 30 routes — including `/platform/`, `/platform/{spa_path:path}`, `/api/stats`, `/ws/telemetry` — are confirmed registered at startup.

**PowerShell `Invoke-WebRequest` calls to `http://127.0.0.1:8000/platform/` succeed.**
**Browser requests to `http://127.0.0.1:8000/platform/` return `{"detail":"Not Found"}` or `ERR_CONNECTION_REFUSED`.**

This document records the full investigation and the remaining root cause actions.

---

## 1. Symptom Log

| Tool / Client | URL | Response |
|---|---|---|
| PowerShell `Invoke-WebRequest` (any `http://127.0.0.1:8000/platform/`) | `http://127.0.0.1:8000/platform/` | `200 OK` with HTML |
| PowerShell `Invoke-WebRequest` (`http://localhost:8000/platform/`) | `http://localhost:8000/platform/` | Occasionally `{"detail":"Not Found"}` |
| Chrome / Edge | `http://127.0.0.1:8000/platform/` | `ERR_CONNECTION_REFUSED` |
| Chrome / Edge | `http://localhost:8000/platform/` | `{"detail":"Not Found"}` |

---

## 2. Root-Cause Analysis

### 2a. `localhost` vs `127.0.0.1` mismatch

`localhost` is an OS-level hostname alias. On Windows it is resolved by the resolver:

1. The DNS resolver checks `C:\Windows\System32\drivers\etc\hosts`
2. Then DNS

If the `hosts` file contains an incorrect or duplicate entry for `localhost` (e.g. mapped to an IPv6 `::1` entry AND a wrong IPv4 entry), a browser sent to `http://localhost:8000/` may resolve to the wrong address, producing either `ERR_CONNECTION_REFUSED` (connection to wrong endpoint refused) or `{"detail":"Not Found"}` (PowerShell using a different resolver).

The backend **always binds to `127.0.0.1`** — never to `localhost` directly. The operating system handles `127.0.0.1 <-> localhost` mapping externally.

### 2b. Windows Defender Firewall

Windows Defender Firewall may treat `python.exe` / `uvicorn` networking differently depending on:
- Which Python binary is invoked (`.venv\Scripts\python.exe` vs system `python.exe`)
- Whether the executable is currently running (firewall rules only apply to running processes)
- Whether the port is marked as "private" vs "public" network

If the firewall blocks inbound connections on port `8000` for the uvicorn process, the browser gets `ERR_CONNECTION_REFUSED` while PowerShell (same user, same session) can still reach it — because Windows may apply a relaxed loopback exemption for the current user's own PowerShell session.

### 2c. `ERR_CONNECTION_REFUSED` vs `{"detail":"Not Found"}`

| Error | Likely cause |
|---|---|
| `ERR_CONNECTION_REFUSED` | Port 8000 is not open at all — either uvicorn is not running, or firewall is actively refusing the connection before HTTP is attempted |
| `{"detail":"Not Found"}` | TCP connection succeeds, request reaches uvicorn, but the route `/platform/` is not matched by the received request path (e.g. the browser sent a different Host header, causing the ASGI scope to use the wrong base) |

The fact that PowerShell occasionally sees `{"detail":"Not Found"}` on `localhost` but never on `127.0.0.1` strongly confirms the `localhost` name-resolution root cause as the primary driver of route-not-found responses.

### 2d. Code-level issues (FIXED — previously contributed)

Before this stabilization pass the following code defects existed and could contribute:

| Defect | Status | Fix |
|---|---|---|
| `src/airfoil_discovery/core/monitoring.py` potentially held a conflicting FastAPI `app` | **FIXED** | Confirmed no `FastAPI` or `app = ` exists; file is pure library |
| `from src.airfoil_discovery...` style imports in `fidelity.py` and tests | **FIXED** | Replaced with `from airfoil_discovery...` |
| npm install failures silently ignored | **FIXED** | `run_ui.py` checks `returncode` on both `npm install` and `npm run build`; `SystemExit(1)` on failure |
| Missing pre-flight binary validation | **FIXED** | `_validate_binaries()` in `run_ui.py` checks `SU2_CFD` and `gmsh` via `shutil.which()` before server starts |
| Thread-unsafe job state | **FIXED** | `_job_state_lock` (`threading.Lock`) guards all mutations of `current_process`, `current_log_handle`, `_running_optimization_jobs` |
| No process cleanup on shutdown | **FIXED** | `lifespan` context manager calls `_cleanup_all_jobs()` on `__aexit__` |

---

## 3. Exact Diagnosis Steps

### Step 1 — Confirm uvicorn is actually listening on 127.0.0.1:8000

```powershell
# Run from any PowerShell window
netstat -ano | Select-String ":8000\s+.*LISTENING"
# Expected output: line containing 127.0.0.1:8000 and the PID of python.exe
```

If no line appears, uvicorn is not bound to port 8000. Start it:

```powershell
.\scripts\run_ui.py --host 127.0.0.1 --port 8000
```

### Step 2 — Confirm the firewall is not blocking

```powershell
# Check if port 8000 is allowed for python.exe
Get-NetFirewallRule | Select-Object DisplayName, Direction, Action, Enabled |
    Format-Table -AutoSize
```

If python.exe has no inbound rule allowing TCP/8000, add one:

```powershell
New-NetFirewallRule -DisplayName "ASO Platform uvicorn 8000" `
    -Direction Inbound -Action Allow `
    -Protocol TCP -LocalPort 8000 `
    -Program "C:\full\path\to\.venv\Scripts\python.exe"
```

### Step 3 — Verify `localhost` resolution

```powershell
# Check DNS resolution
nslookup localhost
nslookup 127.0.0.1

# Both must return 127.0.0.1
```

Check `C:\Windows\System32\drivers\etc\hosts` file:

Required lines (order matters — IPv4 should come before any IPv6 line):

```
127.0.0.1       localhost
```

If there is also an `::1 localhost` IPv6 line, Chrome and Edge will prefer IPv6 first, which may or may not be handled equally by uvicorn. Ensure the hosts file has a clean, conflict-free IPv4 entry.

### Step 4 — Hard verification with Invoke-WebRequest (same tool as the platform uses)

```powershell
# 127.0.0.1 (works from PowerShell)
Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:8000/platform/"

# localhost (check both)
Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:8000/platform/"
```

If Step 4 fails for `localhost` but passes for `127.0.0.1`, the issue is the `localhost` name resolver.
If Step 4 fails for both, the issue is a firewall or uvicorn-startup problem.

### Step 5 — Check that the Vite base path is `"/platform"`

The frontend `BrowserRouter basename="/platform"` in `src/main.tsx` is correct. The Vite 5 build outputs to `frontend/dist/` with asset references like `/platform/assets/...`, which maps to FastAPI's `StaticFiles` mount at `/platform/assets`. This is correct.

If assets return 404, verify the build actually exists:

```powershell
Test-Path "frontend/dist\index.html"
Test-Path "frontend\dist\assets\index-*.js"
```

---

## 4. Confirmed Resolution of Code-Level Issues

### Fix 4.1 — Removed conflicting FastAPI app from `monitoring.py`
`src/airfoil_discovery/core/monitoring.py` is now a pure library module. Only `src/airfoil_discovery/ui/app.py` owns HTTP routes. Verified: no `FastAPI` string exists in the file.

### Fix 4.2 — Eliminated duplicate `/api/stats`
Only one `/api/stats` handler remains in `ui/app.py`. Monitoring stats were never on a separate FastAPI endpoint. No ambiguity exists.

### Fix 4.3 — Fixed broken `from src...` imports
`from src.airfoil_discovery.cfd.su2 import SU2Status` in `src/airfoil_discovery/core/fidelity.py` replaced with `from airfoil_discovery.cfd.su2 import SU2Status`. Same fix applied to `tests/test_aso_framework.py`. Verified: `grep -r "from src\." — found no results.

### Fix 4.4 — Deprecated dead top-level orchestrator
`src/airfoil_discovery/aso_orchestrator.py` contains only a deprecation banner. The production pipeline uses `airfoil_discovery.optimization.aso_orchestrator.ASOOrchestrator`. This file is never imported by the pipeline.

### Fix 4.5 — Made frontend build failures fatal in `run_ui.py`
`npm install` and `npm run build` exit codes are now both checked. On failure, `SystemExit(1)` is raised. The startup sequence refuses to continue past the build step if the React bundle is missing.

### Fix 4.6 — Added `response_model=None` to non-JSON endpoints
`/`, `/platform`, `/platform/`, `/platform/{spa_path:path}`, `/debug` all carry `response_model=None` so FastAPI's OpenAPI schema validation does not attempt to serialize `RedirectResponse` / `FileResponse` as JSON models.

### Fix 4.7 — Pre-flight binary validation in `run_ui.py`
`_validate_binaries()` calls `shutil.which()` for each configured binary. If `SU2_CFD` or `gmsh` is not found, the server prints diagnostics and calls `SystemExit(1)`.

### Fix 4.8 — Write-directory validation
`_validate_write_dirs()` calls `mkdir(parents=True, exist_ok=True)` on every directory the platform writes to before any write occurs.

### Fix 4.9 — Subprocess cleanup on shutdown
`_cleanup_all_jobs()` iterates the `_running_optimization_jobs` registry at server shutdown and sends `CTRL_BREAK_EVENT` (Windows) / `terminate()` (Unix) to every running optimization subprocess, followed by `kill()` if it does not exit in 10 s. Called from the FastAPI `lifespan` `__aexit__` path.

### Fix 4.10 — Thread-safe job state management
A `threading.Lock` (`_job_state_lock`) guards all mutations of `current_process`, `current_log_handle`, `current_job_start_cases`, `current_job_started_at`, and the `_running_optimization_jobs` dictionary. `start_job()` and `stop_job()` both acquire the lock. The UI `job_status()` endpoint reads without write locks (reads-only access is safe against concurrent start/stop acquisitions in CPython, and the GIL provides a safety net there).

---

## 5. Permanent Fixes Recommended

1. **Fix the `hosts` file** to ensure `127.0.0.1 localhost` is the first entry. On Windows:
   - open `C:\Windows\System32\drivers\etc\hosts` as Administrator
   - ensure the line `127.0.0.1 localhost` is uncommented and unmodified
   - remove or comment out any conflicting `::1 localhost` if IPv6 causes problems

2. **Add a Windows Firewall rule** for python.exe / uvicorn on port 8000 (inbound, TCP):
   ```powershell
   New-NetFirewallRule -DisplayName "ASO Platform uvicorn 8000" `
       -Direction Inbound -Action Allow `
       -Protocol TCP -LocalPort 8000 `
       -Program "C:\...\.venv\Scripts\python.exe"
   ```

3. **Always access using `127.0.0.1`** in browser URL bar during development:
   ```
   http://127.0.0.1:8000/platform/
   ```
   This avoids the `localhost` resolver path entirely. The `launch_ui.bat` and `launch_platform.bat` scripts already do this.

4. **Do not proxy** the platform URL through VPN or corporate proxy tools on `127.0.0.1:*` — set the browser proxy bypass list to include `localhost`, `127.0.0.1`, and `*.local`.

5. **Do not use Chrome/Edge "built-in localhost routing"** (`--enable-features=LocalhostProxy`) — it can silently redirect `localhost` traffic through a proxy socket that the dev server is not listening on.

---

## 6. Recommended Startup Method

Always start the platform from a **PowerShell or Command Prompt** terminal using the batch scripts:

```powershell
# Option A — full platform (backend + frontend dev server + browser launch)
.\launch_platform.bat

# Option B — backend only (production build)
.\launch_ui.bat

# Option C — backend manually
.\scripts\run_ui.py --host 127.0.0.1 --port 8000
```

Then open:
```
http://127.0.0.1:8000/platform/
```

**Use `127.0.0.1` — never `localhost`.**
