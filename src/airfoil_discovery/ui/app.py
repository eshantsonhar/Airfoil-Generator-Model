from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from airfoil_discovery.config import Settings, load_settings
from airfoil_discovery.geometry.cst import CSTAirfoil
from airfoil_discovery.schemas import CSTParameters
from airfoil_discovery.storage import ExperimentDatabase
from airfoil_discovery.ui.platform_routes import router as platform_router
from airfoil_discovery.ui.cfd_routes import router as cfd_router
from airfoil_discovery.ui.telemetry_hub import DEFAULT_EVENT_PATH, get_telemetry_hub


def _resolve_project_root() -> Path:
    env_root = os.getenv("AIRFOIL_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return here.parents[3]


PROJECT_ROOT = _resolve_project_root()
SRC_ROOT = PROJECT_ROOT / "src"
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
STATIC_DIR = Path(__file__).resolve().parent / "static"
REACT_DIST = PROJECT_ROOT / "frontend" / "dist"
JOB_LOG_PATH = PROJECT_ROOT / "data" / "logs" / "latest_job.log"
JOB_RUNTIME_PATH = PROJECT_ROOT / "data" / "logs" / "latest_runtime.json"
TELEMETRY_PATH = PROJECT_ROOT / DEFAULT_EVENT_PATH
METHODOLOGY_PATH = PROJECT_ROOT / "lsb_pde_shape_optimization_methodology.md"

# Ensure required data/logs directories exist on import
for _p in [JOB_LOG_PATH.parent, JOB_RUNTIME_PATH.parent, TELEMETRY_PATH.parent]:
    _p.mkdir(parents=True, exist_ok=True)

# Thread lock protecting all mutable module-level job state
_job_state_lock = __import__("threading").Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    hub = get_telemetry_hub()
    hub.event_path = TELEMETRY_PATH
    await hub.start_watcher()
    _running_optimization_jobs.clear()  # reset on startup
    yield
    await hub.stop_watcher()
    _cleanup_all_jobs()


class JobConfig(BaseModel):
    iterations: int = Field(default=2, ge=1)
    batch_size: int = Field(default=1, ge=1)
    n_cores: int = Field(default=0, ge=0)
    use_mpi: bool = False
    mpi_ranks_per_case: int = Field(default=1, ge=1)
    omp_threads_per_rank: int = Field(default=1, ge=1)
    prefer_gpu: bool = False


settings: Settings = load_settings(CONFIG_PATH)
db = ExperimentDatabase(settings.paths.database_path)
app = FastAPI(
    title="Low-Re LSB Airfoil ASO Research Platform",
    description="CFD-driven PDE-constrained airfoil optimization with live telemetry.",
    lifespan=lifespan,
)
app.include_router(platform_router)
app.include_router(cfd_router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
if REACT_DIST.exists():
    app.mount(
        "/platform/assets",
        StaticFiles(directory=REACT_DIST / "assets"),
        name="platform-assets",
    )

# Job execution state — guarded by _job_state_lock
current_process: subprocess.Popen[Any] | None = None
current_log_handle: Any | None = None
current_job_start_cases = 0
current_job_started_at: float | None = None
_running_optimization_jobs: dict[int, subprocess.Popen[Any]] = {}


def _cleanup_all_jobs() -> None:
    """Terminate every tracked optimization subprocess. Called on FastAPI shutdown."""
    with _job_state_lock:
        for pid, proc in dict(_running_optimization_jobs).items():
            try:
                if os.name == "nt":
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            _running_optimization_jobs.pop(pid, None)
    _close_log_handle()


def _close_log_handle() -> None:
    global current_log_handle
    if current_log_handle is not None:
        try:
            current_log_handle.close()
        except Exception:
            pass
        current_log_handle = None


def _validate_write_dirs() -> None:
    """Create every directory that the platform writes to."""
    for _p in [
        (PROJECT_ROOT / "data" / "logs"),
        (PROJECT_ROOT / "data" / "database"),
        (PROJECT_ROOT / "data" / "telemetry"),
        (PROJECT_ROOT / "data" / "diagnostics"),
        (PROJECT_ROOT / "data" / "failures"),
        (PROJECT_ROOT / "data" / "configs"),
        (PROJECT_ROOT / "data" / "plots"),
        TELEMETRY_PATH.parent,
        JOB_LOG_PATH.parent,
        JOB_RUNTIME_PATH.parent,
    ]:
        _p.mkdir(parents=True, exist_ok=True)


def _physical_cpu_count() -> int:
    return max(1, os.cpu_count() or 1)


def _compute_limits() -> dict[str, Any]:
    detected = _physical_cpu_count()
    recommended_cores = max(1, min(4, detected // 2 if detected > 2 else detected))
    return {
        "detected_cpu_cores": detected,
        "recommended_iterations": 2,
        "recommended_batch_size": 1,
        "recommended_n_cores": recommended_cores,
        "recommended_use_mpi": False,
        "recommended_mpi_ranks_per_case": 1,
        "recommended_omp_threads_per_rank": 1,
        "recommended_prefer_gpu": False,
        "max_safe_iterations": 12,
        "max_safe_batch_size": max(1, min(4, recommended_cores)),
        "max_safe_mpi_ranks_per_case": recommended_cores,
        "max_safe_omp_threads_per_rank": recommended_cores,
    }


def _sanitize_job_config(config: JobConfig) -> dict[str, Any]:
    limits = _compute_limits()
    n_cores = limits["recommended_n_cores"] if config.n_cores <= 0 else min(config.n_cores, limits["recommended_n_cores"])
    mpi_ranks = max(1, min(config.mpi_ranks_per_case, n_cores, limits["max_safe_mpi_ranks_per_case"]))
    omp_threads = max(1, min(config.omp_threads_per_rank, max(1, n_cores // mpi_ranks), limits["max_safe_omp_threads_per_rank"]))
    sanitized = {
        "iterations": max(1, min(config.iterations, limits["max_safe_iterations"])),
        "batch_size": max(1, min(config.batch_size, limits["max_safe_batch_size"])),
        "n_cores": n_cores,
        "use_mpi": bool(config.use_mpi and mpi_ranks > 1),
        "mpi_ranks_per_case": mpi_ranks,
        "omp_threads_per_rank": omp_threads,
        "prefer_gpu": bool(config.prefer_gpu),
        "limits": limits,
    }
    return sanitized


def _best_design_frame(limit: int = 1):
    frame = db.best_designs(limit=limit)
    if frame.empty:
        return frame
    return frame.fillna(0)


def _row_to_cst(row: Any) -> CSTParameters:
    return CSTParameters(
        upper=np.array([row["upper_0"], row["upper_1"], row["upper_2"], row["upper_3"]], dtype=float),
        lower=np.array([row["lower_0"], row["lower_1"], row["lower_2"], row["lower_3"]], dtype=float),
        trailing_edge_thickness=float(row["te_thickness"]),
    )


def _airfoil_coordinates(row: Any) -> list[dict[str, float]]:
    cst = CSTAirfoil(settings.geometry)
    coords = cst.full_coordinates(_row_to_cst(row))
    return [{"x": float(x), "y": float(y)} for x, y in coords]


def _polar_for_case(case_key: str) -> list[dict[str, float]]:
    frame = db.training_frame()
    if frame.empty:
        return []
    rows = frame[frame["case_key"] == case_key].sort_values("aoa_deg")
    return [
        {
            "aoa_deg": float(r.aoa_deg),
            "cl": float(r.cl),
            "cd": float(r.cd),
            "efficiency": float(r.cl / max(r.cd, 1.0e-8)),
        }
        for r in rows.itertuples()
    ]


def _dat_text(row: Any) -> str:
    coords = _airfoil_coordinates(row)
    lines = [f"ASO_best_{row['id']}_Re{int(row['reynolds'])}"]
    lines.extend(f"{p['x']:.8f} {p['y']:.8f}" for p in coords)
    return "\n".join(lines) + "\n"


def _platform_index_response() -> FileResponse:
    index_path = REACT_DIST / "index.html"
    if not index_path.exists():
        raise HTTPException(
            status_code=503,
            detail="React UI not built. Run: cd frontend && npm install && npm run build",
        )
    return FileResponse(index_path)


@app.get("/api/health")
def health_check() -> dict:
    """Lightweight health check endpoint for launcher readiness probing."""
    import time
    return {"status": "ok", "timestamp": time.time()}


@app.get("/", response_model=None)
def index() -> RedirectResponse | FileResponse:
    if REACT_DIST.exists():
        return RedirectResponse(url="/platform/", status_code=307)
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/platform", response_model=None)
def platform_redirect() -> RedirectResponse:
    return RedirectResponse(url="/platform/", status_code=307)


@app.get("/platform/", response_model=None)
def platform_index() -> FileResponse:
    return _platform_index_response()


@app.get("/platform/{spa_path:path}", response_model=None)
def platform_spa(spa_path: str) -> FileResponse:
    """Serve built assets or fall back to index.html for client-side routes.
    
    This catch-all handles:
    1. Deep SPA routes (e.g., /platform/geometry) → index.html (for React Router)
    2. Direct file requests (e.g., /platform/vite.svg) → the file
    3. Asset subpaths that bypass the StaticFiles mount (shouldn't happen but defensive)
    """
    # Sanitize: prevent path traversal
    clean_path = spa_path.replace("..", "").lstrip("/")
    if not clean_path:
        return _platform_index_response()
    
    # Try to serve actual file from dist
    candidate = REACT_DIST / clean_path
    if candidate.is_file():
        return FileResponse(candidate)
    
    # For any non-file path under /platform/, serve index.html for SPA routing
    # This ensures deep links like /platform/config work on refresh
    return _platform_index_response()


@app.get("/debug", response_model=None)
def debug() -> FileResponse:
    return FileResponse(STATIC_DIR / "debug.html")


@app.get("/api/limits")
def limits() -> dict[str, Any]:
    return _compute_limits()


@app.get("/api/methodology")
def methodology_status() -> dict[str, Any]:
    return {
        "path": str(METHODOLOGY_PATH),
        "exists": METHODOLOGY_PATH.exists(),
        "framework": "Passive suppression of laminar separation bubbles using PDE-constrained aerodynamic shape optimization",
        "reynolds_range": [settings.flow.reynolds_min, settings.flow.reynolds_max],
        "mach": settings.flow.mach,
        "transition_model_required": settings.solver.transition_model,
        "mesh": {
            "farfield_radius_c": settings.solver.mesh.farfield_radius,
            "wake_length_c": settings.solver.mesh.wake_length,
            "surface_points": settings.solver.mesh.surface_points,
            "y_plus_target": settings.solver.mesh.y_plus_target,
            "inflation_layers": settings.solver.mesh.boundary_layer_layers,
        },
    }


@app.get("/api/stats")
def stats() -> dict[str, Any]:
    best = _best_design_frame(limit=1)
    if best.empty:
        return {"total_cases": db.total_cases(), "best_score": 0.0, "best_efficiency": 0.0}
    row = best.iloc[0]
    polar = _polar_for_case(str(row["case_key"]))
    best_eff = max((p["efficiency"] for p in polar), default=0.0)
    return {"total_cases": db.total_cases(), "best_score": float(row["score"]), "best_efficiency": float(best_eff)}


@app.get("/api/progress")
def progress() -> list[dict[str, float]]:
    frame = db.best_designs(limit=500)
    if frame.empty:
        return []
    ordered = frame.sort_values("id")
    best_so_far = ordered["score"].cummax()
    return [
        {"iteration": int(i + 1), "best_score": float(score)}
        for i, score in enumerate(best_so_far.tolist())
    ]


@app.get("/api/best_airfoil")
def best_airfoil() -> dict[str, Any]:
    best = _best_design_frame(limit=1)
    if best.empty:
        return {"coordinates": [], "score": 0.0, "reynolds": 0.0}
    row = best.iloc[0]
    return {
        "coordinates": _airfoil_coordinates(row),
        "score": float(row["score"]),
        "reynolds": float(row["reynolds"]),
    }


@app.get("/api/best_airfoil_full")
def best_airfoil_full() -> dict[str, Any]:
    best = _best_design_frame(limit=1)
    if best.empty:
        return {}
    row = best.iloc[0]
    params = _row_to_cst(row)
    cst = CSTAirfoil(settings.geometry)
    metrics = cst.geometry_metrics(params)
    polar = _polar_for_case(str(row["case_key"]))
    transitions = db.transition_points_for_case(str(row["case_key"]))
    return {
        "case_key": str(row["case_key"]),
        "signature": str(row["signature"]),
        "reynolds": float(row["reynolds"]),
        "cst_params": {
            "upper": [float(v) for v in params.upper],
            "lower": [float(v) for v in params.lower],
            "te_thickness": float(params.trailing_edge_thickness),
        },
        "geometry": {
            "max_thickness": metrics.max_thickness,
            "max_camber": metrics.max_camber,
            "leading_edge_radius": metrics.leading_edge_radius,
            "smoothness_score": metrics.smoothness_score,
            "curvature_spike": metrics.curvature_spike,
            "prior_score": metrics.prior_score,
        },
        "score_breakdown": {
            "score": float(row["score"]),
            "stall_angle_deg": float(row["stall_angle_deg"]),
            "cd_at_cruise": float(row["cd_at_cruise"]),
            "separation_penalty": float(row["separation_penalty"]),
            "instability_penalty": float(row["instability_penalty"]),
        },
        "polar": polar,
        "transition_points": transitions.to_dict(orient="records") if not transitions.empty else [],
        "dat_text": _dat_text(row),
    }


@app.get("/api/job/runtime")
def job_runtime() -> dict[str, Any]:
    if not JOB_RUNTIME_PATH.exists():
        return {"status": "idle", "stationarity": 0.0, "complementarity": 0.0}
    try:
        return json.loads(JOB_RUNTIME_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "unreadable", "stationarity": 0.0, "complementarity": 0.0}


@app.get("/api/job/log")
def job_log(tail: int = 150) -> dict[str, Any]:
    if not JOB_LOG_PATH.exists():
        return {"lines": []}
    lines = JOB_LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
    tail = max(1, min(int(tail), 1000))
    return {"lines": lines[-tail:]}


@app.get("/api/job/status")
def job_status() -> dict[str, Any]:
    global current_process, current_log_handle
    is_running = current_process is not None and current_process.poll() is None
    if not is_running and current_log_handle:
        _close_log_handle()

    total_cases = db.total_cases()
    job_age_seconds = 0.0 if current_job_started_at is None else max(0.0, time.time() - current_job_started_at)
    log_tail: list[str] = []
    if JOB_LOG_PATH.exists():
        log_tail = JOB_LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()[-12:]

    runtime_data = job_runtime()
    status_detail = "running" if is_running else runtime_data.get("status", "idle")
    if not is_running and status_detail == "completed":
        status_detail = "idle"

    return {
        "is_running": is_running,
        "pid": current_process.pid if current_process else None,
        "return_code": None if is_running or not current_process else current_process.poll(),
        "total_cases": total_cases,
        "new_cases_since_start": max(0, total_cases - current_job_start_cases),
        "job_age_seconds": round(job_age_seconds, 1),
        "log_tail": log_tail,
        "detailed_status": status_detail,
        "runtime_data": runtime_data,
    }


@app.post("/api/job/start")
def start_job(config: JobConfig) -> dict[str, Any]:
    global current_process, current_log_handle, current_job_start_cases, current_job_started_at

    with _job_state_lock:
        if current_process is not None and current_process.poll() is None:
            raise HTTPException(status_code=409, detail="An optimization job is already running.")

        _validate_write_dirs()
        safe = _sanitize_job_config(config)
        JOB_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        JOB_RUNTIME_PATH.write_text(
            json.dumps({"status": "starting", "stationarity": 0.0, "complementarity": 0.0}),
            encoding="utf-8",
        )
        current_job_start_cases = db.total_cases()
        current_job_started_at = time.time()
        current_log_handle = JOB_LOG_PATH.open("w", encoding="utf-8")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC_ROOT)
        env["SU2_N_CORES"] = str(safe["n_cores"])
        env["SU2_USE_MPI"] = "true" if safe["use_mpi"] else "false"
        env["SU2_MPI_RANKS"] = str(safe["mpi_ranks_per_case"])
        env["SU2_OMP_THREADS"] = str(safe["omp_threads_per_rank"])
        env["SU2_PREFER_GPU"] = "true" if safe["prefer_gpu"] else "false"
        env["AIRFOIL_JOB_RUNTIME_PATH"] = str(JOB_RUNTIME_PATH)
        env["AIRFOIL_TELEMETRY_PATH"] = str(TELEMETRY_PATH)
        env["AIRFOIL_RUN_ID"] = f"run_{int(time.time())}"
        TELEMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        TELEMETRY_PATH.write_text("", encoding="utf-8")

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_optimization.py"),
            "--config",
            str(CONFIG_PATH),
            "--iterations",
            str(safe["iterations"]),
            "--batch-size",
            str(safe["batch_size"]),
        ]
        current_process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=current_log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        _running_optimization_jobs[current_process.pid] = current_process

    return {"status": "started", "pid": current_process.pid, **safe}


@app.post("/api/job/stop")
def stop_job() -> dict[str, Any]:
    global current_process
    if current_process is None or current_process.poll() is not None:
        return {"status": "idle"}
    if os.name == "nt":
        current_process.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        current_process.terminate()
    return {"status": "stopping", "pid": current_process.pid}
