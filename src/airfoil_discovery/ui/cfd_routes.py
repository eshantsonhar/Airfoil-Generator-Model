"""
Single CFD run API endpoint: POST /api/cfd/run, GET /api/cfd/status/{id}, GET /api/cfd/result/{id}
"""
from __future__ import annotations
import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import numpy as np

from airfoil_discovery.config import load_settings
from airfoil_discovery.cfd.su2 import SU2Evaluator, SU2Status

router = APIRouter(prefix="/api/cfd", tags=["cfd"])

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"

# In-memory store for CFD runs (simple, single-user)
_cfd_runs: dict[str, dict[str, Any]] = {}
_cfd_lock = threading.Lock()


class CfdRunRequest(BaseModel):
    upper: list[float] = Field(default=[0.18, 0.05, 0.34, 0.10])
    lower: list[float] = Field(default=[-0.19, 0.05, -0.09, 0.03])
    te_thickness: float = Field(default=0.004, ge=0.001, le=0.02)
    scale: float = Field(default=1.0, ge=0.8, le=1.2)
    reynolds: float = Field(default=100000.0, ge=10000, le=1000000)
    aoa: float = Field(default=4.0, ge=-10.0, le=20.0)
    mesh_level: str = Field(default="L0", pattern=r"^L[012]$")


@router.post("/run")
def start_cfd_run(req: CfdRunRequest) -> dict[str, Any]:
    run_id = str(uuid.uuid4())[:8]
    design_vector = np.array(req.upper + req.lower + [req.te_thickness, req.scale])
    
    with _cfd_lock:
        _cfd_runs[run_id] = {
            "status": "queued",
            "run_id": run_id,
            "req": req.model_dump(),
            "design_id": hashlib.sha256(design_vector.tobytes()).hexdigest()[:16],
            "start_ts": 0.0,
            "end_ts": 0.0,
            "result": None,
            "error": None,
        }
    
    # Launch async thread
    thread = threading.Thread(target=_run_cfd, args=(run_id, design_vector, req), daemon=True)
    thread.start()
    
    return {"run_id": run_id, "status": "queued", "design_id": _cfd_runs[run_id]["design_id"]}


def _run_cfd(run_id: str, design_vector: np.ndarray, req: CfdRunRequest) -> None:
    try:
        settings = load_settings(CONFIG_PATH)
        # Temporarily override Reynolds
        settings.flow.reynolds_min = req.reynolds
        
        evaluator = SU2Evaluator(settings)
        case_dir = PROJECT_ROOT / "data" / "cache" / f"cfd_{run_id}"
        
        with _cfd_lock:
            _cfd_runs[run_id]["status"] = "running"
            _cfd_runs[run_id]["start_ts"] = time.time()
        
        time.sleep(5)  # Let status propagate
        
        result = evaluator.run_evaluation(
            design_vector, case_dir, mesh_level=req.mesh_level, aoa=req.aoa,
            design_id=_cfd_runs[run_id]["design_id"],
        )
        
        elapsed = time.time() - _cfd_runs[run_id]["start_ts"]
        
        # Read generated files for response
        files = {}
        if case_dir.exists():
            for f in sorted(case_dir.iterdir()):
                if f.is_file():
                    try:
                        files[f.name] = f.stat().st_size
                    except Exception:
                        files[f.name] = -1
        
        with _cfd_lock:
            _cfd_runs[run_id]["status"] = result.status.value.lower()
            _cfd_runs[run_id]["end_ts"] = time.time()
            rh = result.residual_history
            if rh and len(rh) > 10:
                rh_sample = [float(v) for v in rh[:5]] + [float(v) for v in rh[-5:]]
            elif rh:
                rh_sample = [float(v) for v in rh]
            else:
                rh_sample = []
            _cfd_runs[run_id]["result"] = {
                "cl": float(result.cl),
                "cd": float(result.cd),
                "thickness": float(result.thickness),
                "status": result.status.value,
                "elapsed_s": round(elapsed, 1),
                "converged": bool((result.convergence_report or {}).get("is_valid", False)),
                "residual_converged": bool((result.convergence_report or {}).get("residual_converged", False)),
                "forces_stabilized": bool((result.convergence_report or {}).get("forces_stabilized", False)),
                "failure_stage": str(result.failure_stage) if result.failure_stage else None,
                "failure_reason": str(result.failure_reason) if result.failure_reason else None,
                "files": files,
                "residual_history_sample": rh_sample,
                "n_residual_pts": len(rh or []),
            }
            
    except Exception as e:
        with _cfd_lock:
            _cfd_runs[run_id]["status"] = "error"
            _cfd_runs[run_id]["error"] = f"{type(e).__name__}: {e}"


@router.get("/status/{run_id}")
def get_cfd_status(run_id: str) -> dict[str, Any]:
    with _cfd_lock:
        run = _cfd_runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return {
        "run_id": run_id,
        "status": run["status"],
        "design_id": run["design_id"],
        "elapsed_s": round(time.time() - run["start_ts"], 1) if run["start_ts"] > 0 else 0,
    }


@router.get("/result/{run_id}")
def get_cfd_result(run_id: str) -> dict[str, Any]:
    with _cfd_lock:
        run = _cfd_runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    if run["status"] == "queued" or run["status"] == "running":
        return {"run_id": run_id, "status": run["status"], "result": None}
    if run["error"]:
        return {"run_id": run_id, "status": "error", "error": run["error"], "result": None}
    return {"run_id": run_id, "status": run["status"], "result": run["result"]}