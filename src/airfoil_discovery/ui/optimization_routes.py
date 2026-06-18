"""
Finite Difference Optimization Engine + Web API Integration.
Connects the MMA optimizer to CFD evaluations via finite difference gradients
and streams status updates to the web UI.
"""
import sys, os, time, json, threading, uuid
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import numpy as np
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from airfoil_discovery.config import load_settings
from airfoil_discovery.cfd.su2 import SU2Evaluator
from airfoil_discovery.cfd.mesh import MeshFidelityManager, FidelityParams
from airfoil_discovery.optimization.mma_engine import SvanbergMMA, TrustRegionGovernor
from airfoil_discovery.geometry.validation import AirfoilGeometryValidator, GeometryValidationConfig
from airfoil_discovery.geometry.cst import CSTAirfoil
from airfoil_discovery.schemas import CSTParameters

# Configure fast mesh · 15-25k cells
MeshFidelityManager.REGISTRY["L0"] = FidelityParams("L0", coarse_factor=8.0, y_plus_target=1.0)

router = APIRouter(prefix="/api/optimization", tags=["optimization"])
PROJECT_ROOT = Path(__file__).resolve().parents[3]

_opt_runs: dict[str, dict] = {}
_opt_lock = threading.Lock()

BOUNDS = {
    "x_min": np.array([-0.1, -0.1, -0.1, -0.1, -0.3, -0.3, -0.3, -0.3, 0.001, 0.8]),
    "x_max": np.array([ 0.5,  0.5,  0.5,  0.5,  0.3,  0.3,  0.3,  0.3, 0.020, 1.2]),
}

def extract_metrics(case_dir):
    cl, cd, lsb = 0.0, 0.0, None
    hist = case_dir / "history.csv"
    if hist.exists() and hist.stat().st_size > 0:
        with open(hist) as f:
            lines = f.readlines()
        if len(lines) > 1:
            hdr = [h.strip().strip('"') for h in lines[0].split(',')]
            rows = [l.split(',') for l in lines[1:] if l.strip() and l.strip() != ',']
            if rows:
                if "CL" in hdr: cl = float(rows[-1][hdr.index("CL")])
                if "CD" in hdr: cd = float(rows[-1][hdr.index("CD")])
    # LSB via Cf zero-crossing
    surf = list(case_dir.glob("*surface*.csv"))
    if surf and surf[0].stat().st_size > 100:
        try:
            with open(surf[0]) as f:
                h2 = [h.strip().strip('"').lower() for h in f.readline().strip().split(',')]
            data = np.loadtxt(surf[0], skiprows=1, delimiter=',')
            cf_c = next((i for i,h in enumerate(h2) if h in ('cf','skin_friction_x','skinfriction[0]')),
                        5 if data.shape[1] >= 6 else None)
            if cf_c:
                n = len(data)//2; x = data[n:,0]; cf = data[n:,cf_c]
                idx = np.argsort(x); x, cf = x[idx], cf[idx]
                sep = None; reatt = None
                for i in range(1, len(cf)):
                    if cf[i-1] > 0 and cf[i] < 0 and sep is None:
                        f = cf[i-1]/max(cf[i-1]-cf[i], 1e-30)
                        sep = x[i-1]+f*(x[i]-x[i-1])
                    if cf[i-1] < 0 and cf[i] > 0 and sep and reatt is None:
                        f = cf[i-1]/max(cf[i-1]-cf[i], 1e-30)
                        reatt = x[i-1]+f*(x[i]-x[i-1])
                lsb = (reatt-sep) if (sep and reatt) else None
        except Exception:
            pass
    return cl, cd, lsb

def eval_design(design, tag, settings):
    case_dir = PROJECT_ROOT / "data" / "cache" / f"opt_{tag}"
    evaluator = SU2Evaluator(settings)
    r = evaluator.run_evaluation(design, case_dir, mesh_level="L0", aoa=4.0)
    if r.status.value in ("CONFIG_ERROR", "CRASHED"):
        return None
    cl, cd, lsb = extract_metrics(case_dir)
    return {"cl": cl, "cd": cd, "lsb": lsb, "status": r.status.value}

class OptRequest(BaseModel):
    iterations: int = Field(default=3, ge=1, le=10)
    cl_target: float = Field(default=0.4, ge=0.1, le=1.0)
    lsb_weight: float = Field(default=0.5, ge=0.0, le=5.0)
    upper: list[float] = Field(default=[0.18, 0.05, 0.34, 0.10])
    lower: list[float] = Field(default=[-0.19, 0.05, -0.09, 0.03])
    te_thickness: float = Field(default=0.004, ge=0.001, le=0.02)
    scale: float = Field(default=1.0, ge=0.8, le=1.2)

def _run_opt(run_id: str, req: OptRequest):
    s = load_settings(PROJECT_ROOT / "config" / "default.yaml")
    s.solver.case_timeout_seconds = 1800
    s.solver.stage1_iter = 500
    s.solver.mesh.boundary_layer_first_height = 4.2e-5

    x0 = np.array(req.upper + req.lower + [req.te_thickness, req.scale])
    mma = SvanbergMMA(10, 2, BOUNDS["x_min"], BOUNDS["x_max"])
    mma.initialize(x0)
    x_cur = x0.copy()
    results = []
    baseline_cl = 0.0

    for k in range(req.iterations + 1):
        tag = f"{run_id}_{k}"
        with _opt_lock:
            _opt_runs[run_id]["step"] = f"CFD Evaluation {k+1}/{req.iterations+1}"

        d = eval_design(x_cur, tag, s)
        if d is None:
            with _opt_lock:
                _opt_runs[run_id]["step"] = f"Failed at iteration {k}"
                _opt_runs[run_id]["status"] = "failed"
            break

        J = d["cd"] + req.lsb_weight * (d["lsb"] or 0.0)
        if k == 0:
            baseline_cl = d["cl"]

        entry = {"k": k, "cl": d["cl"], "cd": d["cd"],
                 "lsb": d["lsb"], "J": round(J, 6)}
        results.append(entry)

        with _opt_lock:
            _opt_runs[run_id]["results"] = results
            _opt_runs[run_id]["step"] = f"Iter {k} done: CL={d['cl']:.4f} CD={d['cd']:.6f}"

        if k == req.iterations:
            break

        # FD sensitivity sweep (shape coeffs 0-7)
        eps = 1e-4
        grad = np.zeros(8)
        for i in range(8):
            with _opt_lock:
                _opt_runs[run_id]["step"] = f"FD Sensitivity {i+1}/8"
            xp = x_cur.copy()
            dx = eps * (BOUNDS["x_max"][i] - BOUNDS["x_min"][i])
            xp[i] += dx
            dp = eval_design(xp, f"{run_id}_{k}_sens{i}", s)
            if dp:
                Jp = dp["cd"] + req.lsb_weight * (dp["lsb"] or 0.0)
                grad[i] = (Jp - J) / dx

        # MMA step
        df = np.zeros(10); df[:8] = grad
        cl_def = max(0.0, baseline_cl - d["cl"])
        g = np.array([cl_def, 0.0])
        x_new, accepted, _ = mma.run_optimization_step(f=J, df=df, g=g, dg=np.zeros((2,10)))
        if accepted:
            x_cur = x_new.copy()

    with _opt_lock:
        _opt_runs[run_id]["step"] = "Completed"
        _opt_runs[run_id]["status"] = "completed"

@router.post("/start")
def start_opt(req: OptRequest):
    # Validate geometry
    s = load_settings(PROJECT_ROOT / "config" / "default.yaml")
    a = CSTAirfoil(s.geometry)
    p = CSTParameters(upper=np.array(req.upper), lower=np.array(req.lower),
                       trailing_edge_thickness=req.te_thickness)
    c = a.full_coordinates(p)
    v = AirfoilGeometryValidator(GeometryValidationConfig())
    vr = v.validate_coordinates(c)
    if not vr.can_proceed_to_cfd:
        raise HTTPException(400, detail=f"Invalid geometry: {vr.failure_reasons}")

    rid = str(uuid.uuid4())[:8]
    with _opt_lock:
        _opt_runs[rid] = {"run_id": rid, "status": "running",
                          "step": "Initializing", "results": []}
    threading.Thread(target=_run_opt, args=(rid, req), daemon=True).start()
    return {"run_id": rid, "status": "started"}

@router.get("/status/{run_id}")
def opt_status(run_id: str):
    with _opt_lock:
        r = _opt_runs.get(run_id)
    if not r:
        raise HTTPException(404, detail="Not found")
    return {"run_id": run_id, "status": r["status"],
            "step": r.get("step",""), "results": r.get("results",[])}

@router.get("/result/{run_id}")
def opt_result(run_id: str):
    with _opt_lock:
        r = _opt_runs.get(run_id)
    if not r:
        raise HTTPException(404, detail="Not found")
    return {"run_id": run_id, "results": r.get("results",[])}