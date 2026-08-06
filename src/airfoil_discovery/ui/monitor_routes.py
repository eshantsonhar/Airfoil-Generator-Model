"""
Run Monitor API — live per-case CFD telemetry reader.

Endpoints:
  GET /api/monitor/list               — all known case directories (running + completed)
  GET /api/monitor/{case_id}/history  — full iteration history: Cl, Cd, residuals
  GET /api/monitor/{case_id}/surface  — Cp + Cf distributions from surface CSV
  GET /api/monitor/{case_id}/summary  — convergence quality flag + scalar KPIs
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/monitor", tags=["monitor"])

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CACHE_ROOT = PROJECT_ROOT / "data" / "cache"
RUNTIME_PATH = PROJECT_ROOT / "data" / "logs" / "latest_runtime.json"

# ── helpers ──────────────────────────────────────────────────────────────────

def _parse_history_csv(history_path: Path) -> dict[str, list]:
    """Read an SU2 history.csv and return column→list-of-floats mapping."""
    if not history_path.exists():
        return {}
    try:
        text = history_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 2:
        return {}
    headers = [h.strip().strip('"') for h in lines[0].split(",")]
    result: dict[str, list] = {h: [] for h in headers}
    for line in lines[1:]:
        parts = [p.strip() for p in line.split(",")]
        for i, h in enumerate(headers):
            if i < len(parts):
                try:
                    result[h].append(float(parts[i]))
                except (ValueError, TypeError):
                    pass
    return result


def _is_safe_case_id(case_id: str) -> bool:
    """case_id is used as a single path segment; reject traversal/separators."""
    return bool(re.fullmatch(r"[A-Za-z0-9._+-]+", case_id)) and case_id not in {".", ".."}


def _find_case_dirs(case_id: str) -> list[Path]:
    """Return all stage directories belonging to a case_id, sorted."""
    if not _is_safe_case_id(case_id):
        return []
    # case_id may be like "iter_001_aoa_+04p0"
    # structure: data/cache/run_XXX/<case_id>/  or data/cache/<case_id>/
    matches: list[Path] = []
    if not CACHE_ROOT.exists():
        return matches
    # Search run directories
    for run_dir in CACHE_ROOT.iterdir():
        if not run_dir.is_dir():
            continue
        candidate = run_dir / case_id
        if candidate.is_dir():
            matches.append(candidate)
        # Also check direct children
        if run_dir.name == case_id:
            matches.append(run_dir)
    # Also check direct under cache
    direct = CACHE_ROOT / case_id
    if direct.is_dir() and direct not in matches:
        matches.append(direct)
    return sorted(matches, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)


def _find_latest_history(case_dir: Path) -> Path | None:
    """Find the most recent history.csv inside any stage subdirectory."""
    # Stage directories take priority; newest mtime wins
    candidates: list[Path] = []
    # Direct
    h = case_dir / "history.csv"
    if h.exists():
        candidates.append(h)
    # Stage subdirs
    for stage_dir in sorted(case_dir.iterdir()) if case_dir.exists() else []:
        if stage_dir.is_dir():
            sh = stage_dir / "history.csv"
            if sh.exists():
                candidates.append(sh)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _find_surface_csv(case_dir: Path) -> Path | None:
    """Find a surface_flow CSV file anywhere under the case directory."""
    patterns = [
        "surface_flow.csv",
        "surface_flow_*.csv",
        "*surface*.csv",
    ]
    for stage_dir in [case_dir] + sorted(case_dir.glob("stage_*")):
        if not stage_dir.is_dir():
            continue
        for pat in patterns:
            hits = sorted(stage_dir.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
            if hits:
                return hits[0]
    return None


def _residual_columns(traces: dict[str, list]) -> dict[str, list]:
    """Extract named residual columns from history traces."""
    res: dict[str, list] = {}
    priority = {
        "continuity": ["rms[P]", "RMS_P", "RMS_PRESSURE", "rms[Rho]", "RMS_DENSITY", "RES_RHO"],
        "momentum_x": ["rms[U]", "RMS_U", "RES_RHO-U"],
        "momentum_y": ["rms[V]", "RMS_V", "RES_RHO-V"],
        "turbulence_k": ["rms[k]", "RMS_TKE", "RES_TKE"],
        "turbulence_omega": ["rms[omega]", "RMS_OMEGA", "RES_OMEGA"],
        "transition_gamma": ["rms[gamma]", "RMS_GAMMA", "RES_GAMMA"],
    }
    for label, keys in priority.items():
        for key in keys:
            if key in traces and traces[key]:
                res[label] = traces[key]
                break
    return res


def _compute_delta(series: list[float], window: int = 50) -> list[float]:
    """Compute rolling absolute difference over a window."""
    if not series:
        return []
    result: list[float] = [0.0] * len(series)
    for i in range(window, len(series)):
        result[i] = abs(series[i] - series[i - window])
    return result


def _convergence_flag(traces: dict[str, list], window: int = 50) -> str:
    """
    Return PASS / MARGINAL / FAIL based on:
    - ΔCl and ΔCd stability over last `window` iterations
    - Residual decay magnitude
    """
    cl = traces.get("CL") or traces.get("LIFT") or []
    cd = traces.get("CD") or traces.get("DRAG") or []
    res_cols = _residual_columns(traces)

    if not cl or not cd:
        return "FAIL"

    # Force stability check
    if len(cl) >= window:
        cl_tail = cl[-window:]
        cd_tail = cd[-window:]
        cl_range = max(cl_tail) - min(cl_tail)
        cd_range = max(cd_tail) - min(cd_tail)
        cl_mean = abs(sum(cl_tail) / len(cl_tail)) + 1e-12
        cd_mean = abs(sum(cd_tail) / len(cd_tail)) + 1e-12
        cl_rel = cl_range / cl_mean
        cd_rel = cd_range / cd_mean
    else:
        cl_rel = 1.0
        cd_rel = 1.0

    # Residual decay check
    any_res = next(iter(res_cols.values()), [])
    if len(any_res) >= 2:
        first = abs(any_res[0]) if any_res[0] != 0 else 1e-30
        last = abs(any_res[-1]) if any_res[-1] != 0 else 1e-30
        try:
            drop = math.log10(max(first, 1e-30) / max(last, 1e-30))
        except Exception:
            drop = 0.0
    else:
        drop = 0.0

    forces_stable = cl_rel < 0.01 and cd_rel < 0.02
    forces_marginal = cl_rel < 0.05 and cd_rel < 0.10
    residual_good = drop >= 4.0

    if forces_stable and residual_good:
        return "PASS"
    if forces_marginal or (residual_good and cl_rel < 0.05):
        return "MARGINAL"
    return "FAIL"


def _parse_surface_csv(surface_path: Path) -> dict[str, Any]:
    """
    Parse SU2 surface_flow.csv for Cp and Cf distributions.
    Returns upper/lower surface arrays keyed by x/c.
    """
    if not surface_path.exists():
        return {}
    try:
        text = surface_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 2:
        return {}
    headers = [h.strip().strip('"') for h in lines[0].split(",")]

    # Map common SU2 column names
    _col_map = {
        "x": ["x", "x_coord", "Points:0", "X"],
        "y": ["y", "y_coord", "Points:1", "Y"],
        "cp": ["Pressure_Coefficient", "Cp", "PRESSURE_COEFFICIENT", "CoefficientofPressure"],
        "cf_x": ["Skin_Friction_Coefficient_x", "Cf_x", "SkinFrictionCoefficient:0"],
        "cf_y": ["Skin_Friction_Coefficient_y", "Cf_y", "SkinFrictionCoefficient:1"],
        "cf": ["Skin_Friction_Coefficient", "Cf"],
        "gamma": ["Intermittency", "Gamma", "Turbulent_Intermittency"],
        "p": ["Pressure", "p", "PRESSURE"],
    }

    def find_col(keys: list[str]) -> int | None:
        lh = [h.lower() for h in headers]
        for k in keys:
            if k in headers:
                return headers.index(k)
            kl = k.lower()
            for i, h in enumerate(lh):
                if h == kl:
                    return i
        return None

    col_x = find_col(_col_map["x"])
    col_y = find_col(_col_map["y"])
    col_cp = find_col(_col_map["cp"])
    col_cfx = find_col(_col_map["cf_x"])
    col_cfy = find_col(_col_map["cf_y"])
    col_cf = find_col(_col_map["cf"])
    col_gamma = find_col(_col_map["gamma"])

    if col_x is None or col_cp is None:
        return {}

    rows_x, rows_y, rows_cp, rows_cf, rows_gamma = [], [], [], [], []
    for line in lines[1:]:
        parts = [p.strip() for p in line.split(",")]
        try:
            x = float(parts[col_x]) if col_x < len(parts) else None
            y = float(parts[col_y]) if col_y is not None and col_y < len(parts) else 0.0
            cp = float(parts[col_cp]) if col_cp < len(parts) else None
            # Cf: prefer vector magnitude, then x-component, then scalar
            cf_val = 0.0
            if col_cfx is not None and col_cfy is not None:
                cfx = float(parts[col_cfx]) if col_cfx < len(parts) else 0.0
                cfy = float(parts[col_cfy]) if col_cfy < len(parts) else 0.0
                cf_val = math.sqrt(cfx**2 + cfy**2) * (1.0 if cfx >= 0 else -1.0)
            elif col_cf is not None and col_cf < len(parts):
                cf_val = float(parts[col_cf])
            gamma_val = float(parts[col_gamma]) if col_gamma is not None and col_gamma < len(parts) else None
            if x is not None and cp is not None:
                rows_x.append(x)
                rows_y.append(y)
                rows_cp.append(cp)
                rows_cf.append(cf_val)
                rows_gamma.append(gamma_val)
        except (ValueError, IndexError):
            continue

    if not rows_x:
        return {}

    # Split upper / lower by y coordinate
    upper_indices = sorted([i for i, y in enumerate(rows_y) if y >= 0.0], key=lambda i: rows_x[i])
    lower_indices = sorted([i for i, y in enumerate(rows_y) if y < 0.0], key=lambda i: rows_x[i])

    def build_side(indices: list[int]) -> dict:
        return {
            "x": [rows_x[i] for i in indices],
            "cp": [rows_cp[i] for i in indices],
            "cf": [rows_cf[i] for i in indices],
            "gamma": [rows_gamma[i] for i in indices],
        }

    upper = build_side(upper_indices)
    lower = build_side(lower_indices)

    # Separation / reattachment from Cf = 0 crossings (upper surface)
    x_sep = None
    x_reat = None
    cf_upper = upper["cf"]
    x_upper = upper["x"]
    for i in range(1, len(cf_upper)):
        if cf_upper[i - 1] > 0 and cf_upper[i] <= 0 and x_upper[i] > 0.05:
            x_sep = x_upper[i]
        if x_sep is not None and cf_upper[i - 1] <= 0 and cf_upper[i] > 0:
            x_reat = x_upper[i]
            break

    # Transition location from gamma > 0.1
    x_tr = None
    gamma_upper = upper.get("gamma", [])
    for i, g in enumerate(gamma_upper):
        if g is not None and g > 0.1:
            x_tr = upper["x"][i] if i < len(upper["x"]) else None
            break

    bubble_length = (x_reat - x_sep) if x_sep is not None and x_reat is not None else 0.0

    return {
        "upper": upper,
        "lower": lower,
        "x_sep": x_sep,
        "x_reat": x_reat,
        "x_tr": x_tr,
        "bubble_length": round(bubble_length, 6),
    }


# ── routes ────────────────────────────────────────────────────────────────────

@router.get("/list")
def list_monitor_cases() -> dict[str, Any]:
    """
    Return all cases found in the cache directory, with running/completed status
    from the runtime JSON.
    """
    runtime: dict[str, Any] = {}
    if RUNTIME_PATH.exists():
        try:
            runtime = json.loads(RUNTIME_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    running_ids = {c["case_id"] for c in runtime.get("running_cases", [])}
    completed_cases: list[dict[str, Any]] = []

    if CACHE_ROOT.exists():
        for run_dir in sorted(CACHE_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not run_dir.is_dir():
                continue
            for case_dir in sorted(run_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if not case_dir.is_dir():
                    continue
                h = _find_latest_history(case_dir)
                completed_cases.append({
                    "case_id": case_dir.name,
                    "run_dir": run_dir.name,
                    "status": "running" if case_dir.name in running_ids else "completed",
                    "has_history": h is not None,
                    "modified": case_dir.stat().st_mtime,
                })
                if len(completed_cases) >= 200:
                    break

    return {
        "cases": completed_cases,
        "running": list(runtime.get("running_cases", [])),
    }


@router.get("/{case_id}/history")
def get_case_history(case_id: str) -> dict[str, Any]:
    """
    Return iteration-by-iteration history for a running or completed case.
    Includes: Cl, Cd, Cl/Cd, residuals (all available channels), ΔCl, ΔCd, CFL.
    """
    dirs = _find_case_dirs(case_id)
    if not dirs:
        # Case may not have a cache directory yet (just started)
        return {"case_id": case_id, "ready": False, "traces": {}, "iterations": 0}

    case_dir = dirs[0]
    history_path = _find_latest_history(case_dir)
    if not history_path:
        return {"case_id": case_id, "ready": False, "traces": {}, "iterations": 0}

    raw = _parse_history_csv(history_path)
    if not raw:
        return {"case_id": case_id, "ready": False, "traces": {}, "iterations": 0}

    # Core force channels
    cl = raw.get("CL") or raw.get("LIFT") or []
    cd = raw.get("CD") or raw.get("DRAG") or []
    efficiency = [c / max(d, 1e-10) for c, d in zip(cl, cd)]

    # Residual channels
    residuals = _residual_columns(raw)

    # ΔCl and ΔCd
    delta_window = 50
    delta_cl = _compute_delta(cl, delta_window)
    delta_cd = _compute_delta(cd, delta_window)

    # CFL
    cfl = raw.get("CFL_NUMBER") or raw.get("CFL") or []

    # Iteration numbers
    iters = raw.get("Inner_Iter") or raw.get("Iter") or list(range(1, len(cl) + 1))

    n = len(cl)
    conv_flag = _convergence_flag(raw)

    # Current scalar values (last iteration)
    cl_now = cl[-1] if cl else None
    cd_now = cd[-1] if cd else None
    eff_now = efficiency[-1] if efficiency else None
    delta_cl_now = delta_cl[-1] if delta_cl else None
    delta_cd_now = delta_cd[-1] if delta_cd else None
    cfl_now = cfl[-1] if cfl else None

    return {
        "case_id": case_id,
        "ready": True,
        "iterations": n,
        "history_path": str(history_path.relative_to(PROJECT_ROOT)),
        "traces": {
            "iterations": iters[:n],
            "cl": cl,
            "cd": cd,
            "efficiency": efficiency,
            "delta_cl": delta_cl,
            "delta_cd": delta_cd,
            "cfl": cfl,
            **{f"res_{k}": v for k, v in residuals.items()},
        },
        "scalars": {
            "cl": cl_now,
            "cd": cd_now,
            "efficiency": eff_now,
            "delta_cl": delta_cl_now,
            "delta_cd": delta_cd_now,
            "cfl": cfl_now,
        },
        "convergence_flag": conv_flag,
        "residual_channels": list(residuals.keys()),
    }


@router.get("/{case_id}/surface")
def get_case_surface(case_id: str) -> dict[str, Any]:
    """
    Return Cp and Cf surface distributions from the latest surface_flow.csv.
    Also returns separation, reattachment, and transition locations.
    """
    dirs = _find_case_dirs(case_id)
    if not dirs:
        return {"case_id": case_id, "ready": False}

    case_dir = dirs[0]
    surface_path = _find_surface_csv(case_dir)
    if not surface_path:
        return {"case_id": case_id, "ready": False, "reason": "No surface CSV found"}

    data = _parse_surface_csv(surface_path)
    if not data:
        return {"case_id": case_id, "ready": False, "reason": "Surface CSV parse failed"}

    return {
        "case_id": case_id,
        "ready": True,
        "surface_path": str(surface_path.relative_to(PROJECT_ROOT)),
        **data,
    }


@router.get("/{case_id}/summary")
def get_case_summary(case_id: str) -> dict[str, Any]:
    """
    Return a full convergence summary for a completed case.
    Includes all scalar KPIs, convergence score, and physical diagnostics.
    """
    history_resp = get_case_history(case_id)
    surface_resp = get_case_surface(case_id)

    traces = history_resp.get("traces", {})
    scalars = history_resp.get("scalars", {})
    conv_flag = history_resp.get("convergence_flag", "FAIL")

    cl = traces.get("cl", [])
    cd = traces.get("cd", [])

    # Iteration count
    n_iter = history_resp.get("iterations", 0)

    # Residual drop magnitude
    residual_drop = None
    for key in history_resp.get("residual_channels", []):
        series = traces.get(f"res_{key}", [])
        if len(series) >= 2:
            first = abs(series[0]) if series[0] != 0 else 1e-30
            last = abs(series[-1]) if series[-1] != 0 else 1e-30
            try:
                drop = math.log10(max(first, 1e-30) / max(last, 1e-30))
                residual_drop = round(drop, 2)
            except Exception:
                pass
            break

    # ΔCl and ΔCd stability over last 50 iterations
    window = 50
    cl_stable = None
    cd_stable = None
    if len(cl) >= window:
        cl_stable = round(max(cl[-window:]) - min(cl[-window:]), 6)
        cd_stable = round(max(cd[-window:]) - min(cd[-window:]), 6)

    # Convergence score (0–100)
    score = 0
    if conv_flag == "PASS":
        score = 95
    elif conv_flag == "MARGINAL":
        score = 60
    else:
        score = 20
    if residual_drop is not None and residual_drop >= 6:
        score = min(100, score + 5)

    # Surface diagnostics
    x_sep = surface_resp.get("x_sep")
    x_reat = surface_resp.get("x_reat")
    x_tr = surface_resp.get("x_tr")
    bubble_length = surface_resp.get("bubble_length", 0.0)

    return {
        "case_id": case_id,
        "ready": history_resp.get("ready", False),
        "convergence_flag": conv_flag,
        "convergence_score": score,
        "n_iterations": n_iter,
        "residual_drop_decades": residual_drop,
        "cl_variation_last50": cl_stable,
        "cd_variation_last50": cd_stable,
        "scalars": scalars,
        "lsb": {
            "x_sep": x_sep,
            "x_reat": x_reat,
            "x_tr": x_tr,
            "bubble_length": bubble_length,
        },
        "surface_ready": surface_resp.get("ready", False),
    }
