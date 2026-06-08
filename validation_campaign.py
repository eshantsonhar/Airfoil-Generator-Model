"""
PHASE 2-4: CFD Validation Campaign
Runs NACA 0012 at Re=100k, AoA sweep, and inspects raw output.
"""
import sys, os, json, time, subprocess, urllib.request, csv
from pathlib import Path

PROJECT_ROOT = Path("c:/Eshant_Sonhar/airfoil research paper/airfoil generator model")
os.chdir(PROJECT_ROOT)
os.environ["AIRFOIL_TELEMETRY_PATH"] = str(PROJECT_ROOT / "data/logs/telemetry_events.jsonl")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
from airfoil_discovery.config import load_settings
from airfoil_discovery.cfd.su2 import SU2Evaluator

settings = load_settings(PROJECT_ROOT / "config" / "default.yaml")
settings.solver.case_timeout_seconds = 600

evaluator = SU2Evaluator(settings)

# NACA 0012 CST coefficients
naca0012 = np.array([0.1863, 0.0779, 0.2798, 0.0839, -0.1172, 0.0642, -0.0646, 0.0309, 0.001, 1.0])

def inspect_csv(case_dir):
    """Inspect history.csv and return convergence info."""
    hist = case_dir / "history.csv"
    if not hist.exists():
        return {"error": "history.csv not found"}
    with open(hist) as f:
        lines = f.readlines()
    if len(lines) < 2:
        return {"error": "history.csv too short"}
    
    header = [h.strip().strip('"') for h in lines[0].split(',')]
    data = []
    for line in lines[1:]:
        if line.strip() and line.strip() != ',':
            vals = line.strip().split(',')
            data.append(dict(zip(header, vals)))
    
    result = {
        "n_rows": len(data),
        "last_iter": data[-1].get("ITER", "?") if data else "?",
        "cl": data[-1].get("CL", "?") if data else "?",
        "cd": data[-1].get("CD", "?") if data else "?",
        "residual_rms_start": data[0].get("RMS_PRESSURE", data[0].get("rms[P]", "?")) if data else "?",
        "residual_rms_end": data[-1].get("RMS_PRESSURE", data[-1].get("rms[P]", "?")) if data else "?",
    }
    return result

def inspect_surface(case_dir):
    """Inspect surface.csv."""
    surf = case_dir / "surface.csv"
    if not surf.exists():
        return None
    with open(surf) as f:
        lines = f.readlines()
    return {
        "n_points": len(lines) - 1 if len(lines) > 1 else 0,
        "header": lines[0].strip() if lines else "",
    }

def run_naca_aoa(aoa, mesh_level="L0"):
    """Run NACA 0012 at given AoA and return results."""
    case_dir = PROJECT_ROOT / "data" / "cache" / f"val_naca_aoa{aoa:+.0f}"
    print(f"\n--- NACA 0012 AoA={aoa:+1.0f}°, mesh={mesh_level} ---")
    t0 = time.time()
    result = evaluator.run_evaluation(naca0012, case_dir, mesh_level=mesh_level, aoa=aoa)
    elapsed = time.time() - t0
    print(f"  Status: {result.status.value}  CL={result.cl:.6f}  CD={result.cd:.6f}  Time={elapsed:.0f}s")
    
    csv_info = inspect_csv(case_dir)
    surf_info = inspect_surface(case_dir)
    
    # List all generated files
    files = {}
    if case_dir.exists():
        for f in sorted(case_dir.iterdir()):
            files[f.name] = f.stat().st_size
    
    return {
        "aoa": aoa,
        "status": result.status.value,
        "cl": float(result.cl),
        "cd": float(result.cd),
        "thickness": float(result.thickness),
        "elapsed_s": round(elapsed, 1),
        "converged": (result.convergence_report or {}).get("is_valid", False),
        "failure_stage": result.failure_stage,
        "failure_reason": result.failure_reason,
        "csv_info": csv_info,
        "surface_info": surf_info,
        "files": files,
        "residual_history_len": len(result.residual_history or []),
    }

print("=" * 70)
print("CFD VALIDATION CAMPAIGN")
print("=" * 70)

# Run AoA sweep
results = []
for aoa in [0, 2, 4, 6, 8]:
    r = run_naca_aoa(float(aoa))
    results.append(r)

# Print summary table
print("\n" + "=" * 80)
print("VALIDATION SUMMARY — NACA 0012, Re=100k, L0 mesh")
print("=" * 80)
print(f"{'AoA':>5s} {'Status':>12s} {'CL':>10s} {'CD':>10s} {'Rows':>6s} {'ResStart':>10s} {'ResEnd':>10s} {'Files':>6s}")
print("-" * 80)
for r in results:
    ci = r["csv_info"]
    if "error" not in ci:
        rs = ci.get("residual_rms_start", "?")
        re = ci.get("residual_rms_end", "?")
        rows = ci.get("n_rows", 0)
    else:
        rs = "?"
        re = "?"
        rows = 0
    nf = len(r["files"])
    print(f"{r['aoa']:>5.0f} {r['status']:>12s} {r['cl']:>10.6f} {r['cd']:>10.6f} {rows:>6d} {str(rs):>10s} {str(re):>10s} {nf:>6d}")

# Show generated config
print("\n\n--- GENERATED SU2 CONFIG (AoA=4°) ---")
cfg_path = PROJECT_ROOT / "data" / "cache" / "val_naca_aoa+4" / "config_primal.cfg"
if cfg_path.exists():
    print(cfg_path.read_text())
else:
    print("Config not found")

# Show residual history for AoA=4°
print("\n\n--- RESIDUAL HISTORY (first 10 + last 5, AoA=4°) ---")
case_dir = PROJECT_ROOT / "data" / "cache" / "val_naca_aoa+4"
hist = case_dir / "history.csv"
if hist.exists():
    with open(hist) as f:
        lines = f.readlines()
    header = [h.strip().strip('"') for h in lines[0].split(',')]
    for i, line in enumerate(lines[1:6]):
        print(f"  iter {i}: {line.strip()[:90]}")
    print("  ...")
    for line in lines[-5:]:
        print(f"  iter : {line.strip()[:90]}")

print("\n\n=== VALIDATION CAMPAIGN COMPLETE ===")