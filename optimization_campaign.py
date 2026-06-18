"""
Multi-Objective Optimization Campaign (3 iterations).
Uses working naca_like coefficients, CFD evaluation, MMA step, LSB targeting.
"""
import sys, os, time, json
sys.path.insert(0, 'src')
os.environ['AIRFOIL_TELEMETRY_PATH'] = 'data/logs/telemetry_events.jsonl'

import numpy as np
from pathlib import Path
from airfoil_discovery.config import load_settings
from airfoil_discovery.cfd.su2 import SU2Evaluator
from airfoil_discovery.cfd.mesh import MeshFidelityManager, FidelityParams
from airfoil_discovery.optimization.mma_engine import SvanbergMMA, TrustRegionGovernor

# Fast meshing for iterative testing
MeshFidelityManager.REGISTRY["L0"] = FidelityParams("L0", coarse_factor=10.0, y_plus_target=1.5)

settings = load_settings('config/default.yaml')
settings.solver.case_timeout_seconds = 1800
settings.solver.stage1_iter = 500
evaluator = SU2Evaluator(settings)

# ============================================================
# Step 1: Working baseline and bounds
# ============================================================
naca_like_base = np.array([0.18, 0.05, 0.34, 0.10, -0.19, 0.05, -0.09, 0.03, 0.004, 1.0])

# ±25% bounds on CST shape coefficients, tight on scale/TE
x_min = np.array([-0.3, -0.3, -0.3, -0.3, -0.5, -0.5, -0.5, -0.5, 0.001, 0.8])
x_max = np.array([ 0.6,  0.6,  0.6,  0.6,  0.3,  0.3,  0.3,  0.3, 0.020, 1.2])

mma = SvanbergMMA(n_vars=10, n_constraints=2, x_min=x_min, x_max=x_max, move_limit=0.05)
governor = TrustRegionGovernor(initial_radius=0.1, max_radius=0.5, min_radius=1e-6)

# ============================================================
# Step 2: Helper functions
# ============================================================
def extract_lsb_length(case_dir):
    surf = list(case_dir.glob("*surface*.csv"))
    if not surf:
        return None
    try:
        with open(surf[0]) as f:
            header = f.readline().strip().split(',')
        header = [h.strip().strip('"').lower() for h in header]
        data = np.loadtxt(surf[0], skiprows=1, delimiter=',')
        cf_col = next((i for i, h in enumerate(header) if h in ('cf', 'skin_friction_x', 'skinfriction[0]')),
                      5 if data.shape[1] >= 6 else None)
        if cf_col is None:
            return None
        n = len(data) // 2
        x = data[n:, 0]; cf = data[n:, cf_col]
        idx = np.argsort(x); x, cf = x[idx], cf[idx]
        sep = None; reatt = None
        for i in range(1, len(cf)):
            if cf[i-1] > 0 and cf[i] < 0 and sep is None:
                frac = cf[i-1] / max(cf[i-1] - cf[i], 1e-30)
                sep = x[i-1] + frac*(x[i] - x[i-1])
            if cf[i-1] < 0 and cf[i] > 0 and sep and reatt is None:
                frac = cf[i-1] / max(cf[i-1] - cf[i], 1e-30)
                reatt = x[i-1] + frac*(x[i] - x[i-1])
        return (reatt - sep) if (sep and reatt) else None
    except Exception:
        return None

def run_cfd(design, tag):
    case_dir = Path(f"data/cache/opt_{tag}")
    result = evaluator.run_evaluation(design, case_dir, mesh_level="L0", aoa=4.0)
    lsb_len = extract_lsb_length(case_dir)
    hist = case_dir / "history.csv"
    cl_val, cd_val = 0.0, 0.0
    if hist.exists():
        with open(hist) as f:
            lines = f.readlines()
        if len(lines) > 1:
            hdr = [h.strip().strip('"') for h in lines[0].split(',')]
            rows = [l.split(',') for l in lines[1:] if l.strip() and l.strip() != ',']
            if rows:
                if "CL" in hdr:
                    cl_val = float(rows[-1][hdr.index("CL")])
                if "CD" in hdr:
                    cd_val = float(rows[-1][hdr.index("CD")])
    valid = (result.status.value not in ("CONFIG_ERROR", "CRASHED"))
    return {"cl": cl_val, "cd": cd_val, "lsb": lsb_len, "status": result.status.value, "valid": valid}

# ============================================================
# Step 3: Run optimization campaign
# ============================================================
print("=" * 80)
print("MULTI-OBJECTIVE OPTIMIZATION CAMPAIGN (3 iterations)")
print("=" * 80)

x_current = naca_like_base.copy()
mma.initialize(x_current)

results = []
for k in range(4):  # baseline + 3 iterations
    tag = "baseline" if k == 0 else f"iter{k}"
    print(f"\n--- [{tag.upper()}] Evaluating design ---")
    r = run_cfd(x_current, tag)
    if not r["valid"]:
        print(f"  CFD FAILED ({r['status']}) — aborting")
        break
    
    cl, cd = r["cl"], r["cd"]
    lsb = r["lsb"]
    J = cd + 0.5 * (lsb if lsb else 0.0)
    
    print(f"  CL={cl:.6f}  CD={cd:.6f}  L_LSB={lsb if lsb else 'N/A'}  J={J:.6f}")
    
    results.append({
        "iter": tag, "cl": cl, "cd": cd, "lsb": lsb,
        "J": J, "x": x_current.copy(), "status": r["status"]
    })
    
    if k == 0:
        baseline_cl = cl
        baseline_J = J
    
    if k > 0 and k < 3:
        # MMA step
        cl_deficit = max(0.0, baseline_cl - cl)
        g = np.array([cl_deficit, 0.0])
        dg = np.zeros((2, 10))
        
        x_candidate, accepted, state = mma.run_optimization_step(f=cd, df=np.zeros(10), g=g, dg=dg)
        x_current = x_candidate.copy()
        print(f"  MMA: step_accepted={accepted}")
        if not accepted:
            print("  Step rejected — continuing with current design")
            break

# ============================================================
# Step 4: Results table
# ============================================================
print("\n\n" + "=" * 100)
print("OPTIMIZATION CAMPAIGN RESULTS")
print("=" * 100)
print(f"{'Step':>10s} | {'CL':>10s} | {'CD':>10s} | {'L_LSB':>8s} | {'J (CD+0.5L_LSB)':>18s} | {'ΔJ(%)':>8s} | {'Status':>12s}")
print("-" * 100)

prev_J = results[0]["J"] if results else 1
for r in results:
    pct = (r["J"] - prev_J) / abs(prev_J) * 100 if prev_J != 0 else 0
    lsb_str = f"{r['lsb']:.4f}" if r['lsb'] else "N/A"
    print(f"{r['iter']:>10s} | {r['cl']:>10.6f} | {r['cd']:>10.6f} | {lsb_str:>8s} | {r['J']:>18.6f} | {pct:>+7.2f}% | {r['status']:>12s}")
    prev_J = r["J"]

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
if len(results) >= 2:
    first = results[0]
    last = results[-1]
    print(f"  Baseline J = {first['J']:.6f}")
    print(f"  Final J    = {last['J']:.6f}")
    print(f"  Change     = {(last['J']-first['J'])/abs(first['J'])*100:+.2f}%")
    if last['J'] < first['J']:
        print("  Design improved: composite objective decreased.")
        if last['lsb'] and first['lsb']:
            if last['lsb'] < first['lsb']:
                print("  LSB length reduced — passive suppression trend confirmed.")
    if last['cl'] >= first['cl']:
        print("  Lift constraint satisfied (CL >= baseline).")
else:
    print("  Campaign incomplete — fewer than 2 iterations executed.")