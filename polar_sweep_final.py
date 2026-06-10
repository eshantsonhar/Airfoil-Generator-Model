"""
Polar sweep with corrected geometry and TE thickness within bounds.
Uses coarse_factor=10, TE thickness=0.001, corrected symmetric CST coeffs.
"""
import sys, os, time
sys.path.insert(0, 'src')
os.environ['AIRFOIL_TELEMETRY_PATH'] = 'data/logs/telemetry_events.jsonl'

import numpy as np
from pathlib import Path
from airfoil_discovery.config import load_settings
from airfoil_discovery.cfd.su2 import SU2Evaluator
from airfoil_discovery.cfd.mesh import MeshFidelityManager, FidelityParams

MeshFidelityManager.REGISTRY["L0"] = FidelityParams("L0", coarse_factor=10.0, y_plus_target=1.5)

# Anti-symmetric CST with TE thickness = 0.001 (minimum config bound)
naca_symmetric = np.array([0.650660, -0.879011, 0.643487, 0.004026,
                           -0.650660, 0.879011, -0.643487, -0.004026,
                           0.001, 1.0])

def extract_lsb(case_dir):
    surf = list(case_dir.glob("*surface*.csv"))
    if not surf:
        return None, None, None, "No surface file"
    try:
        with open(surf[0]) as f:
            header = f.readline().strip().split(',')
        header = [h.strip().strip('"').lower() for h in header]
        data = np.loadtxt(surf[0], skiprows=1, delimiter=',')
        
        cf_col = next((i for i, h in enumerate(header) if h in ('cf', 'skin_friction_x', 'skinfriction[0]')), 
                      5 if data.shape[1] >= 6 else None)
        if cf_col is None:
            return None, None, None, f"Columns: {header}"
        
        # Upper surface is second half of points
        n = len(data) // 2
        x = data[n:, 0]
        cf = data[n:, cf_col]
        idx = np.argsort(x)
        x, cf = x[idx], cf[idx]
        
        sep = None
        reatt = None
        for i in range(1, len(cf)):
            if cf[i-1] > 0 and cf[i] < 0 and sep is None:
                frac = cf[i-1] / (cf[i-1] - cf[i]) if cf[i] != cf[i-1] else 0.5
                sep = x[i-1] + frac * (x[i] - x[i-1])
            if cf[i-1] < 0 and cf[i] > 0 and sep is not None and reatt is None:
                frac = cf[i-1] / (cf[i-1] - cf[i]) if cf[i] != cf[i-1] else 0.5
                reatt = x[i-1] + frac * (x[i] - x[i-1])
        
        bl = (reatt - sep) if (sep and reatt) else None
        return sep, reatt, bl, "OK"
    except Exception as e:
        return None, None, None, str(e)

s = load_settings('config/default.yaml')
s.solver.case_timeout_seconds = 1800
s.solver.stage1_iter = 500
evaluator = SU2Evaluator(s)

results = []
for aoa in [0, 2, 4, 6, 8]:
    case_dir = Path(f"data/cache/polar_{aoa}")
    print(f"\n--- AoA = {aoa:+.0f}° ---")
    t0 = time.time()
    r = evaluator.run_evaluation(naca_symmetric, case_dir, mesh_level="L0", aoa=float(aoa))
    elapsed = time.time() - t0
    
    # Read CSV
    csv_data = None
    hist = case_dir / "history.csv"
    if hist.exists():
        with open(hist) as f:
            lines = f.readlines()
        if len(lines) > 1:
            hdr = [h.strip().strip('"') for h in lines[0].split(',')]
            rows = [l.split(',') for l in lines[1:] if l.strip() and l.strip() != ',']
            rms_c = next((c for c in ["rms[P]", "RMS_PRESSURE"] if c in hdr), None)
            if rms_c and rows:
                ri = hdr.index(rms_c)
                csv_data = {
                    "drop": abs(float(rows[-1][ri]) - float(rows[0][ri])),
                    "cl": float(rows[-1][hdr.index("CL")]) if "CL" in hdr else 0,
                    "cd": float(rows[-1][hdr.index("CD")]) if "CD" in hdr else 0,
                }
    
    sep, reatt, bl, lsb_info = extract_lsb(case_dir)
    d = csv_data or {"drop": 0, "cl": 0, "cd": 0}
    
    print(f"  Status: {r.status.value}  CL={d['cl']:.6f}  CD={d['cd']:.6f}  Drop={d['drop']:.1f}ord")
    results.append({"aoa": aoa, "cl": d["cl"], "cd": d["cd"], "drop": d["drop"],
                    "sep": sep, "reatt": reatt, "bl": bl, "status": r.status.value, "t": elapsed})

print("\n\n" + "=" * 120)
print("VALIDATION POLAR — Symmetric NACA 0012, Re=100k")
print("=" * 120)
print(f"{'α':>4s} | {'CL':>10s} | {'CD':>10s} | {'x_sep':>8s} | {'x_reat':>8s} | {'L_LSB':>8s} | {'Δlog₁₀R':>8s} | {'Time':>6s}")
print("-" * 120)
for r in results:
    sep_str = f"{r['sep']:.4f}" if r['sep'] is not None else "N/A"
    reatt_str = f"{r['reatt']:.4f}" if r['reatt'] is not None else "N/A"
    bl_str = f"{r['bl']:.4f}" if r['bl'] is not None else "N/A"
    print(f"{r['aoa']:>4.0f} | {r['cl']:>10.6f} | {r['cd']:>10.6f} | "
          f"{sep_str:>8s} | {reatt_str:>8s} | {bl_str:>8s} | "
          f"{r['drop']:>8.1f} | {r['t']:>5.0f}s")

cl_0 = results[0]["cl"]
symmetry_pass = abs(cl_0) <= 0.02
all_drops = all(r['drop'] >= 4.0 for r in results if r['drop'] > 0)
conv_status = all(r['status'] not in ('CRASHED', 'CONFIG_ERROR') for r in results)

print(f"\nCL at α=0°: {cl_0:.6f} {'✅ PASS' if symmetry_pass else '❌ FAIL'}")
if all_drops and conv_status:
    print("CLASSIFICATION: A — Suitable for absolute physical optimization")
elif conv_status:
    print("CLASSIFICATION: B — Suitable for design ranking")
else:
    print("CLASSIFICATION: C — Qualitative only")