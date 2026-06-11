"""
Compute NACA 0012 CST coefficients with exactly t/c=0.12 for symmetry.
Run 5-point polar sweep at Re=100k with coarse_factor=10 mesh.
Extract Cf-based LSB from surface data.
"""
import sys, os, time
sys.path.insert(0, 'src')
os.environ['AIRFOIL_TELEMETRY_PATH'] = 'data/logs/telemetry_events.jsonl'

import numpy as np
from pathlib import Path
from airfoil_discovery.config import load_settings
from airfoil_discovery.cfd.su2 import SU2Evaluator
from airfoil_discovery.cfd.mesh import MeshFidelityManager, FidelityParams
from airfoil_discovery.geometry.cst import CSTAirfoil, cosine_spacing
from airfoil_discovery.schemas import CSTParameters

# Use coarse_factor=10 for ~8k cells, fast meshing
MeshFidelityManager.REGISTRY["L0"] = FidelityParams("L0", coarse_factor=10.0, y_plus_target=1.5)

# -----------------------------------------------------------------------
# Step 1: Compute CST coefficients producing exactly t/c = 0.12
# Anti-symmetric: upper = -lower (for NACA 00xx symmetric)
# -----------------------------------------------------------------------
print("=" * 60)
print("STEP 1: Computing anti-symmetric CST coeffs for NACA 0012 (t/c=0.12)")
print("=" * 60)

def naca00xx_thickness(x, t=0.12):
    """NACA 00xx thickness distribution formula."""
    return 5.0 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)

# Generate half-thickness at chord stations (cosine spacing concentrates at LE/TE)
x_pts = cosine_spacing(400)
half_t = naca00xx_thickness(x_pts) / 2.0  # y_u = +half_t for symmetric

# CST: y(x) = C(x) * sum(au_k * B_k(x)) + x * te/2
# For symmetric with te=0: y_u = C(x) * sum(au_k * B_k(x))
C = np.clip(x_pts**0.5 * (1.0 - x_pts)**1.0, 1e-12, None)

# Degree-3 Bernstein basis (4 coefficients)
B3 = np.array([(1-x_pts)**3, 3*x_pts*(1-x_pts)**2, 3*x_pts**2*(1-x_pts), x_pts**3])

# Fit: half_t / C = sum(au_k * B_k)
# Use only x where C is not too small (avoid LE singularity)
valid = C > 1e-6
A = B3[:, valid].T
target = (half_t[valid] / C[valid])

coeffs, _, _, _ = np.linalg.lstsq(A, target, rcond=None)
au = coeffs  # upper coefficients
al = -coeffs  # lower = anti-symmetric

# Scale to exactly t/c=0.12
settings = load_settings('config/default.yaml')
airfoil = CSTAirfoil(settings.geometry)
params = CSTParameters(upper=au, lower=al, trailing_edge_thickness=0.0)
coords = airfoil.full_coordinates(params)
yu = coords[:len(coords)//2, 1]
yl_new = coords[len(coords)//2:, 1]
yl_rev = yl_new[::-1]
n_min = min(len(yu), len(yl_rev))
max_t = np.max(yu[:n_min] - yl_rev[:n_min])

# Scale factor to achieve exactly 12% thickness
scale_factor = 0.12 / max_t
au_final = au * scale_factor
al_final = -au_final  # anti-symmetric

# Verify final thickness
params2 = CSTParameters(upper=au_final, lower=al_final, trailing_edge_thickness=0.001)
coords2 = airfoil.full_coordinates(params2)
yu2 = coords2[:len(coords2)//2, 1]
yl2 = coords2[len(coords2)//2:, 1][::-1]
n2 = min(len(yu2), len(yl2))
max_t_final = np.max(yu2[:n2] - yl2[:n2])
asym = np.max(np.abs(yu2[:n2] + yl2[:n2]))

print(f"  Fitted: au = [{au[0]:.6f}, {au[1]:.6f}, {au[2]:.6f}, {au[3]:.6f}]")
print(f"  Raw max thickness: {max_t:.6f} (target 0.12)")
print(f"  Scale factor: {scale_factor:.6f}")
print(f"  Final au = [{au_final[0]:.6f}, {au_final[1]:.6f}, {au_final[2]:.6f}, {au_final[3]:.6f}]")
print(f"  Final al = [{al_final[0]:.6f}, {al_final[1]:.6f}, {al_final[2]:.6f}, {al_final[3]:.6f}]")
print(f"  Final max thickness: {max_t_final:.6f} (target 0.12)")
print(f"  Max asymmetry: {asym:.8e}")
assert abs(max_t_final - 0.12) < 0.002, f"Thickness {max_t_final} != 0.12!"
assert asym < 1e-6, f"Asymmetry {asym} too large!"
assert max_t_final < 0.13, f"Thickness {max_t_final} exceeds validator bounds [0.06, 0.13]!"

design_vector = np.array([au_final[0], au_final[1], au_final[2], au_final[3],
                          al_final[0], al_final[1], al_final[2], al_final[3],
                          0.001, 1.0])
print(f"\n  Final design vector:")
print(f"  np.array([{design_vector[0]:.6f}, {design_vector[1]:.6f}, {design_vector[2]:.6f}, {design_vector[3]:.6f},")
print(f"            {design_vector[4]:.6f}, {design_vector[5]:.6f}, {design_vector[6]:.6f}, {design_vector[7]:.6f},")
print(f"            {design_vector[8]}, {design_vector[9]}])")

# -----------------------------------------------------------------------
# Step 2: Run 5-point polar sweep
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: 5-Point Polar Sweep — NACA 0012, Re=100k")
print("=" * 60)

s = load_settings('config/default.yaml')
s.solver.case_timeout_seconds = 1800
s.solver.stage1_iter = 500
evaluator = SU2Evaluator(s)

def extract_lsb(case_dir):
    """Cf zero-crossing LSB detection."""
    surf = list(case_dir.glob("*surface*.csv"))
    if not surf:
        return None, None, None, "No surface file"
    try:
        with open(surf[0]) as f:
            header = f.readline().strip().split(',')
        header = [h.strip().strip('"').lower() for h in header]
        data = np.loadtxt(surf[0], skiprows=1, delimiter=',')
        
        cf_col = next((i for i, h in enumerate(header) 
                       if h in ('cf', 'skin_friction_x', 'skinfriction[0]')),
                      5 if data.shape[1] >= 6 else None)
        if cf_col is None:
            return None, None, None, f"No Cf col in {header}"
        
        n = len(data) // 2
        x = data[n:, 0]
        cf = data[n:, cf_col]
        idx = np.argsort(x)
        x, cf = x[idx], cf[idx]
        
        sep, reatt = None, None
        for i in range(1, len(cf)):
            if cf[i-1] > 0 and cf[i] < 0 and sep is None:
                frac = cf[i-1] / max(cf[i-1] - cf[i], 1e-30)
                sep = x[i-1] + frac * (x[i] - x[i-1])
            if cf[i-1] < 0 and cf[i] > 0 and sep is not None and reatt is None:
                frac = cf[i-1] / max(cf[i-1] - cf[i], 1e-30)
                reatt = x[i-1] + frac * (x[i] - x[i-1])
        bl = (reatt - sep) if (sep and reatt) else None
        return sep, reatt, bl, "OK"
    except Exception as e:
        return None, None, None, str(e)

def read_csv(case_dir):
    hist = case_dir / "history.csv"
    if not hist.exists():
        return None
    with open(hist) as f:
        lines = f.readlines()
    if len(lines) < 2:
        return None
    hdr = [h.strip().strip('"') for h in lines[0].split(',')]
    rows = [l.split(',') for l in lines[1:] if l.strip() and l.strip() != ',']
    if not rows:
        return None
    rms_c = next((c for c in ["rms[P]", "RMS_PRESSURE"] if c in hdr), None)
    if rms_c and len(rows) > 1:
        ri = hdr.index(rms_c)
        return {
            "drop": abs(float(rows[-1][ri]) - float(rows[0][ri])),
            "cl": float(rows[-1][hdr.index("CL")]) if "CL" in hdr else 0,
            "cd": float(rows[-1][hdr.index("CD")]) if "CD" in hdr else 0,
        }
    return None

results = []
for aoa in [0, 2, 4, 6, 8]:
    case_dir = Path(f"data/cache/sweep_{aoa}")
    print(f"\n--- α = {aoa:+1.0f}° ---")
    t0 = time.time()
    r = evaluator.run_evaluation(design_vector, case_dir, mesh_level="L0", aoa=float(aoa))
    elapsed = time.time() - t0
    
    d = read_csv(case_dir) or {"drop": 0, "cl": 0, "cd": 0}
    sep, reatt, bl, lsbinfo = extract_lsb(case_dir)
    
    print(f"  {r.status.value:12s}  CL={d['cl']:.6f}  CD={d['cd']:.6f}  "
          f"Δlog10R={d['drop']:.1f}  LSB_len={bl if bl else 'N/A'}  Time={elapsed:.0f}s")
    
    results.append({"aoa": aoa, "cl": d["cl"], "cd": d["cd"], "drop": d["drop"],
                    "sep": sep, "reatt": reatt, "bl": bl, "status": r.status.value})

# -----------------------------------------------------------------------
# Step 3: Print final results table
# -----------------------------------------------------------------------
print("\n\n" + "=" * 120)
print("VALIDATION TABLE — Symmetric NACA 0012 (t/c=0.12), Re=100k, L0 mesh")
print("=" * 120)
print(f"{'α':>4s} | {'CL':>10s} | {'CD':>10s} | {'x_sep/c':>10s} | {'x_reat/c':>10s} | {'L_LSB/c':>10s} | {'Δlog₁₀R':>10s} | {'Status':>12s}")
print("-" * 120)
for r in results:
    sep_s = f"{r['sep']:.4f}" if r['sep'] is not None else "N/A"
    reatt_s = f"{r['reatt']:.4f}" if r['reatt'] is not None else "N/A"
    bl_s = f"{r['bl']:.4f}" if r['bl'] is not None else "N/A"
    print(f"{r['aoa']:>4.0f} | {r['cl']:>10.6f} | {r['cd']:>10.6f} | {sep_s:>10s} | {reatt_s:>10s} | {bl_s:>10s} | {r['drop']:>10.1f} | {r['status']:>12s}")

# -----------------------------------------------------------------------
# Final classification
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
sym_ok = abs(results[0]["cl"]) <= 0.02
conv_ok = all(r["drop"] >= 3.0 for r in results if r["status"] not in ("CONFIG_ERROR", "CRASHED"))
all_ran = all(r["status"] not in ("CONFIG_ERROR", "CRASHED") for r in results)

print(f"CL at α=0°: {results[0]['cl']:.6f}  {'✅' if sym_ok else '❌'}")
print(f"All AoA ran: {'✅' if all_ran else '❌'}")
if all_ran and conv_ok and sym_ok:
    print("CLASSIFICATION: A — Suitable for absolute physical optimization")
elif all_ran:
    print("CLASSIFICATION: B — Suitable for design ranking")
else:
    print("CLASSIFICATION: C — Qualitative only")