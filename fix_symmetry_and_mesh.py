"""
Fix NACA 0012 CST asymmetry + deploy high-density mesh.
Computes mathematically correct anti-symmetric CST coefficients
and increases mesh resolution for proper convergence.
"""
import sys, os, time
sys.path.insert(0, 'src')
os.environ['AIRFOIL_TELEMETRY_PATH'] = 'data/logs/telemetry_events.jsonl'

import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
from airfoil_discovery.geometry.cst import CSTAirfoil, cosine_spacing
from airfoil_discovery.schemas import CSTParameters
from airfoil_discovery.config import load_settings
from airfoil_discovery.cfd.su2 import SU2Evaluator

settings = load_settings('config/default.yaml')

# =====================================================================
# PART 1: Compute correct anti-symmetric CST coefficients for NACA 0012
# =====================================================================
print("=" * 60)
print("PART 1: Anti-symmetric NACA 0012 CST Coefficients")
print("=" * 60)

# NACA 0012 analytical thickness distribution
def naca0012_thickness(x, t=0.12):
    """NACA 00xx thickness distribution."""
    return 5.0 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)

# Generate half-thickness at CST-distributed x points
x_samples = cosine_spacing(200)
half_t = naca0012_thickness(x_samples) / 2.0  # y_u = +half_t, y_l = -half_t for symmetric

# CST class function
C = (x_samples ** 0.5) * ((1.0 - x_samples) ** 1.0)
C_clipped = np.clip(C, 1e-10, None)

# For symmetric airfoil with zero TE thickness:
# y_u(x) = C(x) * sum(au_k * B_k(x))  (since half_t*te*x = 0 for te=0)
# So we fit: half_t / C(x) = sum(au_k * B_k(x))

Bernstein_3 = np.array([x_samples**3, 
                         3*x_samples**2*(1-x_samples),
                         3*x_samples*(1-x_samples)**2,
                         (1-x_samples)**3])  # degree 3 Bernstein basis

shape_target = half_t / C_clipped

# Least squares fit for 4 upper coefficients
A = Bernstein_3.T  # N x 4 design matrix
coeffs, residuals, rank, s = np.linalg.lstsq(A[C_clipped > 1e-6], shape_target[C_clipped > 1e-6], rcond=None)

au_fitted = coeffs  # upper coefficients
al_fitted = -coeffs  # lower = negative of upper (anti-symmetric)

print(f"  Fitted upper CST coeffs: au = [{au_fitted[0]:.6f}, {au_fitted[1]:.6f}, {au_fitted[2]:.6f}, {au_fitted[3]:.6f}]")
print(f"  Fitted lower CST coeffs: al = [{al_fitted[0]:.6f}, {al_fitted[1]:.6f}, {al_fitted[2]:.6f}, {al_fitted[3]:.6f}]")

# Verify symmetry: maximum asymmetry should be machine-zero
params = CSTParameters(upper=au_fitted, lower=al_fitted, trailing_edge_thickness=0.0)
airfoil = CSTAirfoil(settings.geometry)
coords = airfoil.full_coordinates(params)
n = len(coords) // 2
yu = coords[:n, 1]
yl = coords[n:, 1]
yl_rev = yl[::-1]
min_len = min(len(yu), len(yl_rev))
asymmetry = np.max(np.abs(yu[:min_len] + yl_rev[:min_len]))
print(f"  Max asymmetry |yu + yl[flipped]|: {asymmetry:.8e}")

# Comparison with original coefficients
orig_coeffs = np.array([0.1863, 0.0779, 0.2798, 0.0839, -0.1172, 0.0642, -0.0646, 0.0309, 0.001, 1.0])
print(f"\n  Original:    upper=[0.1863, 0.0779, 0.2798, 0.0839] lower=[-0.1172, 0.0642, -0.0646, 0.0309]")
print(f"  Corrected:   upper=[{au_fitted[0]:.4f}, {au_fitted[1]:.4f}, {au_fitted[2]:.4f}, {au_fitted[3]:.4f}] lower=[{al_fitted[0]:.4f}, {al_fitted[1]:.4f}, {al_fitted[2]:.4f}, {al_fitted[3]:.4f}]")

# =====================================================================
# PART 2: High-density mesh with proper resolution
# =====================================================================
print("\n" + "=" * 60)
print("PART 2: High-Density Mesh Generation")
print("=" * 60)

# Update mesh fidelity levels for real CFD resolution
from airfoil_discovery.cfd.mesh import MeshFidelityManager, FidelityParams

# New mesh levels with proper cell counts
MeshFidelityManager.REGISTRY = {
    "L0": FidelityParams("L0", coarse_factor=3.0, y_plus_target=1.0),
    "L1": FidelityParams("L1", coarse_factor=2.0, y_plus_target=0.8),
    "L2": FidelityParams("L2", coarse_factor=1.5, y_plus_target=0.5),
    "L3": FidelityParams("L3", coarse_factor=1.0, y_plus_target=0.3),
}

print("  Mesh fidelity levels reconfigured:")
for level, params in MeshFidelityManager.REGISTRY.items():
    print(f"    {level}: coarse_factor={params.coarse_factor}, y+<{params.y_plus_target}")

# Also update the config's first cell height for better BL resolution
print(f"\n  First cell height (Re=100k): will be computed per-reynolds by physics module")
print(f"  Target: y+<1.0 at Re=100k => Δy₁ ≈ 4.2e-5 chord")

# =====================================================================
# PART 3: Run test with corrected coefficients and denser mesh
# =====================================================================
print("\n" + "=" * 60)
print("PART 3: CFD Test — Symmetric NACA 0012, Re=100k, AoA=0°")
print("=" * 60)

# Use corrected symmetric coefficients
symmetric_design = np.array([au_fitted[0], au_fitted[1], au_fitted[2], au_fitted[3],
                             al_fitted[0], al_fitted[1], al_fitted[2], al_fitted[3],
                             0.0, 1.0])  # TE thickness = 0 for closed TE

s = load_settings('config/default.yaml')
s.solver.case_timeout_seconds = 3600
s.solver.stage1_iter = 1000  # More iterations for convergence
s.solver.stage1_cfl = 3.0

evaluator = SU2Evaluator(s)

case_dir = Path('data/cache/sym_test')
t0 = time.time()
result = evaluator.run_evaluation(symmetric_design, case_dir, mesh_level="L0", aoa=0.0)
elapsed = time.time() - t0

c = result.convergence_report or {}
print(f"\n  Status: {result.status.value}")
print(f"  CL = {result.cl:.8f}  (should be ~0.0 for symmetric airfoil at AoA=0°)")
print(f"  CD = {result.cd:.8f}")
print(f"  Time: {elapsed:.0f}s")

# Check residual drop
hist = case_dir / "history.csv"
if hist.exists():
    with open(hist) as f:
        lines = f.readlines()
    header = [h.strip().strip('"') for h in lines[0].split(',')]
    rms_col = 'rms[P]' if 'rms[P]' in header else 'RMS_PRESSURE'
    if rms_col in header:
        rms_idx = header.index(rms_col)
        data = [l.split(',') for l in lines[1:] if l.strip() and l.strip() != ',']
        if data and len(data) > 1:
            start_log = float(data[0][rms_idx])
            end_log = float(data[-1][rms_idx])
            drop = abs(end_log - start_log)
            print(f"  Residual drop: {drop:.1f} orders")
            print(f"  Start: {start_log:.2f} → End: {end_log:.2f}")

print(f"\n  Generated files in {case_dir}:")
for f in sorted(case_dir.iterdir()):
    size = f.stat().st_size
    print(f"    {f.name:30s} {size:>8d} bytes")

print("\n=== FIXES COMPLETE ===")
print(f"Corrected NACA 0012 design vector:")
print(f"  np.array([{symmetric_design[0]:.6f}, {symmetric_design[1]:.6f}, {symmetric_design[2]:.6f}, {symmetric_design[3]:.6f},")
print(f"            {symmetric_design[4]:.6f}, {symmetric_design[5]:.6f}, {symmetric_design[6]:.6f}, {symmetric_design[7]:.6f},")
print(f"            {symmetric_design[8]}, {symmetric_design[9]}])")