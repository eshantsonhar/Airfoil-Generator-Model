import sys
sys.path.insert(0, ".")
import numpy as np
from pathlib import Path
from airfoil_discovery.verification.convergence import (
    ResidualConvergenceAnalyzer, IterativeConvergenceMonitor
)

h = Path('data/failures/iter_001_aoa_+02p0/history.csv')
text = h.read_text(encoding='utf-8')
lines = text.splitlines()
headers = [it.strip().strip('"') for it in lines[0].split(',')]
traces = {h2: [] for h2 in headers}
for line in lines[1:]:
    if not line.strip() or line.strip() == ',': continue
    lvs = [it.strip() for it in line.split(',')]
    for i, h2 in enumerate(headers):
        if i < len(lvs):
            try: traces[h2].append(float(lvs[i]))
            except: pass

residual_history = (
    traces.get("rms[P]") or traces.get("RMS_PRESSURE") or
    traces.get("rms[Rho]") or traces.get("RMS_DENSITY") or
    traces.get("RES_RHO") or traces.get("RES_RMS") or []
)
cl_history = traces.get("CL") or []
cd_history = traces.get("CD") or []
print(f"residuals: {len(residual_history)}, CL: {len(cl_history)}, CD: {len(cd_history)}")

# Fix 1: check convergence with analyzer
analyzer = ResidualConvergenceAnalyzer(residual_threshold=1e-4, stagnation_threshold=1e-3,
                                       stagnation_iterations=30, min_iterations=50)
metrics = analyzer.analyze(residual_history)
residual_converged = metrics.below_threshold
print(f"\nFix1: residual converged: {residual_converged}, final_residual_abs: {abs(metrics.final_residual):.4e}")

# Fix 2: check forces stabilization with improved mean-vs-mean comparison
fm = IterativeConvergenceMonitor(
    force_stabilization_threshold=0.005,
    force_oscillation_threshold=0.01,
    force_drift_threshold=0.002,
    stabilization_window=30
)
f_metrics = fm.analyze_forces(cl_history, cd_history)
forces_stabilized = f_metrics.forces_stabilized
print(f"\nFix2: forces_stabilized: {forces_stabilized}")
print(f"  final_cl={f_metrics.final_cl:.6f}, final_cd={f_metrics.final_cd:.6f}")
print(f"  force_oscillation_acceptable: {f_metrics.force_oscillation_acceptable}")
print(f"  force_drift_acceptable: {f_metrics.force_drift_acceptable}")
print(f"  cl_relative_oscillation: {f_metrics.cl_relative_oscillation:.4f}")
print(f"  cd_relative_oscillation: {f_metrics.cd_relative_oscillation:.4f}")
print(f"  cl_trend: {f_metrics.cl_trend:.6f}, cd_trend: {f_metrics.cd_trend:.6f}")

# is_valid
is_valid = residual_converged and forces_stabilized
print(f"\nis_valid (residual AND forces): {is_valid}")
