"""
Test single iteration with continuous adjoint to verify rc=0 and non-zero gradient.
"""
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from airfoil_discovery.aso.optimizer import run_primal_and_adjoint
from airfoil_discovery.aso.cst import N_DESIGN_VARS, compute_airfoil_coordinates

# Use baseline design vector (NACA 0012-like)
baseline_dv = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Upper surface coefficients
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Lower surface coefficients
])

# Paths
workspace = PROJECT_ROOT
mesh_path = workspace / "data" / "mesh_fixed.su2"
case_dir = workspace / "test_adjoint_iter"
case_dir.mkdir(exist_ok=True)

# SU2 binary
su2_cfd_bin = str(workspace / "bin" / "SU2_CFD.exe")

print("=" * 80)
print("TESTING SINGLE ITERATION WITH CONTINUOUS ADJOINT")
print("=" * 80)
print(f"Mesh: {mesh_path}")
print(f"Case dir: {case_dir}")
print(f"SU2 binary: {su2_cfd_bin}")
print(f"Design vector: {baseline_dv}")
print()

# Run primal + adjoint
result = run_primal_and_adjoint(
    su2_cfd_bin=su2_cfd_bin,
    su2_adj_bin=su2_cfd_bin,
    mesh_path=mesh_path,
    dv=baseline_dv,
    case_dir=case_dir,
    aoa_deg=4.0,
    reynolds=1e5,
    mach=0.1,
    n_iter_primal=1000,  # More iterations for convergence
    n_iter_adjoint=200,
    cfl_primal=1.0,
    cfl_adjoint=0.5,
    transition_model=False,  # SST-only for stability
    turbulence_intensity=0.05,
    turb_viscosity_ratio=10.0,
    objective="DRAG",
    timeout_primal=3600,
    timeout_adjoint=600,
    use_adjoint=True,
)

print("=" * 80)
print("RESULTS")
print("=" * 80)
print(f"Primal converged: {result.primal_converged}")
print(f"Adjoint converged: {result.adjoint_converged}")
print(f"Overall converged: {result.converged}")
print(f"CL: {result.cl:.6f}")
print(f"CD: {result.cd:.6f}")
print(f"Gradient valid: {result.gradient_valid}")
if result.adjoint_gradient is not None:
    print(f"Gradient norm: {np.linalg.norm(result.adjoint_gradient):.6e}")
    print(f"Gradient: {result.adjoint_gradient}")
else:
    print(f"Gradient norm: N/A (gradient is None)")
    print(f"Gradient: None")
print(f"Failure reason: {result.failure_reason}")
print("=" * 80)

# Check acceptance criteria
if result.primal_converged and result.adjoint_converged:
    if result.gradient_valid and result.adjoint_gradient is not None and np.linalg.norm(result.adjoint_gradient) > 1e-12:
        print("✅ SUCCESS: Adjoint converged with non-zero gradient")
        sys.exit(0)
    else:
        print("❌ FAIL: Gradient is invalid or zero")
        sys.exit(1)
else:
    print("❌ FAIL: Primal or adjoint did not converge")
    sys.exit(1)
