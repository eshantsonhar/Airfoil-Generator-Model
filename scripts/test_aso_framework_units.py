#!/usr/bin/env python3
"""
Unit tests for the PDE-Constrained ASO Framework.
Tests all components except actual SU2 execution (requires binaries).
"""

import sys
import json
import tempfile
from pathlib import Path

import numpy as np

# Add project src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from airfoil_discovery.aso import (
    N_DESIGN_VARS,
    CSTBounds,
    compute_airfoil_coordinates,
    check_geometry_validity,
    design_vector_to_surface_coefficients,
    surface_coefficients_to_design_vector,
    project_surface_sensitivity_to_cst,
)
from airfoil_discovery.aso.config_primal import generate_primal_config
from airfoil_discovery.aso.config_adjoint import generate_adjoint_config
from airfoil_discovery.aso.adjoint import detect_upper_lower_split, verify_adjoint_gradient
from airfoil_discovery.aso.mesh_deform import (
    generate_su2_def_config,
    compute_mesh_displacement,
    write_airfoil_dat,
)
from airfoil_discovery.aso.optimizer import ConvergenceHistory, IterationRecord, CFDResult

n_pass = 0
n_total = 0


def check(name: str, condition: bool, detail: str = ""):
    global n_pass, n_total
    n_total += 1
    if condition:
        n_pass += 1
        print(f"  [PASS] {name}")
    else:
        msg = f"  [FAIL] {name}"
        if detail:
            msg += f" ({detail})"
        print(msg)


# ── 1. CST Bounds ──────────────────────────────────────────────────────────
print("\n=== CST: Bounds ===")
b = CSTBounds.default()
check("CSTOrder", CSTBounds.default().upper_min.shape == (6,))
check("UpperBounds", float(b.upper_min[0]) == 0.0 and float(b.upper_max[0]) == 0.5)
check("LowerBounds", float(b.lower_min[0]) == -0.4 and float(b.lower_max[0]) == 0.1)
check("ThicknessBounds", b.min_thickness == 0.06 and b.max_thickness == 0.18)

# ── 2. CST: Design Vector Split ────────────────────────────────────────────
print("\n=== CST: Design Vector Split ===")
dv = np.array([0.18, 0.28, 0.34, 0.25, 0.15, 0.08, -0.19, -0.12, -0.09, -0.05, -0.02, -0.01])
upper, lower = design_vector_to_surface_coefficients(dv)
check("UpperSplit", len(upper) == 6)
check("LowerSplit", len(lower) == 6)
dv_roundtrip = surface_coefficients_to_design_vector(upper, lower)
check("Roundtrip", np.allclose(dv, dv_roundtrip))

# ── 3. CST: Airfoil Generation ─────────────────────────────────────────────
print("\n=== CST: Airfoil Generation ===")
coords = compute_airfoil_coordinates(dv, bounds=b)
check("NumPoints", len(coords) >= 100)
is_valid, reason = check_geometry_validity(dv, bounds=b)
check("ValidGeometry", is_valid, detail=reason)

# Test invalid geometry (swap upper/lower signs)
dv_invalid = np.array([-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
valid_inv, reason_inv = check_geometry_validity(dv_invalid, bounds=b)
check("InvalidDetected", not valid_inv, detail=reason_inv)

# ── 4. Primal Config ──────────────────────────────────────────────────────
print("\n=== Primal Config Generation ===")
cfg = generate_primal_config(mesh_filename="mesh.su2", aoa_deg=4.0, reynolds=1e5)
check("Solver", "SOLVER= INC_RANS" in cfg)
check("SST", "KIND_TURB_MODEL= SST" in cfg)
check("Transition", "KIND_TRANS_MODEL= LM" in cfg)
check("MUSCL", "MUSCL_FLOW= YES" in cfg)
check("Limiter", "VENKATAKRISHNAN_WANG" in cfg)
check("Roe", "CONV_NUM_METHOD_FLOW= FDS" in cfg)
check("CFL", "CFL_NUMBER= " in cfg)
check("AoA", "AOA= 4.0" in cfg)
check("Re", "REYNOLDS_NUMBER= 100000.0" in cfg)
check("TurbulenceIntensity", "FREESTREAM_TURBULENCEINTENSITY= 0.001" in cfg)

# Without transition
cfg_no_trans = generate_primal_config(mesh_filename="mesh.su2", aoa_deg=4.0, reynolds=1e5, transition_model=False)
check("NoTransition", "KIND_TRANS_MODEL= NONE" in cfg_no_trans)

# ── 5. Adjoint Config ─────────────────────────────────────────────────────
print("\n=== Adjoint Config Generation ===")
with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False, encoding="utf-8") as f:
    f.write("SOLVER= INC_RANS\nAOA= 4.0\n")
    primal_path = Path(f.name)

adj_cfg = generate_adjoint_config(
    mesh_filename="mesh.su2",
    primal_config_filename=str(primal_path),
    objective="DRAG",
)
check("AdjSolver", "MATH_PROBLEM= DISCRETE_ADJOINT" in adj_cfg)
check("Objective", "OBJECTIVE_FUNCTION= DRAG" in adj_cfg)
check("SurfaceDV", "MARKER_MONITORING= ( airfoil )" in adj_cfg)
check("AdjCFL", "CFL_ADAPT= NO" in adj_cfg)
check("AdjOutput", "SURFACE_FILENAME= surface_adjoint" in adj_cfg)
primal_path.unlink()

# ── 6. Mesh Deformation Config ────────────────────────────────────────────
print("\n=== Mesh Deformation Config ===")
def_cfg = generate_su2_def_config(mesh_input="mesh.su2", mesh_output="mesh_def.su2")
check("DefSolver", "SOLVER= ELASTICITY" in def_cfg)
check("Elasticity", "MATH_PROBLEM= DIRECT" in def_cfg)
check("Stiffness", "DEFORM_STIFFNESS_TYPE= INVERSE_VOLUME" in def_cfg)
check("DefIter", "DEFORM_NONLINEAR_ITER= 500" in def_cfg)

# ── 7. Surface Sensitivity Split ──────────────────────────────────────────
print("\n=== Surface Split Detection ===")
# Simulate SU2 surface ordering: TE upper -> LE -> TE lower
n_upper = 50
n_lower = 50
x_surf = np.concatenate([
    np.linspace(1.0, 0.0, n_upper),  # upper: TE to LE
    np.linspace(0.0, 1.0, n_lower),  # lower: LE to TE
])
y_surf = np.concatenate([
    0.1 * np.sin(np.linspace(0, np.pi, n_upper)),      # upper positive
    -0.05 * np.sin(np.linspace(0, np.pi, n_lower)),     # lower negative
])
up_idx, low_idx = detect_upper_lower_split(x_surf, y_surf)
check("UpperCount", np.sum(up_idx) >= n_upper - 1)
check("LowerCount", np.sum(low_idx) >= n_lower - 1)
check("Separation", np.sum(up_idx & low_idx) == 0)  # no overlap
check("MeanUpperAboveZero", np.mean(y_surf[up_idx]) > 0)
check("MeanLowerBelowZero", np.mean(y_surf[low_idx]) < 0)


# ── 8. Sensitivity Projection ─────────────────────────────────────────────
print("\n=== Sensitivity Projection ===")
dJ_dx = np.random.randn(len(x_surf)) * 0.01
dJ_dy = np.concatenate([
    np.random.randn(n_upper) * 0.01,
    np.random.randn(n_lower) * 0.01,
])
grad = project_surface_sensitivity_to_cst(dJ_dx, dJ_dy, x_surf, up_idx, low_idx)
check("GradShape", grad.shape == (12,))
check("GradNotNaN", not np.any(np.isnan(grad)))

# ── 9. Gradient Verification ──────────────────────────────────────────────
print("\n=== Gradient Verification ===")
g = np.array([0.1, 0.05, 0.03, 0.01, -0.02, -0.01, -0.05, -0.03, 0.02, 0.01, 0.005, 0.001])
report = verify_adjoint_gradient(g)
check("GradValid", report["is_valid"])
check("GradNormPositive", report["adjoint_norm"] > 0)

# FD verification
g_fd = g + np.random.randn(12) * 0.01
report_fd = verify_adjoint_gradient(g, grad_fd=g_fd)
check("FDValid", report_fd["is_valid"] or len(report_fd["warnings"]) > 0)

# ── 10. Mesh Displacement ─────────────────────────────────────────────────
print("\n=== Mesh Displacement ===")
dv2 = dv + 0.01
disp = compute_mesh_displacement(dv, dv2)
check("DisplacementPositive", disp > 0)
check("DisplacementReasonable", disp < 1.0)

# Larger perturbation -> larger displacement
dv3 = dv + 0.05
disp_large = compute_mesh_displacement(dv, dv3)
check("LargerDisplacement", disp_large > disp)

# ── 11. Convergence History ───────────────────────────────────────────────
print("\n=== Convergence History ===")
hist = ConvergenceHistory()
r1 = IterationRecord(
    iteration=1, cd=0.025, cl=0.6, objective=0.025, grad_norm=0.1,
    step_accepted=True, trust_radius=0.1, max_thickness=0.12,
    design_vector=[0.1] * 12,
)
r2 = IterationRecord(
    iteration=2, cd=0.020, cl=0.65, objective=0.020, grad_norm=0.05,
    step_accepted=True, trust_radius=0.15, max_thickness=0.115,
    design_vector=[0.1] * 12,
)
hist.add(r1)
hist.add(r2)
hist.finalize(converged=True)
check("HistoryCount", hist.total_iterations == 2)
check("CdHistory", hist.cd_history == [0.025, 0.020])
check("GradNormHistory", hist.grad_norm_history == [0.1, 0.05])
check("ClHistory", hist.cl_history == [0.6, 0.65])

# JSON serialization
with tempfile.TemporaryDirectory() as tmp:
    hist_path = Path(tmp) / "history.json"
    hist.save(hist_path)
    loaded = json.loads(hist_path.read_text())
    check("JSON_Iterations", len(loaded["iterations"]) == 2)
    check("JSON_Converged", loaded["converged"])
    check("JSON_DVs", loaded["n_design_vars"] == N_DESIGN_VARS)

# ── 12. CFD Result ────────────────────────────────────────────────────────
print("\n=== CFD Result ===")
res = CFDResult(
    cl=0.6, cd=0.025, converged=True,
    adjoint_gradient=np.ones(12), gradient_valid=True,
    primal_converged=True, adjoint_converged=True,
)
check("CFD_Values", abs(res.cl - 0.6) < 1e-10 and abs(res.cd - 0.025) < 1e-10)
check("CFD_Converged", res.converged)
check("CFD_GradShape", res.adjoint_gradient.shape == (12,))

# ── 13. Write DAT File ────────────────────────────────────────────────────
print("\n=== Write DAT file ===")
with tempfile.TemporaryDirectory() as tmp:
    dat_path = Path(tmp) / "airfoil.dat"
    write_airfoil_dat(dv, dat_path)
    content = dat_path.read_text()
    lines = content.strip().split("\n")
    check("DAT_HasHeader", lines[0] == "airfoil")
    check("DAT_HasPoints", len(lines) >= 100)

print(f"\n{'='*50}")
print(f"Results: {n_pass}/{n_total} tests passed")
print(f"{'='*50}")
sys.exit(0 if n_pass == n_total else 1)
