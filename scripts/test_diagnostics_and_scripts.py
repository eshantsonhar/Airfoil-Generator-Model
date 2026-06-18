#!/usr/bin/env python3
"""
Integration tests for diagnostics module and new scripts.
Tests all components except SU2 execution (requires binaries).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import json
import tempfile

print("=" * 60)
print("DIAGNOSTICS MODULE & SCRIPTS VERIFICATION")
print("=" * 60)

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


# ── 1. Diagnostics imports ────────────────────────────────────────────────
print("\n=== Diagnostics Module Imports ===")
from airfoil_discovery.aso.diagnostics import (
    SurfaceFlowData, LSBResult, AerodynamicMetrics,
    parse_surface_flow, extract_lsb_from_cf, extract_lsb_from_cp,
    compute_aerodynamic_metrics, compare_baseline_optimized,
)
check("Module loaded", True)

# ── 2. LSB extraction from Cf ────────────────────────────────────────────
print("\n=== LSB Extraction from Cf ===")
n = 100
x = np.linspace(0, 1, n)

# No LSB (fully attached)
cf_attached = 0.01 * (1 - x * 0.5)
lsb_none = extract_lsb_from_cf(x, cf_attached)
check("NoLSB_detected", not lsb_none.lsb_detected)
check("NoLSB_no_separation", lsb_none.separation_point is None)

# Clear LSB: separation at x=0.3, reattachment at x=0.55
cf_lsb = np.zeros_like(x)
for i in range(n):
    if x[i] < 0.3:
        cf_lsb[i] = 0.005 * (1 - x[i] * 2)  # positive, decreasing
    elif x[i] < 0.55:
        cf_lsb[i] = -0.003 * np.sin((x[i] - 0.3) / 0.25 * np.pi)  # negative
    else:
        cf_lsb[i] = 0.004 * (x[i] - 0.55) / 0.45  # positive recovering

lsb_yes = extract_lsb_from_cf(x, cf_lsb)
check("LSB_detected", lsb_yes.lsb_detected)
check("LSB_separation", lsb_yes.separation_point is not None and lsb_yes.separation_point > 0.25)
check("LSB_reattachment", lsb_yes.reattachment_point is not None and lsb_yes.reattachment_point < 0.6)
if lsb_yes.bubble_length is not None:
    check("LSB_length_positive", lsb_yes.bubble_length > 0)

# Fully separated (Cf always negative)
cf_sep = -0.005 * np.ones_like(x)
lsb_sep = extract_lsb_from_cf(x, cf_sep)
check("FullySeparated_no_LSB", not lsb_sep.lsb_detected)
check("FullySeparated_has_warnings", len(lsb_sep.warnings) > 0)

# ── 3. LSB extraction from Cp ────────────────────────────────────────────
print("\n=== LSB Extraction from Cp ===")
x_short = np.linspace(0, 1, 50)

# No plateau (smooth Cp distribution)
cp_smooth = -2.0 * (1 - x_short) ** 2 + 0.5
lsb_cp_none = extract_lsb_from_cp(x_short, cp_smooth, min_plateau_length=0.01)
check("Cp_NoLSB", not lsb_cp_none.lsb_detected)

# Plateau (constant Cp region indicating LSB)
cp_plateau = np.zeros_like(x_short)
for i in range(50):
    if x_short[i] < 0.2:
        cp_plateau[i] = -3.0 * (x_short[i] / 0.2)
    elif x_short[i] < 0.5:
        cp_plateau[i] = -3.0  # plateau
    else:
        cp_plateau[i] = -3.0 + 5.0 * ((x_short[i] - 0.5) / 0.5)  # recovery

lsb_cp_yes = extract_lsb_from_cp(x_short, cp_plateau, min_plateau_length=0.02)
check("Cp_LSB_detected", lsb_cp_yes.lsb_detected)
check("Cp_plateau_exists", lsb_cp_yes.plateau_start is not None)

# ── 4. AerodynamicMetrics ────────────────────────────────────────────────
print("\n=== Aerodynamic Metrics ===")
metrics = AerodynamicMetrics(cl=0.6, cd=0.025, cm=-0.05)
check("Efficiency", abs(metrics.efficiency - 24.0) < 0.01)
check("CL_valid", abs(metrics.cl - 0.6) < 1e-10)
check("CD_valid", abs(metrics.cd - 0.025) < 1e-10)

opt = AerodynamicMetrics(cl=0.65, cd=0.020, cm=-0.04)
check("Opt_Efficiency", abs(opt.efficiency - 32.5) < 0.01)

comp = compare_baseline_optimized(metrics, opt)
check("Comparison_has_deltas", "delta_cd" in comp)
check("Cd_reduction_positive", comp["cd_reduction_percent"] > 0)
check("Cl_improvement_positive", comp["cl_improvement_percent"] > 0)

# ── 5. SurfaceFlowData Construction ──────────────────────────────────────
print("\n=== SurfaceFlowData ===")
x = np.concatenate([np.linspace(1, 0, 50), np.linspace(0, 1, 50)])
y = np.concatenate([0.1 * np.sin(np.linspace(0, np.pi, 50)), -0.05 * np.sin(np.linspace(0, np.pi, 50))])

data = SurfaceFlowData(
    x=x, y=y, cp=np.zeros(100), cf=np.zeros(100),
    pressure=np.zeros(100), density=np.ones(100),
    velocity=np.ones(100), mach=np.zeros(100), temperature=np.ones(100) * 300,
    n_nodes=100,
)
data._detect_upper_lower_split()
check("Split_detected", data.has_upper_lower_split)
check("Upper_points", np.sum(data.upper_indices) > 0)
check("Lower_points", np.sum(data.lower_indices) > 0)

if data.has_upper_lower_split:
    check("Upper_y_positive", np.mean(data.y_upper) > 0)
    check("Lower_y_negative", np.mean(data.y_lower) < 0)

# ── 6. parse_aero_from_history ────────────────────────────────────────────
print("\n=== History Parsing ===")
from airfoil_discovery.aso.diagnostics import _parse_aero_from_history

with tempfile.TemporaryDirectory() as tmp:
    hist_path = Path(tmp) / "history.csv"
    hist_path.write_text(
        '"ITER","CL","CD","CMz"\n'
        '1,0.500,0.030,-0.050\n'
        '2,0.550,0.028,-0.048\n'
        '3,0.600,0.025,-0.045\n'
    )
    cl, cd, cm = _parse_aero_from_history(hist_path)
    check("History_CL", abs(cl - 0.6) < 1e-10)
    check("History_CD", abs(cd - 0.025) < 1e-10)
    check("History_CM", abs(cm + 0.045) < 1e-10)

# ── 7. Script imports (no execution) ──────────────────────────────────────
print("\n=== Script Imports ===")
# verify_adjoint_gradients uses same imports as existing framework
from airfoil_discovery.aso.optimizer import run_primal_and_adjoint
check("run_primal_and_adjoint_imported", True)

# plot_optimization_diagnostics uses parse_surface_flow and extract_lsb_from_cf
# already verified above

# run_production_sweeps uses PDEOptimizer
from airfoil_discovery.aso import PDEOptimizer, ConvergenceHistory
check("PDEOptimizer_imported", True)

# ── 8. Compare with 12 DV interface ──────────────────────────────────────
print("\n=== 12-DV Interface ===")
from airfoil_discovery.aso import N_DESIGN_VARS, CST_ORDER
check("12_DV", N_DESIGN_VARS == 12)
check("6th_order", CST_ORDER == 6)

dv = np.array([0.18, 0.28, 0.34, 0.25, 0.15, 0.08,
               -0.19, -0.12, -0.09, -0.05, -0.02, -0.01])
from airfoil_discovery.aso import compute_airfoil_coordinates, check_geometry_validity
coords = compute_airfoil_coordinates(dv)
valid, _ = check_geometry_validity(dv)
check("CST_airfoil_generated", len(coords) > 100 and valid)

# ── 9. Config generation ─────────────────────────────────────────────────
print("\n=== Config Generation ===")
from airfoil_discovery.aso import generate_primal_config, generate_adjoint_config
cfg = generate_primal_config("mesh.su2", aoa_deg=4.0, reynolds=1e5)
check("Primal_has_SST", "KIND_TURB_MODEL= SST" in cfg)
check("Primal_has_LM", "KIND_TRANS_MODEL= LM" in cfg)
check("Primal_has_MUSCL", "MUSCL_FLOW= YES" in cfg)
check("Primal_has_Roe", "CONV_NUM_METHOD_FLOW= FDS" in cfg)

# ── Results ──────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Results: {n_pass}/{n_total} tests passed")
print(f"{'='*50}")
sys.exit(0 if n_pass == n_total else 1)