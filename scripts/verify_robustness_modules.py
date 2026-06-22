#!/usr/bin/env python3
"""
Verification tests for preflight, smoke_test, and optimizer fault tolerance.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

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


print("=" * 60)
print("ROBUSTNESS MODULES VERIFICATION")
print("=" * 60)

# ── 1. Preflight module ──────────────────────────────────────────────────
print("\n=== Preflight Module ===")
from airfoil_discovery.aso.preflight import (
    PreflightReport, run_preflight_checks, check_su2_binary, check_su2_mesh,
    check_output_directory, check_baseline_design,
)
check("preflight imports", True)

# Baseline design check
from airfoil_discovery.aso.cst import CSTBounds
dv = np.array([0.18, 0.28, 0.34, 0.25, 0.15, 0.08,
               -0.19, -0.12, -0.09, -0.05, -0.02, -0.01])
ok, info = check_baseline_design(dv, bounds=CSTBounds.default())
check("baseline design valid", ok, detail=info)

# Output directory check
with tempfile.TemporaryDirectory() as tmp:
    ok, free_gb, msg = check_output_directory(Path(tmp))
    check("output directory writable", ok, detail=msg)

# Mesh check - nonexistent
ok_mesh, info_mesh = check_su2_mesh(Path("/nonexistent/file.su2"))
check("mesh not-found detected", not ok_mesh)

# Binary check - nonexistent
ok_bin, info_bin = check_su2_binary("/nonexistent/su2_cfd", "SU2_CFD")
check("binary not-found detected", not ok_bin)

# PreflightReport dataclass
r = PreflightReport()
check("preflight report defaults", not r.all_checks_passed)
r.all_checks_passed = True
check("preflight report modified", r.all_checks_passed)

# ── 2. Smoke Test Module ─────────────────────────────────────────────────
print("\n=== Smoke Test Module ===")
from airfoil_discovery.aso.smoke_test import (
    SmokeTestOverrides, smoke_test_message, apply_smoke_overrides,
    get_smoke_overrides, is_smoke_mode,
)
check("smoke_test imports", True)

# Default overrides
o = get_smoke_overrides()
check("smoke override n_iter_primal", o.n_iter_primal == 20)
check("smoke override n_iter_adjoint", o.n_iter_adjoint == 10)
check("smoke override max_iter", o.max_iterations == 2)
check("smoke override move_limit", o.move_limit == 0.01)

# Apply overrides to kwargs dict
kwargs = {
    "n_iter_primal": 3000, "n_iter_adjoint": 500,
    "max_iterations": 50, "convergence_tolerance": 1e-4,
    "cfl_primal": 3.0, "cfl_adjoint": 1.0,
}
result = apply_smoke_overrides(kwargs)
check("override primal iters", result["n_iter_primal"] == 20)
check("override adjoint iters", result["n_iter_adjoint"] == 10)
check("override max iter", result["max_iterations"] == 2)
check("override cfl_primal", result["cfl_primal"] == 5.0)

# Smoke test message
msg = smoke_test_message()
check("smoke test message format", "SMOKE TEST MODE" in msg)

# to_dict
s = SmokeTestOverrides()
d = s.to_dict()
check("to_dict has primal_iters", d["primal_iters"] == 20)
check("to_dict has adjoint_iters", d["adjoint_iters"] == 10)

# is_smoke_mode
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--smoke-test", action="store_true")
args_no = parser.parse_args([])
check("no smoke mode", not is_smoke_mode(args_no))
args_yes = parser.parse_args(["--smoke-test"])
check("smoke mode detected", is_smoke_mode(args_yes))

# ── 3. Optimizer fault tolerance ────────────────────────────────────────
print("\n=== Optimizer Fault Tolerance ===")
from airfoil_discovery.aso.optimizer import (
    setup_signal_handlers, shutdown_requested, update_emergency_state,
    _emergency_state,
)
check("optimizer fault tolerance imports", True)

# Emergency state update
update_emergency_state(current_dv=dv, iteration=5)
check("emergency state updated", _emergency_state["current_dv"] is not None)
check("emergency state iteration", _emergency_state["iteration"] == 5)

# shutdown check
check("no shutdown requested", not shutdown_requested())

# setup_signal_handlers (just verify it doesn't crash)
setup_signal_handlers()
check("signal handlers installed", True)

# ── 4. Entry point script ─────────────────────────────────────────────
print("\n=== Entry Point Script ===")
import importlib.util
spec = importlib.util.spec_from_file_location(
    "aso_script",
    PROJECT_ROOT / "scripts" / "run_aso_pde_optimization.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
check("entry point script loads", hasattr(mod, "main"))
check("entry point has parse_args", hasattr(mod, "parse_args"))

# ── 5. Integration: Full import from aso package ─────────────────────
print("\n=== Full Package Import ===")
from airfoil_discovery.aso import (
    # Preflight
    PreflightReport, run_preflight_checks,
    check_su2_binary, check_su2_mesh,
    check_output_directory, check_baseline_design,
    # Smoke test
    SmokeTestOverrides, get_smoke_overrides,
    is_smoke_mode, apply_smoke_overrides, smoke_test_message,
    # Optimizer
    setup_signal_handlers, shutdown_requested, update_emergency_state,
)
check("all robustness symbols exported", True)

# ── Results ────────────────────────────────────────────────────────────
print(f"\n{'=' * 50}")
print(f"Results: {n_pass}/{n_total} tests passed")
print(f"{'=' * 50}")
sys.exit(0 if n_pass == n_total else 1)