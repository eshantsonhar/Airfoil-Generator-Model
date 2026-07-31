#!/usr/bin/env python3
"""
Test the patched config_primal.py by generating a config and comparing to baseline.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from airfoil_discovery.aso.config_primal import generate_primal_config

# Generate config with default parameters (should match baseline_verification)
config_text = generate_primal_config(
    mesh_filename="airfoil.su2",
    aoa_deg=4.0,
    reynolds=1e5,
    mach=0.1,
    n_iter=200,
    cfl_initial=0.5,
    cfl_final=3.0,
    cfl_adapt=True,
)

# Write to test file
test_config_path = PROJECT_ROOT / "test_generated_config.cfg"
test_config_path.write_text(config_text)

print(f"Generated config saved to: {test_config_path}")

# Load baseline config for comparison
baseline_config_path = PROJECT_ROOT / "phase5_output" / "baseline_verification" / "config_primal.cfg"
baseline_text = baseline_config_path.read_text()

# Compare key lines
print("\n=== COMPARISON ===")
test_lines = config_text.split('\n')
baseline_lines = baseline_text.split('\n')

# Check critical parameters
critical_params = [
    "MU_CONSTANT",
    "INC_VELOCITY_INIT",
    "CFL_NUMBER",
    "CFL_ADAPT",
    "CFL_ADAPT_PARAM",
    "ITER",
    "LINEAR_SOLVER_ERROR",
    "LINEAR_SOLVER_ITER",
    "CONV_CAUCHY_EPS",
]

print("\nCritical parameters comparison:")
for param in critical_params:
    test_line = [l for l in test_lines if param in l]
    baseline_line = [l for l in baseline_lines if param in l]
    if test_line and baseline_line:
        match = "✓" if test_line[0] == baseline_line[0] else "✗"
        print(f"{match} {param}:")
        print(f"  Generated: {test_line[0]}")
        print(f"  Baseline:  {baseline_line[0]}")
    else:
        print(f"✗ {param}: Missing in one or both configs")

print("\n=== TEST COMPLETE ===")
