#!/usr/bin/env python3
"""
Quick integration verification: ensures the ASO framework
loads cleanly and coexists with the existing pipeline modules.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

print("1. Importing ASO framework...")
from airfoil_discovery.aso import (
    PDEOptimizer, CSTBounds, N_DESIGN_VARS, CST_ORDER,
    compute_airfoil_coordinates, check_geometry_validity,
    generate_primal_config, generate_adjoint_config,
)
print("   OK")

print("2. Importing existing pipeline modules...")
from airfoil_discovery.optimization.mma_engine import SvanbergMMA
from airfoil_discovery.geometry.cst import CSTAirfoil
from airfoil_discovery.cfd.su2_config import build_stage_config
from airfoil_discovery.config import Settings
print("   OK")

print("3. Verifying ASO entry point script...")
import importlib.util
spec = importlib.util.spec_from_file_location(
    "aso_script",
    PROJECT_ROOT / "scripts" / "run_aso_pde_optimization.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print(f"   OK - has main={hasattr(mod, 'main')}")

print("4. Design variable consistency...")
print(f"   CST_ORDER={CST_ORDER}, N_DESIGN_VARS={N_DESIGN_VARS}")
assert CST_ORDER == 6
assert N_DESIGN_VARS == 12

print("5. Quick CST geometry test...")
import numpy as np
dv = np.array([0.18, 0.28, 0.34, 0.25, 0.15, 0.08,
               -0.19, -0.12, -0.09, -0.05, -0.02, -0.01])
coords = compute_airfoil_coordinates(dv)
valid, reason = check_geometry_validity(dv)
print(f"   Points={len(coords)}, valid={valid}")
assert len(coords) >= 100
assert valid

print("6. Config generation...")
import tempfile
cfg = generate_primal_config("mesh.su2", aoa_deg=4.0, reynolds=1e5)
assert "KIND_TURB_MODEL= SST" in cfg
assert "SOLVER= INC_RANS" in cfg

# Write a dummy primal config for adjoint generation
with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False, encoding="utf-8") as f:
    f.write("SOLVER= INC_RANS\n")
    tmp_path = f.name

adj_cfg = generate_adjoint_config("mesh.su2", primal_config_filename=tmp_path)
assert "MATH_PROBLEM= DISCRETE_ADJOINT" in adj_cfg
Path(tmp_path).unlink()
print("   OK")

print("\n=== INTEGRATION VERIFIED SUCCESSFULLY ===")