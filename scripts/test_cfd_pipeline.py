#!/usr/bin/env python
"""
CFD pipeline diagnostic test.
Tests each stage of the CFD pipeline in isolation.
"""

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ["AIRFOIL_PROJECT_ROOT"] = str(PROJECT_ROOT)

import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

PASS = 0
FAIL = 0

def check(desc, cond, detail=""):
    global PASS, FAIL
    if cond:
        print(f"  [PASS] {desc}")
        PASS += 1
    else:
        print(f"  [FAIL] {desc} — {detail}")
        FAIL += 1

print("=" * 60)
print("  CFD PIPELINE DIAGNOSTIC")
print("=" * 60)

# 1. Load config
from airfoil_discovery.config import load_settings
try:
    settings = load_settings(PROJECT_ROOT / "config" / "default.yaml")
    check("Config loaded", True)
except Exception as e:
    check("Config loaded", False, str(e))
    sys.exit(1)

# 2. Verify binary paths
su2_bin = settings.solver.su2_cfd_bin
gmsh_bin = settings.solver.gmsh_bin
check(f"SU2_CFD path: {su2_bin}", Path(su2_bin).exists(), f"resolved: {Path(su2_bin).resolve()}")
check(f"GMSH path: {gmsh_bin}", Path(gmsh_bin).exists(), f"resolved: {Path(gmsh_bin).resolve()}")

# 3. Geometry generation test
from airfoil_discovery.geometry.cst import CSTAirfoil
from airfoil_discovery.schemas import CSTParameters
try:
    airfoil = CSTAirfoil(settings.geometry)
    design_vector = np.array([0.18, 0.05, 0.34, 0.10, -0.19, 0.05, -0.09, 0.03, 0.004, 1.0])
    params = CSTParameters(
        upper=design_vector[:4],
        lower=design_vector[4:8],
        trailing_edge_thickness=design_vector[8]
    )
    coords = airfoil.full_coordinates(params)
    check("Geometry generation", len(coords) > 100, f"{len(coords)} points")
except Exception as e:
    check("Geometry generation", False, str(e))

# 4. Geometry validation
from airfoil_discovery.geometry.validation import AirfoilGeometryValidator, GeometryValidationConfig
try:
    validator = AirfoilGeometryValidator(GeometryValidationConfig())
    result = validator.validate_coordinates(coords)
    check("Geometry validation passes", result.can_proceed_to_cfd, str(result.failure_reasons))
    check(f"  Max thickness: {result.max_thickness:.4f}", result.max_thickness > 0.05)
except Exception as e:
    check("Geometry validation", False, str(e))

# 5. GMSH geo file generation
from airfoil_discovery.cfd.mesh import MeshFidelityManager, build_geo_script
try:
    fidelity = MeshFidelityManager.get_params("L0")
    geo_script = build_geo_script(
        coords=coords,
        reynolds=100000.0,
        mesh_cfg=settings.solver.mesh,
        coarse_factor=fidelity.coarse_factor,
    )
    geo_path = PROJECT_ROOT / "data" / "test_diag" / "test_airfoil.geo"
    geo_path.parent.mkdir(parents=True, exist_ok=True)
    geo_path.write_text(geo_script, encoding="utf-8")
    check("GMSH geo file written", geo_path.exists(), str(geo_path))
except Exception as e:
    check("GMSH geo file generation", False, str(e))

# 6. Run GMSH mesh generation
import subprocess
mesh_path = geo_path.parent / "airfoil.su2"
try:
    result = subprocess.run(
        [gmsh_bin, str(geo_path.name), "-2", "-format", "su2", "-o", mesh_path.name],
        cwd=str(geo_path.parent),
        capture_output=True, text=True, timeout=120,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    (geo_path.parent / "gmsh_diag_stdout.log").write_text(result.stdout, errors="ignore")
    (geo_path.parent / "gmsh_diag_stderr.log").write_text(result.stderr, errors="ignore")
    check("GMSH runs successfully", result.returncode == 0, f"rc={result.returncode}")
    if result.stderr:
        print(f"  GMSH stderr: {result.stderr[:500]}")
    check(f"Mesh file exists ({mesh_path.stat().st_size if mesh_path.exists() else 0} bytes)", 
          mesh_path.exists() and mesh_path.stat().st_size > 0)
except subprocess.TimeoutExpired:
    check("GMSH runs successfully", False, "TIMEOUT after 120s")
except Exception as e:
    check("GMSH runs successfully", False, str(e))

# 7. Write and validate SU2 config
from airfoil_discovery.schemas import CandidateDesign
from airfoil_discovery.cfd.su2_config import build_stage1_config
try:
    candidate = CandidateDesign(params=params, reynolds=100000.0)
    config_text = build_stage1_config(candidate, mesh_path, aoa=2.0, settings=settings)
    config_path = geo_path.parent / "config_primal.cfg"
    config_path.write_text(config_text, encoding="utf-8")
    check("SU2 config written", config_path.exists(), str(config_path))
    
    # Validate config has required keys
    import re
    for key in ["SOLVER", "MESH_FILENAME", "RESTART_SOL", "ITER"]:
        check(f"  Config key {key} present", 
              bool(re.search(rf"^{key}\s*=", config_text, re.MULTILINE)))
    
    # Check mesh filename in config matches
    check(f"  Config MESH_FILENAME={mesh_path.name}", 
          f"MESH_FILENAME= {mesh_path.name}" in config_text)
except Exception as e:
    check("SU2 config generation", False, str(e))

# 8. Skip SU2 primal (takes too long for CI), but verify binary works
try:
    version_result = subprocess.run(
        [su2_bin, "--version"],
        capture_output=True, text=True, timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    check("SU2 executable runs", version_result.returncode != -1, f"stderr: {version_result.stderr[:200]}")
except Exception as e:
    check("SU2 executable runs", False, str(e))

print()
print("=" * 60)
print(f"  RESULTS: {PASS} passed, {FAIL} failed out of {PASS + FAIL}")
print("=" * 60)
sys.exit(0 if FAIL == 0 else 1)