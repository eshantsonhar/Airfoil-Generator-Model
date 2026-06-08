"""Quick CFD execution test — verify SU2 can run an evaluation."""
import sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

os.environ["AIRFOIL_TELEMETRY_PATH"] = str(PROJECT_ROOT / "data" / "logs" / "telemetry_events.jsonl")
os.environ["AIRFOIL_JOB_RUNTIME_PATH"] = str(PROJECT_ROOT / "data" / "logs" / "latest_runtime.json")
os.environ["AIRFOIL_RUN_ID"] = "test_cfd_run"

import numpy as np
from airfoil_discovery.config import load_settings

settings = load_settings(PROJECT_ROOT / "config" / "default.yaml")
print(f"SU2: {settings.solver.su2_cfd_bin}  exists={os.path.exists(settings.solver.su2_cfd_bin)}")
print(f"Gmsh: {settings.solver.gmsh_bin}  exists={os.path.exists(settings.solver.gmsh_bin)}")
print(f"stage1_iter={settings.solver.stage1_iter}")

# Test SU2 binary responds
import subprocess
r = subprocess.run([str(settings.solver.su2_cfd_bin), "--help"], capture_output=True, text=True, timeout=10,
                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
print(f"SU2 help rc={r.returncode}, stderr[:200]={r.stderr[:200]}")

# Test Gmsh binary responds  
r = subprocess.run([str(settings.solver.gmsh_bin), "--version"], capture_output=True, text=True, timeout=10,
                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
print(f"Gmsh version rc={r.returncode}, stdout={r.stdout.strip()}")

print("\n=== All binaries OK. Testing CFD evaluation ===")

# Create a minimal test case directory
case_dir = PROJECT_ROOT / "data" / "cache" / "test_quick"
case_dir.mkdir(parents=True, exist_ok=True)

# Run single evaluation on NACA-like initial design
from airfoil_discovery.cfd.su2 import SU2Evaluator

evaluator = SU2Evaluator(settings)
design_vector = np.array([
    0.18, 0.05, 0.34, 0.10,    # upper CST
    -0.19, 0.05, -0.09, 0.03,   # lower CST
    0.004,                      # TE thickness
    1.0                         # scale
])

print("Starting CFD evaluation (this takes several minutes)...")
import time
t0 = time.time()
result = evaluator.run_evaluation(design_vector, case_dir, mesh_level="L1", aoa=4.0)
elapsed = time.time() - t0

print(f"\n{'='*60}")
print(f"CFD Evaluation completed in {elapsed:.1f}s")
print(f"Status: {result.status.value}")
print(f"CL = {result.cl:.6f}")
print(f"CD = {result.cd:.6f}")
print(f"Max thickness = {result.thickness:.6f}")
if result.convergence_report:
    print(f"Converged: {result.convergence_report.get('is_valid', False)}")
    print(f"Residual converged: {result.convergence_report.get('residual_converged', False)}")
    print(f"Forces stabilized: {result.convergence_report.get('forces_stabilized', False)}")
if result.failure_stage:
    print(f"Failure stage: {result.failure_stage}")
    print(f"Failure reason: {result.failure_reason}")

# Show generated files
print(f"\nFiles in {case_dir}:")
for f in sorted(case_dir.iterdir()):
    size = f.stat().st_size if f.is_file() else 0
    print(f"  {f.name:40s} {size:>10d} bytes")