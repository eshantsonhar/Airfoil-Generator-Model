"""Quick CFD test with L0 mesh level to minimize meshing time."""
import sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

os.environ["AIRFOIL_TELEMETRY_PATH"] = str(PROJECT_ROOT / "data" / "logs" / "telemetry_events.jsonl")
os.environ["AIRFOIL_JOB_RUNTIME_PATH"] = str(PROJECT_ROOT / "data" / "logs" / "latest_runtime.json")
os.environ["AIRFOIL_RUN_ID"] = "test_cfd_l0"

import numpy as np
from airfoil_discovery.config import load_settings

settings = load_settings(PROJECT_ROOT / "config" / "default.yaml")
settings.solver.case_timeout_seconds = 600  # 10 minute timeout

print(f"SU2: {settings.solver.su2_cfd_bin} exists={os.path.exists(settings.solver.su2_cfd_bin)}")
print(f"Gmsh: {settings.solver.gmsh_bin} exists={os.path.exists(settings.solver.gmsh_bin)}")
print(f"stage1_iter={settings.solver.stage1_iter}")

from airfoil_discovery.cfd.su2 import SU2Evaluator

evaluator = SU2Evaluator(settings)
design_vector = np.array([
    0.18, 0.05, 0.34, 0.10,    # upper CST
    -0.19, 0.05, -0.09, 0.03,   # lower CST
    0.004,                      # TE thickness
    1.0                         # scale
])

case_dir = PROJECT_ROOT / "data" / "cache" / "test_l0"
print("Starting CFD evaluation with L0 mesh...")
import time
t0 = time.time()
result = evaluator.run_evaluation(design_vector, case_dir, mesh_level="L0", aoa=4.0)
elapsed = time.time() - t0

print(f"\n{'='*60}")
print(f"CFD Evaluation completed in {elapsed:.1f}s")
print(f"Status: {result.status.value}")
print(f"CL = {result.cl:.6f}")
print(f"CD = {result.cd:.6f}")
if result.failure_stage:
    print(f"Failure stage: {result.failure_stage}")
    print(f"Failure reason: {result.failure_reason}")
print(f"\nFiles in {case_dir}:")
if case_dir.exists():
    for f in sorted(case_dir.iterdir()):
        size = f.stat().st_size if f.is_file() else 0
        print(f"  {f.name:40s} {size:>10d} bytes")