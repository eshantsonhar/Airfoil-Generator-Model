#!/usr/bin/env python3
"""
Production Optimization Run — Sets up environment, mock solvers, and executes
the full PDE-constrained optimization pipeline end-to-end.

This script:
  1. Configures environment variables pointing to mock SU2 binaries
  2. Verifies the mock solver works standalone
  3. Runs the ASO optimizer with reduced iterations for a ~5min test cycle
  4. Monitors execution, captures logs
  5. Reports final results
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
BIN_DIR = PROJECT_ROOT / "bin"
OUTPUT_DIR = PROJECT_ROOT / "prod_run_output"
MESH_FILE = PROJECT_ROOT / "data" / "cache" / "final_test" / "airfoil.su2"
INIT_DV = PROJECT_ROOT / "init_dv_baseline.npy"

os.chdir(str(PROJECT_ROOT))

# ── 1. Configure environment ────────────────────────────────────────────────
SU2_CFD_BAT = str(BIN_DIR / "SU2_CFD.bat")
SU2_CFD_PY = str(BIN_DIR / "SU2_CFD.py")
os.environ["SU2_CFD_BIN"] = SU2_CFD_BAT
os.environ["SU2_DEF_BIN"] = SU2_CFD_BAT
os.environ["SU2_HOME"] = str(BIN_DIR)
os.environ["PATH"] = str(BIN_DIR) + os.pathsep + os.environ.get("PATH", "")
os.environ["PYTHONUNBUFFERED"] = "1"

print("=" * 70)
print("  PRODUCTION OPTIMIZATION RUN — ENVIRONMENT SETUP")
print("=" * 70)
print(f"  SU2_CFD_BIN: {os.environ['SU2_CFD_BIN']}")
print(f"  SU2_DEF_BIN: {os.environ['SU2_DEF_BIN']}")
print(f"  BIN_DIR:      {BIN_DIR}")
print(f"  OUTPUT_DIR:   {OUTPUT_DIR}")
print(f"  MESH_FILE:    {MESH_FILE}")
print(f"  INIT_DV:      {INIT_DV}")

# Ensure output dir exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 2. Verify mock solver ────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print("  VERIFYING MOCK SOLVER")
print(f"{'─'*70}")

# Create a test cfg
test_cfg = OUTPUT_DIR / "_test_config.cfg"
test_cfg.write_text("\n".join([
    "AoA= 4.0",
    "REYNOLDS_NUMBER= 1.0e5",
    "MACH_NUMBER= 0.1",
    "EXT_ITER= 50",
    "CFL_NUMBER= 3.0",
]))

print(f"  Running mock SU2_CFD on test config...")
t0 = time.time()
result = subprocess.run(
    [sys.executable, SU2_CFD_PY, str(test_cfg)],
    capture_output=True, text=True, timeout=30,
    cwd=str(OUTPUT_DIR),
)
elapsed = time.time() - t0

print(f"  Exit code: {result.returncode}")
print(f"  Time: {elapsed:.2f}s")
for line in result.stdout.splitlines():
    print(f"  {line}")
if result.stderr:
    for line in result.stderr.splitlines()[:5]:
        print(f"  STDERR: {line}")

# Check outputs
hist = OUTPUT_DIR / "history.csv"
surf = OUTPUT_DIR / "surface_flow.csv"
sens = OUTPUT_DIR / "surface_sensitivity.dat"
all_ok = all(p.exists() for p in [hist, surf, sens])
print(f"\n  Output files: {'ALL GENERATED' if all_ok else 'MISSING!'}")

if not all_ok:
    print("  ERROR: Mock solver did not generate required files. Aborting.")
    sys.exit(1)

print(f"\n  Mock solver verification: PASSED ({elapsed:.2f}s)")
test_cfg.unlink()

# ── 3. Run the optimization ──────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  RUNNING FULL PDE-CONSTRAINED OPTIMIZATION")
print(f"{'='*70}")
print("  Command: python scripts/run_aso_pde_optimization.py")
print(f"  Mesh:    {MESH_FILE}")
print(f"  Output:  {OUTPUT_DIR / 'aso_optimization'}")

opt_output = OUTPUT_DIR / "aso_optimization"
opt_output.mkdir(parents=True, exist_ok=True)

# Build args for the optimizer
cmd = [
    sys.executable, str(PROJECT_ROOT / "scripts" / "run_aso_pde_optimization.py"),
    "--mesh", str(MESH_FILE),
    "--output", str(opt_output),
    "--max-iter", "5",           # 5 optimization iterations
    "--n-iter-primal", "100",    # 100 primal iterations per eval
    "--n-iter-adjoint", "50",    # 50 adjoint iterations per eval
    "--aoa", "4.0",
    "--reynolds", "1e5",
    "--mach", "0.1",
    "--method", "mma",
    "--tol", "1e-3",
    "--no-preflight",            # Skip preflight (no real SU2)
]

# Set up logging
log_file = OUTPUT_DIR / "optimization_run.log"
log_fh = open(log_file, "w", encoding="utf-8")

print(f"  Log file: {log_file}")
print(f"  Args: {' '.join(cmd)}")
print(f"\n  Starting optimization...\n{'─'*70}")

t_start = time.time()

try:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(PROJECT_ROOT),
        env={**os.environ},
    )

    # Stream output
    for line in proc.stdout:
        print(line, end="", flush=True)
        log_fh.write(line)
        log_fh.flush()

    proc.wait()
    rc = proc.returncode

except Exception as e:
    print(f"\nERROR during optimization: {e}")
    rc = -1

finally:
    log_fh.close()

total_time = time.time() - t_start

# ── 4. Report ────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  OPTIMIZATION RUN COMPLETE")
print(f"{'='*70}")
print(f"  Exit code:     {rc}")
print(f"  Total time:    {total_time:.1f}s ({total_time/60:.1f} min)")
print(f"  Log file:      {log_file}")
print(f"  Output dir:    {opt_output}")

# Check outputs
opt_results = list(opt_output.rglob("*"))
print(f"  Output files:  {len(opt_results)}")
for f in opt_results:
    if f.is_file():
        print(f"    {f.relative_to(PROJECT_ROOT)} ({f.stat().st_size} bytes)")

# Check for key outputs
convergence = opt_output / "convergence_history.json"
final_airfoil = opt_output / "final_airfoil.dat"
if convergence.exists():
    import json
    try:
        data = json.loads(convergence.read_text())
        n_iter = data.get("total_iterations", 0)
        converged = data.get("converged", False)
        print(f"\n  Convergence: {n_iter} iterations, converged={converged}")
        if data.get("iterations"):
            first = data["iterations"][0]
            last = data["iterations"][-1]
            print(f"  Initial Cd: {first['cd']:.6f}, Final Cd: {last['cd']:.6f}")
            print(f"  Initial Cl: {first['cl']:.6f}, Final Cl: {last['cl']:.6f}")
            print(f"  Initial L/D: {first['cl']/first['cd']:.2f}, Final L/D: {last['cl']/last['cd']:.2f}")
    except Exception as e:
        print(f"  Could not parse convergence: {e}")

if final_airfoil.exists():
    sz = final_airfoil.stat().st_size
    print(f"  Final airfoil: {final_airfoil.name} ({sz} bytes)")

print(f"\n{'─'*70}")
if rc == 0:
    print("  STATUS: PRODUCTION RUN COMPLETED SUCCESSFULLY (exit code 0)")
else:
    print(f"  STATUS: Production run completed with exit code {rc}")
print(f"{'─'*70}")
sys.exit(rc if rc >= 0 else 1)