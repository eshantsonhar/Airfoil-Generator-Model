#!/usr/bin/env python3
"""
Run a single baseline CFD solve to verify convergence and log CL/CD values.
"""
import sys
import os
import logging
from pathlib import Path
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from airfoil_discovery.aso.optimizer import run_primal_and_adjoint, CFDResult
from airfoil_discovery.aso.config_primal import write_primal_config

def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("BASELINE CFD SOLVE")
    logger.info("=" * 60)
    
    # Paths - use the original baseline mesh
    project_root = Path(__file__).resolve().parents[1]
    mesh_path = project_root / "phase5_output" / "mesh_baseline.su2"
    dv_path = project_root / "init_dv_baseline.npy"
    output_dir = project_root / "baseline_cfd_run"
    
    # Verify files exist
    if not mesh_path.exists():
        logger.error(f"Mesh not found: {mesh_path}")
        sys.exit(1)
    
    if not dv_path.exists():
        logger.error(f"Initial DV not found: {dv_path}")
        sys.exit(1)
    
    # Load initial design vector
    dv = np.load(dv_path)
    logger.info(f"Loaded design vector: shape={dv.shape}")
    
    # SU2 binary
    su2_cfd_bin = "bin/SU2_CFD.exe"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use the patched config generator to run a fresh CFD solve
    from airfoil_discovery.aso.config_primal import write_primal_config
    
    # Copy mesh to output dir
    import shutil
    mesh_in_case = output_dir / "airfoil.su2"
    shutil.copy2(mesh_path, mesh_in_case)
    
    # Generate config with patched settings
    write_primal_config(
        output_path=output_dir / "config_primal.cfg",
        mesh_filename="airfoil.su2",
        aoa_deg=4.0,
        reynolds=1e5,
        mach=0.1,
        n_iter=100,
        cfl_initial=1.5,
        cfl_final=50.0,
        cfl_adapt=True,
        transition_model=True,
        turbulence_intensity=0.001,
        turb_viscosity_ratio=5.0,
        slope_limiter_flow="VENKATAKRISHNAN_WANG",
        slope_limiter_turb="VENKATAKRISHNAN",
    )
    logger.info("Generated patched config using updated config_primal.py")
    
    # Run SU2 directly with patched config
    import subprocess
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    result = subprocess.run(
        [su2_cfd_bin, "config_primal.cfg"],
        cwd=str(output_dir),
        capture_output=True,
        text=True,
        timeout=3600,
        creationflags=flags
    )
    
    (output_dir / "su2_primal_stdout.log").write_text(result.stdout, encoding="utf-8", errors="ignore")
    (output_dir / "su2_primal_stderr.log").write_text(result.stderr, encoding="utf-8", errors="ignore")
    
    logger.info(f"SU2_CFD completed with rc={result.returncode}")
    
    # Parse history file
    from airfoil_discovery.aso.optimizer import _parse_history
    history_file = output_dir / "history.csv"
    cl, cd, converged = _parse_history(history_file)
    
    # Create CFDResult-like object
    from airfoil_discovery.aso.optimizer import CFDResult
    result_obj = CFDResult(
        cl=cl,
        cd=cd,
        converged=converged,
        adjoint_gradient=np.zeros(12),
        gradient_valid=False,
        primal_converged=converged,
        adjoint_converged=False,
        case_dir=output_dir,
        mesh_path=mesh_in_case,
    )
    
    # Report results
    logger.info("=" * 60)
    logger.info("BASELINE CFD RESULTS")
    logger.info("=" * 60)
    logger.info(f"CL: {result_obj.cl:.6f}")
    logger.info(f"CD: {result_obj.cd:.6f}")
    logger.info(f"Converged: {result_obj.converged}")
    logger.info(f"Primal Converged: {result_obj.primal_converged}")
    
    if result_obj.failure_reason:
        logger.warning(f"Failure Reason: {result_obj.failure_reason}")
    
    # Sanity checks
    logger.info("\nSANITY CHECKS:")
    if result_obj.cd > 0:
        logger.info(f"L/D: {result_obj.cl/result_obj.cd:.2f}")
    else:
        logger.warning(f"Cannot compute L/D (CD={result_obj.cd:.6f})")
    
    if 0.4 <= result_obj.cl <= 0.8:
        logger.info(f"✓ CL is physically reasonable: {result_obj.cl:.6f}")
    else:
        logger.warning(f"✗ CL is out of expected range [0.4, 0.8]: {result_obj.cl:.6f}")
    
    if 0.005 <= result_obj.cd <= 0.020:
        logger.info(f"✓ CD is physically reasonable: {result_obj.cd:.6f}")
    else:
        logger.warning(f"✗ CD is out of expected range [0.005, 0.020]: {result_obj.cd:.6f}")
    
    logger.info("=" * 60)
    
    # Save results to file
    results_file = output_dir / "baseline_results.txt"
    ld_ratio = result_obj.cl/result_obj.cd if result_obj.cd > 0 else 0.0
    results_file.write_text(
        f"CL: {result_obj.cl:.6f}\n"
        f"CD: {result_obj.cd:.6f}\n"
        f"L/D: {ld_ratio:.2f}\n"
        f"Converged: {result_obj.converged}\n"
        f"Primal Converged: {result_obj.primal_converged}\n"
    )
    logger.info(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main()
