#!/usr/bin/env python3
"""
Quick validation CFD run for benchmark comparison.
Runs a single CFD case and extracts results for validation.
"""
import sys
import os
import logging
import json
import subprocess
from pathlib import Path
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

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
    
    logger.info("=" * 70)
    logger.info("VALIDATION CFD RUN")
    logger.info("=" * 70)
    
    # Use the existing baseline mesh and design vector
    mesh_path = PROJECT_ROOT / "data" / "mesh_fixed.su2"
    dv_path = PROJECT_ROOT / "init_dv_baseline.npy"
    output_dir = PROJECT_ROOT / "validation_cfd_run"
    
    # Check if files exist
    if not mesh_path.exists():
        logger.error(f"Mesh not found: {mesh_path}")
        logger.info("Trying alternative mesh path...")
        mesh_path = PROJECT_ROOT / "phase5_output" / "mesh_baseline.su2"
        if not mesh_path.exists():
            logger.error("Alternative mesh also not found. Exiting.")
            sys.exit(1)
    
    if not dv_path.exists():
        logger.error(f"Initial DV not found: {dv_path}")
        sys.exit(1)
    
    # Load initial design vector
    dv = np.load(dv_path)
    logger.info(f"Loaded design vector: shape={dv.shape}")
    logger.info(f"Design vector: {dv}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy mesh to output dir
    import shutil
    mesh_in_case = output_dir / "airfoil.su2"
    shutil.copy2(mesh_path, mesh_in_case)
    logger.info(f"Copied mesh to: {mesh_in_case}")
    
    # Generate SU2 config using the patched config generator
    from airfoil_discovery.aso.config_primal import write_primal_config
    
    write_primal_config(
        output_path=output_dir / "config_primal.cfg",
        mesh_filename="airfoil.su2",
        aoa_deg=4.0,
        reynolds=1e5,
        mach=0.1,
        n_iter=500,  # Reduced iterations for faster validation
        cfl_initial=1.5,
        cfl_final=3.0,
        cfl_adapt=True,
        transition_model=True,
        turbulence_intensity=0.001,
        turb_viscosity_ratio=5.0,
        slope_limiter_flow="VENKATAKRISHNAN_WANG",
        slope_limiter_turb="VENKATAKRISHNAN",
    )
    logger.info("Generated SU2 config")
    
    # Run SU2 CFD
    su2_cfd_bin = "bin/SU2_CFD.exe"
    logger.info(f"Running SU2_CFD: {su2_cfd_bin}")
    
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    result = subprocess.run(
        [su2_cfd_bin, "config_primal.cfg"],
        cwd=str(output_dir),
        capture_output=True,
        text=True,
        timeout=1800,  # 30 minutes timeout
        creationflags=flags
    )
    
    (output_dir / "su2_stdout.log").write_text(result.stdout, encoding="utf-8", errors="ignore")
    (output_dir / "su2_stderr.log").write_text(result.stderr, encoding="utf-8", errors="ignore")
    
    logger.info(f"SU2_CFD completed with rc={result.returncode}")
    
    if result.returncode != 0:
        logger.error("SU2_CFD failed!")
        logger.error(f"STDERR: {result.stderr[:500]}")
        sys.exit(1)
    
    # Parse history file
    history_file = output_dir / "history.csv"
    if not history_file.exists():
        logger.error(f"History file not found: {history_file}")
        sys.exit(1)
    
    # Parse CSV to get final CL and CD
    with open(history_file, 'r') as f:
        lines = f.readlines()
    
    # Clean up header - remove quotes and whitespace
    header = [h.strip().strip('"') for h in lines[0].strip().split(',')]
    logger.info(f"History headers: {header}")
    
    # Find CL and CD column indices
    try:
        cl_idx = header.index('CL')
        cd_idx = header.index('CD')
    except ValueError as e:
        logger.error(f"Could not find CL/CD columns: {e}")
        logger.info(f"Available columns: {header}")
        sys.exit(1)
    
    # Get last line for final values
    last_line = [val.strip().strip('"') for val in lines[-1].strip().split(',')]
    cl = float(last_line[cl_idx])
    cd = float(last_line[cd_idx])
    
    logger.info(f"Final iteration values: CL={cl}, CD={cd}")
    
    logger.info("=" * 70)
    logger.info("VALIDATION CFD RESULTS")
    logger.info("=" * 70)
    logger.info(f"Test Conditions:")
    logger.info(f"  Airfoil: CST-generated baseline")
    logger.info(f"  Reynolds: 100,000")
    logger.info(f"  AoA: 4.0°")
    logger.info(f"  Mach: 0.1")
    logger.info(f"  Transition Model: γ-Reθ (LM)")
    logger.info(f"  Turbulence Intensity: 0.001")
    logger.info(f"  Viscosity Ratio: 5.0")
    logger.info(f"")
    logger.info(f"Computed SU2 Values:")
    logger.info(f"  C_L: {cl:.6f}")
    logger.info(f"  C_D: {cd:.6f}")
    logger.info(f"  L/D: {cl/cd:.3f}")
    logger.info("=" * 70)
    
    logger.info("=" * 70)
    logger.info("VALIDATION CFD RESULTS")
    logger.info("=" * 70)
    logger.info(f"Test Conditions:")
    logger.info(f"  Airfoil: CST-generated baseline")
    logger.info(f"  Reynolds: 100,000")
    logger.info(f"  AoA: 4.0°")
    logger.info(f"  Mach: 0.1")
    logger.info(f"  Transition Model: γ-Reθ (LM)")
    logger.info(f"  Turbulence Intensity: 0.001")
    logger.info(f"  Viscosity Ratio: 5.0")
    logger.info(f"")
    logger.info(f"Computed SU2 Values:")
    logger.info(f"  C_L: {cl:.6f}")
    logger.info(f"  C_D: {cd:.6f}")
    logger.info(f"  L/D: {cl/cd:.3f}")
    logger.info("=" * 70)
    
    # Load reference data for comparison
    ref_data_dir = SRC_ROOT / "airfoil_discovery" / "validation" / "literature_benchmarks"
    
    # Try to find a matching reference case
    reference_found = False
    for ref_file in ref_data_dir.glob("*.json"):
        try:
            ref = json.loads(ref_file.read_text())
            if ref.get("reynolds") == 100000 and 4.0 in ref.get("aoa", []):
                # Find the AoA=4.0 values
                aoa_idx = ref["aoa"].index(4.0)
                ref_cl = ref["cl"][aoa_idx]
                ref_cd = ref["cd"][aoa_idx]
                
                logger.info(f"")
                logger.info(f"Reference Data ({ref['airfoil_name']}):")
                logger.info(f"  Source: {ref.get('source', 'Unknown')}")
                logger.info(f"  Experimental C_L: {ref_cl:.6f}")
                logger.info(f"  Experimental C_D: {ref_cd:.6f}")
                logger.info(f"  Experimental L/D: {ref_cl/ref_cd:.3f}")
                logger.info(f"")
                logger.info(f"Comparison:")
                logger.info(f"  C_L Error: {(cl - ref_cl)/ref_cl*100:+.1f}%")
                logger.info(f"  C_D Error: {(cd - ref_cd)/ref_cd*100:+.1f}%")
                
                reference_found = True
                break
        except Exception as e:
            logger.warning(f"Could not load reference file {ref_file}: {e}")
    
    if not reference_found:
        logger.info("No matching reference data found for Re=100k, AoA=4°")
        logger.info("Available reference files:")
        for ref_file in ref_data_dir.glob("*.json"):
            try:
                ref = json.loads(ref_file.read_text())
                logger.info(f"  {ref_file.name}: Re={ref.get('reynolds')}, AoA={ref.get('aoa')}")
            except:
                pass
    
    # Save results
    results = {
        "test_conditions": {
            "airfoil": "CST-generated baseline",
            "reynolds": 100000,
            "aoa": 4.0,
            "mach": 0.1,
            "transition_model": "gamma-Re_theta (LM)",
            "turbulence_intensity": 0.001,
            "viscosity_ratio": 5.0
        },
        "computed_values": {
            "cl": cl,
            "cd": cd,
            "ld_ratio": cl/cd
        }
    }
    
    results_file = output_dir / "validation_results.json"
    results_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info(f"Results saved to: {results_file}")
    
    logger.info("=" * 70)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()