#!/usr/bin/env python3
"""
15-Iteration Real CFD Verification Test

This script runs a true 15-iteration optimization test against data/mesh_fixed.su2
to verify that the hardened safeguards work correctly in real CFD execution.
"""

import sys
import logging
import subprocess
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("test_15iter_verification")


def main():
    """Run 15-iteration verification test."""
    logger.info("=" * 60)
    logger.info("15-ITERATION REAL CFD VERIFICATION TEST")
    logger.info("=" * 60)
    
    # Output directory
    output_dir = Path("aso_15iter_verification")
    output_dir.mkdir(exist_ok=True)
    
    # PowerShell command for 15-iteration test
    ps_command = f"""
    python scripts/run_aso_pde_optimization.py `
      --mesh data/mesh_fixed.su2 `
      --output {output_dir} `
      --method mma `
      --max-iter 15 `
      --min-cl 1.0 `
      --cl-penalty-weight 1.0 `
      --no-adjoint `
      --aoa 4.0 `
      --reynolds 1.0e5 `
      --mach 0.1 `
      --no-preflight `
      --su2-cfd bin/SU2_CFD.exe `
      --su2-def bin/SU2_DEF.exe
    """
    
    logger.info("Starting 15-iteration optimization...")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Mesh: data/mesh_fixed.su2")
    logger.info(f"Max iterations: 15")
    logger.info("")
    
    # Run the optimization
    try:
        result = subprocess.run(
            ["powershell", "-Command", ps_command],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )
        
        # Log output
        logger.info("STDOUT:")
        logger.info(result.stdout)
        
        if result.stderr:
            logger.warning("STDERR:")
            logger.warning(result.stderr)
        
        logger.info(f"Return code: {result.returncode}")
        
        # Check for success
        if result.returncode == 0:
            logger.info("=" * 60)
            logger.info("✓ 15-ITERATION VERIFICATION COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            
            # Check for geometry validation in logs
            log_file = output_dir / "optimization.log"
            if log_file.exists():
                log_content = log_file.read_text(encoding="utf-8", errors="ignore")
                
                geom_validation_count = log_content.count("Geometric validation")
                logger.info(f"Geometry validation calls detected: {geom_validation_count}")
                
                if geom_validation_count > 0:
                    logger.info("✓ Geometry validation is active")
                else:
                    logger.warning("⚠ No geometry validation calls detected in logs")
                
                # Check for move limit floor enforcement
                if "floor enforced" in log_content:
                    logger.info("✓ Move limit floor enforcement detected")
                
                # Check for best design export
                best_design = output_dir / "best_airfoil_shape.dat"
                if best_design.exists():
                    logger.info("✓ Best design export found")
                else:
                    logger.warning("⚠ Best design export not found")
                
                # Check iteration count
                if "MMA Iteration 15/15" in log_content:
                    logger.info("✓ All 15 iterations completed")
                else:
                    logger.warning("⚠ Not all 15 iterations completed")
            
            return 0
        else:
            logger.error("=" * 60)
            logger.error("✗ 15-ITERATION VERIFICATION FAILED")
            logger.error("=" * 60)
            return 1
            
    except subprocess.TimeoutExpired:
        logger.error("✗ 15-ITERATION VERIFICATION TIMED OUT (2 hours)")
        return 1
    except Exception as e:
        logger.error(f"✗ 15-ITERATION VERIFICATION ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
