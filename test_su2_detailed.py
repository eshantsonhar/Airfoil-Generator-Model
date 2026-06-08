#!/usr/bin/env python3
"""
Detailed test of SU2 execution to find where it's failing
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from airfoil_discovery.config import load_settings
from airfoil_discovery.cfd.su2 import SU2Evaluator, SU2Status, SU2ExecutionError
import numpy as np

def main():
    print("Testing SU2 execution step by step...")
    
    # Load settings
    settings = load_settings('config/default.yaml')
    print(f"SU2 path: {settings.solver.su2_cfd_bin}")
    print(f"GMSH path: {settings.solver.gmsh_bin}")
    
    # Create evaluator
    evaluator = SU2Evaluator(settings)
    print("Evaluator created")
    
    # Test with a simple design vector
    design_vector = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.005, 0.0])
    case_dir = Path("data/test_case")
    
    try:
        print("Running evaluation...")
        result = evaluator.run_evaluation(design_vector, case_dir, "L1")
        print(f"Status: {result.status}")
        if result.status == SU2Status.OK:
            print(f"Success! CL={result.cl}, CD={result.cd}")
        else:
            print(f"Failed with status: {result.status}")
    except Exception as e:
        print(f"Exception occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()