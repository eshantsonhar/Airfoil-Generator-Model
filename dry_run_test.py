"""
Phase 2: Dry Run Verification Test
Tests each component of the CFD optimization pipeline before full run.
"""

import os
import subprocess
import tempfile
import time
from pathlib import Path
import numpy as np

# Add project to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

def test_su2_cfd():
    """Test 1: Run single flow solve with SU2_CFD"""
    print("=" * 60)
    print("TEST 1: Single Flow Solve with SU2_CFD")
    print("=" * 60)
    
    # Create temporary directory
    test_dir = Path("test_dry_run")
    test_dir.mkdir(exist_ok=True)
    
    # Copy config and mesh
    config_path = Path("config.cfg")
    mesh_path = Path("airfoil_perfect.su2")
    
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        return False
    
    if not mesh_path.exists():
        print(f"ERROR: Mesh file not found: {mesh_path}")
        return False
    
    # Copy to test directory
    test_config = test_dir / "test_config.cfg"
    test_mesh = test_dir / "airfoil_perfect.su2"
    
    import shutil
    shutil.copy2(config_path, test_config)
    shutil.copy2(mesh_path, test_mesh)
    
    # Update config to use local mesh and reduce iterations for test
    config_text = test_config.read_text()
    config_text = config_text.replace("ITER= 1000", "ITER= 100")  # Reduce iterations for test
    test_config.write_text(config_text)
    
    # Run SU2_CFD
    su2_cfd_bin = "bin/SU2_CFD.exe"
    
    if not Path(su2_cfd_bin).exists():
        print(f"ERROR: SU2_CFD not found: {su2_cfd_bin}")
        return False
    
    print(f"Running SU2_CFD with config: {test_config}")
    print(f"Working directory: {test_dir}")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            [su2_cfd_bin, test_config.name],
            cwd=test_dir,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
    except subprocess.TimeoutExpired:
        print("ERROR: SU2_CFD timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"ERROR: SU2_CFD execution failed: {e}")
        return False
    
    elapsed_time = time.time() - start_time
    
    # Write logs
    (test_dir / "su2_cfd_stdout.log").write_text(result.stdout)
    (test_dir / "su2_cfd_stderr.log").write_text(result.stderr)
    
    print(f"Return code: {result.returncode}")
    print(f"Elapsed time: {elapsed_time:.1f}s")
    
    if result.returncode == 0:
        print("✓ SU2_CFD completed successfully")
        
        # Check for output files
        history_file = test_dir / "history.csv"
        if history_file.exists():
            print(f"✓ History file created: {history_file}")
            # Check if it has data
            history_content = history_file.read_text()
            lines = history_content.strip().split('\n')
            print(f"  History has {len(lines)} lines of data")
        else:
            print("✗ No history file created")
            
        surface_file = test_dir / "surface_flow.csv"
        if surface_file.exists():
            print(f"✓ Surface file created: {surface_file}")
        else:
            print("✗ No surface file created")
            
        return True
    else:
        print(f"✗ SU2_CFD failed with return code {result.returncode}")
        print(f"Stderr: {result.stderr[:500]}")
        return False

def test_adjoint_config():
    """Test 2: Generate and verify adjoint configuration"""
    print("\n" + "=" * 60)
    print("TEST 2: Adjoint Configuration Generation")
    print("=" * 60)
    
    try:
        from src.airfoil_discovery.aso.config_adjoint import generate_adjoint_config
        
        config_text = generate_adjoint_config(
            mesh_filename="airfoil_perfect.su2",
            primal_config_filename="config.cfg",
            objective="DRAG",
            n_iter=50,  # Reduced for test
            cfl_adjoint=1.0
        )
        
        # Write for inspection
        test_dir = Path("test_dry_run")
        test_dir.mkdir(exist_ok=True)
        adj_config_path = test_dir / "config_adjoint_test.cfg"
        adj_config_path.write_text(config_text)
        
        print(f"✓ Adjoint config generated: {adj_config_path}")
        print(f"  Length: {len(config_text)} characters")
        print(f"  Lines: {len(config_text.splitlines())}")
        
        # Check for critical parameters
        required_params = [
            "MATH_PROBLEM= DISCRETE_ADJOINT",
            "SOLVER= RANS",
            "OBJECTIVE_FUNCTION=",
            "MARKER_HEATFLUX= ( airfoil, 0.0 )",
            "MARKER_FAR= ( farfield )"
        ]
        
        for param in required_params:
            if param in config_text:
                print(f"  ✓ Contains: {param}")
            else:
                print(f"  ✗ Missing: {param}")
                return False
                
        return True
        
    except Exception as e:
        print(f"✗ Adjoint config generation failed: {e}")
        return False

def test_mesh_deformation():
    """Test 3: Mesh deformation with SU2_DEF"""
    print("\n" + "=" * 60)
    print("TEST 3: Mesh Deformation with SU2_DEF")
    print("=" * 60)
    
    try:
        from src.airfoil_discovery.aso.mesh_deform import deform_mesh
        
        # Create test design vectors
        dv_old = np.ones(12) * 0.1  # Baseline design
        dv_new = np.ones(12) * 0.15  # Slightly different design
        
        test_dir = Path("test_dry_run_mesh")
        test_dir.mkdir(exist_ok=True)
        
        su2_def_bin = "bin/SU2_DEF.exe"
        mesh_path = Path("airfoil_perfect.su2")
        
        if not Path(su2_def_bin).exists():
            print(f"ERROR: SU2_DEF not found: {su2_def_bin}")
            return False
            
        if not mesh_path.exists():
            print(f"ERROR: Mesh file not found: {mesh_path}")
            return False
        
        print("Testing mesh deformation...")
        print(f"  Original mesh: {mesh_path}")
        print(f"  SU2_DEF binary: {su2_def_bin}")
        print(f"  Test directory: {test_dir}")
        
        # Try deformation
        deformed_path = deform_mesh(
            su2_def_bin=su2_def_bin,
            original_mesh_path=mesh_path,
            dv_old=dv_old,
            dv_new=dv_new,
            work_dir=test_dir,
            marker="airfoil",
            n_iter_def=50  # Reduced for test
        )
        
        if deformed_path and deformed_path.exists():
            print(f"✓ Mesh deformation successful!")
            print(f"  Deformed mesh: {deformed_path}")
            print(f"  File size: {deformed_path.stat().st_size} bytes")
            return True
        else:
            print("✗ Mesh deformation failed")
            # Check what files were created
            su2_files = list(test_dir.glob("*.su2"))
            print(f"  SU2 files in test directory: {[f.name for f in su2_files]}")
            return False
            
    except Exception as e:
        print(f"✗ Mesh deformation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_finite_difference_gradient():
    """Test 4: Finite difference gradient fallback"""
    print("\n" + "=" * 60)
    print("TEST 4: Finite Difference Gradient Computation")
    print("=" * 60)
    
    try:
        from src.airfoil_discovery.aso.optimizer import ASOObjectiveFunction
        
        # Create a mock objective function for testing
        class MockObjectiveFunction:
            def __init__(self):
                self.call_count = 0
                
            def __call__(self, dv):
                self.call_count += 1
                # Simple quadratic function for testing
                return np.sum(dv**2)
        
        # Test the _finite_difference_gradient method
        test_obj = MockObjectiveFunction()
        
        # Monkey patch the method to use our mock
        orig_call = ASOObjectiveFunction.__call__
        ASOObjectiveFunction.__call__ = lambda self, dv: test_obj(dv)
        
        try:
            # Create instance
            obj_func = ASOObjectiveFunction(
                su2_cfd_bin="dummy",
                mesh_path=Path("dummy"),
                case_root=Path("dummy")
            )
            
            # Test design vector
            dv_test = np.ones(12) * 0.5
            
            print("Testing finite difference gradient...")
            print(f"  Test design vector shape: {dv_test.shape}")
            
            # Call the method
            grad = obj_func._finite_difference_gradient(dv_test, eps=1e-3)
            
            print(f"✓ Finite difference gradient computed")
            print(f"  Gradient shape: {grad.shape}")
            print(f"  Gradient norm: {np.linalg.norm(grad):.6e}")
            print(f"  Function evaluations: {test_obj.call_count}")
            
            # Check gradient values (should be approximately 2*dv for f = sum(dv^2))
            expected_grad = 2 * dv_test
            error = np.max(np.abs(grad - expected_grad))
            print(f"  Max error vs analytic: {error:.6e}")
            
            if error < 5e-3:  # Relaxed tolerance for finite difference with eps=1e-3
                print("  ✓ Gradient accuracy acceptable")
                return True
            else:
                print("  ✗ Gradient accuracy poor")
                return False
                
        finally:
            # Restore original method
            ASOObjectiveFunction.__call__ = orig_call
            
    except Exception as e:
        print(f"✗ Finite difference gradient test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all dry run tests"""
    print("=" * 80)
    print("PHASE 2: DRY RUN VERIFICATION")
    print("=" * 80)
    print()
    
    tests = [
        ("SU2_CFD Flow Solve", test_su2_cfd),
        ("Adjoint Config Generation", test_adjoint_config),
        ("Mesh Deformation", test_mesh_deformation),
        ("Finite Difference Gradient", test_finite_difference_gradient)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"Exception in {test_name}: {e}")
            results.append((test_name, False))
        
        print()  # Blank line between tests
    
    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n✓ All tests passed! Ready for Phase 3: Full Production Run.")
        return True
    else:
        print(f"\n✗ {len(results) - passed} tests failed. Fix issues before full run.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)