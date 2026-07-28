import os
import sys
import numpy as np
from pathlib import Path
import json

sys.path.insert(0, 'src')

from airfoil_discovery.config import load_settings
from airfoil_discovery.cfd.su2 import SU2Runner, SU2ExecutionError, SU2ConfigurationError
from airfoil_discovery.aso.mesh_deform import deform_mesh
from airfoil_discovery.geometry.cst import compute_airfoil_coordinates

results = []

def log_test(name, passed, detail=""):
    results.append({"test": name, "passed": passed, "detail": detail})
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")

print("Starting Deep Audit...")
# 1. Config resilience
try:
    s = load_settings('config/default.yaml')
    log_test("Load default config", True)
except Exception as e:
    log_test("Load default config", False, str(e))

# 2. Geometry coordinate limits & NaN propagation
try:
    dv_nan = np.array([np.nan]*12)
    coords = compute_airfoil_coordinates(dv_nan)
    if np.any(np.isnan(coords)):
        log_test("Geometry NaN propagation", True, "NaNs safely propagated to coords without hard crash")
    else:
        log_test("Geometry NaN propagation", False, "NaNs did not propagate")
except Exception as e:
    log_test("Geometry NaN propagation", False, f"Crash on NaN: {e}")

# 3. SU2 Runner resilience: Malformed history
runner = SU2Runner(s)
malformed_history_path = Path("malformed_history.csv")
malformed_history_path.write_text("ITER,CL,CD\n1,0.5,0.01\n2,NaN,0.02\n")
try:
    cl, cd = runner._read_results(malformed_history_path)
    log_test("SU2Runner parse malformed history (NaN)", False, "Should have raised SU2ExecutionError")
except SU2ExecutionError as e:
    log_test("SU2Runner parse malformed history (NaN)", True, str(e))
except Exception as e:
    log_test("SU2Runner parse malformed history (NaN)", False, f"Wrong exception: {e}")

# 4. SU2Runner resilience: empty history
empty_history_path = Path("empty_history.csv")
empty_history_path.write_text("")
try:
    runner._read_results(empty_history_path)
    log_test("SU2Runner parse empty history", False, "Should have raised SU2ExecutionError")
except SU2ExecutionError as e:
    log_test("SU2Runner parse empty history", True, str(e))

# 5. SU2Runner resilience: unphysical results
unphys_history_path = Path("unphys_history.csv")
unphys_history_path.write_text("ITER,CL,CD\n1,0.5,0.01\n2,150.0,-5.0\n")
try:
    runner._read_results(unphys_history_path)
    log_test("SU2Runner parse unphysical history", False, "Should have raised SU2ExecutionError")
except SU2ExecutionError as e:
    log_test("SU2Runner parse unphysical history", True, str(e))

# 6. Mesh deformation edge cases
dv_valid = np.zeros(12)
dv_wrong_shape = np.zeros(10)
res = deform_mesh("su2_def", Path("test.su2"), dv_valid, dv_wrong_shape, Path("."))
if res is None:
    log_test("Mesh deform wrong shape", True, "Handled safely")
else:
    log_test("Mesh deform wrong shape", False, "Did not handle safely")

with open("audit_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Deep Audit Complete.")
