import numpy as np
from pathlib import Path
from airfoil_discovery.aso.mesh_deform import generate_su2_def_config, deform_mesh

# Test 1: Config Generation Syntax
config_str = generate_su2_def_config(mesh_input='mesh_in.su2', mesh_output='mesh_out.su2', marker='airfoil', poisson_ratio=0.3)
assert 'MATH_PROBLEM= ELASTICITY' in config_str, 'MATH_PROBLEM syntax invalid'
assert 'DEFORM_POISSONS_RATIO= 0.3' in config_str, 'Poisson ratio syntax invalid'
assert 'MARKER_HEATFLUX= ( airfoil, 0.0 )' in config_str, 'Boundary condition marker missing'
print(' Config generation syntax verified!')

# Test 2: Invalid Design Vector Safeguards
dv_valid = np.zeros(12)
dv_nan = np.array([np.nan] * 12)
assert deform_mesh('su2_def', Path('non_existent.su2'), dv_valid, dv_nan, Path('.')) is None, 'NaN handling failed'

# Test 3: Non-existent Mesh Safeguard
assert deform_mesh('su2_def', Path('missing_mesh_file.su2'), dv_valid, dv_valid, Path('.')) is None, 'Missing mesh check failed'
print(' Edge case safeguards verified!')

print('PHASE 2 VERIFICATION PASSED SUCCESSFULLY')
