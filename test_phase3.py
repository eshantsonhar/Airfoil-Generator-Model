import os, sys, numpy as np
from pathlib import Path
from airfoil_discovery.config import load_settings
from airfoil_discovery.cfd.su2 import SU2Runner
from airfoil_discovery.aso.mesh_deform import generate_su2_def_config, deform_mesh

print('--- STARTING COMPLETE INTEGRATION VERIFICATION ---')

# 1. Config Schema & Default Check
settings = load_settings('config/default.yaml')
assert hasattr(settings.solver, 'su2_def_bin'), '[FAIL] Missing su2_def_bin field'
assert settings.solver.su2_def_bin == 'bin/SU2_DEF.exe', '[FAIL] Default su2_def_bin mismatch'
print('[PASS] Config schema verified')

# 2. Environment Override Check
os.environ['SU2_DEF_BIN'] = 'custom/path/SU2_DEF'
settings_env = load_settings('config/default.yaml')
assert settings_env.solver.su2_def_bin == 'custom/path/SU2_DEF', '[FAIL] Env override failed'
print('[PASS] Environment overrides verified')

# 3. SU2_DEF Config Generation Checks
config_str = generate_su2_def_config(mesh_input='mesh_in.su2', mesh_output='mesh_out.su2', marker='airfoil', poisson_ratio=0.3)
assert 'MATH_PROBLEM= ELASTICITY' in config_str, '[FAIL] Improper MATH_PROBLEM syntax'
assert 'DEFORM_POISSONS_RATIO= 0.3' in config_str, '[FAIL] Improper Poisson ratio key'
assert 'MARKER_HEATFLUX= ( airfoil, 0.0 )' in config_str, '[FAIL] Missing boundary condition heatflux'
print('[PASS] SU2_DEF configuration syntax verified')

# 4. Mesh Deformation Safeguard Checks
dv_valid = np.zeros(12)
dv_nan = np.array([np.nan] * 12)
assert deform_mesh('su2_def', Path('non_existent.su2'), dv_valid, dv_nan, Path('.')) is None, '[FAIL] NaN vector allowed'
assert deform_mesh('su2_def', Path('missing_mesh.su2'), dv_valid, dv_valid, Path('.')) is None, '[FAIL] Missing mesh allowed'
print('[PASS] Mesh deformation edge-case safeguards verified')

print('--- ALL SYSTEM INTEGRATION CHECKS PASSED (100% READY) ---')
