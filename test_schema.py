import os
from airfoil_discovery.config import load_settings
settings = load_settings('config/default.yaml')
assert hasattr(settings.solver, 'su2_def_bin'), 'Missing su2_def_bin field'
assert settings.solver.su2_def_bin == 'bin/SU2_DEF.exe', 'Default value mismatch'

os.environ['SU2_DEF_BIN'] = 'custom/path/SU2_DEF'
settings_env = load_settings('config/default.yaml')
assert settings_env.solver.su2_def_bin == 'custom/path/SU2_DEF', 'Env override failed'
print('PHASE 1 VERIFICATION PASSED SUCCESSFULLY')
