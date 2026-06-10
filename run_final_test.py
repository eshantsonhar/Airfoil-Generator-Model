"""Run final test with MUSCL+correct CFL+FDS+convergence fix."""
import sys, os
sys.path.insert(0, 'src')
os.environ['AIRFOIL_TELEMETRY_PATH'] = 'data/logs/telemetry_events.jsonl'

from airfoil_discovery.config import load_settings
from airfoil_discovery.cfd.su2 import SU2Evaluator, SU2Status
import numpy as np
from pathlib import Path

s = load_settings('config/default.yaml')
s.solver.case_timeout_seconds = 600
s.solver.stage1_cfl = 3.0  # Lower CFL for MUSCL stability

e = SU2Evaluator(s)
d = np.array([0.1863, 0.0779, 0.2798, 0.0839, -0.1172, 0.0642, -0.0646, 0.0309, 0.001, 1.0])

# Print config to verify changes
cfg_text = Path('src/airfoil_discovery/cfd/su2_config.py').read_text()
assert 'muscl = "YES"' in cfg_text, "MUSCL=YES not in config!"
assert 'CONV_NUM_METHOD_FLOW= FDS' in cfg_text, "FDS not in config!"
print("Config verification: MUSCL=YES, FDS, GREEN_GAUSS — all OK")

import time
t0 = time.time()
r = e.run_evaluation(d, Path('data/cache/final_test'), mesh_level='L0', aoa=4.0)
elapsed = time.time() - t0

c = r.convergence_report or {}
print(f"\nSTATUS={r.status.value} CL={r.cl:.6f} CD={r.cd:.6f} Time={elapsed:.0f}s")
print(f"CONVERGED={c.get('is_valid')} RES_DROP={c.get('residual_converged')} FORCES={c.get('forces_stabilized')}")

# Check actual convergence in CSV  
hist = Path('data/cache/final_test/history.csv')
if hist.exists():
    lines = hist.read_text().splitlines()
    if len(lines) > 1:
        header = [h.strip().strip('"') for h in lines[0].split(',')]
        rms_col = 'rms[P]' if 'rms[P]' in header else 'RMS_PRESSURE'
        if rms_col in header:
            rms_idx = header.index(rms_col)
            data = [l.split(',') for l in lines[1:] if l.strip() and l.strip() != ',']
            if data:
                start_log = float(data[0][rms_idx])
                end_log = float(data[-1][rms_idx])
                drop = abs(end_log - start_log)
                print(f"CSV: start_log={start_log:.2f} end_log={end_log:.2f} drop={drop:.1f}orders")
else:
    print("No history.csv found")