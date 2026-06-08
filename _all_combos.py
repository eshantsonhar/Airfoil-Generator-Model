import sys
sys.path.insert(0,'.')
import numpy as np
from pathlib import Path
from airfoil_discovery.verification.convergence import IterativeConvergenceMonitor

# Test with sw=10 + osc_threshold=0.02 vs default sw=30
for name, sw, osc_th in [
    ('L0 sw10 osc0.02', 10, 0.02),
    ('L0 sw15 osc0.015', 15, 0.015),
    ('L0 sw12 osc0.018', 12, 0.018),
    ('def sw30 osc0.01', 30, 0.01),
]:
    print(f'\n{name}:')
    fm = IterativeConvergenceMonitor(
        force_stabilization_threshold=0.005,
        force_oscillation_threshold=osc_th,
        stabilization_window=sw)
    h = Path('data/failures/iter_001_aoa_+02p0/history.csv')
    lines = h.read_text().splitlines()
    headers = [it.strip().strip(chr(34)) for it in lines[0].split(',')]
    traces = {h2: [] for h2 in headers}
    for line in lines[1:]:
        if not line.strip() or line.strip()==',': continue
        lvs = [it.strip() for it in line.split(',')]
        for i, h2 in enumerate(headers):
            if i<len(lvs):
                try: traces[h2].append(float(lvs[i]))
                except: pass
    cl = np.array(traces.get('CL',[]), dtype=float)
    cd = np.array(traces.get('CD',[]), dtype=float)
    f = fm.analyze_forces(cl.tolist(), cd.tolist())
    print(f'  stab={f.forces_stabilized}, osc={f.force_oscillation_acceptable}, '
          f'cl_rosc={f.cl_relative_oscillation:.4f}, cd_rosc={f.cd_relative_oscillation:.4f}')
    is_ok = f.forces_stabilized and f.force_oscillation_acceptable
    print(f'  FORCE CHECK PASSES: {is_ok}')
