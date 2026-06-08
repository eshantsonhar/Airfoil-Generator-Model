import sys
sys.path.insert(0,'.')
import numpy as np
from pathlib import Path
from airfoil_discovery.verification.convergence import ResidualConvergenceAnalyzer

h = Path('data/cache/iter_001_cand_00_aoa_+02p0/history.csv')
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

rmsP = traces.get('rms[P]', [])
n = len(rmsP)
print(f'History length: {n}')
if n < 5:
    print('Only header row')
else:
    print(f'First 5: {[round(v, 3) for v in rmsP[:5]]}')
    print(f'Last 5: {[round(v, 3) for v in rmsP[-5:]]}')
    initial = abs(rmsP[0])
    final = abs(rmsP[-1])
