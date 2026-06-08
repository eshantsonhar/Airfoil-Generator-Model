import math
import numpy as np
from pathlib import Path

h = Path('data/failures/iter_001_aoa_+02p0/history.csv')
text = h.read_text(encoding='utf-8')
lines = text.splitlines()
headers = [it.strip().strip('"') for it in lines[0].split(',')]
traces = {h2: [] for h2 in headers}
for line in lines[1:]:
    if not line.strip() or line.strip() == ',': continue
    lvs = [it.strip() for it in line.split(',')]
    for i, h2 in enumerate(headers):
        if i < len(lvs):
            try: traces[h2].append(float(lvs[i]))
            except: pass
cl = np.array(traces.get('CL', []))
cd = np.array(traces.get('CD', []))

print("L1 target check (n=30, Ideally sw=10 like L1/ITER=30):\n")
sw = min(10, len(cl))
rc = cl[-sw:]
cc = cd[-sw:]
cl_mean = float(np.mean(rc))
cd_mean = float(np.mean(cc))
# Oscillation check centered on mean of sw subset
cl_std_sw = float(np.std(rc))
cd_std_sw = float(np.std(cc))
cl_rel_sw = cl_std_sw / (abs(cl_mean) + 1e-15)
cd_rel_sw = cd_std_sw / (abs(cd_mean) + 1e-15)
# Stabilization centered on mean
cl_change = abs(rc[-1] - cl_mean) / (abs(cl_mean) + 1e-15)
cd_change = abs(cc[-1] - cd_mean) / (abs(cd_mean) + 1e-15)
print(f"With sw=10 subset:")
print(f"  cl_mean={cl_mean:.6f}, cd_mean={cd_mean:.6f}")
print(f"  cl_std_sw={cl_std_sw:.6f}, cd_std_sw={cd_std_sw:.6f}")
print(f"  cl_rel_osc={cl_rel_sw:.4f} (<0.01={cl_rel_sw<0.01}), cd_rel_osc={cd_rel_sw:.4f} (<0.01={cd_rel_sw<0.01})")
print(f"  cl_change={cl_change:.6f} (<0.005={cl_change<0.005}), cd_change={cd_change:.6f} (<0.005={cd_change<0.005})")
print(f"\nWith sw=10, stabilization PASS: {cl_change<0.005 and cd_change<0.005}")
