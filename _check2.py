import numpy as np
from pathlib import Path
import sys

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
print(f"CL range: {cl.min():.4f}..{cl.max():.4f}, mean={cl.mean():.4f}")
print(f"CD range: {cd.min():.4f}..{cd.max():.4f}, mean={cd.mean():.4f}")
# Check var-vs-mean stabilization
cl_chg_vm = abs(cl[-1]-np.mean(cl))/(abs(np.mean(cl))+1e-15)
cd_chg_vm = abs(cd[-1]-np.mean(cd))/(abs(np.mean(cd))+1e-15)
print(f"var-vs-mean: cl_chg={cl_chg_vm:.4f}, cd_chg={cd_chg_vm:.4f}")
print(f"force_stab (var-vs-mean 0.005): {cl_chg_vm < 0.005 and cd_chg_vm < 0.005}")
# Check with var/mean threshold = 0.2
print(f"force_stab (0.2 threshold): {cl_chg_vm < 0.2 and cd_chg_vm < 0.2}")
# Check oscillation
cl_std, cd_std = np.std(cl), np.std(cd)
cl_mean, cd_mean = np.mean(cl), np.mean(cd)
cl_rosc = cl_std / (abs(cl_mean) + 1e-15)
cd_rosc = cd_std / (abs(cd_mean) + 1e-15)
print(f"oscilation: cl_rosc={cl_rosc:.4f}, cd_rosc={cd_rosc:.4f}")
# Check trend
if len(cl) > 10:
    sl_cl, _ = np.polyfit(np.arange(15,30), cl[15:], 1)
    sl_cd, _ = np.polyfit(np.arange(15,30), cd[15:], 1)
    print(f"Trend last 15: cl_slope={sl_cl:.6f}, cd_slope={sl_cd:.6f}")
    print(f"drift OK (<0.002): {abs(sl_cl)<0.002 and abs(sl_cd)<0.002}")
