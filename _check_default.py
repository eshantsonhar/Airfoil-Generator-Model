import numpy as np
from pathlib import Path

# Check default run with ITER=30 at L0, and compare to expected behavior at L1 ITER=80
# by checking runs in cache if they have usable residuals.

h = Path('data/cache/iter_009_cand_00_aoa_+04p0/history.csv')
lines = h.read_text(encoding='utf-8').splitlines()
headers = [it.strip().strip('"') for it in lines[0].split(',')]
traces = {h2: [] for h2 in headers}
for line in lines[1:]:
    if not line.strip() or line.strip() == ',': continue
    lvs = [it.strip() for it in line.split(',')]
    for i, h2 in enumerate(headers):
        if i < len(lvs):
            try: traces[h2].append(float(lvs[i]))
            except: pass
rmsP = np.array(traces.get('rms[P]', []), dtype=float)
print(f"iter_009_cand_00 hist_len={len(rmsP)}, abs(rmsP[0])={abs(rmsP[0]):.3f}, abs(rmsP[-1])={abs(rmsP[-1]):.3f}")

h2 = Path('data/cache/iter_001_cand_00_aoa_+02p0/history.csv')
lines2 = h2.read_text(encoding='utf-8').splitlines()
headers2 = [it.strip().strip('"') for it in lines2[0].split(',')]
traces2 = {h2: [] for h2 in headers2}
for line in lines2[1:]:
    if not line.strip() or line.strip() == ',': continue
    lvs = [it.strip() for it in line.split(',')]
    for i, h2 in enumerate(headers2):
        if i < len(lvs):
            try: traces2[h2].append(float(lvs[i]))
            except: pass
rmsP2 = np.array(traces2.get('rms[P]', []), dtype=float)
print(f"iter_001_cand_00 hist_len={len(rmsP2)}, abs(rmsP2[0])={abs(rmsP2[0]):.3f}, abs(rmsP2[-1])={abs(rmsP2[-1]):.3f}")

# Check: threshold test with various cutoff values
print("\nThreshold analysis:")
print(f"  abs(rms[-1]) < 1.0: {abs(rmsP[-1]) < 1.0}")
print(f"  abs(rms[-1]) < 2.0: {abs(rmsP[-1]) < 2.0}")
print(f"  abs(rms[-1]) < 5.0: {abs(rmsP[-1]) < 5.0}")
print(f"  abs(rms[-1]) < 10.0: {abs(rmsP[-1]) < 10.0}")
