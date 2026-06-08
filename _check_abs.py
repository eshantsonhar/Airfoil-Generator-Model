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

rmsP = np.array(traces.get('rms[P]', []), dtype=float)
abs_rmsP = np.abs(rmsP)

print('Raw rms[P]:', [f'{v:.3f}' for v in rmsP])
print('abs(rms[P]):', [f'{v:.3f}' for v in abs_rmsP])

# Check per-iteration delta
deltas = np.abs(np.diff(abs_rmsP))
print('Delta abs (iteration-to-iteration):', [f'{v:.4f}' for v in deltas])
print(f'Mean delta: {np.mean(deltas):.4f}, Max delta: {np.max(deltas):.4f}')

# Alternative: check if abs(rms) monotonically DECREASES
# (convergence means residual magnitude always shrinking)
monotonic = bool(np.all(np.diff(abs_rmsP) <= 0))
print(f'abs(rms) is monotonically decreasing: {monotonic}')
improvements = np.diff(abs_rmsP)
improvements_positive = np.sum(improvements < 0)
print(f'Improvements (abs decrease): {improvements_positive}/{len(improvements)}')

# Check if abs(rms) decreased overall
overall_reduction = abs_rmsP[-1] - abs_rmsP[0]
print(f'Overall abs change: {overall_reduction:.4f} (negative = converging)')

# What threshold would make this pass?
print(f'Final abs(rms): {abs_rmsP[-1]:.4f}')
print(f'Passes abs < 5.0: {abs_rmsP[-1] < 5.0}')
print(f'Passes abs < 7.0: {abs_rmsP[-1] < 7.0}')
print(f'Passes abs < 10.0: {abs_rmsP[-1] < 10.0}')
