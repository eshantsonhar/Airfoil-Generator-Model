from pathlib import Path
import re

root = Path('.')
mesh = root / 'data' / 'cache' / 'final_test' / 'airfoil_perfect.su2'
print('mesh exists', mesh.exists())
text = mesh.read_text(errors='replace')
lines = text.splitlines()
print('total lines', len(lines))
for i, line in enumerate(lines[:60], start=1):
    if any(tok in line.upper() for tok in ['NDIME', 'NELEM', 'NPOIN2', 'NELEM2', 'NMARK', 'MARKER']):
        print(i, line)

for i, line in enumerate(lines[60:300], start=61):
    if line.strip().startswith('NPOIN2') or line.strip().startswith('NELEM2'):
        print('found at', i, line)
        break

npoints = None
for i, line in enumerate(lines):
    if line.strip().startswith('NPOIN2'):
        parts=line.strip().split()
        if len(parts) > 1:
            npoints=int(parts[1])
        print('NPOIN2 at', i+1, 'npoints=', npoints)
        pos = i+1
        break

if npoints is not None:
    coords=[]
    for line in lines[pos:pos+npoints]:
        parts=line.strip().split()
        if len(parts) >= 2:
            try:
                coords.append((float(parts[0]), float(parts[1])))
            except Exception:
                pass
    print('parsed coords count', len(coords))
    if coords:
        xs = [x for x,y in coords]
        ys = [y for x,y in coords]
        print('xmin', min(xs), 'xmax', max(xs), 'ymin', min(ys), 'ymax', max(ys))

hist = root / 'aso_results_final' / 'cfd_cases' / 'eval_1785335113' / 'history.csv'
print('history exists', hist.exists())
if hist.exists():
    hist_text = hist.read_text(errors='replace')
    hist_lines = hist_text.splitlines()
    print('history total lines', len(hist_lines))
    for i, line in enumerate(hist_lines[:15], start=1):
        print('H', i, line)
    print('--- last 10 ---')
    for i, line in enumerate(hist_lines[-10:], start=1):
        print('H', len(hist_lines)-10+i, line)
