from pathlib import Path
root = Path('.')
mesh = root / 'data' / 'cache' / 'final_test' / 'airfoil_perfect.su2'
print('mesh exists', mesh.exists())
text = mesh.read_text(errors='replace')
lines = text.splitlines()
print('total lines', len(lines))
markers = [(i + 1, line) for i, line in enumerate(lines) if 'MARKER' in line.upper()]
print('marker lines:', len(markers))
for line_num, line in markers[:40]:
    print(line_num, line)

npo = None
ne = None
for i, line in enumerate(lines):
    s = line.strip().upper()
    if s.startswith('NPOIN2'):
        parts = line.split()
        if len(parts) > 1:
            npo = int(parts[1])
        print('NPOIN2 at', i + 1, 'value', npo)
    if s.startswith('NELEM2'):
        parts = line.split()
        if len(parts) > 1:
            ne = int(parts[1])
        print('NELEM2 at', i + 1, 'value', ne)
print('NPOIN2', npo, 'NELEM2', ne)

# Find first coordinate block after NPOIN2
coord_start = None
for i, line in enumerate(lines):
    if line.strip().upper().startswith('NPOIN2'):
        coord_start = i + 1
        break
if coord_start is not None and npo is not None:
    coords = []
    for line in lines[coord_start:coord_start + min(npo, 200)]:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                coords.append((float(parts[0]), float(parts[1])))
            except ValueError:
                pass
    print('parsed coords preview', coords[:5])
    if coords:
        xs = [x for x, y in coords]
        ys = [y for x, y in coords]
        print('coord head xmin', min(xs), 'xmax', max(xs), 'ymin', min(ys), 'ymax', max(ys))
