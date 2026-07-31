import re
data = open(r'c:\Eshant_Sonhar\airfoil research paper\airfoil generator model\data\mesh_highres_fixed.su2').read()
lines = data.split('\n')

# Find NPOIN line
npoin_line = None
for i, line in enumerate(lines):
    if line.startswith('NPOIN='):
        npoin_line = i
        break

npoin = int(lines[npoin_line].split('=')[1])
node_start = npoin_line + 1

# First 199 nodes are the airfoil surface
af_coords = []
for i in range(node_start, node_start + 199):
    parts = lines[i].split()
    af_coords.append((float(parts[0]), float(parts[1])))
xs = [c[0] for c in af_coords]
ys = [c[1] for c in af_coords]
print(f'Airfoil chord: {max(xs) - min(xs):.6f} m')
print(f'X range: [{min(xs):.6f}, {max(xs):.6f}]')
print(f'Y range: [{min(ys):.6f}, {max(ys):.6f}]')
