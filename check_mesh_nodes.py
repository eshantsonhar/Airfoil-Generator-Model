lines = open('data/mesh_highres_fixed.su2').readlines()
npoin_idx = None
nelem_idx = None
for i, line in enumerate(lines):
    if line.startswith('NPOIN='):
        npoin_idx = i
    elif line.startswith('NELEM='):
        nelem_idx = i
        break

if npoin_idx is not None and nelem_idx is not None:
    node_lines = nelem_idx - npoin_idx - 1
    print(f'NPOIN line: {npoin_idx}')
    print(f'NELEM line: {nelem_idx}')
    print(f'Node lines between: {node_lines}')
    print(f'NPOIN value: {lines[npoin_idx].split("=")[1].strip()}')
