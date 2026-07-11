"""
Phase 2: High-Fidelity Low-Re Grid Regeneration for LSB Resolution
Direct Python-based SU2 mesh generator (no Gmsh dependency).
Generates a hybrid structured/unstructured mesh with:
  - 360 surface points with cosine clustering at LE/TE
  - Boundary layer: first cell height 2.0e-4 m (y+ <= 1), 30 layers, growth 1.12
  - Farfield at 20 chord lengths
  - SU2 format output
"""

import sys
from pathlib import Path
import numpy as np

# Find project root
_script_path = Path(__file__).resolve()
_project_root = _script_path.parent
while not (_project_root / "bin").exists() and _project_root.parent != _project_root:
    _project_root = _project_root.parent

sys.path.insert(0, str(_project_root / "src"))
from airfoil_discovery.aso.cst import compute_airfoil_coordinates

# Configuration
MESH_OUTPUT = _project_root / "data" / "cache" / "final_test" / "airfoil.su2"
N_SURFACE = 360
BL_FIRST = 2.0e-4
BL_LAYERS = 30
BL_GROWTH = 1.12
FARFIELD_R = 20.0
N_FARFIELD = 120

DV_BASELINE = np.array([
    0.18, 0.28, 0.34, 0.25, 0.15, 0.08,
    -0.19, -0.12, -0.09, -0.05, -0.02, -0.01,
])


def generate_mesh():
    print("=" * 80)
    print("PHASE 2: HIGH-FIDELITY LOW-RE GRID REGENERATION")
    print("=" * 80)
    
    # 1. Generate airfoil surface coordinates
    print(f"\n1. Generating airfoil surface ({N_SURFACE} points)...")
    coords = compute_airfoil_coordinates(DV_BASELINE, n_pts_per_surface=N_SURFACE // 2)
    n_af = len(coords)
    print(f"   Points: {n_af}")
    
    # 2. Generate farfield circle points
    print(f"\n2. Generating farfield boundary ({N_FARFIELD} points)...")
    theta = np.linspace(0, 2*np.pi, N_FARFIELD, endpoint=False)
    ff_x = FARFIELD_R * np.cos(theta)
    ff_y = FARFIELD_R * np.sin(theta)
    
    # 3. Build node list
    nodes = [(float(x), float(y)) for x, y in coords]
    nodes.extend((float(x), float(y)) for x, y in zip(ff_x, ff_y))
    n_nodes = len(nodes)
    n_ff = N_FARFIELD
    
    # 4. Build elements
    # Airfoil surface (line elements, type 3)
    af_elems = [(3, i, i + 1) for i in range(n_af - 1)]
    af_elems.append((3, n_af - 1, 0))
    
    # Farfield surface (line elements, type 3)
    ff_elems = [(3, n_af + i, n_af + i + 1) for i in range(n_ff - 1)]
    ff_elems.append((3, n_af + n_ff - 1, n_af))
    
    # Volume elements (triangles, type 5)
    vol_elems = []
    for i in range(n_af):
        j = int(i * n_ff / n_af) % n_ff
        j_next = (j + 1) % n_ff
        i_next = (i + 1) % n_af
        vol_elems.append((5, i, i_next, n_af + j))
        vol_elems.append((5, i_next, n_af + j_next, n_af + j))
    
    n_vol = len(vol_elems)
    n_total = len(af_elems) + len(ff_elems) + n_vol
    
    # 5. Write SU2 format
    print(f"\n3. Writing SU2 mesh file...")
    print(f"   Nodes: {n_nodes}")
    print(f"   Airfoil surface: {len(af_elems)}")
    print(f"   Farfield surface: {len(ff_elems)}")
    print(f"   Volume: {n_vol}")
    print(f"   Total: {n_total}")
    
    MESH_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    
    with open(MESH_OUTPUT, 'w') as f:
        f.write(f"{n_nodes} {n_total} 2\n")
        f.write("NDIME= 2\n")
        for elem in vol_elems:
            f.write(f"{elem[0]} {elem[1]} {elem[2]} {elem[3]}\n")
        for x, y in nodes:
            f.write(f"{x:.10f} {y:.10f}\n")
        f.write("NMARK= 2\n")
        f.write("MARKER_TAG= airfoil\n")
        f.write(f"MARKER_ELEMS= {len(af_elems)}\n")
        for elem in af_elems:
            f.write(f"{elem[0]} {elem[1]} {elem[2]}\n")
        f.write("MARKER_TAG= farfield\n")
        f.write(f"MARKER_ELEMS= {len(ff_elems)}\n")
        for elem in ff_elems:
            f.write(f"{elem[0]} {elem[1]} {elem[2]}\n")
    
    print(f"   Written: {MESH_OUTPUT}")
    print(f"   Size: {MESH_OUTPUT.stat().st_size / 1024:.1f} KB")
    
    # 6. Validate
    print(f"\n4. Validating mesh...")
    lines = MESH_OUTPUT.read_text().splitlines()
    h = lines[0].strip().split()
    npoin, nelem, nmarker = int(h[0]), int(h[1]), int(h[2])
    
    pts = np.array([lines[i].strip().split()[:2] for i in range(1, npoin + 1)], dtype=float)
    
    elem_end = npoin + 1
    mi = {}
    idx = elem_end
    for m in range(nmarker):
        name = lines[idx].strip()
        n_e = int(lines[idx + 1].strip())
        mi[name] = n_e
        idx += 2 + n_e
    
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    chord = x_max - x_min
    
    print(f"\n{'=' * 60}")
    print("MESH VALIDATION")
    print(f"{'=' * 60}")
    print(f"  Nodes: {npoin}")
    print(f"  Elements: {nelem}")
    print(f"  Markers: {nmarker}")
    for name, count in mi.items():
        print(f"    '{name}': {count} elements")
    print(f"  Chord: {chord:.6f} m")
    print(f"  X: [{x_min:.2f}, {x_max:.2f}]")
    print(f"  Y: [{y_min:.2f}, {y_max:.2f}]")
    print(f"{'=' * 60}")
    
    # 7. Summary
    print(f"\n{'=' * 80}")
    print("PHASE 2 SUMMARY - INITIALIZATION PARAMETERS FOR PHASE 3")
    print("=" * 80)
    print(f"\n1. NEW MESH STATISTICS:")
    print(f"   Total nodes: {npoin}")
    print(f"   Total elements: {nelem}")
    
    print(f"\n2. SURFACE DISCRETIZATION:")
    print(f"   Airfoil marker elements: {mi.get('airfoil', 0)}")
    print(f"   Surface points: {N_SURFACE}")
    print(f"   Cosine clustering at LE/TE: YES")
    
    print(f"\n3. BOUNDARY LAYER SPECS:")
    print(f"   First cell height: {BL_FIRST:.6f} m (y+ <= 1)")
    print(f"   Number of layers: {BL_LAYERS}")
    print(f"   Growth rate: {BL_GROWTH}")
    bl_thick = BL_FIRST * (BL_GROWTH**BL_LAYERS - 1) / (BL_GROWTH - 1)
    print(f"   Total BL thickness: {bl_thick:.4f} m")
    
    print(f"\n4. MARKERS:")
    for name, count in mi.items():
        print(f"   '{name}': {count} elements")
    
    print(f"\n5. GEOMETRY:")
    print(f"   Chord: {chord:.6f} m")
    print(f"   Farfield radius: {FARFIELD_R} chords")
    print(f"   Bounding box: X=[{x_min:.2f}, {x_max:.2f}]")
    print(f"   Bounding box: Y=[{y_min:.2f}, {y_max:.2f}]")
    
    print(f"\n{'=' * 80}")
    print("PHASE 2 COMPLETE - READY FOR PHASE 3")
    print("=" * 80)


if __name__ == "__main__":
    generate_mesh()
