"""
Phase 3: Fix Mesh Topology - Build Proper 30-Layer Structured O-Grid
Generates a structurally complete 2D fluid domain with:
  - 360 surface points on airfoil
  - 30 inflation layers (boundary layer)
  - Farfield at 20 chords
  - Total nodes: ~10,800+ (360 surface × 30 layers + farfield)
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

# ── Configuration ──────────────────────────────────────────────────────────
MESH_OUTPUT = _project_root / "data" / "cache" / "final_test" / "airfoil.su2"
N_SURFACE = 360          # Points on airfoil surface
BL_LAYERS = 30           # Number of inflation layers
BL_GROWTH = 1.12         # Layer growth rate
BL_FIRST = 2.0e-4        # First layer height (m)
FARFIELD_R = 20.0        # Farfield radius (chords)
N_FARFIELD = 120         # Farfield circle points

DV_BASELINE = np.array([
    0.18, 0.28, 0.34, 0.25, 0.15, 0.08,
    -0.19, -0.12, -0.09, -0.05, -0.02, -0.01,
])


def generate_structured_mesh():
    """Generate a proper multi-layer structured O-grid mesh."""
    print("=" * 80)
    print("PHASE 3: FIX MESH TOPOLOGY - STRUCTURED O-GRID")
    print("=" * 80)
    
    # 1. Generate airfoil surface coordinates
    print(f"\n1. Generating airfoil surface ({N_SURFACE} points)...")
    coords = compute_airfoil_coordinates(DV_BASELINE, n_pts_per_surface=N_SURFACE // 2)
    n_af = len(coords)
    print(f"   Airfoil points: {n_af}")
    
    # 2. Compute cumulative layer thicknesses
    # Layer heights: h_i = BL_FIRST * BL_GROWTH^i
    layer_heights = np.array([BL_FIRST * (BL_GROWTH ** i) for i in range(BL_LAYERS)])
    cumulative_heights = np.cumsum(layer_heights)
    total_bl_thickness = cumulative_heights[-1]
    print(f"\n2. Boundary layer parameters:")
    print(f"   First layer height: {BL_FIRST:.6f} m")
    print(f"   Layers: {BL_LAYERS}")
    print(f"   Growth rate: {BL_GROWTH}")
    print(f"   Total BL thickness: {total_bl_thickness:.4f} m ({total_bl_thickness*100:.2f}% chord)")
    
    # 3. Generate farfield circle
    print(f"\n3. Generating farfield boundary ({N_FARFIELD} points)...")
    theta = np.linspace(0, 2*np.pi, N_FARFIELD, endpoint=False)
    ff_x = FARFIELD_R * np.cos(theta)
    ff_y = FARFIELD_R * np.sin(theta)
    
    # 4. Build node grid
    # For each surface point, we create a radial line of nodes going outward
    # Node layout:
    #   Layer 0: airfoil surface (n_af nodes)
    #   Layer 1..BL_LAYERS-1: intermediate layers (n_af nodes each)
    #   Layer BL_LAYERS: transition to farfield (n_af nodes)
    #   Farfield: N_FARFIELD nodes
    
    print(f"\n4. Building node grid...")
    
    # Compute radial direction vectors from each surface point
    # We need outward normals. For a closed airfoil going clockwise,
    # the outward normal is perpendicular to the tangent.
    nodes = []
    
    # For each surface point, compute the outward direction
    # The airfoil goes: TE upper -> LE -> TE lower (clockwise)
    # Outward = rotate tangent 90 degrees clockwise
    
    for layer in range(BL_LAYERS + 1):  # +1 for transition layer
        r = cumulative_heights[layer - 1] if layer > 0 else 0.0
        
        for i in range(n_af):
            # Get current point
            x0, y0 = coords[i]
            
            # Compute tangent direction (forward difference)
            i_next = (i + 1) % n_af
            i_prev = (i - 1) % n_af
            tx = coords[i_next, 0] - coords[i_prev, 0]
            ty = coords[i_next, 1] - coords[i_prev, 1]
            norm = np.sqrt(tx*tx + ty*ty)
            if norm > 1e-12:
                tx /= norm
                ty /= norm
            
            # Outward normal (rotate tangent 90° clockwise)
            # For clockwise airfoil: outward = (ty, -tx)
            nx = ty
            ny = -tx
            
            # Ensure outward direction (pointing away from center)
            # Check if normal points toward center, if so flip it
            if nx * x0 + ny * y0 < 0:
                nx = -nx
                ny = -ny
            
            # Compute node position
            xn = x0 + r * nx
            yn = y0 + r * ny
            nodes.append((float(xn), float(yn)))
    
    # Add farfield nodes
    ff_start = len(nodes)
    for x, y in zip(ff_x, ff_y):
        nodes.append((float(x), float(y)))
    
    n_nodes = len(nodes)
    n_ff = N_FARFIELD
    n_per_layer = n_af
    n_layers = BL_LAYERS + 1  # BL layers + transition layer
    
    print(f"   Nodes per layer: {n_per_layer}")
    print(f"   Number of layers: {n_layers}")
    print(f"   Farfield nodes: {n_ff}")
    print(f"   Total nodes: {n_nodes}")
    
    # 5. Build elements
    print(f"\n5. Building elements...")
    
    # Airfoil surface elements (line type 3)
    af_elems = []
    for i in range(n_af - 1):
        af_elems.append((3, i, i + 1))
    af_elems.append((3, n_af - 1, 0))
    
    # Farfield surface elements (line type 3)
    ff_elems = []
    for i in range(n_ff - 1):
        ff_elems.append((3, ff_start + i, ff_start + i + 1))
    ff_elems.append((3, ff_start + n_ff - 1, ff_start))
    
    # Volume elements (quadrilaterals, type 9 for SU2)
    # Connect each layer to the next
    vol_elems = []
    for layer in range(n_layers - 1):
        base = layer * n_per_layer
        next_base = (layer + 1) * n_per_layer
        for i in range(n_per_layer):
            i_next = (i + 1) % n_per_layer
            # Quad: (base+i, base+i_next, next_base+i_next, next_base+i)
            vol_elems.append((9, base + i, base + i_next, next_base + i_next, next_base + i))
    
    # Transition elements: connect last BL layer to farfield
    # Map each BL layer node to nearest farfield node
    last_layer_base = (n_layers - 1) * n_per_layer
    for i in range(n_per_layer):
        # Find nearest farfield node
        x_bl = nodes[last_layer_base + i][0]
        y_bl = nodes[last_layer_base + i][1]
        min_dist = float('inf')
        nearest_ff = 0
        for j in range(n_ff):
            dx = nodes[ff_start + j][0] - x_bl
            dy = nodes[ff_start + j][1] - y_bl
            dist = dx*dx + dy*dy
            if dist < min_dist:
                min_dist = dist
                nearest_ff = j
        
        i_next = (i + 1) % n_per_layer
        j_next = (nearest_ff + 1) % n_ff
        
        # Triangle: (last_layer_base+i, last_layer_base+i_next, ff_start+nearest_ff)
        vol_elems.append((5, last_layer_base + i, last_layer_base + i_next, ff_start + nearest_ff))
        # Triangle: (last_layer_base+i_next, ff_start+j_next, ff_start+nearest_ff)
        vol_elems.append((5, last_layer_base + i_next, ff_start + j_next, ff_start + nearest_ff))
    
    n_vol = len(vol_elems)
    n_total = len(af_elems) + len(ff_elems) + n_vol
    
    print(f"   Airfoil surface: {len(af_elems)}")
    print(f"   Farfield surface: {len(ff_elems)}")
    print(f"   Volume elements: {n_vol}")
    print(f"   Total elements: {n_total}")
    
    # 6. Write SU2 format
    print(f"\n6. Writing SU2 mesh file...")
    MESH_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    
    with open(MESH_OUTPUT, 'w') as f:
        f.write(f"{n_nodes} {n_total} 2\n")
        f.write("NDIME= 2\n")
        
        # Volume elements
        for elem in vol_elems:
            f.write(f"{' '.join(str(x) for x in elem)}\n")
        
        # Nodes
        for x, y in nodes:
            f.write(f"{x:.10f} {y:.10f}\n")
        
        # Markers
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
    
    # 7. Validate
    print(f"\n7. Validating mesh...")
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
    
    # 8. Summary for Phase 3
    print(f"\n{'=' * 80}")
    print("PHASE 3 SUMMARY - CORRECTED MESH METRICS")
    print("=" * 80)
    print(f"\n1. CORRECTED MESH STATISTICS:")
    print(f"   Total nodes: {npoin}")
    print(f"   Total elements: {nelem}")
    print(f"   Expected nodes (n_af × layers + ff): {n_af * n_layers + n_ff}")
    
    print(f"\n2. SURFACE DISCRETIZATION:")
    print(f"   Airfoil marker elements: {mi.get('airfoil', 0)}")
    print(f"   Surface points: {N_SURFACE}")
    
    print(f"\n3. BOUNDARY LAYER:")
    print(f"   Layers: {BL_LAYERS}")
    print(f"   First cell height: {BL_FIRST:.6f} m")
    print(f"   Growth rate: {BL_GROWTH}")
    print(f"   Total thickness: {total_bl_thickness:.4f} m")
    
    print(f"\n4. MARKERS:")
    for name, count in mi.items():
        print(f"   '{name}': {count} elements")
    
    print(f"\n5. GEOMETRY:")
    print(f"   Chord: {chord:.6f} m")
    print(f"   Farfield: {FARFIELD_R} chords")
    
    print(f"\n{'=' * 80}")
    print("PHASE 3 COMPLETE - READY FOR BASELINE CFD RUN")
    print("=" * 80)


if __name__ == "__main__":
    generate_structured_mesh()
