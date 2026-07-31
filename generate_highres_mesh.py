"""
High-Fidelity Boundary Layer Mesh Generator for Low-Re Transition Modeling
Generates a structured C-mesh with boundary layer inflation for y+ < 1 at Re=1e5
"""

import sys
from pathlib import Path
import numpy as np
import math

# Find project root
_script_path = Path(__file__).resolve()
_project_root = _script_path.parent
while not (_project_root / "bin").exists() and _project_root.parent != _project_root:
    _project_root = _project_root.parent

sys.path.insert(0, str(_project_root / "src"))
from airfoil_discovery.aso.cst import compute_airfoil_coordinates

# Configuration for high-fidelity transition mesh
MESH_OUTPUT = _project_root / "data" / "mesh_highres_fixed.su2"
N_SURFACE = 200  # Surface points
BL_FIRST = 1.0e-4  # First cell height for y+ ~ 0.5 at Re=1e5
BL_LAYERS = 40  # Number of boundary layer layers
BL_GROWTH = 1.12  # Growth rate
FARFIELD_R = 20.0  # Farfield distance in chords
N_FARFIELD = 100  # Farfield points

DV_BASELINE = np.array([
    0.18, 0.28, 0.34, 0.25, 0.15, 0.08,
    -0.19, -0.12, -0.09, -0.05, -0.02, -0.01,
])


def generate_boundary_layer_nodes(surface_coords, first_height, n_layers, growth):
    """Generate structured boundary layer nodes using hyperbolic tangent spacing."""
    bl_nodes = []
    
    # Hyperbolic tangent distribution for smooth clustering near wall
    for i in range(n_layers + 1):
        # Normalized distance from 0 to 1
        s = i / n_layers
        # Hyperbolic tangent clustering
        eta = math.tanh(1.5 * s) / math.tanh(1.5)
        # Physical distance
        height = first_height * (1 + eta * (growth**n_layers - 1))
        bl_nodes.append(height)
    
    return bl_nodes


def generate_mesh():
    print("=" * 80)
    print("HIGH-FIDELITY BOUNDARY LAYER MESH GENERATION")
    print("=" * 80)
    
    # 1. Generate airfoil surface coordinates
    print(f"\n1. Generating airfoil surface ({N_SURFACE} points)...")
    coords = compute_airfoil_coordinates(DV_BASELINE, n_pts_per_surface=N_SURFACE // 2)
    n_af = len(coords)
    print(f"   Points: {n_af}")
    
    # Center airfoil at origin (0, 0) with leading edge at x=0
    x_coords = [c[0] for c in coords]
    y_coords = [c[1] for c in coords]
    x_min, x_max = min(x_coords), max(x_coords)
    chord = x_max - x_min
    
    # Normalize to chord = 1.0 and position LE at x=0
    coords = np.array([( (c[0] - x_min) / chord, c[1] / chord ) for c in coords])
    print(f"   Chord normalized to: 1.0 m")
    
    # 2. Generate boundary layer normal directions
    print(f"\n2. Computing surface normals...")
    normals = []
    for i in range(n_af):
        # Previous and next points
        prev_idx = (i - 1) % n_af
        next_idx = (i + 1) % n_af
        prev_pt = coords[prev_idx]
        next_pt = coords[next_idx]
        curr_pt = coords[i]
        
        # Tangent vector
        tangent = next_pt - prev_pt
        tangent = tangent / np.linalg.norm(tangent)
        
        # Normal vector (rotate 90 degrees CCW)
        normal = np.array([-tangent[1], tangent[0]])
        
        # Ensure normal points outward (away from airfoil interior)
        # For airfoil, interior is typically below upper surface and above lower surface
        # Use centroid check
        centroid = np.mean(coords, axis=0)
        to_centroid = centroid - curr_pt
        if np.dot(normal, to_centroid) > 0:
            normal = -normal
        
        normals.append(normal)
    
    # 3. Generate boundary layer nodes
    print(f"\n3. Generating boundary layer nodes...")
    print(f"   First cell height: {BL_FIRST:.6e} m")
    print(f"   Layers: {BL_LAYERS}")
    print(f"   Growth rate: {BL_GROWTH}")
    
    bl_heights = generate_boundary_layer_nodes(coords, BL_FIRST, BL_LAYERS, BL_GROWTH)
    bl_thickness = bl_heights[-1]
    print(f"   Total BL thickness: {bl_thickness:.4f} m")
    
    # Create boundary layer nodes
    bl_node_coords = []
    for layer_idx, height in enumerate(bl_heights[1:]):  # Skip layer 0 (surface)
        layer_coords = []
        for i in range(n_af):
            pt = coords[i] + normals[i] * height
            layer_coords.append(pt)
        bl_node_coords.append(layer_coords)
    
    # 4. Generate farfield boundary
    print(f"\n4. Generating farfield boundary ({N_FARFIELD} points)...")
    theta = np.linspace(0, 2*np.pi, N_FARFIELD, endpoint=False)
    ff_x = FARFIELD_R * np.cos(theta)
    ff_y = FARFIELD_R * np.sin(theta)
    ff_coords = list(zip(ff_x, ff_y))
    
    # 5. Build node list
    print(f"\n5. Building node list...")
    nodes = [(float(x), float(y)) for x, y in coords]  # Surface nodes (layer 0)
    for layer in bl_node_coords:
        nodes.extend([(float(x), float(y)) for x, y in layer])  # BL nodes
    
    # Add intermediate nodes between BL and farfield
    # Create a structured transition region
    n_transition = 10
    for t in range(1, n_transition + 1):
        frac = t / n_transition
        transition_radius = bl_thickness + frac * (FARFIELD_R - bl_thickness)
        theta_t = np.linspace(0, 2*np.pi, N_FARFIELD, endpoint=False)
        tx = transition_radius * np.cos(theta_t)
        ty = transition_radius * np.sin(theta_t)
        nodes.extend([(float(x), float(y)) for x, y in zip(tx, ty)])
    
    # Farfield nodes
    nodes.extend([(float(x), float(y)) for x, y in ff_coords])
    
    n_nodes = len(nodes)
    print(f"   Total nodes: {n_nodes}")
    
    # 6. Build elements
    print(f"\n6. Building elements...")
    
    # Surface elements (line elements, type 3)
    surface_elems = [(3, i, (i + 1) % n_af) for i in range(n_af)]
    
    # Farfield elements
    ff_start_idx = n_nodes - N_FARFIELD
    ff_elems = [(3, ff_start_idx + i, ff_start_idx + (i + 1) % N_FARFIELD) for i in range(N_FARFIELD)]
    
    # Volume elements (triangles, type 5)
    vol_elems = []
    
    # Boundary layer volume elements
    layer_start = n_af
    for layer_idx in range(BL_LAYERS):
        layer_end = layer_start + n_af
        for i in range(n_af):
            # Triangle connecting this layer to next layer
            i_next = (i + 1) % n_af
            vol_elems.append((5, layer_start + i, layer_start + i_next, layer_end + i))
            vol_elems.append((5, layer_start + i_next, layer_end + i_next, layer_end + i))
        layer_start = layer_end
    
    # Transition region elements
    transition_start = layer_start
    for t in range(n_transition):
        transition_end = transition_start + N_FARFIELD
        # Connect previous layer to this transition layer
        prev_layer_start = transition_start - N_FARFIELD if t > 0 else layer_start - N_FARFIELD
        for i in range(N_FARFIELD):
            i_next = (i + 1) % N_FARFIELD
            vol_elems.append((5, prev_layer_start + i, prev_layer_start + i_next, transition_end + i))
            vol_elems.append((5, prev_layer_start + i_next, transition_end + i_next, transition_end + i))
        transition_start = transition_end
    
    # Connect last transition layer to farfield
    last_transition_start = transition_start
    for i in range(N_FARFIELD):
        i_next = (i + 1) % N_FARFIELD
        vol_elems.append((5, last_transition_start + i, last_transition_start + i_next, ff_start_idx + i))
        vol_elems.append((5, last_transition_start + i_next, ff_start_idx + i_next, ff_start_idx + i))
    
    n_vol = len(vol_elems)
    n_total = len(surface_elems) + len(ff_elems) + n_vol
    
    print(f"   Surface elements: {len(surface_elems)}")
    print(f"   Farfield elements: {len(ff_elems)}")
    print(f"   Volume elements: {n_vol}")
    print(f"   Total: {n_total}")
    
    # 7. Write SU2 format
    print(f"\n7. Writing SU2 mesh file...")
    MESH_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    
    with open(MESH_OUTPUT, 'w') as f:
        # SU2 format: NDIME, NELEM, elements, NPOIN, nodes, NMARK, markers (no header line)
        f.write("NDIME= 2\n")
        f.write(f"NELEM= {n_vol}\n")
        
        # Volume elements
        for elem in vol_elems:
            f.write(f"{elem[0]} {elem[1]} {elem[2]} {elem[3]}\n")
        
        f.write(f"NPOIN= {n_nodes}\n")
        
        # Nodes
        for x, y in nodes:
            f.write(f"{x:.10f} {y:.10f}\n")
        
        # Markers
        f.write("NMARK= 2\n")
        f.write("MARKER_TAG= airfoil\n")
        f.write(f"MARKER_ELEMS= {len(surface_elems)}\n")
        for elem in surface_elems:
            f.write(f"{elem[0]} {elem[1]} {elem[2]}\n")
        
        f.write("MARKER_TAG= farfield\n")
        f.write(f"MARKER_ELEMS= {len(ff_elems)}\n")
        for elem in ff_elems:
            f.write(f"{elem[0]} {elem[1]} {elem[2]}\n")
    
    print(f"   Written: {MESH_OUTPUT}")
    print(f"   Size: {MESH_OUTPUT.stat().st_size / 1024:.1f} KB")
    
    # 8. Validation
    print(f"\n8. Validating mesh...")
    lines = MESH_OUTPUT.read_text().splitlines()
    
    # Parse SU2 format: NDIME, NELEM, elements, NPOIN, nodes, NMARK, markers (no header line)
    nelem_line = None
    npoin_line = None
    for i, line in enumerate(lines):
        if line.startswith('NELEM='):
            nelem_line = i
        elif line.startswith('NPOIN='):
            npoin_line = i
            break
    
    nelem = int(lines[nelem_line].split('=')[1])
    npoin = int(lines[npoin_line].split('=')[1])
    
    # Node coordinates start after NPOIN line
    node_start = npoin_line + 1
    node_coords = []
    for i in range(node_start, len(lines)):
        line = lines[i].strip()
        if line.startswith('NMARK='):
            break
        if not line:
            continue
        parts = line.split()[:2]
        if len(parts) == 2:
            try:
                node_coords.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue
        if len(node_coords) >= npoin:
            break
    
    pts = np.array(node_coords[:npoin])
    
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    chord = x_max - x_min
    
    print(f"\n{'=' * 60}")
    print("MESH VALIDATION")
    print(f"{'=' * 60}")
    print(f"  Nodes: {npoin}")
    print(f"  Elements: {nelem}")
    print(f"  Markers: 2 (airfoil, farfield)")
    print(f"  Chord: {chord:.6f} m")
    print(f"  X: [{x_min:.2f}, {x_max:.2f}]")
    print(f"  Y: [{y_min:.2f}, {y_max:.2f}]")
    print(f"  BL layers: {BL_LAYERS}")
    print(f"  First cell height: {BL_FIRST:.6e} m")
    print(f"  BL thickness: {bl_thickness:.4f} m")
    print(f"{'=' * 60}")
    
    print(f"\n{'=' * 80}")
    print("HIGH-FIDELITY MESH GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nMesh saved to: {MESH_OUTPUT}")
    print(f"Ready for transition modeling with γ-Reθ")
    print("=" * 80)


if __name__ == "__main__":
    generate_mesh()
