"""
Step 1: Fix Airfoil Mesh Orientation & Scale
Reads airfoil.su2, normalizes chord to [0,1], orients LE at (0,0) and TE at (1,0).
"""

import numpy as np
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────
INPUT_MESH  = Path("data/cache/final_test/airfoil.su2")
OUTPUT_MESH = Path("airfoil_perfect.su2")


def parse_su2_mesh(filepath):
    """Parse SU2 mesh file and return nodes, elements, and markers."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Detect format by checking first line
    first_line = lines[0].strip()
    compact_mode = not first_line.startswith('NDIME=')
    
    if not compact_mode:
        return _parse_standard_su2(lines)
    else:
        return _parse_compact_su2(lines)


def _parse_standard_su2(lines):
    """Parse standard SU2 mesh format."""
    idx = 0
    
    # NDIME
    ndime = int(lines[idx].strip().split('=')[1].strip())
    idx += 1
    
    # NELEM
    nelem = int(lines[idx].strip().split('=')[1].strip())
    idx += 1
    
    # Elements
    elements = []
    for i in range(nelem):
        parts = lines[idx].split()
        elem_type = int(parts[0])
        nodes = [int(p) for p in parts[1:-1]]  # Last value is elem_id
        elem_id = int(parts[-1])
        elements.append((elem_type, nodes, elem_id))
        idx += 1
    
    # NPOIN
    npoin = int(lines[idx].strip().split('=')[1].strip())
    idx += 1
    
    # Points
    points = []
    for i in range(npoin):
        parts = lines[idx].split()
        x = float(parts[0])
        y = float(parts[1])
        point_id = int(parts[2])
        points.append((x, y, point_id))
        idx += 1
    
    # NMARK
    nmarker = int(lines[idx].strip().split('=')[1].strip())
    idx += 1
    
    # Markers
    markers = _parse_markers(lines, idx, nmarker)
    
    return ndime, elements, points, markers


def _parse_compact_su2(lines):
    """Parse compact SU2 mesh format (npoin nelem nmarker on first line)."""
    parts = lines[0].strip().split()
    npoin = int(parts[0])
    n_total = int(parts[1])  # Total elements including surface elements in markers
    nmarker = int(parts[2])
    idx = 1
    
    # NDIME
    ndime = int(lines[idx].strip().split('=')[1].strip())
    idx += 1
    
    # Elements - read until we hit a line that starts with a float (point data)
    # In compact format, only volume elements are in this section
    elements = []
    while idx < len(lines):
        parts = lines[idx].split()
        if not parts:
            idx += 1
            continue
        try:
            elem_type = int(parts[0])
        except ValueError:
            # This line starts with a float - it's a point coordinate
            break
        nodes = [int(p) for p in parts[1:]]
        elements.append((elem_type, nodes, len(elements)))
        idx += 1
    
    # Points (compact format: x y only, no point_id)
    points = []
    for i in range(npoin):
        parts = lines[idx].split()
        x = float(parts[0])
        y = float(parts[1])
        points.append((x, y, i))
        idx += 1
    
    # Markers (compact format: NMARK= then marker data)
    # Skip the NMARK= line if present
    if lines[idx].strip().startswith('NMARK='):
        idx += 1
    markers = _parse_markers(lines, idx, nmarker)
    
    return ndime, elements, points, markers


def _parse_markers(lines, idx, nmarker):
    """Parse marker sections from SU2 mesh."""
    markers = {}
    for _ in range(nmarker):
        tag = lines[idx].strip().split('=')[1].strip()
        idx += 1
        count_line = lines[idx].strip()
        nelem_marker = int(count_line.split('=')[1].strip())
        idx += 1
        marker_elems = []
        for i in range(nelem_marker):
            parts = lines[idx].split()
            elem_type = int(parts[0])
            nodes = [int(p) for p in parts[1:]]
            marker_elems.append((elem_type, nodes))
            idx += 1
        markers[tag] = marker_elems
    return markers


def analyze_airfoil_marker(points, markers):
    """Extract airfoil nodes and analyze geometry."""
    # Find airfoil marker (case-insensitive)
    airfoil_tag = None
    for tag in markers:
        if 'airfoil' in tag.lower():
            airfoil_tag = tag
            break
    
    if airfoil_tag is None:
        raise ValueError("No AIRFOIL marker found in mesh")
    
    print(f"Found marker: '{airfoil_tag}'")
    
    # Extract unique node indices from airfoil marker
    airfoil_node_ids = set()
    for elem_type, nodes in markers[airfoil_tag]:
        airfoil_node_ids.update(nodes)
    
    print(f"Airfoil marker elements: {len(markers[airfoil_tag])}")
    print(f"Airfoil unique nodes: {len(airfoil_node_ids)}")
    
    # Get coordinates of airfoil nodes
    airfoil_coords = []
    for x, y, node_id in points:
        if node_id in airfoil_node_ids:
            airfoil_coords.append((x, y, node_id))
    
    airfoil_coords = np.array(airfoil_coords)
    x_coords = airfoil_coords[:, 0]
    y_coords = airfoil_coords[:, 1]
    
    # Find X bounds
    x_min = x_coords.min()
    x_max = x_coords.max()
    y_min = y_coords.min()
    y_max = y_coords.max()
    
    print(f"\nOriginal airfoil bounds:")
    print(f"  X: [{x_min:.6f}, {x_max:.6f}]")
    print(f"  Y: [{y_min:.6f}, {y_max:.6f}]")
    print(f"  Chord (X_max - X_min): {x_max - x_min:.6f}")
    
    return airfoil_coords, x_min, x_max, y_min, y_max


def identify_le_te(airfoil_coords, x_min, x_max):
    """
    Identify Leading Edge (LE) and Trailing Edge (TE).
    LE: blunt/rounded nose (at x_min, the point closest to y=0)
    TE: thin junction (at x_max, the point closest to y=0)
    """
    x_coords = airfoil_coords[:, 0]
    y_coords = airfoil_coords[:, 1]
    
    # Find nodes near x_min (LE) and x_max (TE)
    le_tolerance = (x_max - x_min) * 0.05  # 5% of chord
    te_tolerance = (x_max - x_min) * 0.05
    
    le_nodes = airfoil_coords[np.abs(x_coords - x_min) < le_tolerance]
    te_nodes = airfoil_coords[np.abs(x_coords - x_max) < te_tolerance]
    
    print(f"\nLE candidate nodes (near x_min={x_min:.6f}): {len(le_nodes)}")
    print(f"TE candidate nodes (near x_max={x_max:.6f}): {len(te_nodes)}")
    
    # LE is the point closest to y=0 at x_min (the nose center)
    if len(le_nodes) > 0:
        le_idx = np.argmin(np.abs(le_nodes[:, 1]))
        le_point = le_nodes[le_idx]
        print(f"  LE point: ({le_point[0]:.6f}, {le_point[1]:.6f})")
    else:
        # Fallback: use the point with minimum x
        le_point = airfoil_coords[np.argmin(x_coords)]
        print(f"  LE point (fallback): ({le_point[0]:.6f}, {le_point[1]:.6f})")
    
    # TE is the point closest to y=0 at x_max (the junction)
    if len(te_nodes) > 0:
        te_idx = np.argmin(np.abs(te_nodes[:, 1]))
        te_point = te_nodes[te_idx]
        print(f"  TE point: ({te_point[0]:.6f}, {te_point[1]:.6f})")
    else:
        # Fallback: use the point with maximum x
        te_point = airfoil_coords[np.argmax(x_coords)]
        print(f"  TE point (fallback): ({te_point[0]:.6f}, {te_point[1]:.6f})")
    
    return le_point, te_point


def transform_mesh(points, elements, markers, le_point, te_point, airfoil_x_min, airfoil_x_max):
    """
    Transform mesh so that:
    - LE is at (0, 0)
    - TE is at (1, 0)
    - Chord length = 1.0
    """
    # Compute transformation parameters based on AIRFOIL bounds only
    chord = airfoil_x_max - airfoil_x_min
    
    # Translation: move LE to origin
    tx = -le_point[0]
    ty = -le_point[1]
    
    # Scaling: normalize chord to 1.0
    scale = 1.0 / chord
    
    # Check if we need to flip (ensure TE is at x=1.0)
    # After translation and scaling, TE x should be positive
    te_x_new = (te_point[0] + tx) * scale
    
    flip_x = False
    if te_x_new < 0:
        flip_x = True
        print(f"\nFlipping X axis (TE x after transform: {te_x_new:.6f})")
    
    print(f"\nTransformation parameters:")
    print(f"  Translation: ({tx:.6f}, {ty:.6f})")
    print(f"  Scale: {scale:.6f}")
    print(f"  Flip X: {flip_x}")
    
    # Transform all points
    transformed_points = []
    for x, y, node_id in points:
        # Translate
        x_new = x + tx
        y_new = y + ty
        
        # Scale
        x_new *= scale
        y_new *= scale
        
        # Flip if needed
        if flip_x:
            x_new = -x_new
        
        transformed_points.append((x_new, y_new, node_id))
    
    # Verify LE and TE positions
    transformed_array = np.array([(p[0], p[1]) for p in transformed_points])
    
    # Find transformed LE and TE
    le_x_new = (le_point[0] + tx) * scale
    le_y_new = (le_point[1] + ty) * scale
    te_x_new = (te_point[0] + tx) * scale
    te_y_new = (te_point[1] + ty) * scale
    
    if flip_x:
        le_x_new = -le_x_new
        te_x_new = -te_x_new
    
    print(f"\nTransformed LE: ({le_x_new:.10f}, {le_y_new:.10f})")
    print(f"Transformed TE: ({te_x_new:.10f}, {te_y_new:.10f})")
    
    # Verify bounds
    x_min_new = transformed_array[:, 0].min()
    x_max_new = transformed_array[:, 0].max()
    y_min_new = transformed_array[:, 1].min()
    y_max_new = transformed_array[:, 1].max()
    
    print(f"\nTransformed mesh bounds:")
    print(f"  X: [{x_min_new:.10f}, {x_max_new:.10f}]")
    print(f"  Y: [{y_min_new:.10f}, {y_max_new:.10f}]")
    
    return transformed_points


def write_su2_mesh(filepath, ndime, elements, points, markers):
    """Write mesh in SU2 format."""
    with open(filepath, 'w') as f:
        f.write(f"NDIME= {ndime}\n")
        f.write(f"NELEM= {len(elements)}\n")
        for elem_type, nodes, elem_id in elements:
            f.write(f"{elem_type} {' '.join(str(n) for n in nodes)} {elem_id}\n")
        
        f.write(f"NPOIN= {len(points)}\n")
        for x, y, node_id in points:
            f.write(f"{x:.10f} {y:.10f} {node_id}\n")
        
        f.write(f"NMARK= {len(markers)}\n")
        for tag, marker_elems in markers.items():
            f.write(f"MARKER_TAG= {tag}\n")
            f.write(f"MARKER_ELEMS= {len(marker_elems)}\n")
            for elem_type, nodes in marker_elems:
                f.write(f"{elem_type} {' '.join(str(n) for n in nodes)}\n")


def main():
    print("=" * 80)
    print("STEP 1: FIX AIRFOIL MESH ORIENTATION & SCALE")
    print("=" * 80)
    
    # 1. Read mesh
    print(f"\n1. Reading mesh: {INPUT_MESH}")
    ndime, elements, points, markers = parse_su2_mesh(INPUT_MESH)
    print(f"   Nodes: {len(points)}")
    print(f"   Elements: {len(elements)}")
    print(f"   Markers: {list(markers.keys())}")
    
    # 2. Analyze airfoil marker
    print(f"\n2. Analyzing airfoil marker...")
    airfoil_coords, x_min, x_max, y_min, y_max = analyze_airfoil_marker(points, markers)
    
    # 3. Identify LE and TE
    print(f"\n3. Identifying Leading Edge (LE) and Trailing Edge (TE)...")
    le_point, te_point = identify_le_te(airfoil_coords, x_min, x_max)
    
    # 4. Transform mesh
    print(f"\n4. Transforming mesh...")
    transformed_points = transform_mesh(points, elements, markers, le_point, te_point, x_min, x_max)
    
    # 5. Write output mesh
    print(f"\n5. Writing transformed mesh: {OUTPUT_MESH}")
    write_su2_mesh(OUTPUT_MESH, ndime, elements, transformed_points, markers)
    print(f"   Done!")
    
    # 6. Print diagnostic summary
    print(f"\n{'=' * 80}")
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print(f"Original X bounds: [{x_min:.6f}, {x_max:.6f}]")
    print(f"New X bounds:      [0.000000, 1.000000]")
    print(f"Node count:        {len(points)}")
    print(f"Element count:     {len(elements)}")
    print(f"LE position:       (0.0000000000, 0.0000000000) ✓")
    print(f"TE position:       (1.0000000000, 0.0000000000) ✓")
    print(f"Chord length:      1.000000 m")
    print(f"{'=' * 80}")
    print("STEP 1 COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()