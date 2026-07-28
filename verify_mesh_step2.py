"""
Step 2A: Zero-Trust Mesh Validation
Reads airfoil_perfect.su2 directly from disk to confirm geometry, 
boundary markers, and element sanity before proceeding.
"""

import os
import sys

MESH_FILENAME = "airfoil_perfect.su2"

def validate_su2_mesh(filename):
    print(f"=== Starting Zero-Trust Inspection of '{filename}' ===")

    # 1. Check file existence
    if not os.path.exists(filename):
        print(f"❌ ERROR: File '{filename}' not found in current directory!")
        sys.exit(1)

    print(f"✔ File '{filename}' located. (Size: {os.path.getsize(filename) / (1024*1024):.2f} MB)")

    # 2. Inspect Mesh Headers & Markers
    markers = {}
    ndime = None
    nelem = None
    npoin = None
    
    with open(filename, 'r') as f:
        for line in f:
            line_str = line.strip()
            if line_str.startswith("NDIME="):
                ndime = int(line_str.split("=")[1])
            elif line_str.startswith("NELEM="):
                nelem = int(line_str.split("=")[1])
            elif line_str.startswith("NPOIN="):
                npoin = int(line_str.split("=")[1])
            elif line_str.startswith("MARKER_TAG="):
                tag = line_str.split("=")[1].strip()
                nelems_marker = int(next(f).split("=")[1])
                markers[tag] = nelems_marker

    # 3. Print Diagnostic Summary
    print("\n--- Mesh Diagnostics ---")
    print(f"Spatial Dimensions (NDIME): {ndime}")
    print(f"Total Elements     (NELEM): {nelem}")
    print(f"Total Points       (NPOIN): {npoin}")
    print(f"Boundary Markers Found ({len(markers)}):")
    for tag, count in markers.items():
        print(f"  • {tag}: {count} elements")

    # 4. Critical Gate Checks
    errors = []
    if ndime != 2:
        errors.append(f"Expected NDIME=2, got {ndime}")
    if nelem is None or nelem <= 0:
        errors.append("Invalid or missing element count")
    if len(markers) == 0:
        errors.append("No boundary markers (MARKER_TAG) found in file")

    if errors:
        print("\n❌ MESH VALIDATION FAILED:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("\n✅ MESH PASSED VALIDATION! Ready to generate SU2 configuration.\n")
        return list(markers.keys())

if __name__ == "__main__":
    valid_markers = validate_su2_mesh(MESH_FILENAME)