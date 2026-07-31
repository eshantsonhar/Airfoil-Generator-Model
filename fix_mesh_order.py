#!/usr/bin/env python3
"""
Fix SU2 mesh section ordering to comply with SU2 format requirements.
Correct order: NDIME → NPOIN → NELEM → NMARK
"""

def fix_su2_mesh_order(input_path, output_path):
    """
    Reorder SU2 mesh sections to correct format.
    """
    with open(input_path, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    # Find section boundaries
    ndime_idx = None
    nelem_idx = None
    npoin_idx = None
    nmark_idx = None
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('NDIME='):
            ndime_idx = i
        elif stripped.startswith('NELEM='):
            nelem_idx = i
        elif stripped.startswith('NPOIN='):
            npoin_idx = i
        elif stripped.startswith('NMARK='):
            nmark_idx = i
    
    if None in [ndime_idx, nelem_idx, npoin_idx, nmark_idx]:
        print(f"Error: Could not find all required sections. Indices: NDIME={ndime_idx}, NELEM={nelem_idx}, NPOIN={npoin_idx}, NMARK={nmark_idx}")
        return []
    
    # Extract sections based on their positions in the file
    # Original order appears to be: NDIME, NELEM, NPOIN, NMARK
    # We need: NDIME, NPOIN, NELEM, NMARK
    
    # NDIME section (just the header line)
    ndime_section = [lines[ndime_idx]]
    
    # NELEM section (header + data until NPOIN)
    nelem_section = lines[nelem_idx:npoin_idx]
    
    # NPOIN section (header + data until NMARK)
    npoin_section = lines[npoin_idx:nmark_idx]
    
    # NMARK section (header + data until end)
    nmark_section = lines[nmark_idx:]
    
    # Write in correct order: NDIME, NPOIN, NELEM, NMARK
    with open(output_path, 'w') as f:
        f.write('\n'.join(ndime_section + npoin_section + nelem_section + nmark_section))
    
    print(f"Fixed mesh written to {output_path}")
    
    # Extract and print marker names
    marker_names = []
    for line in nmark_section:
        stripped = line.strip()
        if stripped.startswith('MARKER_TAG='):
            marker_name = stripped.split('=')[1].strip()
            marker_names.append(marker_name)
    
    print(f"Boundary markers found: {marker_names}")
    return marker_names

if __name__ == "__main__":
    input_mesh = r"c:\Eshant_Sonhar\airfoil research paper\airfoil generator model\data\pilot_run\airfoil.su2"
    output_mesh = r"c:\Eshant_Sonhar\airfoil research paper\airfoil generator model\data\mesh_fixed.su2"
    
    markers = fix_su2_mesh_order(input_mesh, output_mesh)
    print(f"\nVerification complete. Markers: {markers}")
