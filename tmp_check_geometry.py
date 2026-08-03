import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve() / 'src'))
from airfoil_discovery.aso.cst import CSTBounds, compute_surface_coordinates
from airfoil_discovery.aso.mesh_deform import validate_geometric_integrity

dv = np.array([0.18, 0.28, 0.34, 0.25, 0.15, 0.08, -0.19, -0.12, -0.09, -0.05, -0.02, -0.01])
print(validate_geometric_integrity(dv, te_thickness=CSTBounds.default().te_thickness))
upper, lower = compute_surface_coordinates(dv, te_thickness=CSTBounds.default().te_thickness)
thickness = upper[:, 1] - lower[:, 1]
print('min', thickness.min())
print('mid', thickness[(upper[:, 0] >= 0.05) & (upper[:, 0] <= 0.95)].min())
print('struct', thickness[(upper[:, 0] >= 0.2) & (upper[:, 0] <= 0.8)].min())
print('max', thickness.max())
