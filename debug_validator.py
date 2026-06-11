"""Debug geometry validation issue for NACA 0012 coefficients."""
import sys
sys.path.insert(0, 'src')
from airfoil_discovery.geometry.cst import CSTAirfoil
from airfoil_discovery.schemas import CSTParameters
from airfoil_discovery.config import load_settings
from airfoil_discovery.geometry.validation import AirfoilGeometryValidator, GeometryValidationConfig
import numpy as np

s = load_settings('config/default.yaml')
a = CSTAirfoil(s.geometry)

# Our coefficients (scaled to t/c=0.12)
p = CSTParameters(
    upper=np.array([-0.050550, 0.778580, -1.175346, 0.783176]),
    lower=np.array([0.050550, -0.778580, 1.175346, -0.783176]),
    trailing_edge_thickness=0.001
)
c = a.full_coordinates(p)
print(f"Coords shape: {c.shape}")
print(f"x range: [{c[:,0].min():.4f}, {c[:,0].max():.4f}]")
print(f"y range: [{c[:,1].min():.4f}, {c[:,1].max():.4f}]")

v = AirfoilGeometryValidator(GeometryValidationConfig())
vr = v.validate_coordinates(c)
print(f"can_proceed_to_cfd: {vr.can_proceed_to_cfd}")
print(f"max_thickness: {vr.max_thickness}")
print(f"failure_reasons: {vr.failure_reasons}")
if hasattr(vr, 'violations'):
    for vv in vr.violations:
        print(f"Violation: {getattr(vv, 'value', vv)}")
print(f"\nConfig thickness_bounds: {s.geometry.thickness_bounds}")
print(f"Config camber_bounds: {s.geometry.camber_bounds}")