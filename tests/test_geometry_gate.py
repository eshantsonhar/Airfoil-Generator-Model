import numpy as np

from airfoil_discovery.aso.cst import CSTBounds
from airfoil_discovery.aso.mesh_deform import validate_geometric_integrity


def test_default_initial_design_passes_structural_thickness_gate() -> None:
    dv = np.array([
        0.18, 0.28, 0.34, 0.25, 0.15, 0.08,
        -0.19, -0.12, -0.09, -0.05, -0.02, -0.01,
    ])

    is_valid, reason = validate_geometric_integrity(
        dv,
        te_thickness=CSTBounds.default().te_thickness,
        min_thickness_fraction=0.02,
    )

    assert is_valid, reason
