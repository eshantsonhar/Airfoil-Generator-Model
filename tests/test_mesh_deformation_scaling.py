from pathlib import Path

import numpy as np

from src.airfoil_discovery.aso.mesh_deform import write_surface_positions_file


def test_surface_positions_file_enforces_minimum_target_displacement(tmp_path):
    mesh_path = Path("data/mesh_fixed.su2")
    dv_old = np.array([
        0.18, 0.28, 0.34, 0.25, 0.15, 0.08,
        -0.19, -0.12, -0.09, -0.05, -0.02, -0.01,
    ])
    dv_new = dv_old.copy()
    dv_new[0] += 1e-4

    output_path = tmp_path / "surface_positions.dat"
    max_target_displacement = write_surface_positions_file(
        mesh_path=mesh_path,
        dv_old=dv_old,
        dv_new=dv_new,
        output_path=output_path,
        marker="airfoil",
        te_thickness=0.003,
    )

    assert max_target_displacement >= 1e-4
