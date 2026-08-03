from pathlib import Path

import numpy as np
import pytest

from src.airfoil_discovery.aso import mesh_deform
from src.airfoil_discovery.aso.optimizer import _parse_history


def test_parse_history_falls_back_to_surface_flow_csv(tmp_path: Path) -> None:
    surface_flow = tmp_path / "surface_flow.csv"
    surface_flow.write_text("CL,CD\n1.2345,0.0567\n", encoding="utf-8")

    cl, cd, converged = _parse_history(tmp_path / "history.csv")

    assert cl == pytest.approx(1.2345)
    assert cd == pytest.approx(0.0567)
    assert converged is False


def test_write_surface_positions_preserves_small_displacements(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mesh_path = tmp_path / "mesh.su2"
    mesh_path.write_text("NPOIN=2\n0 0\n1 0\n", encoding="utf-8")

    monkeypatch.setattr(mesh_deform, "_parse_su2_nodes", lambda path: np.array([[0.0, 0.0], [1.0, 0.0]]))
    monkeypatch.setattr(mesh_deform, "_parse_marker_node_ids", lambda path, marker: [0, 1])

    old_surface = {"x": np.array([0.0, 1.0]), "upper": np.array([0.0, 0.0]), "lower": np.array([-0.1, -0.1])}
    new_surface = {"x": np.array([0.0, 1.0]), "upper": np.array([1.0e-5, 1.0e-5]), "lower": np.array([-0.1, -0.1])}

    surfaces = {"old": old_surface, "new": new_surface}

    def fake_lookup(dv, te_thickness, n_pts: int = 1200):
        return surfaces["old" if np.allclose(dv, 0.0) else "new"]

    monkeypatch.setattr(mesh_deform, "_surface_y_lookup", fake_lookup)

    output_path = tmp_path / "surface_positions.dat"
    max_disp = mesh_deform.write_surface_positions_file(
        mesh_path=mesh_path,
        dv_old=np.zeros(12),
        dv_new=np.ones(12),
        output_path=output_path,
        marker="airfoil",
    )

    assert max_disp == pytest.approx(1.0e-5)
