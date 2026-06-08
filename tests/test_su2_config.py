from __future__ import annotations

import numpy as np

from airfoil_discovery.cfd.su2_config import build_stage_config
from airfoil_discovery.config import load_settings
from airfoil_discovery.schemas import CandidateDesign, CSTParameters


def test_stage_config_uses_su2_comment_syntax(tmp_path) -> None:
    settings = load_settings("config/default.yaml")
    candidate = CandidateDesign(
        params=CSTParameters(
            upper=np.array([0.1, 0.2, 0.1, 0.0], dtype=float),
            lower=np.array([-0.1, -0.05, -0.02, 0.0], dtype=float),
            trailing_edge_thickness=0.004,
        ),
        reynolds=25000.0,
    )

    mesh_path = tmp_path / "airfoil.su2"
    mesh_path.write_text("dummy", encoding="utf-8")

    config = build_stage_config(1, candidate, mesh_path, aoa=2.0, settings=settings)

    assert "% stage: 1" in config
    assert "// stage: 1" not in config
    assert all(line.startswith("%") or "=" in line or not line for line in config.splitlines())
