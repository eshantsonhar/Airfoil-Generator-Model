from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from airfoil_discovery.cfd.extractor import ExtractionError, PhysicsExtractor
from airfoil_discovery.cfd.mesh import build_geo_script, compute_mesh_parameters
from airfoil_discovery.cfd.multi_stage import _check_convergence_coefficients, _check_convergence_residual, compute_gci
from airfoil_discovery.cfd.su2_config import build_stage3_config
from airfoil_discovery.cfd.verifier import TransitionVerifier
from airfoil_discovery.config import load_settings
from airfoil_discovery.optimization.scoring import AirfoilScorer
from airfoil_discovery.reporting import ReportGenerator
from airfoil_discovery.schemas import CandidateDesign, CSTParameters, GeometryMetrics, PolarPoint
from airfoil_discovery.storage import ExperimentDatabase
from airfoil_discovery.validation import ValidationCase, ValidationFailedError, Validator


def _candidate() -> CandidateDesign:
    return CandidateDesign(
        params=CSTParameters(
            upper=np.array([0.18, 0.05, 0.34, 0.10]),
            lower=np.array([-0.19, 0.05, -0.09, 0.03]),
            trailing_edge_thickness=0.004,
        ),
        reynolds=30000.0,
        geometry_metrics=GeometryMetrics(
            max_thickness=0.1,
            max_camber=0.04,
            leading_edge_radius=0.01,
            smoothness_score=0.9,
            curvature_spike=10.0,
            prior_score=0.8,
            is_valid=True,
        ),
    )


def test_mesh_script_contains_required_low_re_parameters() -> None:
    settings = load_settings("config/default.yaml")
    coords = np.array([[1.0, 0.0], [0.5, 0.05], [0.0, 0.0], [0.5, -0.05]])
    script = build_geo_script(coords, 30000.0, settings.solver.mesh)
    assert "Point(5) = {-10.00, -10.00" in script
    assert "Point(6) = {20.00, -10.00" in script
    assert "Field[3].SizeMin = 0.0020;" in script
    assert "Field[4].XMax = 15.000;" in script
    params = compute_mesh_parameters(30000.0, settings.solver.mesh)
    assert 20 <= params.n_layers <= 60
    assert params.growth_rate <= 1.15


def test_stage3_config_uses_transition_and_farfield_only() -> None:
    settings = load_settings("config/default.yaml")
    config = build_stage3_config(_candidate(), Path("mesh.su2"), 4.0, settings, Path("stage2/restart_flow.dat"))
    assert "FREESTREAM_TURBULENCEINTENSITY= 0.001" in config
    assert "FREESTREAM_TURB_VISCOSITY_RATIO= 5.0" in config
    assert "MARKER_FAR= ( farfield )" in config
    assert "MARKER_INLET" not in config
    assert "MARKER_SUPERSONIC_INLET" not in config


def test_extractor_and_verifier_detect_transition_features(tmp_path: Path) -> None:
    surface_path = tmp_path / "surface_flow.json"
    x = np.linspace(0.0, 1.0, 220)
    y = np.concatenate([np.linspace(0.1, 0.0, 110), np.linspace(-0.1, 0.0, 110)])
    cp_upper = np.concatenate([np.linspace(-3.0, -2.9, 40), np.full(30, -2.9), np.linspace(-2.8, -0.6, 40)])
    cp_lower = np.linspace(0.2, -0.1, 110)
    cf_upper = np.concatenate([np.full(60, 0.01), np.full(25, -0.005), np.full(25, 0.008)])
    cf_lower = np.full(110, 0.006)
    gamma_upper = np.concatenate([np.zeros(70), np.linspace(0.2, 1.0, 40)])
    gamma_lower = np.zeros(110)
    payload = {
        "x": x.tolist(),
        "y": y.tolist(),
        "cp": np.concatenate([cp_upper, cp_lower]).tolist(),
        "cf": np.concatenate([cf_upper, cf_lower]).tolist(),
        "gamma": np.concatenate([gamma_upper, gamma_lower]).tolist(),
    }
    surface_path.write_text(json.dumps(payload), encoding="utf-8")
    extractor = PhysicsExtractor()
    dist = extractor.extract(surface_path, 4.0)
    assert len(dist.upper_cp) >= 200
    assert dist.x_tr is not None
    assert dist.x_sep is not None
    verifier = TransitionVerifier().verify(dist, 20000.0)
    assert verifier.lsb_detected


def test_extractor_raises_for_missing_surface_file(tmp_path: Path) -> None:
    extractor = PhysicsExtractor()
    try:
        extractor.extract(tmp_path / "missing.json", 4.0)
    except ExtractionError:
        return
    raise AssertionError("Expected ExtractionError")


def test_scorer_adds_surface_penalties() -> None:
    settings = load_settings("config/default.yaml")
    scorer = AirfoilScorer(settings.scoring)
    polar = [PolarPoint(aoa_deg=float(i), cl=0.1 * i, cd=0.02 + 0.001 * i) for i in range(0, 10, 2)]
    surface = type("Surface", (), {"bubble_length": 0.22, "cp_min": -4.8})()
    verification = type("Verification", (), {"physics_violation_penalty": 2.0})()
    scored = scorer.score(polar, surface, verification)
    assert scored["large_bubble_penalty"] > 0.0
    assert scored["suction_peak_penalty"] > 0.0
    assert scored["physics_violation_penalty"] == 2.0


def test_database_migration_and_report_generation(tmp_path: Path) -> None:
    db = ExperimentDatabase(tmp_path / "airfoil.sqlite")
    best = db.best_designs(limit=10)
    assert "transition_points" in db.metadata.tables
    assert "uq_runs" in db.metadata.tables
    assert "mis_results" in db.metadata.tables
    report = ReportGenerator(load_settings("config/default.yaml"), db).generate("testrun", tmp_path)
    text = report.read_text(encoding="utf-8")
    assert "# Abstract" in text
    assert "# Conclusions" in text


def test_validation_pass_and_failure_paths(tmp_path: Path) -> None:
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    case_payload = {
        "airfoil_name": "NACA0012",
        "reynolds": 10000,
        "aoa": [0, 2, 4],
        "cl": [0.0, 0.2, 0.4],
        "cd": [0.02, 0.025, 0.03],
    }
    (reference_dir / "case.json").write_text(json.dumps(case_payload), encoding="utf-8")
    settings = load_settings("config/default.yaml")
    validator = Validator(settings, reference_dir)
    report = validator.run()
    assert report.passed
    passed_path = tmp_path / "validation_passed.json"
    validator.write_passed_file(report, passed_path)
    written = json.loads(passed_path.read_text(encoding="utf-8"))
    assert written["cases"][0]["airfoil_name"] == "NACA0012"

    def bad_run(case: ValidationCase) -> dict[str, list[float]]:
        return {"cl": [10.0, 10.0, 10.0], "cd": [1.0, 1.0, 1.0]}

    bad_validator = Validator(settings, reference_dir, run_case=bad_run)
    failed_report = bad_validator.run()
    assert not failed_report.passed


def test_convergence_helpers_and_gci() -> None:
    history = [{"CL": 0.5 + 1e-4 * i, "CD": 0.02 + 1e-5 * i, "RMS_RES": 1e-1 / (10 ** (i / 20))} for i in range(120)]
    coeff_ok, _ = _check_convergence_coefficients(history, window=100, tol=0.05)
    resid_ok, _ = _check_convergence_residual(history, required_drop=4.0)
    assert coeff_ok
    assert resid_ok
    gci = compute_gci(1.0, 0.95, 0.92, 1.5)
    assert gci >= 0.0
