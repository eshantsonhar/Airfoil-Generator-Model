from __future__ import annotations

import numpy as np
import pytest

from airfoil_discovery.geometry.validation import (
    AirfoilGeometryValidator,
    GeometryValidationConfig,
    GeometryValidationStatus,
    GeometryValidationSuite,
    GeometryViolationType,
)


def naca_symmetric(thickness: float = 0.12, n_points: int = 100) -> np.ndarray:
    """Build a NACA 00xx airfoil in TE -> LE -> TE point ordering."""
    x = np.linspace(0.0, 1.0, n_points)
    half_thickness = 5 * thickness * (
        0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4
    )
    upper = np.column_stack([x[::-1], half_thickness[::-1]])
    lower = np.column_stack([x[1:], -half_thickness[1:]])
    return np.vstack([upper, lower])


def surfaces_to_coordinates(x: np.ndarray, y_upper: np.ndarray, y_lower: np.ndarray) -> np.ndarray:
    return np.vstack(
        [np.column_stack([x[::-1], y_upper[::-1]]), np.column_stack([x[1:], y_lower[1:]])]
    )


@pytest.fixture
def validator() -> AirfoilGeometryValidator:
    return AirfoilGeometryValidator()


def test_clean_airfoil_is_valid_and_metrics_are_physical(validator):
    result = validator.validate_coordinates(naca_symmetric())

    assert result.status is GeometryValidationStatus.VALID
    assert result.is_valid and result.can_proceed_to_cfd
    assert result.violations == []
    assert result.max_thickness == pytest.approx(0.12, abs=5e-3)
    assert result.max_camber == pytest.approx(0.0, abs=1e-9)
    assert validator.config.min_le_radius < result.le_radius < validator.config.max_le_radius


def test_none_coordinates_are_rejected_without_crashing(validator):
    result = validator.validate_coordinates(None)

    assert not result.is_valid
    assert result.failure_reasons == ["Coordinates are None"]
    assert result.coordinates.size == 0


@pytest.mark.parametrize("bad", [np.zeros(10), np.zeros((10, 3))])
def test_wrong_shaped_arrays_are_rejected(validator, bad):
    result = validator.validate_coordinates(bad)

    assert result.status is GeometryValidationStatus.INVALID
    assert "Invalid coordinate shape" in result.failure_reasons[0]


def test_nan_and_inf_short_circuit_before_geometric_analysis(validator):
    coords = naca_symmetric()
    coords[5, 1] = np.nan
    coords[7, 1] = np.inf

    result = validator.validate_coordinates(coords)

    assert result.violations == [
        GeometryViolationType.NaN_COORDINATES,
        GeometryViolationType.INFINITE_COORDINATES,
    ]
    assert result.thickness_distribution is None


def test_too_few_points_is_reported_as_insufficient(validator):
    coords = naca_symmetric(n_points=8)

    result = validator.validate_coordinates(coords)

    assert GeometryViolationType.INSUFFICIENT_POINTS in result.violations
    assert not result.can_proceed_to_cfd


def test_crossed_surfaces_are_flagged_as_self_intersection(validator):
    x = np.linspace(0.0, 1.0, 60)
    result = validator.validate_coordinates(
        surfaces_to_coordinates(x, 0.02 * (1 - x), 0.08 * (1 - x))
    )

    assert GeometryViolationType.SELF_INTERSECTION in result.violations
    assert GeometryViolationType.SURFACE_CROSSING in result.violations
    assert GeometryViolationType.NEGATIVE_THICKNESS in result.violations


def test_oscillatory_upper_surface_is_flagged(validator):
    x = np.linspace(0.0, 1.0, 50)
    y_upper = 0.05 * np.sin(20 * np.pi * x) * (1 - x)

    result = validator.validate_coordinates(
        surfaces_to_coordinates(x, y_upper, -0.05 * np.ones_like(x))
    )

    assert GeometryViolationType.OSCILLATORY_SURFACE in result.violations


def test_duplicate_points_are_detected(validator):
    coords = naca_symmetric()
    coords[30] = coords[29]

    result = validator.validate_coordinates(coords)

    assert GeometryViolationType.DUPLICATE_POINTS in result.violations


def test_thickness_bounds_come_from_config():
    coords = naca_symmetric(thickness=0.20)
    strict = AirfoilGeometryValidator(GeometryValidationConfig(max_thickness_ratio=0.15))

    result = strict.validate_coordinates(coords)

    assert GeometryViolationType.THICKNESS_OUT_OF_BOUNDS in result.violations
    assert AirfoilGeometryValidator().validate_coordinates(coords).is_valid


def test_thin_trailing_edge_is_flagged_as_degenerate():
    coords = naca_symmetric()
    validator = AirfoilGeometryValidator(GeometryValidationConfig(min_te_thickness=0.01))

    result = validator.validate_coordinates(coords)

    assert GeometryViolationType.DEGENERATE_TE in result.violations
    assert GeometryViolationType.INVALID_TE_THICKNESS in result.violations


def test_result_to_dict_is_json_friendly(validator):
    result = validator.validate_coordinates(naca_symmetric())

    payload = result.to_dict()

    assert payload["status"] == "VALID"
    assert payload["is_valid"] is True
    assert payload["violations"] == []
    assert payload["max_thickness"] == result.max_thickness


def test_smooth_cst_parameters_within_bounds_are_valid(validator):
    result = validator.validate_cst_parameters(
        np.array([0.16, 0.18, 0.20, 0.22]), np.array([-0.16, -0.14, -0.12, -0.10]), 0.004
    )

    assert result.status is GeometryValidationStatus.VALID
    assert result.is_valid and result.can_proceed_to_cfd


def test_large_cst_coefficients_only_warn(validator):
    result = validator.validate_cst_parameters(
        np.array([2.0, 3.0, -1.0, 0.5]), np.array([-2.0, 1.0, 0.5, -0.3]), 0.004
    )

    assert result.status is GeometryValidationStatus.WARNING
    assert result.is_valid
    # Warnings are advisory but still block the automatic hand-off to CFD.
    assert not result.can_proceed_to_cfd


def test_negative_te_thickness_invalidates_cst_parameters(validator):
    result = validator.validate_cst_parameters(np.array([0.18, 0.05]), np.array([-0.19, 0.05]), -0.01)

    assert result.status is GeometryValidationStatus.INVALID
    assert GeometryViolationType.INVALID_TE_THICKNESS in result.violations
    assert not result.is_valid


def test_validation_suite_reports_per_case_results():
    summary = GeometryValidationSuite().run_all_tests()

    assert summary["total_tests"] == len(summary["results"])
    assert summary["passed"] + summary["failed"] == summary["total_tests"]
    assert summary["pass_rate"] == pytest.approx(summary["passed"] / summary["total_tests"])

    by_name = {case["name"]: case for case in summary["results"]}
    assert by_name["valid_airfoil"]["passed"]
    assert by_name["self_intersection"]["passed"]
    assert by_name["nan_coordinates"]["passed"]
