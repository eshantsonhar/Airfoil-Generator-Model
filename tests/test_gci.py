from __future__ import annotations

import pytest

from airfoil_discovery.verification.gci import (
    GCIResult,
    GridConvergenceIndex,
    RichardsonExtrapolation,
)

# Grid sizes are consumed as h-refinement ratios (sizes[1] / sizes[0]), so a
# doubling sequence expresses "each grid is twice as coarse as the previous".
GRID_SIZES = [100, 200, 400]


def _second_order_series(
    exact: float, h_fine: float, coefficient: float, ratio: float = 2.0
) -> tuple[float, float, float]:
    """Build three solutions carrying an exact second-order discretization error."""
    fine = exact + coefficient * h_fine**2
    medium = exact + coefficient * (ratio * h_fine) ** 2
    coarse = exact + coefficient * (ratio**2 * h_fine) ** 2
    return fine, medium, coarse


def test_compute_recovers_second_order_and_exact_value() -> None:
    analyzer = GridConvergenceIndex()
    fine, medium, coarse = _second_order_series(exact=1.5, h_fine=0.01, coefficient=3.0)

    result = analyzer.compute(fine, medium, coarse, GRID_SIZES)

    assert result.observed_order == pytest.approx(2.0, abs=1e-9)
    assert result.extrapolated_value == pytest.approx(1.5, abs=1e-9)
    assert result.extrapolation_error == pytest.approx(abs(fine - 1.5), abs=1e-9)
    assert result.is_monotonic
    assert result.passed_order_check
    assert result.refinement_ratios == [1.0, 2.0, 2.0]


def test_observed_order_sign_tracks_grid_size_ordering() -> None:
    """Passing decreasing cell counts inverts the refinement ratio and the order sign."""
    analyzer = GridConvergenceIndex()
    fine, medium, coarse = _second_order_series(exact=1.5, h_fine=0.01, coefficient=3.0)

    increasing = analyzer.compute(fine, medium, coarse, [100, 200, 400])
    decreasing = analyzer.compute(fine, medium, coarse, [400, 200, 100])

    assert decreasing.observed_order == pytest.approx(-increasing.observed_order)
    assert not decreasing.passed_order_check
    # r**p is unchanged by the inversion, so the extrapolation is unaffected.
    assert decreasing.extrapolated_value == pytest.approx(increasing.extrapolated_value)
    assert decreasing.gci_fine_medium == pytest.approx(increasing.gci_fine_medium)


def test_compute_marks_non_monotonic_series() -> None:
    result = GridConvergenceIndex().compute(1.0, 1.2, 1.1, GRID_SIZES)

    assert not result.is_monotonic


def test_compute_falls_back_to_theoretical_order_when_solutions_are_identical() -> None:
    result = GridConvergenceIndex().compute(2.0, 2.0, 2.0, GRID_SIZES, theoretical_order=3.0)

    assert result.observed_order == pytest.approx(3.0)
    assert result.extrapolated_value == pytest.approx(2.0)
    assert result.gci_fine_medium == pytest.approx(0.0)
    assert result.numerical_uncertainty == pytest.approx(0.0)


def test_gci_scales_linearly_with_safety_factor() -> None:
    fine, medium, coarse = _second_order_series(exact=1.5, h_fine=0.01, coefficient=3.0)

    conservative = GridConvergenceIndex(safety_factor=3.0).compute(fine, medium, coarse, GRID_SIZES)
    standard = GridConvergenceIndex(safety_factor=1.25).compute(fine, medium, coarse, GRID_SIZES)

    assert conservative.gci_fine_medium == pytest.approx(standard.gci_fine_medium * (3.0 / 1.25))
    assert conservative.relative_uncertainty == conservative.gci_fine_medium
    assert conservative.numerical_uncertainty == pytest.approx(
        abs(fine) * conservative.gci_fine_medium
    )


def test_asymptotic_range_flag_uses_convergence_ratio_bounds() -> None:
    fine, medium, coarse = _second_order_series(exact=1.5, h_fine=0.01, coefficient=3.0)

    inside = GridConvergenceIndex().compute(fine, medium, coarse, GRID_SIZES)
    # Error grows under refinement, so the GCI is not positive and no ratio is formed.
    outside = GridConvergenceIndex().compute(1.0, 1.5, 1.52, GRID_SIZES)

    assert 0.5 <= inside.convergence_ratio <= 1.5
    assert inside.is_asymptotic and inside.passed_asymptotic_range
    assert outside.convergence_ratio == 0.0
    assert not outside.is_asymptotic and not outside.passed_asymptotic_range


def test_compute_multi_variable_analyses_each_variable_independently() -> None:
    analyzer = GridConvergenceIndex()
    cl = _second_order_series(exact=0.8, h_fine=0.01, coefficient=1.0)
    cd = _second_order_series(exact=0.02, h_fine=0.01, coefficient=-0.5)

    results = analyzer.compute_multi_variable({"cl": cl, "cd": cd}, GRID_SIZES)

    assert set(results) == {"cl", "cd"}
    assert results["cl"].extrapolated_value == pytest.approx(0.8, abs=1e-9)
    assert results["cd"].extrapolated_value == pytest.approx(0.02, abs=1e-9)


def test_result_to_dict_exposes_every_field() -> None:
    result = GridConvergenceIndex().compute(1.0, 1.1, 1.3, GRID_SIZES)

    payload = result.to_dict()

    assert set(payload) == set(GCIResult.__dataclass_fields__)
    assert payload["grid_sizes"] == GRID_SIZES
    assert payload["fine_value"] == 1.0
    assert payload["observed_order"] == result.observed_order
    assert bool(payload["is_monotonic"]) == bool(result.is_monotonic)


def test_richardson_extrapolation_matches_analytic_limit() -> None:
    fine, medium, _ = _second_order_series(exact=1.5, h_fine=0.01, coefficient=3.0)

    extrapolated, uncertainty = RichardsonExtrapolation().extrapolate(
        fine_value=fine,
        medium_value=medium,
        refinement_ratio=2.0,
        observed_order=2.0,
    )

    assert extrapolated == pytest.approx(1.5, abs=1e-9)
    assert uncertainty == pytest.approx(abs(fine - 1.5), abs=1e-9)


def test_richardson_extrapolation_is_identity_for_zero_order() -> None:
    extrapolated, uncertainty = RichardsonExtrapolation().extrapolate(
        1.23, 1.4, refinement_ratio=2.0, observed_order=0.0
    )

    assert extrapolated == 1.23
    assert uncertainty == 0.0


def test_richardson_extrapolation_is_identity_for_unit_refinement_ratio() -> None:
    extrapolated, uncertainty = RichardsonExtrapolation().extrapolate(
        1.23, 1.4, refinement_ratio=1.0, observed_order=2.0
    )

    assert extrapolated == 1.23
    assert uncertainty == 0.0


def test_estimate_order_matches_analytic_value_and_defaults_when_flat() -> None:
    richardson = RichardsonExtrapolation()
    fine, medium, coarse = _second_order_series(exact=1.5, h_fine=0.01, coefficient=3.0)

    assert richardson.estimate_order(fine, medium, coarse, r12=2.0, r23=2.0) == pytest.approx(
        2.0, abs=1e-9
    )
    assert richardson.estimate_order(1.0, 1.0, 1.0, r12=2.0, r23=2.0) == 2.0
