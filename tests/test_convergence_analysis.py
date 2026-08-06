from __future__ import annotations

import numpy as np
import pytest

from airfoil_discovery.verification.convergence import (
    ConvergenceStatus,
    IterativeConvergenceMonitor,
    ResidualConvergenceAnalyzer,
)


def decaying_residuals(n: int = 200, start_exp: float = -1.0, end_exp: float = -9.0) -> list[float]:
    """Smoothly decaying residual history spanning the given decades."""
    return list(10.0 ** np.linspace(start_exp, end_exp, n))


def steady_forces(n: int = 200, cl: float = 0.5, cd: float = 0.01) -> tuple[list, list]:
    return [cl] * n, [cd] * n


@pytest.fixture
def analyzer() -> ResidualConvergenceAnalyzer:
    return ResidualConvergenceAnalyzer()


@pytest.fixture
def monitor() -> IterativeConvergenceMonitor:
    return IterativeConvergenceMonitor()


def test_empty_residual_history_raises(analyzer):
    with pytest.raises(ValueError, match="Empty residual history"):
        analyzer.analyze([])


def test_smooth_decay_is_converged_monotonic_and_asymptotic(analyzer):
    history = decaying_residuals()

    metrics = analyzer.analyze(history)

    assert metrics.iteration_count == len(history)
    assert metrics.final_residual == pytest.approx(1e-9)
    assert metrics.max_residual == pytest.approx(1e-1)
    assert metrics.below_threshold
    assert metrics.monotonic_decrease
    assert metrics.asymptotic_behavior
    assert not metrics.stagnation_detected
    assert metrics.stagnation_start_iteration is None
    # log10 residual drops by 8 decades over 199 steps.
    assert metrics.convergence_rate == pytest.approx(8 / 199, rel=1e-6)
    assert metrics.log_residual_slope == pytest.approx(-metrics.convergence_rate)


def test_flat_history_is_reported_as_stagnation(analyzer):
    metrics = analyzer.analyze([1e-3] * 200)

    assert metrics.stagnation_detected
    assert metrics.stagnation_start_iteration == 0
    assert metrics.convergence_rate == pytest.approx(0.0, abs=1e-12)


def test_short_history_skips_rate_and_asymptotic_analysis(analyzer):
    metrics = analyzer.analyze([1e-2, 1e-3, 1e-4])

    assert metrics.convergence_rate == 0.0
    assert metrics.log_residual_slope == 0.0
    assert not metrics.asymptotic_behavior
    assert metrics.monotonic_decrease
    assert not metrics.stagnation_detected


def test_residual_bump_breaks_monotonicity(analyzer):
    history = decaying_residuals()
    history[-10] = 1.0

    metrics = analyzer.analyze(history)

    assert not metrics.monotonic_decrease
    assert metrics.max_residual == pytest.approx(1.0)


def test_threshold_is_applied_to_the_magnitude_of_the_final_residual():
    # SU2 reports log10(residual), so negative values must still be compared by magnitude.
    metrics = ResidualConvergenceAnalyzer(residual_threshold=1e-6).analyze([-1e-9] * 5)

    assert metrics.below_threshold


def test_empty_force_history_raises(monitor):
    with pytest.raises(ValueError, match="Empty force history"):
        monitor.analyze_forces([], [])


def test_steady_forces_are_stabilized(monitor):
    cl_history, cd_history = steady_forces()

    metrics = monitor.analyze_forces(cl_history, cd_history)

    assert metrics.final_cl == pytest.approx(0.5)
    assert metrics.final_cd == pytest.approx(0.01)
    assert metrics.cl_std == pytest.approx(0.0)
    assert metrics.cl_amplitude == pytest.approx(0.0)
    assert metrics.cl_relative_oscillation == pytest.approx(0.0)
    assert metrics.cl_trend == pytest.approx(0.0, abs=1e-12)
    assert metrics.forces_stabilized
    assert metrics.force_oscillation_acceptable
    assert metrics.force_drift_acceptable


def test_oscillation_metrics_ignore_the_initial_transient(monitor):
    # First half ramps from zero; only the converged tail should drive the ratios.
    n = 200
    cl = list(np.linspace(0.0, 0.5, n // 2)) + [0.5] * (n // 2)
    cd = list(np.linspace(0.0, 0.01, n // 2)) + [0.01] * (n // 2)

    metrics = monitor.analyze_forces(cl, cd)

    assert metrics.cl_std > 0.1  # full-history spread is dominated by the ramp
    assert metrics.cl_relative_oscillation == pytest.approx(0.0, abs=1e-9)
    assert metrics.force_oscillation_acceptable


def test_oscillating_forces_fail_the_oscillation_check(monitor):
    n = 200
    t = np.arange(n)
    cl = list(0.5 + 0.05 * np.sin(0.3 * t))
    cd = list(0.01 + 0.001 * np.sin(0.3 * t))

    metrics = monitor.analyze_forces(cl, cd)

    assert metrics.cl_relative_oscillation > monitor.force_oscillation_threshold
    assert not metrics.force_oscillation_acceptable
    assert not metrics.forces_stabilized


def test_drifting_forces_fail_the_drift_check(monitor):
    n = 200
    cl = list(0.5 + 0.01 * np.arange(n))
    cd = list(0.01 + 1e-4 * np.arange(n))

    metrics = monitor.analyze_forces(cl, cd)

    assert metrics.cl_trend == pytest.approx(0.01)
    assert not metrics.force_drift_acceptable


def test_spectral_analysis_is_skipped_for_short_histories(monitor):
    cl_history, cd_history = steady_forces(n=32)

    spectral = monitor.analyze_spectral(cl_history, cd_history)

    assert spectral.cl_dominant_frequency == 0.0
    assert spectral.cl_spectral_power == 0.0
    assert not spectral.periodic_shedding_detected
    assert not spectral.metastable_behavior_detected
    assert spectral.shedding_frequency is None


def test_single_tone_history_is_detected_as_periodic_shedding(monitor):
    n = 256
    t = np.arange(n)
    cl = list(0.5 + 0.05 * np.sin(2 * np.pi * 8 * t / n))
    cd = list(0.01 + 0.001 * np.sin(2 * np.pi * 8 * t / n))

    spectral = monitor.analyze_spectral(cl, cd)

    assert spectral.cl_dominant_frequency == pytest.approx(8 / n)
    assert spectral.cl_spectral_power > 0.0
    assert spectral.periodic_shedding_detected
    assert spectral.shedding_frequency == pytest.approx(spectral.cl_dominant_frequency)


def test_report_marks_a_clean_run_as_converged_and_valid(monitor, analyzer):
    residual = analyzer.analyze(decaying_residuals())
    force = monitor.analyze_forces(*steady_forces())

    report = monitor.generate_report(residual, force)

    assert report.status is ConvergenceStatus.CONVERGED
    assert report.is_valid
    assert report.failure_reasons == []
    assert report.recommended_actions == []


def test_report_requires_min_iterations_for_validity(analyzer):
    monitor = IterativeConvergenceMonitor(min_iterations=500)
    residual = analyzer.analyze(decaying_residuals())
    force = monitor.analyze_forces(*steady_forces())

    report = monitor.generate_report(residual, force)

    assert report.status is ConvergenceStatus.CONVERGED
    assert not report.is_valid


def test_report_flags_divergence_from_a_large_final_residual(monitor, analyzer):
    residual = analyzer.analyze(decaying_residuals(start_exp=-1.0, end_exp=1.0))
    force = monitor.analyze_forces(*steady_forces())

    report = monitor.generate_report(residual, force)

    assert report.status is ConvergenceStatus.DIVERGED
    assert not report.is_valid
    assert "Residuals did not converge below threshold" in report.failure_reasons


def test_report_flags_stalled_runs(monitor, analyzer):
    residual = analyzer.analyze([1e-3] * 200)
    force = monitor.analyze_forces(*steady_forces())

    report = monitor.generate_report(residual, force)

    assert report.status is ConvergenceStatus.STALLED
    assert "Residual stagnation detected" in report.failure_reasons


def test_report_flags_oscillating_runs_from_spectral_evidence(monitor, analyzer):
    n = 256
    t = np.arange(n)
    cl = list(0.5 + 0.05 * np.sin(2 * np.pi * 8 * t / n))
    cd = list(0.01 + 0.001 * np.sin(2 * np.pi * 8 * t / n))
    residual = analyzer.analyze(decaying_residuals())
    force = monitor.analyze_forces(cl, cd)
    spectral = monitor.analyze_spectral(cl, cd)

    report = monitor.generate_report(residual, force, spectral)

    assert report.status is ConvergenceStatus.OSCILLATING
    assert "Periodic shedding detected (unsteady flow)" in report.failure_reasons
    assert not report.is_valid


def test_report_to_dict_flattens_residual_and_force_sections(monitor, analyzer):
    residual = analyzer.analyze(decaying_residuals())
    force = monitor.analyze_forces(*steady_forces())

    payload = monitor.generate_report(residual, force).to_dict()

    assert payload["status"] == "CONVERGED"
    assert payload["is_valid"] is True
    assert payload["residual"]["final_residual"] == pytest.approx(residual.final_residual)
    assert payload["force"]["final_cl"] == pytest.approx(force.final_cl)
    assert bool(payload["force"]["forces_stabilized"])
