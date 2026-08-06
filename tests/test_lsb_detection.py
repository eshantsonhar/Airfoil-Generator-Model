from __future__ import annotations

import numpy as np
import pytest

from airfoil_discovery.physics.lsb_detection import (
    LSBClassification,
    LSBDetector,
    LSBMetrics,
    LSBType,
)

X = np.linspace(0.0, 1.0, 201)


def bubble_cp(separation: float = 0.3, reattachment: float = 0.5) -> np.ndarray:
    """Cp with a favourable ramp, a plateau over the bubble, then pressure recovery."""
    plateau_level = -1.0 - 2.0 * separation
    return np.where(
        X < separation,
        -1.0 - 2.0 * X,
        np.where(X < reattachment, plateau_level, plateau_level + 3.0 * (X - reattachment)),
    )


def bubble_cf(separation: float = 0.3, reattachment: float = 0.5) -> np.ndarray:
    return np.where((X > separation) & (X < reattachment), -1e-3, 2e-3)


def bubble_intermittency(onset: float = 0.4, length: float = 0.1) -> np.ndarray:
    return np.clip((X - onset) / length, 0.0, 1.0)


def attached_cp() -> np.ndarray:
    """Steep monotonic Cp with no plateau region anywhere."""
    return -3.0 + 4.0 * X


def empty_metrics(**overrides) -> LSBMetrics:
    base = {
        "lsb_detected": True,
        "separation_location": None,
        "transition_onset": None,
        "transition_completion": None,
        "reattachment_location": None,
        "bubble_length": None,
        "bubble_height_proxy": None,
        "bubble_area_proxy": None,
        "plateau_start": None,
        "plateau_end": None,
        "plateau_length": None,
        "plateau_pressure_level": None,
        "cf_reversal_location": None,
        "cf_recovery_location": None,
        "min_cf": None,
        "intermittency_onset": None,
        "intermittency_completion": None,
        "intermittency_growth_rate": None,
        "apg_severity": 0.0,
        "apg_region_start": None,
        "apg_region_end": None,
        "wall_shear_collapse_detected": False,
        "wall_shear_collapse_location": None,
        "reattachment_strength": None,
        "physically_consistent": True,
    }
    base.update(overrides)
    return LSBMetrics(**base)


@pytest.fixture
def detector() -> LSBDetector:
    return LSBDetector()


def test_pressure_plateau_bounds_match_the_flat_cp_region(detector):
    start, end, level = detector.detect_pressure_plateau(X, bubble_cp())

    assert start == pytest.approx(0.3, abs=0.01)
    assert end == pytest.approx(0.5, abs=0.01)
    assert level == pytest.approx(-1.6, abs=1e-6)


def test_no_plateau_is_reported_for_a_steep_monotonic_cp(detector):
    assert detector.detect_pressure_plateau(X, attached_cp()) == (None, None, None)


def test_plateau_detection_needs_a_minimum_number_of_samples(detector):
    x = np.linspace(0.0, 1.0, 5)
    assert detector.detect_pressure_plateau(x, np.full_like(x, -1.0)) == (None, None, None)


def test_separation_prefers_the_skin_friction_reversal(detector):
    location = detector.detect_separation(X, cf=bubble_cf(), cp=bubble_cp())

    assert location == pytest.approx(0.305, abs=1e-6)


def test_separation_falls_back_to_the_pressure_gradient(detector):
    location = detector.detect_separation(X, cf=None, cp=bubble_cp())

    assert location is not None
    assert location >= 0.3


def test_separation_returns_none_without_any_distribution(detector):
    assert detector.detect_separation(X) is None
    assert detector.detect_separation(X, cf=np.full_like(X, 2e-3)) is None


def test_reattachment_is_the_first_positive_cf_after_separation(detector):
    location = detector.detect_reattachment(X, cf=bubble_cf(), separation_location=0.3)

    assert location == pytest.approx(0.5, abs=1e-6)


def test_reattachment_requires_a_known_separation_location(detector):
    assert detector.detect_reattachment(X, cf=bubble_cf()) is None


def test_transition_locations_come_from_the_intermittency_thresholds(detector):
    onset, completion = detector.detect_transition(X, intermittency=bubble_intermittency())

    assert onset == pytest.approx(0.41, abs=1e-6)
    assert completion == pytest.approx(0.495, abs=1e-6)


def test_transition_is_none_when_the_flow_stays_laminar(detector):
    onset, completion = detector.detect_transition(X, intermittency=np.zeros_like(X))

    assert onset is None and completion is None


def test_transition_without_any_distribution_is_undetermined(detector):
    assert detector.detect_transition(X) == (None, None)


def test_apg_severity_is_zero_for_a_fully_favourable_gradient(detector):
    severity, start, end = detector.compute_apg_severity(X, -1.0 - 2.0 * X)

    assert severity == 0.0
    assert start is None and end is None


def test_apg_severity_integrates_the_adverse_region(detector):
    severity, start, end = detector.compute_apg_severity(X, bubble_cp())

    assert severity > 0.0
    assert start == pytest.approx(0.5, abs=0.01)
    assert end == pytest.approx(1.0)


def test_classification_of_a_run_without_a_bubble(detector):
    classification = detector.classify_bubble(empty_metrics(lsb_detected=False))

    assert classification.bubble_type is LSBType.NO_BUBBLE
    assert classification.bursting_risk_score == 0.0
    assert classification.stability_indicator == 1.0
    assert classification.suppression_mechanisms == []


def test_short_and_long_bubbles_are_separated_by_the_configured_threshold(detector):
    short = detector.classify_bubble(empty_metrics(bubble_length=0.05))
    long = detector.classify_bubble(empty_metrics(bubble_length=0.25))

    assert short.bubble_type is LSBType.SHORT_BUBBLE
    assert short.hysteresis_index == pytest.approx(0.3)
    assert long.bubble_type is LSBType.LONG_BUBBLE
    assert long.hysteresis_index == pytest.approx(0.7)
    assert long.bursting_risk_score > short.bursting_risk_score
    assert long.stability_indicator < short.stability_indicator
    assert long.drag_amplification_factor == pytest.approx(1.5)
    assert "transition_promotion" in long.suppression_mechanisms


def test_missing_bubble_length_leaves_the_bubble_unclassified(detector):
    classification = detector.classify_bubble(empty_metrics())

    assert classification.bubble_type is LSBType.UNCLASSIFIED
    assert classification.drag_amplification_factor is None
    assert classification.effective_camber_distortion is None


def test_bursting_risk_saturates_at_one(detector):
    classification = detector.classify_bubble(empty_metrics(bubble_length=0.9, apg_severity=50.0))

    assert classification.bursting_risk_score == 1.0
    assert classification.stability_indicator == 0.0


def test_severe_apg_adds_a_pressure_shaping_recommendation(detector):
    classification = detector.classify_bubble(
        empty_metrics(bubble_length=0.05, apg_severity=8.0, separation_location=0.3)
    )

    assert "pressure_gradient_shaping" in classification.suppression_mechanisms
    assert "leading_edge_contouring" in classification.suppression_mechanisms
    assert "high_apg_region" in classification.critical_regions
    assert "separation_at_x=0.300" in classification.critical_regions


def test_full_detection_of_a_long_bubble(detector):
    report = detector.detect(X, bubble_cp(), bubble_cf(), bubble_intermittency())

    assert report.metrics.lsb_detected
    assert report.metrics.separation_location == pytest.approx(0.305, abs=1e-6)
    assert report.metrics.reattachment_location == pytest.approx(0.5, abs=1e-6)
    assert report.metrics.bubble_length == pytest.approx(0.195, abs=1e-6)
    assert report.metrics.bubble_height_proxy == pytest.approx(0.1 * 0.195, abs=1e-6)
    assert report.metrics.wall_shear_collapse_detected
    assert report.classification.bubble_type is LSBType.LONG_BUBBLE
    assert report.is_valid
    assert report.warnings == []
    # cf data present gives the highest confidence tier.
    assert report.confidence == pytest.approx(0.95)


def test_confidence_tiers_depend_on_the_available_distributions(detector):
    cp, cf, gamma = bubble_cp(), bubble_cf(), bubble_intermittency()

    assert detector.detect(X, cp).confidence == pytest.approx(0.8)
    assert detector.detect(X, cp, intermittency_upper=gamma).confidence == pytest.approx(0.9)
    assert detector.detect(X, cp, cf_upper=cf).confidence == pytest.approx(0.95)
    assert detector.detect(X, attached_cp()).confidence == pytest.approx(0.5)


def test_attached_flow_produces_no_bubble(detector):
    report = detector.detect(X, attached_cp())

    assert not report.metrics.lsb_detected
    assert report.classification.bubble_type is LSBType.NO_BUBBLE
    assert report.metrics.plateau_length is None


def test_transition_upstream_of_separation_is_flagged_as_inconsistent(detector):
    report = detector.detect(
        X, bubble_cp(), bubble_cf(), bubble_intermittency(onset=0.05, length=0.05)
    )

    assert not report.metrics.physically_consistent
    assert "transition_before_separation" in report.metrics.consistency_flags
    assert not report.is_valid
    assert "Physical inconsistency detected in LSB metrics" in report.warnings


def test_report_to_dict_summarises_metrics_and_classification(detector):
    report = detector.detect(X, bubble_cp(), bubble_cf(), bubble_intermittency())

    payload = report.to_dict()

    assert payload["lsb_detected"]
    assert payload["bubble_type"] == "LONG_BUBBLE"
    assert payload["bubble_length"] == pytest.approx(report.metrics.bubble_length)
    assert payload["apg_severity"] == pytest.approx(report.metrics.apg_severity)
    assert payload["confidence"] == pytest.approx(report.confidence)
    assert payload["warnings"] == []


def test_classification_is_returned_as_a_dataclass_instance(detector):
    report = detector.detect(X, bubble_cp(), bubble_cf())

    assert isinstance(report.classification, LSBClassification)
    assert report.detection_method == "pressure_plateau_cf_intermittency"
    assert report.x_coordinates is X
