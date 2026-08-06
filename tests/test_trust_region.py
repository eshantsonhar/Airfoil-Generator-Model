from __future__ import annotations

import pytest

from airfoil_discovery.optimization.trust_region import TrustRegionGovernor


def test_high_agreement_step_is_accepted_and_expands():
    result = TrustRegionGovernor().evaluate_step(-0.95, -1.0)

    assert result["rho"] == pytest.approx(0.95)
    assert result["accepted"]
    assert result["action"] == "EXPAND"


def test_partial_agreement_keeps_move_limits():
    result = TrustRegionGovernor().evaluate_step(-0.5, -1.0)

    assert result["rho"] == pytest.approx(0.5)
    assert result["accepted"]
    assert result["action"] == "KEEP"


def test_poor_agreement_is_accepted_but_shrinks():
    result = TrustRegionGovernor().evaluate_step(-0.2, -1.0)

    assert result["rho"] == pytest.approx(0.2)
    assert result["accepted"]
    assert result["action"] == "SHRINK"


def test_wrong_sign_actual_change_is_rejected_and_shrinks():
    result = TrustRegionGovernor().evaluate_step(0.4, -1.0)

    assert result["rho"] == pytest.approx(-0.4)
    assert not result["accepted"]
    assert result["action"] == "SHRINK"


def test_negligible_prediction_is_treated_as_perfect_agreement():
    result = TrustRegionGovernor().evaluate_step(-5.0, 1e-15)

    assert result["rho"] == 1.0
    assert result["accepted"]
    assert result["action"] == "EXPAND"


@pytest.mark.parametrize(
    ("rho_accept", "rho_shrink", "expected_accepted", "expected_action"),
    [
        (0.4, 0.25, False, "KEEP"),
        (0.1, 0.5, True, "SHRINK"),
    ],
)
def test_thresholds_are_configurable(rho_accept, rho_shrink, expected_accepted, expected_action):
    governor = TrustRegionGovernor(rho_accept=rho_accept, rho_shrink=rho_shrink)

    result = governor.evaluate_step(-0.3, -1.0)

    assert result["accepted"] is expected_accepted
    assert result["action"] == expected_action


def test_acceptance_boundary_is_strict():
    governor = TrustRegionGovernor(rho_accept=0.1)

    assert not governor.evaluate_step(-0.1, -1.0)["accepted"]
    assert governor.evaluate_step(-0.1001, -1.0)["accepted"]
