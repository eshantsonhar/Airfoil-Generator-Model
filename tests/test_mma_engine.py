import numpy as np
import pytest

from airfoil_discovery.optimization.mma_engine import SvanbergMMA


def test_mma_accepts_small_objective_improvement_even_when_prediction_is_pessimistic() -> None:
    mma = SvanbergMMA(
        n_vars=2,
        n_constraints=0,
        x_min=np.array([-1.0, -1.0]),
        x_max=np.array([1.0, 1.0]),
        move_limit=0.1,
    )
    mma.initialize(np.array([0.0, 0.0]))

    mma.state.f_val = 1.0
    mma.state.f_prev = 1.0
    mma.state.g_vals = np.zeros(0)
    mma.state.g_prev = np.zeros(0)

    x_candidate = np.array([0.02, -0.01])
    x_accepted, accepted, stagnated = mma.step(
        x_new=x_candidate,
        f_new=0.8,
        f_pred=1.2,
        g_new=None,
    )

    assert accepted is True
    assert stagnated is False
    assert np.allclose(x_accepted, x_candidate)


def test_mma_rejects_step_when_candidate_is_worse_than_reference() -> None:
    mma = SvanbergMMA(
        n_vars=2,
        n_constraints=0,
        x_min=np.array([-1.0, -1.0]),
        x_max=np.array([1.0, 1.0]),
        move_limit=0.1,
    )
    mma.initialize(np.array([0.0, 0.0]))
    mma.state.f_val = 0.20

    x_candidate = np.array([0.05, 0.0])
    x_accepted, accepted, stagnated = mma.step(
        x_new=x_candidate,
        f_new=0.25,
        f_pred=0.18,
        g_new=None,
    )

    assert accepted is False
    assert stagnated is False
    assert np.allclose(x_accepted, np.zeros(2))


def test_mma_sync_state_resets_stagnation_counter() -> None:
    mma = SvanbergMMA(
        n_vars=2,
        n_constraints=1,
        x_min=np.array([-1.0, -1.0]),
        x_max=np.array([1.0, 1.0]),
        move_limit=0.1,
    )
    mma.initialize(np.array([0.1, -0.1]))
    mma.state.stagnated_counter = 7
    mma.state.f_val = 0.5

    x_best = np.array([0.2, -0.05])
    g_best = np.array([-0.01])
    mma.sync_state(x_best, 0.21, g_best)

    assert np.allclose(mma.state.x, x_best)
    assert mma.state.f_val == pytest.approx(0.21)
    assert mma.state.stagnated_counter == 0
    assert np.allclose(mma.state.g_vals, g_best)


def test_mma_propose_step_returns_nonzero_move() -> None:
    mma = SvanbergMMA(
        n_vars=3,
        n_constraints=0,
        x_min=np.full(3, -0.5),
        x_max=np.full(3, 0.5),
        move_limit=0.2,
    )
    mma.initialize(np.zeros(3))
    mma.sync_state(np.zeros(3), 1.0)

    x_candidate, f_pred, _lambd, _state = mma.propose_step(
        f=1.0,
        df=np.array([-1.0, -0.5, 0.25]),
        g=None,
        dg=None,
    )

    assert np.linalg.norm(x_candidate) > 1e-6
    assert f_pred < 1.0


def test_mma_reject_step_triggers_stagnation_after_threshold() -> None:
    mma = SvanbergMMA(
        n_vars=2,
        n_constraints=0,
        x_min=np.array([-1.0, -1.0]),
        x_max=np.array([1.0, 1.0]),
        move_limit=0.1,
    )
    mma.initialize(np.array([0.0, 0.0]))
    mma.state.f_val = 0.2
    mma.state.stagnated_counter = 9

    _, accepted, stagnated = mma.step(
        x_new=np.array([0.2, 0.0]),
        f_new=0.5,
        f_pred=0.15,
        g_new=None,
    )

    assert accepted is False
    assert stagnated is True
