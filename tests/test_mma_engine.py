import numpy as np

from airfoil_discovery.optimization.mma_engine import SvanbergMMA


def test_mma_accepts_small_objective_improvement_even_when_prediction_is_pessimistic() -> None:
    mma = SvanbergMMA(n_vars=2, n_constraints=0, x_min=np.array([-1.0, -1.0]), x_max=np.array([1.0, 1.0]), move_limit=0.1)
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
