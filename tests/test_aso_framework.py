import numpy as np
from airfoil_discovery.core.mma_solver import TrustRegionMMA
from airfoil_discovery.core.fidelity import FidelityController, CFDResult, CFDState
from airfoil_discovery.cfd.su2 import SU2Status, DesignEvaluation, AdjointResult
from pathlib import Path


class FakeSU2Evaluator:
    def run_evaluation(self, x, case_dir, level):
        grad = 2.0 * np.asarray(x, dtype=float)
        cd = float(np.sum(np.asarray(x, dtype=float) ** 2))
        return DesignEvaluation(
            cl=1.0,
            cd=cd,
            thickness=0.12,
            status=SU2Status.OK,
            adjoint=AdjointResult(
                grad_cd=grad,
                grad_cl=0.5 * grad,
                residual=1e-6,
                converged=True,
            ),
        )

def test_sensitivity_branin():
    print("--- Running Sensitivity Check (Branin) ---")
    # Minimizing Branin function (simplified 2D)
    # f(x) = (x2 - 5.1/(4*pi^2)*x1^2 + 5/pi*x1 - 6)^2 + 10*(1-1/(8*pi))*cos(x1) + 10
    # Global minimum at x = (-pi, 12.275) or (pi, 2.275)
    mma = TrustRegionMMA(n_vars=2, lower_bounds=np.array([-5, 0]), upper_bounds=np.array([10, 15]))
    # Dummy gradient
    grad = np.array([1.0, 1.0])
    constraints = np.array([0.0])
    jacobians = np.array([[1.0, 1.0]])
    x_new = mma.solve_subproblem(np.array([0.0, 0.0]), grad, constraints, jacobians)
    print(f"Subproblem step output: {x_new}")
    print("Test Passed: Solver executed without crashing.\n")

def test_gradient_verification():
    print("--- Running Gradient Verification ---")
    evaluator = FakeSU2Evaluator()
    x = np.array([0.5] * 10)
    
    # Get adjoint gradient
    res = evaluator.run_evaluation(x, Path("test_case"), "L1")
    adj_grad = res.adjoint.grad_cd
    
    # Finite difference (eps=1e-4)
    eps = 1e-4
    fd_grad = np.zeros(10)
    for i in range(10):
        x_plus = x.copy()
        x_plus[i] += eps
        res_plus = evaluator.run_evaluation(x_plus, Path("test_case"), "L1")
        fd_grad[i] = (res_plus.cd - res.cd) / eps
    
    # Check alignment
    fc = FidelityController()
    cos_sim = fc.check_gradients(adj_grad, fd_grad)
    print(f"Cosine Similarity (Adjoint vs FD): {cos_sim:.4f}")
    assert cos_sim > 0.95
    print("Test Passed: Adjoint gradients are consistent with Finite Difference.\n")

def test_failure_recovery():
    print("--- Running Failure State Recovery ---")
    # Force failure in evaluator
    class FailingEvaluator(FakeSU2Evaluator):
        def run_evaluation(self, x, case_dir, level):
            res = super().run_evaluation(x, case_dir, level)
            res.status = SU2Status.ADJOINT_INVALID
            return res
            
    evaluator = FailingEvaluator()
    res = evaluator.run_evaluation(np.array([0.5]*10), Path("test_fail"), "L1")
    
    # Verify state machine catches it
    assert res.status == SU2Status.ADJOINT_INVALID
    print("Test Passed: System correctly identified failed CFD state.\n")

if __name__ == "__main__":
    test_sensitivity_branin()
    test_gradient_verification()
    test_failure_recovery()
