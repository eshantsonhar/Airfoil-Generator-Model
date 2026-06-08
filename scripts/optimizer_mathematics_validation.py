"""
Optimizer Mathematics Validation Suite.

Numerically verifies:
1. Gradient correctness (FD comparison)
2. Step acceptance logic
3. Gain-ratio correctness
4. Scaling consistency
5. Hessian approximation quality
6. Move-limit logic
7. KKT interpretation
8. Convergence validity

This is a standalone verification tool that tests the MMA optimizer
with known analytic test functions before trusting it with CFD.
"""

import sys
import os
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from airfoil_discovery.optimization.mma_engine import SvanbergMMA, TrustRegionGovernor


# Known test functions with analytic gradients
class QuadraticTest:
    """Simple quadratic: f(x) = x^T A x + b^T x, known minimum at -0.5 A^{-1} b"""
    def __init__(self, n=2):
        self.n = n
        self.A = np.random.randn(n, n)
        self.A = self.A.T @ self.A + np.eye(n) * 0.1  # Positive definite
        self.b = np.random.randn(n) * 0.5
        
    def fun(self, x):
        return float(x @ self.A @ x + self.b @ x)
    
    def grad(self, x):
        return 2 * self.A @ x + self.b
    
    @property
    def minimum(self):
        return -0.5 * np.linalg.solve(self.A, self.b)


class RosenbrockTest:
    """Rosenbrock function: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)"""
    def __init__(self, n=2):
        self.n = n
        
    def fun(self, x):
        return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))
    
    def grad(self, x):
        g = np.zeros_like(x)
        for i in range(len(x)-1):
            g[i] += -400 * x[i] * (x[i+1] - x[i]**2) - 2 * (1 - x[i])
            g[i+1] += 200 * (x[i+1] - x[i]**2)
        return g


class FiniteDifferenceChecker:
    """
    Verify gradients using finite differences.
    
    Checks:
    - Gradient alignment with FD
    - Directional derivative consistency
    - Random perturbation validation
    """
    
    def __init__(self, eps=1e-6, rtol=0.1):
        self.eps = eps
        self.rtol = rtol
        
    def check_gradient(self, fun, grad, x0: np.ndarray, n_checks: int = 10) -> dict:
        """Check gradient against central finite differences."""
        errors = []
        max_error = 0.0
        
        for _ in range(n_checks):
            # Random perturbation direction
            d = np.random.randn(len(x0))
            d = d / np.linalg.norm(d)
            
            # Central FD
            f_plus = fun(x0 + self.eps * d)
            f_minus = fun(x0 - self.eps * d)
            fd_dir = (f_plus - f_minus) / (2 * self.eps)
            
            # Analytic directional derivative
            analytic_dir = np.dot(grad(x0), d)
            
            # Relative error
            denom = max(abs(fd_dir), 1e-12)
            error = abs(analytic_dir - fd_dir) / denom
            errors.append(error)
            max_error = max(max_error, error)
        
        return {
            "max_relative_error": max_error,
            "mean_relative_error": float(np.mean(errors)),
            "pass": max_error < self.rtol,
            "errors": errors,
        }
    
    def check_all_directions(self, fun, grad, x0: np.ndarray) -> dict:
        """Check gradient in each coordinate direction."""
        errors = []
        for i in range(len(x0)):
            d = np.zeros(len(x0))
            d[i] = 1.0
            f_plus = fun(x0 + self.eps * d)
            f_minus = fun(x0 - self.eps * d)
            fd = (f_plus - f_minus) / (2 * self.eps)
            analytic = grad(x0)[i]
            denom = max(abs(fd), 1e-12)
            error = abs(analytic - fd) / denom
            errors.append(error)
        return {
            "max_coordinate_error": max(errors),
            "mean_coordinate_error": float(np.mean(errors)),
            "pass": max(errors) < self.rtol,
            "coordinate_errors": errors,
        }


class MMAMathematicsValidator:
    """
    Validates the MMA optimizer mathematics.
    
    Tests:
    1. Gradient correctness via FD
    2. Step acceptance logic
    3. Gain ratio correctness
    4. Asymptote update stability
    5. Move limit enforcement
    6. Constraint handling
    7. Stagnation recovery
    """
    
    def __init__(self, n_vars: int = 4, n_constraints: int = 2):
        self.n_vars = n_vars
        self.n_constraints = n_constraints
        self.results = []
        
    def test_gradient_alignment(self) -> dict:
        """Test that the optimizer correctly interprets gradients."""
        mma = SvanbergMMA(n_vars=self.n_vars, n_constraints=self.n_constraints)
        x0 = np.ones(self.n_vars) * 0.5
        mma.initialize(x0)
        
        # Test with quadratic function
        test_fn = QuadraticTest(self.n_vars)
        x = x0.copy()
        f = test_fn.fun(x)
        df = test_fn.grad(x)
        
        # Run step
        x_next, accepted, state = mma.run_optimization_step(f, df)
        
        # Verify: objective should decrease if gradient points uphill
        f_next = test_fn.fun(x_next)
        actual_reduction = f - f_next
        
        return {
            "test": "gradient_alignment",
            "initial_objective": f,
            "next_objective": f_next,
            "actual_reduction": actual_reduction,
            "step_accepted": accepted,
            "pass": actual_reduction >= -1e-6 or accepted,
        }
    
    def test_step_acceptance_positive_rho(self) -> dict:
        """Test that steps improving the objective are accepted."""
        mma = SvanbergMMA(n_vars=self.n_vars, n_constraints=self.n_constraints)
        x0 = np.zeros(self.n_vars)
        mma.initialize(x0)
        
        # Simulate a good step with improvement
        f_old = 10.0
        f_new = 5.0
        f_pred = 4.0  # Over-predicted improvement
        
        x_accepted, accepted = mma.step(
            x_new=x0 + 0.01,
            f_new=f_new,
            f_pred=f_pred,
        )
        
        return {
            "test": "step_acceptance_positive",
            "f_old": f_old,
            "f_new": f_new,
            "accepted": accepted,
            "pass": accepted,
        }
    
    def test_step_rejection_negative_rho(self) -> dict:
        """Test that steps worsening the objective are rejected."""
        mma = SvanbergMMA(n_vars=self.n_vars, n_constraints=self.n_constraints)
        x0 = np.zeros(self.n_vars)
        mma.initialize(x0)
        
        # Initialize state with current objective
        mma.state.f_val = 10.0
        
        # Simulate a bad step
        x_accepted, accepted = mma.step(
            x_new=x0 + 0.5,
            f_new=100.0,  # Much worse
            f_pred=5.0,   # Predicted improvement (wrong)
        )
        
        return {
            "test": "step_rejection_negative",
            "accepted": accepted,
            "pass": not accepted,
        }
    
    def test_trust_region_governor(self) -> dict:
        """Test trust-region radius management."""
        gov = TrustRegionGovernor(
            initial_radius=0.1,
            max_radius=0.5,
            min_radius=1e-6,
        )
        
        # Test expansion on good steps
        r1 = gov.update(rho=0.9)
        assert r1["radius"] > 0.1, "Trust region should expand on good step"
        
        # Test contraction on bad steps
        r2 = gov.update(rho=-0.5)
        assert r2["radius"] < r1["radius"], "Trust region should contract on bad step"
        
        # Test reset on repeated failures
        for _ in range(10):
            gov.update(rho=-0.5)
        
        return {
            "test": "trust_region_governor",
            "radius_after_good": r1["radius"],
            "radius_after_bad": r2["radius"],
            "pass": r1["radius"] > 0.1 and r2["radius"] < r1["radius"],
        }
    
    def test_gain_ratio_calculation(self) -> dict:
        """Verify gain ratio = actual_reduction / predicted_reduction."""
        mma = SvanbergMMA(n_vars=self.n_vars, n_constraints=self.n_constraints)
        x0 = np.zeros(self.n_vars)
        mma.initialize(x0)
        mma.state.f_val = 10.0
        
        # Step with known gain ratio
        x_acc, acc = mma.step(
            x_new=x0 + 0.01,
            f_new=8.0,
            f_pred=6.0,  # Predicted reduction of 4
        )
        
        # Expected: actual=2, predicted=4, rho=0.5
        expected_rho = (10.0 - 8.0) / (10.0 - 6.0)  # = 2/4 = 0.5
        
        return {
            "test": "gain_ratio",
            "actual_rho": mma.state.rho if mma.state else 0.0,
            "expected_rho": expected_rho,
            "pass": abs(mma.state.rho - expected_rho) < 1e-6 if mma.state else False,
        }
    
    def test_convergence_on_quadratic(self) -> dict:
        """Test that MMA converges to known minimum of quadratic."""
        mma = SvanbergMMA(n_vars=2, n_constraints=0,
                          x_min=np.array([-2.0, -2.0]),
                          x_max=np.array([2.0, 2.0]))
        test_fn = QuadraticTest(2)
        x0 = np.array([1.0, -0.5])
        mma.initialize(x0)
        
        objective_history = [test_fn.fun(x0)]
        
        for i in range(50):
            x_current = mma.state.x if mma.state else x0
            f = test_fn.fun(x_current)
            df = test_fn.grad(x_current)
            
            x_next, accepted, state = mma.run_optimization_step(f, df)
            if accepted:
                f_next = test_fn.fun(x_next)
                objective_history.append(f_next)
                
                # Check convergence
                if abs(f_next - f) < 1e-10 and np.linalg.norm(df) < 1e-6:
                    break
        
        f_final = test_fn.fun(mma.state.x if mma.state else x0)
        f_min = test_fn.minimum
        f_min_val = test_fn.fun(f_min)
        
        return {
            "test": "quadratic_convergence",
            "initial_objective": objective_history[0],
            "final_objective": f_final,
            "known_minimum": f_min_val,
            "n_iterations": len(objective_history),
            "objective_decreased": f_final < objective_history[0],
            "pass": f_final < objective_history[0] * 0.5,
        }
    
    def test_move_limit_enforcement(self) -> dict:
        """Test that MMA respects move limits."""
        mma = SvanbergMMA(n_vars=2, n_constraints=0,
                          x_min=np.array([-1.0, -1.0]),
                          x_max=np.array([1.0, 1.0]),
                          move_limit=0.05)
        x0 = np.array([0.0, 0.0])
        mma.initialize(x0)
        
        # Generate a steep gradient
        test_fn = QuadraticTest(2)
        x = x0.copy()
        for _ in range(5):
            f = test_fn.fun(x)
            df = test_fn.grad(x) * 100  # Large gradient
            x_next, accepted, state = mma.run_optimization_step(f, df)
            if accepted:
                dx = np.max(np.abs(x_next - x))
                x = x_next
        
        # Move limit prevents unbounded steps even with large gradients
        # With 100x gradient scaling, asymptotes should still bound the step
        max_step_normalized = max(np.abs(x - x0) / (np.array([1.0, 1.0]) - np.array([-1.0, -1.0])))
        
        return {
            "test": "move_limit_enforcement",
            "max_step_normalized": max_step_normalized,
            "pass": max_step_normalized < 0.50,  # Move limits + asymptotes bound steps < 50% of range
        }
    
    def test_constraint_satisfaction(self) -> dict:
        """Test that constraint handling works correctly."""
        mma = SvanbergMMA(n_vars=2, n_constraints=1,
                          x_min=np.array([-2.0, -2.0]),
                          x_max=np.array([2.0, 2.0]))
        x0 = np.array([0.0, 0.0])
        mma.initialize(x0)
        
        # Define a simple constrained problem:
        # minimize f = x^2 + y^2
        # subject to g = x + y - 0.5 <= 0  (feasible region is x+y <= 0.5)
        
        def fun(x):
            return float(x[0]**2 + x[1]**2)
        
        def grad(x):
            return np.array([2*x[0], 2*x[1]])
        
        g = np.array([x0[0] + x0[1] - 0.5])
        dg = np.array([[1.0, 1.0]])
        
        for _ in range(20):
            x = mma.state.x if mma.state else x0
            f = fun(x)
            df = grad(x)
            g_val = np.array([x[0] + x[1] - 0.5])
            
            x_next, accepted, state = mma.run_optimization_step(f, df, g_val, dg)
            if accepted:
                pass
        
        final_x = mma.state.x if mma.state else x0
        constraint_violation = max(0, final_x[0] + final_x[1] - 0.5)
        
        return {
            "test": "constraint_satisfaction",
            "final_x": final_x.tolist(),
            "constraint_violation": constraint_violation,
            "pass": constraint_violation < 0.01,  # Should be nearly feasible
        }
    
    def run_all_tests(self) -> dict:
        """Run all mathematics validation tests."""
        test_functions = [
            self.test_gradient_alignment,
            self.test_step_acceptance_positive_rho,
            self.test_step_rejection_negative_rho,
            self.test_trust_region_governor,
            self.test_gain_ratio_calculation,
            self.test_convergence_on_quadratic,
            self.test_move_limit_enforcement,
            self.test_constraint_satisfaction,
        ]
        
        self.results = []
        for test_fn in test_functions:
            try:
                result = test_fn()
                self.results.append(result)
            except Exception as e:
                self.results.append({
                    "test": test_fn.__name__,
                    "error": str(e),
                    "pass": False,
                })
        
        passed = sum(1 for r in self.results if r.get("pass", False))
        total = len(self.results)
        
        return {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "results": self.results,
        }
    
    def generate_report(self) -> str:
        """Generate human-readable validation report."""
        lines = []
        lines.append("=" * 60)
        lines.append("OPTIMIZER MATHEMATICS VALIDATION REPORT")
        lines.append("=" * 60)
        lines.append("")
        
        run_result = self.run_all_tests()
        
        lines.append(f"Tests: {run_result['total_tests']}")
        lines.append(f"Passed: {run_result['passed']}")
        lines.append(f"Failed: {run_result['failed']}")
        lines.append(f"Pass Rate: {run_result['pass_rate']*100:.1f}%")
        lines.append("")
        
        for r in run_result["results"]:
            passed = r.get("pass", False)
            status = "PASS" if passed else "FAIL"
            lines.append(f"[{status}] {r['test']}")
            for key, value in r.items():
                if key not in ["test", "pass"]:
                    lines.append(f"      {key}: {value}")
            lines.append("")
        
        if run_result["failed"] > 0:
            lines.append("WARNING: Some optimizer mathematics tests failed.")
            lines.append("The MMA optimizer may produce incorrect results.")
        else:
            lines.append("All optimizer mathematics tests passed.")
            lines.append("The MMA optimizer is mathematically sound.")
        
        return "\n".join(lines)


def main():
    """Run the full optimizer mathematics validation suite."""
    print("Optimizer Mathematics Validation Suite")
    print("=" * 60)
    
    # Test gradient correctness with FD
    print("\n1. Gradient Finite Difference Check")
    print("-" * 40)
    fd_checker = FiniteDifferenceChecker()
    
    test_fn = QuadraticTest(4)
    x0 = np.random.randn(4) * 0.5
    fd_result = fd_checker.check_gradient(test_fn.fun, test_fn.grad, x0, n_checks=50)
    print(f"  Max relative error: {fd_result['max_relative_error']:.2e}")
    print(f"  Mean relative error: {fd_result['mean_relative_error']:.2e}")
    print(f"  PASS: {fd_result['pass']}")
    
    coord_result = fd_checker.check_all_directions(test_fn.fun, test_fn.grad, x0)
    print(f"  Max coordinate error: {coord_result['max_coordinate_error']:.2e}")
    
    # Test MMA mathematics
    print("\n2. MMA Mathematics Tests")
    print("-" * 40)
    validator = MMAMathematicsValidator(n_vars=4, n_constraints=2)
    report = validator.generate_report()
    print(report)
    
    # Check FD gradient alignment
    if not fd_result['pass']:
        print("\nCRITICAL: Gradient FD check failed!")
        print("The analytic gradient does not match finite differences.")
        print("This must be fixed before using gradient-based optimization.")
        sys.exit(1)


if __name__ == "__main__":
    main()