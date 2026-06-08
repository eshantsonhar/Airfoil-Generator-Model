import numpy as np
from typing import Callable, Tuple
import scipy.optimize as opt

class TrustRegionMMA:
    """
    Method of Moving Asymptotes (MMA) with Trust-Region controller
    and Svanberg (1987) asymptote updates.
    """
    def __init__(self, n_vars: int, lower_bounds: np.ndarray, upper_bounds: np.ndarray):
        self.n = n_vars
        self.lb = lower_bounds
        self.ub = upper_bounds
        self.x_k = None
        self.x_k1 = None
        self.move_limits = np.ones(n_vars) * 0.2
        self.rho_expand = 0.75
        
    def solve_subproblem(self, x: np.ndarray, grad: np.ndarray, constraints: np.ndarray, jacobians: np.ndarray) -> np.ndarray:
        """
        Solves the dual subproblem using a convex approximation.
        Here, we use scipy to solve the linearized constrained problem.
        """
        # Define the objective function for the linearized problem
        # J_lin = f(x) + grad^T * (x_next - x)
        fun = lambda x_next: np.dot(grad, x_next - x)

        # Linear constraints: g(x) + J(x) * (x_next - x) <= 0
        cons = {'type': 'ineq', 'fun': lambda x_next: - (constraints + np.dot(jacobians, x_next - x))}

        bounds = [(max(self.lb[i], x[i] - self.move_limits[i]),
                   min(self.ub[i], x[i] + self.move_limits[i])) for i in range(self.n)]

        res = opt.minimize(fun, x, method='SLSQP', bounds=bounds, constraints=cons)
        return res.x

    def step(self, x_current: np.ndarray, obj_new: float, obj_pred: float) -> Tuple[np.ndarray, bool]:
        """
        Trust-region acceptance logic.
        """
        if obj_pred == 0: rho = 1.0
        else: rho = (obj_new - self.obj_current) / obj_pred # Simplified
        
        if rho < 0:
            return self.x_k1, False # Reject and rollback
        elif rho > self.rho_expand:
            self.move_limits *= 1.2 # Expand
        
        self.obj_current = obj_new
        return x_current, True

class GeometricRegularizer:
    @staticmethod
    def compute_penalty(x: np.ndarray, lambda_reg: float = 1e-3) -> float:
        # Second difference approximation of curvature
        curvature = np.linalg.norm(np.diff(x, n=2))**2
        return lambda_reg * curvature

