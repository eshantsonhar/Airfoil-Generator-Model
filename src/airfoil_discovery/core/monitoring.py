from pathlib import Path

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class ASOMetrics:
    # 1. CFD Convergence
    primal_residual: float
    adjoint_residual: float
    
    # 2. Optimization
    objective_value: float
    constraint_residuals: np.ndarray
    
    # 3. KKT
    stationarity_norm: float
    complementarity: float
    
    # 4. Gradient Health
    gradient_norm: float
    gradient_angle_change: float
    
    # 5. Trust Region
    gain_ratio: float
    step_size: float
    
    # 6. Geometric
    smoothness_metric: float
    gradient_fd_error: float = 0.0

class RealTimeMonitor:
    """
    Tracks and logs ASO pipeline health.
    """
    def __init__(self):
        self.history: List[ASOMetrics] = []
        
    def log(self, metrics: ASOMetrics):
        self.history.append(metrics)
        # Check alerts
        if metrics.gain_ratio < 0:
            print(f"ALERT: Bad trust region step detected! Ratio: {metrics.gain_ratio}")
        if metrics.primal_residual > 1e-3:
            print(f"ALERT: Poor CFD convergence! Residual: {metrics.primal_residual}")

    def get_dashboard_data(self) -> Dict:
        return {
            "primal_residuals": [m.primal_residual for m in self.history],
            "adjoint_residuals": [m.adjoint_residual for m in self.history],
            "objective": [m.objective_value for m in self.history],
            "stationarity": [m.stationarity_norm for m in self.history],
            "complementarity": [m.complementarity for m in self.history],
            "gradient_norm": [m.gradient_norm for m in self.history],
            "gain_ratio": [m.gain_ratio for m in self.history],
            "smoothness": [m.smoothness_metric for m in self.history],
            "grad_fd_error": [m.gradient_fd_error for m in self.history]
        }

class GradientAuditor:
    """
    Adaptive gradient validator.
    """
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.failure_count = 0

    def check_along_step(self, x: np.ndarray, step: np.ndarray, grad: np.ndarray, J_current: float) -> float:
        """
        Directional derivative check: phi'(0) = grad^T * step
        """
        eps = 1e-4
        x_new = x + eps * step
        # Need to re-evaluate base if not provided
        J_new = self.evaluator.run_evaluation(x_new, Path("./temp_fd"), "L1").cd
        directional_fd = (J_new - J_current) / eps
        directional_adj = np.dot(grad, step)

        return abs(directional_adj - directional_fd) / (abs(directional_fd) + 1e-12)

    def multi_dim_check(self, x: np.ndarray, grad: np.ndarray, J_base: float) -> float:
        """
        Samples k=3 random dimensions for statistical confidence.
        """
        errors = []
        for _ in range(3):
            idx = np.random.randint(0, len(x))
            eps = 1e-4
            x_plus = x.copy()
            x_plus[idx] += eps
            J_plus = self.evaluator.run_evaluation(x_plus, Path("./temp_fd"), "L1").cd
            fd = (J_plus - J_base) / eps
            errors.append(abs(grad[idx] - fd) / (abs(fd) + 1e-12))
        return float(np.mean(errors))
