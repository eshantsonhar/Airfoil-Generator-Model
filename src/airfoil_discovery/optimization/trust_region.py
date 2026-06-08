import numpy as np

class TrustRegionGovernor:
    """
    Implements a gain-ratio (rho) based trust region check.
    Ensures the CFD truth aligns with the optimizer's linear/reciprocal model prediction.
    """
    def __init__(self, rho_accept: float = 0.1, rho_shrink: float = 0.25):
        self.rho_accept = rho_accept
        self.rho_shrink = rho_shrink
        
    def evaluate_step(self, f_actual_delta: float, f_predicted_delta: float) -> dict:
        """
        Calculates rho = actual_reduction / predicted_reduction.
        Returns whether to accept the step and how to adjust the move limits.
        """
        if abs(f_predicted_delta) < 1e-12:
            rho = 1.0
        else:
            rho = f_actual_delta / f_predicted_delta
            
        accepted = rho > self.rho_accept
        action = "KEEP"
        
        if rho < self.rho_shrink:
            action = "SHRINK"
        elif rho > 0.75:
            action = "EXPAND"
            
        return {
            "rho": rho,
            "accepted": accepted,
            "action": action
        }
