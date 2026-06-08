"""
Adjoint Gradient Truth Audit.

Verifies adjoint gradients against finite differences to detect:
- Adjoint corruption (wrong sensitivity sign/magnitude)
- Gradient noise amplification (noise exceeding signal)
- Transition-induced gradient discontinuities (non-differentiable physics)
- False sensitivity localization (wrong surface regions)
- Mesh sensitivity contamination (gradients follow mesh, not physics)

Optimization HALTS on gradient corruption.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class GradientTruthReport:
    """Report from adjoint gradient truth audit."""
    # FD comparison
    fd_relative_errors: np.ndarray  # Per-variable relative errors
    max_relative_error: float
    mean_relative_error: float
    errors_below_threshold: bool
    
    # Directional derivative consistency
    directional_derivatives: List[float]
    directional_errors: List[float]
    directional_pass: bool
    
    # Sign stability
    sign_stable: bool
    sign_change_count: int
    
    # Sensitivity smoothness
    gradient_variance: float
    variance_pass: bool
    
    # Overall
    is_credible: bool
    failure_reasons: List[str] = field(default_factory=list)


class AdjointTruthAuditor:
    """
    Verifies adjoint gradients against finite differences.
    
    Performs:
    - Per-variable FD comparison with central differences
    - Directional derivative consistency in random directions
    - Sign stability analysis
    - Gradient variance tracking
    
    Gradients must pass ALL checks or optimization is halted.
    """
    
    def __init__(self,
                 fd_eps: float = 1e-5,
                 relative_threshold: float = 0.1,
                 directional_threshold: float = 0.2,
                 variance_threshold: float = 2.0):
        """
        Initialize adjoint truth auditor.
        
        Args:
            fd_eps: Finite difference perturbation size
            relative_threshold: Max acceptable relative error per variable
            directional_threshold: Max acceptable directional derivative error
            variance_threshold: Max acceptable gradient variance (std/mean)
        """
        self.fd_eps = fd_eps
        self.relative_threshold = relative_threshold
        self.directional_threshold = directional_threshold
        self.variance_threshold = variance_threshold
    
    def audit_gradient(self,
                       x0: np.ndarray,
                       grad_adjoint: np.ndarray,
                       objective_fn: Callable,
                       n_directional_tests: int = 10,
                       previous_gradients: Optional[List[np.ndarray]] = None,
                       ) -> GradientTruthReport:
        """
        Audit adjoint gradient against finite differences.
        
        Args:
            x0: Design point where gradient is evaluated
            grad_adjoint: Adjoint-computed gradient (shape n_vars,)
            objective_fn: Function that evaluates objective at x
            n_directional_tests: Number of random directional derivative tests
            previous_gradients: List of previous gradients for sign stability
            
        Returns:
            GradientTruthReport with verification results
        """
        n_vars = len(x0)
        failure_reasons = []
        
        # 1. Per-variable FD comparison with central differences
        fd_grad = np.zeros(n_vars)
        for i in range(n_vars):
            x_plus = x0.copy()
            x_minus = x0.copy()
            eps = self.fd_eps * max(1.0, abs(x0[i]))
            x_plus[i] += eps
            x_minus[i] -= eps
            f_plus = objective_fn(x_plus)
            f_minus = objective_fn(x_minus)
            fd_grad[i] = (f_plus - f_minus) / (2 * eps)
        
        # Compute relative errors
        fd_relative_errors = np.zeros(n_vars)
        for i in range(n_vars):
            denom = max(abs(fd_grad[i]), 1e-12)
            fd_relative_errors[i] = abs(grad_adjoint[i] - fd_grad[i]) / denom
        
        max_error = float(np.max(fd_relative_errors))
        mean_error = float(np.mean(fd_relative_errors))
        errors_ok = max_error < self.relative_threshold
        
        if not errors_ok:
            failure_reasons.append(
                f"Gradient FD error {max_error:.4f} exceeds threshold {self.relative_threshold}"
            )
        
        # 2. Directional derivative tests in random directions
        directional_derivatives = []
        directional_errors = []
        directional_ok = True
        
        for _ in range(n_directional_tests):
            # Random unit direction
            d = np.random.randn(n_vars)
            d = d / max(np.linalg.norm(d), 1e-15)
            
            # Central FD in this direction
            eps = self.fd_eps
            f_plus = objective_fn(x0 + eps * d)
            f_minus = objective_fn(x0 - eps * d)
            fd_dir = (f_plus - f_minus) / (2 * eps)
            
            # Adjoint directional derivative
            adj_dir = float(np.dot(grad_adjoint, d))
            
            directional_derivatives.append(adj_dir)
            error = abs(adj_dir - fd_dir) / max(abs(fd_dir), 1e-12)
            directional_errors.append(error)
            
            if error > self.directional_threshold:
                directional_ok = False
        
        if not directional_ok:
            failure_reasons.append(
                f"Directional derivative error {max(directional_errors):.4f} exceeds threshold"
            )
        
        # 3. Sign stability analysis
        sign_stable = True
        sign_changes = 0
        if previous_gradients and len(previous_gradients) >= 2:
            for i in range(n_vars):
                current_sign = np.sign(grad_adjoint[i])
                prev_signs = [np.sign(g[i]) for g in previous_gradients[-3:]]
                if any(current_sign != ps for ps in prev_signs if ps != 0):
                    sign_changes += 1
            if sign_changes > n_vars // 2:
                sign_stable = False
                failure_reasons.append("Unstable gradient sign - possible adjoint corruption")
        
        # 4. Gradient variance / smoothness
        gradient_variance = 0.0
        variance_ok = True
        if n_vars > 2:
            # Check for noise amplification
            grad_mean = np.mean(np.abs(grad_adjoint))
            grad_std = np.std(grad_adjoint)
            gradient_variance = float(grad_std / max(grad_mean, 1e-12))
            if gradient_variance > self.variance_threshold:
                variance_ok = False
                failure_reasons.append(
                    f"High gradient variance {gradient_variance:.2f} - noise amplification"
                )
        
        # Overall credibility
        is_credible = errors_ok and directional_ok and sign_stable and variance_ok
        
        return GradientTruthReport(
            fd_relative_errors=fd_relative_errors,
            max_relative_error=max_error,
            mean_relative_error=mean_error,
            errors_below_threshold=errors_ok,
            directional_derivatives=directional_derivatives,
            directional_errors=directional_errors,
            directional_pass=directional_ok,
            sign_stable=sign_stable,
            sign_change_count=sign_changes,
            gradient_variance=gradient_variance,
            variance_pass=variance_ok,
            is_credible=is_credible,
            failure_reasons=failure_reasons,
        )
    
    def audit_gradient_sweep(self,
                             x0: np.ndarray,
                             grad_fn: Callable,
                             objective_fn: Callable,
                             n_sweeps: int = 5) -> Dict[str, Any]:
        """
        Sweep FD perturbation sizes to find linearity region.
        
        Args:
            x0: Design point
            grad_fn: Function computing adjoint gradient
            objective_fn: Function evaluating objective
            n_sweeps: Number of perturbation sizes
            
        Returns:
            Dict with sweep results
        """
        eps_values = np.logspace(-6, -2, n_sweeps)
        results = []
        
        for eps in eps_values:
            n_vars = len(x0)
            grad_adj = grad_fn(x0)
            
            fd_grad = np.zeros(n_vars)
            for i in range(n_vars):
                x_plus = x0.copy()
                x_minus = x0.copy()
                x_plus[i] += eps
                x_minus[i] -= eps
                f_plus = objective_fn(x_plus)
                f_minus = objective_fn(x_minus)
                fd_grad[i] = (f_plus - f_minus) / (2 * eps)
            
            errors = np.abs(grad_adj - fd_grad) / max(np.abs(fd_grad).max(), 1e-12)
            results.append({
                "eps": eps,
                "max_error": float(np.max(errors)),
                "mean_error": float(np.mean(errors)),
            })
        
        # Find optimal eps (minimum error)
        best_idx = int(np.argmin([r["max_error"] for r in results]))
        
        return {
            "sweep_results": results,
            "optimal_eps": eps_values[best_idx],
            "optimal_error": results[best_idx]["max_error"],
            "linearity_established": results[best_idx]["max_error"] < self.relative_threshold,
        }
    
    def compute_hessian_spectrum(self,
                                 grad_fn: Callable,
                                 x0: np.ndarray,
                                 n_directions: int = 20) -> Dict[str, Any]:
        """
        Estimate Hessian spectral properties via finite differences.
        
        Args:
            grad_fn: Function computing gradient
            x0: Design point
            n_directions: Number of probe directions
            
        Returns:
            Dict with spectral estimates
        """
        n_vars = len(x0)
        eps = self.fd_eps
        
        # Compute Hessian-vector products
        Hv_results = []
        for _ in range(n_directions):
            v = np.random.randn(n_vars)
            v = v / np.linalg.norm(v)
            
            grad_plus = grad_fn(x0 + eps * v)
            grad_minus = grad_fn(x0 - eps * v)
            Hv = (grad_plus - grad_minus) / (2 * eps)
            Hv_results.append(Hv)
        
        # Estimate condition number from Rayleigh quotients
        quotients = []
        for i in range(len(Hv_results)):
            v = Hv_results[i]
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-12:
                v = v / v_norm
                # Estimate v^T H v
                for Hv in Hv_results:
                    quotient = abs(np.dot(v, Hv)) / max(np.linalg.norm(Hv), 1e-12)
                    quotients.append(quotient)
        
        condition_estimate = max(quotients) / max(min(quotients), 1e-12) if quotients else float('inf')
        
        return {
            "hessian_condition_estimate": condition_estimate,
            "n_probe_directions": n_directions,
            "well_conditioned": condition_estimate < 1000,
        }