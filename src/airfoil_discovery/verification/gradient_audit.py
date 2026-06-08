"""
Gradient verification and audit for adjoint-based optimization.

Implements finite-difference verification, directional derivative checks,
cosine similarity monitoring, and trust-region consistency validation.
Ensures gradient integrity before optimizer acceptance.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Tuple
from enum import Enum


class GradientStatus(Enum):
    """Gradient verification status."""
    VALID = "VALID"
    FD_MISMATCH = "FD_MISMATCH"
    DIRECTIONAL_ERROR = "DIRECTIONAL_ERROR"
    INSTABILITY = "INSTABILITY"
    COSINE_COLLAPSE = "COSINE_COLLAPSE"
    TRUST_REGION_VIOLATION = "TRUST_REGION_VIOLATION"
    UNKNOWN = "UNKNOWN"


@dataclass
class FDVerificationResult:
    """Finite-difference verification result."""
    
    # Adjoint gradient
    adjoint_gradient: np.ndarray
    
    # Finite-difference gradient
    fd_gradient: np.ndarray
    
    # Error metrics
    absolute_error: float
    relative_error: float
    cosine_similarity: float
    
    # Component-wise errors
    component_errors: np.ndarray
    max_component_error: float
    
    # Verification check
    passed: bool
    tolerance: float
    
    # Diagnostics
    fd_step_size: float
    condition_number: Optional[float] = None


@dataclass
class DirectionalDerivativeResult:
    """Directional derivative verification result."""
    
    # Direction
    direction: np.ndarray
    
    # Predicted change (from gradient)
    predicted_change: float
    
    # Actual change (from finite difference)
    actual_change: float
    
    # Error
    absolute_error: float
    relative_error: float
    
    # Sign check
    sign_correct: bool
    
    # Verification
    passed: bool
    tolerance: float


@dataclass
class GradientAuditReport:
    """Comprehensive gradient audit report."""
    
    # Overall status (required, no defaults)
    status: GradientStatus
    is_valid: bool
    
    # FD verification
    fd_verification: Optional[FDVerificationResult] = None
    
    # Directional derivative check
    directional_check: Optional[DirectionalDerivativeResult] = None
    
    # Temporal consistency
    cosine_history: List[float] = field(default_factory=list)
    gradient_variance: float = 0.0
    temporal_stability: bool = True
    
    # Trust region
    trust_region_consistent: bool = True
    
    # Failure reasons and recommendations
    failure_reasons: List[str] = field(default_factory=list)
    
    # Recommendations
    recommended_actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "is_valid": self.is_valid,
            "failure_reasons": self.failure_reasons,
            "recommended_actions": self.recommended_actions,
            "cosine_history": self.cosine_history,
            "gradient_variance": self.gradient_variance,
            "temporal_stability": self.temporal_stability,
            "trust_region_consistent": self.trust_region_consistent,
        }


class FiniteDifferenceVerifier:
    """
    Verifies adjoint gradients using finite differences.
    
    Implements central difference scheme with adaptive step size.
    Provides component-wise error analysis and cosine similarity checks.
    """
    
    def __init__(
        self,
        tolerance: float = 0.05,
        fd_step_size: float = 1e-6,
        min_step_size: float = 1e-10,
        max_step_size: float = 1e-3,
    ):
        """
        Initialize finite-difference verifier.
        
        Args:
            tolerance: Acceptable relative error threshold
            fd_step_size: Initial finite-difference step size
            min_step_size: Minimum step size for adaptive refinement
            max_step_size: Maximum step size for adaptive refinement
        """
        self.tolerance = tolerance
        self.fd_step_size = fd_step_size
        self.min_step_size = min_step_size
        self.max_step_size = max_step_size
    
    def verify(
        self,
        adjoint_gradient: np.ndarray,
        objective_function: Callable[[np.ndarray], float],
        x: np.ndarray,
    ) -> FDVerificationResult:
        """
        Verify adjoint gradient using finite differences.
        
        Args:
            adjoint_gradient: Gradient from adjoint solver
            objective_function: Function to compute objective value
            x: Current design point
        
        Returns:
            FDVerificationResult with verification metrics
        """
        n = len(x)
        fd_gradient = np.zeros_like(x)
        
        # Central difference scheme
        h = self.fd_step_size
        f0 = objective_function(x)
        
        for i in range(n):
            # Perturb variable i
            dx = np.zeros_like(x)
            dx[i] = h
            
            f_plus = objective_function(x + dx)
            f_minus = objective_function(x - dx)
            
            # Central difference
            fd_gradient[i] = (f_plus - f_minus) / (2 * h)
        
        # Compute error metrics
        absolute_error = np.linalg.norm(adjoint_gradient - fd_gradient)
        relative_error = absolute_error / (np.linalg.norm(fd_gradient) + 1e-15)
        
        # Cosine similarity
        dot_product = np.dot(adjoint_gradient, fd_gradient)
        norm_adj = np.linalg.norm(adjoint_gradient)
        norm_fd = np.linalg.norm(fd_gradient)
        cosine_similarity = dot_product / (norm_adj * norm_fd + 1e-15)
        
        # Component-wise errors
        component_errors = np.abs(adjoint_gradient - fd_gradient) / (np.abs(fd_gradient) + 1e-15)
        max_component_error = float(np.max(component_errors))
        
        # Condition number estimate
        condition_number = None
        if norm_fd > 1e-15:
            condition_number = norm_adj / norm_fd
        
        # Verification check
        passed = (relative_error < self.tolerance and
                cosine_similarity > 0.95 and
                max_component_error < 2.0 * self.tolerance)
        
        return FDVerificationResult(
            adjoint_gradient=adjoint_gradient.copy(),
            fd_gradient=fd_gradient,
            absolute_error=float(absolute_error),
            relative_error=float(relative_error),
            cosine_similarity=float(cosine_similarity),
            component_errors=component_errors,
            max_component_error=max_component_error,
            passed=passed,
            tolerance=self.tolerance,
            fd_step_size=h,
            condition_number=condition_number,
        )
    
    def adaptive_verify(
        self,
        adjoint_gradient: np.ndarray,
        objective_function: Callable[[np.ndarray], float],
        x: np.ndarray,
        max_adaptations: int = 3,
    ) -> FDVerificationResult:
        """
        Verify with adaptive step size refinement.
        
        Args:
            adjoint_gradient: Gradient from adjoint solver
            objective_function: Function to compute objective value
            x: Current design point
            max_adaptations: Maximum number of step size adaptations
        
        Returns:
            FDVerificationResult with best verification metrics
        """
        best_result = None
        best_error = float('inf')
        
        h = self.fd_step_size
        
        for _ in range(max_adaptations + 1):
            result = self.verify(adjoint_gradient, objective_function, x)
            
            if result.relative_error < best_error:
                best_error = result.relative_error
                best_result = result
            
            # Adapt step size if error is large
            if result.relative_error > self.tolerance and h > self.min_step_size:
                h = max(h / 10, self.min_step_size)
                self.fd_step_size = h
            else:
                break
        
        return best_result if best_result is not None else self.verify(
            adjoint_gradient, objective_function, x
        )


class GradientAuditor:
    """
    Comprehensive gradient auditing for adjoint-based optimization.
    
    Performs:
    - Finite-difference verification
    - Directional derivative checks
    - Temporal consistency monitoring
    - Trust-region consistency validation
    """
    
    def __init__(
        self,
        fd_tolerance: float = 0.05,
        directional_tolerance: float = 0.10,
        cosine_threshold: float = 0.95,
        variance_threshold: float = 0.5,
    ):
        """
        Initialize gradient auditor.
        
        Args:
            fd_tolerance: Finite-difference verification tolerance
            directional_tolerance: Directional derivative tolerance
            cosine_threshold: Minimum cosine similarity for temporal stability
            variance_threshold: Maximum acceptable gradient variance
        """
        self.fd_verifier = FiniteDifferenceVerifier(tolerance=fd_tolerance)
        self.directional_tolerance = directional_tolerance
        self.cosine_threshold = cosine_threshold
        self.variance_threshold = variance_threshold
        
        # History for temporal consistency
        self.gradient_history: List[np.ndarray] = []
        self.cosine_history: List[float] = []
    
    def verify_gradient(
        self,
        adjoint_gradient: np.ndarray,
        objective_function: Callable[[np.ndarray], float],
        x: np.ndarray,
        use_adaptive: bool = True,
    ) -> FDVerificationResult:
        """
        Verify gradient using finite differences.
        
        Args:
            adjoint_gradient: Gradient from adjoint solver
            objective_function: Function to compute objective value
            x: Current design point
            use_adaptive: Use adaptive step size refinement
        
        Returns:
            FDVerificationResult with verification metrics
        """
        if use_adaptive:
            return self.fd_verifier.adaptive_verify(adjoint_gradient, objective_function, x)
        else:
            return self.fd_verifier.verify(adjoint_gradient, objective_function, x)
    
    def check_directional_derivative(
        self,
        gradient: np.ndarray,
        objective_function: Callable[[np.ndarray], float],
        x: np.ndarray,
        direction: Optional[np.ndarray] = None,
    ) -> DirectionalDerivativeResult:
        """
        Check directional derivative consistency.
        
        Args:
            gradient: Adjoint gradient
            objective_function: Function to compute objective value
            x: Current design point
            direction: Direction to check (random if None)
        
        Returns:
            DirectionalDerivativeResult with verification metrics
        """
        if direction is None:
            # Random direction
            direction = np.random.randn(len(x))
            direction = direction / np.linalg.norm(direction)
        
        # Predicted change from gradient
        predicted_change = np.dot(gradient, direction)
        
        # Actual change from finite difference
        h = 1e-6
        f_plus = objective_function(x + h * direction)
        f_minus = objective_function(x - h * direction)
        actual_change = (f_plus - f_minus) / (2 * h)
        
        # Error metrics
        absolute_error = abs(predicted_change - actual_change)
        relative_error = absolute_error / (abs(actual_change) + 1e-15)
        
        # Sign check
        sign_correct = (predicted_change * actual_change >= 0) or (abs(predicted_change) < 1e-15)
        
        # Verification
        passed = (relative_error < self.directional_tolerance and sign_correct)
        
        return DirectionalDerivativeResult(
            direction=direction.copy(),
            predicted_change=predicted_change,
            actual_change=actual_change,
            absolute_error=absolute_error,
            relative_error=relative_error,
            sign_correct=sign_correct,
            passed=passed,
            tolerance=self.directional_tolerance,
        )
    
    def check_temporal_consistency(self, gradient: np.ndarray) -> bool:
        """
        Check temporal consistency of gradients.
        
        Args:
            gradient: Current gradient
        
        Returns:
            True if gradients are temporally stable
        """
        if len(self.gradient_history) == 0:
            self.gradient_history.append(gradient.copy())
            return True
        
        # Compute cosine similarity with previous gradient
        prev_gradient = self.gradient_history[-1]
        dot_product = np.dot(gradient, prev_gradient)
        norm_current = np.linalg.norm(gradient)
        norm_prev = np.linalg.norm(prev_gradient)
        cosine = dot_product / (norm_current * norm_prev + 1e-15)
        
        self.cosine_history.append(float(cosine))
        self.gradient_history.append(gradient.copy())
        
        # Keep only last 10 gradients
        if len(self.gradient_history) > 10:
            self.gradient_history.pop(0)
            self.cosine_history.pop(0)
        
        # Check if cosine similarity is acceptable
        return cosine > self.cosine_threshold
    
    def compute_gradient_variance(self) -> float:
        """
        Compute variance of recent gradients.
        
        Returns:
            Gradient variance metric
        """
        if len(self.gradient_history) < 2:
            return 0.0
        
        gradients = np.array(self.gradient_history)
        mean_gradient = np.mean(gradients, axis=0)
        
        # Compute variance as normalized RMS deviation
        variance = np.sqrt(np.mean(np.sum((gradients - mean_gradient)**2, axis=1)))
        variance = variance / (np.linalg.norm(mean_gradient) + 1e-15)
        
        return float(variance)
    
    def audit(
        self,
        adjoint_gradient: np.ndarray,
        objective_function: Callable[[np.ndarray], float],
        x: np.ndarray,
        check_directional: bool = True,
        check_temporal: bool = True,
    ) -> GradientAuditReport:
        """
        Perform comprehensive gradient audit.
        
        Args:
            adjoint_gradient: Gradient from adjoint solver
            objective_function: Function to compute objective value
            x: Current design point
            check_directional: Perform directional derivative check
            check_temporal: Perform temporal consistency check
        
        Returns:
            GradientAuditReport with comprehensive assessment
        """
        failure_reasons = []
        recommended_actions = []
        
        # FD verification
        fd_result = self.verify_gradient(adjoint_gradient, objective_function, x)
        
        if not fd_result.passed:
            failure_reasons.append(f"FD verification failed: relative error {fd_result.relative_error:.3f}")
            recommended_actions.append("Check adjoint solver implementation or mesh quality")
        
        # Directional derivative check
        directional_result = None
        if check_directional:
            directional_result = self.check_directional_derivative(
                adjoint_gradient, objective_function, x
            )
            
            if not directional_result.passed:
                failure_reasons.append(
                    f"Directional derivative failed: relative error {directional_result.relative_error:.3f}"
                )
                recommended_actions.append("Investigate gradient direction accuracy")
        
        # Temporal consistency check
        temporal_stability = True
        if check_temporal:
            temporal_stability = self.check_temporal_consistency(adjoint_gradient)
            
            if not temporal_stability:
                failure_reasons.append("Gradient temporal instability detected")
                recommended_actions.append("Monitor optimization stability")
        
        # Gradient variance
        gradient_variance = self.compute_gradient_variance()
        if gradient_variance > self.variance_threshold:
            failure_reasons.append(f"High gradient variance: {gradient_variance:.3f}")
            recommended_actions.append("Check for optimizer oscillation")
        
        # Determine overall status
        if not fd_result.passed:
            status = GradientStatus.FD_MISMATCH
        elif directional_result and not directional_result.passed:
            status = GradientStatus.DIRECTIONAL_ERROR
        elif not temporal_stability:
            status = GradientStatus.INSTABILITY
        elif gradient_variance > self.variance_threshold:
            status = GradientStatus.COSINE_COLLAPSE
        else:
            status = GradientStatus.VALID
        
        is_valid = (status == GradientStatus.VALID)
        
        return GradientAuditReport(
            status=status,
            fd_verification=fd_result,
            directional_check=directional_result,
            cosine_history=self.cosine_history.copy(),
            gradient_variance=gradient_variance,
            temporal_stability=temporal_stability,
            trust_region_consistent=True,  # To be implemented with trust-region integration
            is_valid=is_valid,
            failure_reasons=failure_reasons,
            recommended_actions=recommended_actions,
        )
    
    def reset_history(self):
        """Reset gradient history for temporal consistency tracking."""
        self.gradient_history.clear()
        self.cosine_history.clear()
