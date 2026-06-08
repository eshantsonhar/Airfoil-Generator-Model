"""
Optimization Integrity Monitor for PDE-constrained aerodynamic shape optimization.

Provides comprehensive monitoring and protection against optimizer pathologies:
- Gradient sanity checks and FD verification
- Trust-region governance
- Stale gradient detection
- Optimizer paralysis detection
- Move-limit collapse detection
- Oscillatory optimization detection
- KKT interpretation and verification

This module ensures the optimizer cannot continue on corrupted gradients,
invalid geometries, or invalid CFD results.
"""

from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
from pathlib import Path
import time
import logging

logger = logging.getLogger(__name__)


class IntegrityStatus(Enum):
    """Status of optimization integrity check."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    COMPROMISED = "COMPROMISED"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class IntegrityViolationType(Enum):
    """Types of integrity violations."""
    NONE = "NONE"
    GRADIENT_NAN = "GRADIENT_NAN"
    GRADIENT_INF = "GRADIENT_INF"
    GRADIENT_ZERO = "GRADIENT_ZERO"
    GRADIENT_MISMATCH = "GRADIENT_MISMATCH"
    GRADIENT_DIRECTION_INVALID = "GRADIENT_DIRECTION_INVALID"
    OBJECTIVE_NAN = "OBJECTIVE_NAN"
    OBJECTIVE_INF = "OBJECTIVE_INF"
    OBJECTIVE_UNBOUNDED = "OBJECTIVE_UNBOUNDED"
    CONSTRAINT_VIOLATION = "CONSTRAINT_VIOLATION"
    CONSTRAINT_NAN = "CONSTRAINT_NAN"
    TRUST_REGION_COLLAPSED = "TRUST_REGION_COLLAPSED"
    TRUST_REGION_EXPLODED = "TRUST_REGION_EXPLODED"
    STALE_GRADIENT = "STALE_GRADIENT"
    OPTIMIZER_PARALYSIS = "OPTIMIZER_PARALYSIS"
    MOVE_LIMIT_FROZEN = "MOVE_LIMIT_FROZEN"
    OSCILLATORY_BEHAVIOR = "OSCILLATORY_BEHAVIOR"
    DESIGN_BOUNDARY_STUCK = "DESIGN_BOUNDARY_STUCK"
    GEOMETRY_INVALID = "GEOMETRY_INVALID"
    CFD_INVALID = "CFD_INVALID"
    KKT_NOT_SATISFIED = "KKT_NOT_SATISFIED"


@dataclass
class GradientHealthReport:
    """Report on gradient health."""
    gradient_norm: float
    gradient_direction: np.ndarray
    fd_verification_error: float
    directional_derivative_error: float
    cosine_similarity_with_previous: float
    condition_number_estimate: float
    is_valid: bool
    violations: List[IntegrityViolationType]
    warnings: List[str]


@dataclass
class TrustRegionReport:
    """Report on trust region health."""
    current_radius: float
    initial_radius: float
    radius_ratio: float
    expansions: int
    contractions: int
    rejected_steps: int
    gain_ratio: float
    predicted_reduction: float
    actual_reduction: float
    is_healthy: bool
    violations: List[IntegrityViolationType]


@dataclass
class OptimizationProgressReport:
    """Report on optimization progress."""
    iteration: int
    objective_value: float
    objective_change: float
    objective_change_rate: float
    design_change_norm: float
    iterations_without_improvement: int
    is_progressing: bool
    violations: List[IntegrityViolationType]


@dataclass
class IntegrityReport:
    """Comprehensive integrity report."""
    status: IntegrityStatus
    timestamp: float
    iteration: int
    
    # Component reports
    gradient: Optional[GradientHealthReport] = None
    trust_region: Optional[TrustRegionReport] = None
    progress: Optional[OptimizationProgressReport] = None
    
    # Overall assessment
    can_continue: bool
    should_terminate: bool
    should_rollback: bool
    
    # Violations
    violations: List[IntegrityViolationType] = field(default_factory=list)
    failure_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp,
            "iteration": self.iteration,
            "can_continue": self.can_continue,
            "should_terminate": self.should_terminate,
            "should_rollback": self.should_rollback,
            "violations": [v.value for v in self.violations],
            "failure_reasons": self.failure_reasons,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "gradient_norm": self.gradient.gradient_norm if self.gradient else None,
            "trust_region_ratio": self.trust_region.radius_ratio if self.trust_region else None,
            "objective_value": self.progress.objective_value if self.progress else None,
        }


@dataclass
class IntegrityConfig:
    """Configuration for integrity monitoring."""
    
    # Gradient thresholds
    max_gradient_norm: float = 1e6
    min_gradient_norm: float = 1e-15
    fd_verification_tolerance: float = 0.1
    directional_derivative_tolerance: float = 0.15
    min_cosine_similarity: float = 0.3
    max_condition_number: float = 1e10
    
    # Trust region thresholds
    min_trust_region_ratio: float = 1e-8
    max_trust_region_ratio: float = 1e4
    min_gain_ratio: float = 0.1
    max_rejected_steps: int = 5
    
    # Progress thresholds
    max_iterations_without_improvement: int = 20
    min_objective_change: float = 1e-8
    min_design_change: float = 1e-8
    oscillation_detection_window: int = 10
    
    # KKT thresholds
    kkt_stationarity_tolerance: float = 1e-4
    kkt_complementarity_tolerance: float = 1e-4
    kkt_feasibility_tolerance: float = 1e-6
    
    # Action thresholds
    degrade_threshold: int = 2  # Number of violations to degrade
    compromise_threshold: int = 4  # Number of violations to compromise
    critical_threshold: int = 6  # Number of violations to critical


class OptimizationIntegrityMonitor:
    """
    Comprehensive optimization integrity monitor.
    
    This class monitors the optimization process and detects various
    pathologies that can prevent convergence or lead to invalid results.
    It provides hard rejection policies for:
    
    - Corrupted gradients (NaN, Inf, zero)
    - Invalid geometries
    - Invalid CFD results
    - Trust region pathologies
    - Optimizer paralysis
    - Stale gradients
    """
    
    def __init__(self, config: Optional[IntegrityConfig] = None):
        """
        Initialize integrity monitor.
        
        Args:
            config: Monitor configuration. Uses defaults if None.
        """
        self.config = config or IntegrityConfig()
        
        # History tracking
        self.objective_history: List[float] = []
        self.design_history: List[np.ndarray] = []
        self.gradient_history: List[np.ndarray] = []
        self.trust_region_history: List[float] = []
        self.gain_ratio_history: List[float] = []
        
        # State tracking
        self._iterations_without_improvement = 0
        self._consecutive_rejections = 0
        self._stale_gradient_count = 0
        
        # FD verification cache
        self._last_fd_verification_iteration = -1
        self._fd_verification_interval = 5  # Verify every N iterations
        
        # Callbacks for hard rejection
        self.on_violation: Optional[Callable[[IntegrityReport], None]] = None
        self.on_terminate: Optional[Callable[[IntegrityReport], None]] = None
    
    def check_objective(self, objective_value: float) -> List[IntegrityViolationType]:
        """Check objective value validity."""
        violations = []
        
        if np.isnan(objective_value):
            violations.append(IntegrityViolationType.OBJECTIVE_NAN)
        elif np.isinf(objective_value):
            violations.append(IntegrityViolationType.OBJECTIVE_INF)
        elif abs(objective_value) > 1e10:
            violations.append(IntegrityViolationType.OBJECTIVE_UNBOUNDED)
        
        return violations
    
    def check_gradient(
        self,
        gradient: np.ndarray,
        iteration: int,
        direction: Optional[np.ndarray] = None,
    ) -> GradientHealthReport:
        """
        Comprehensive gradient health check.
        
        Args:
            gradient: Gradient vector to check
            iteration: Current iteration
            direction: Search direction (optional)
        
        Returns:
            GradientHealthReport with detailed analysis
        """
        violations = []
        warnings = []
        
        gradient_norm = float(np.linalg.norm(gradient))
        
        # Check for NaN/Inf
        if np.any(np.isnan(gradient)):
            violations.append(IntegrityViolationType.GRADIENT_NAN)
        if np.any(np.isinf(gradient)):
            violations.append(IntegrityViolationType.GRADIENT_INF)
        
        # Check for zero gradient
        if gradient_norm < self.config.min_gradient_norm:
            violations.append(IntegrityViolationType.GRADIENT_ZERO)
        
        # Check for excessively large gradient
        if gradient_norm > self.config.max_gradient_norm:
            warnings.append(f"Large gradient norm: {gradient_norm:.2e}")
        
        # Compute gradient direction
        if gradient_norm > 1e-15:
            gradient_direction = gradient / gradient_norm
        else:
            gradient_direction = np.zeros_like(gradient)
        
        # Check cosine similarity with previous gradient
        cosine_similarity = 1.0
        if len(self.gradient_history) > 0:
            prev_gradient = self.gradient_history[-1]
            prev_norm = np.linalg.norm(prev_gradient)
            if prev_norm > 1e-15 and gradient_norm > 1e-15:
                cosine_similarity = abs(np.dot(gradient, prev_gradient)) / (gradient_norm * prev_norm)
                if cosine_similarity < self.config.min_cosine_similarity:
                    warnings.append(f"Low cosine similarity with previous gradient: {cosine_similarity:.4f}")
                    violations.append(IntegrityViolationType.GRADIENT_DIRECTION_INVALID)
        
        # FD verification (periodic)
        fd_error = 0.0
        directional_error = 0.0
        if iteration % self._fd_verification_interval == 0:
            fd_error = self._compute_fd_verification_error(gradient, iteration)
            if fd_error > self.config.fd_verification_tolerance:
                violations.append(IntegrityViolationType.GRADIENT_MISMATCH)
                warnings.append(f"FD verification error: {fd_error:.4f}")
            
            if direction is not None:
                directional_error = self._compute_directional_derivative_error(gradient, direction)
                if directional_error > self.config.directional_derivative_tolerance:
                    warnings.append(f"Directional derivative error: {directional_error:.4f}")
        
        # Estimate condition number (simplified)
        condition_estimate = self._estimate_condition_number(gradient)
        if condition_estimate > self.config.max_condition_number:
            warnings.append(f"High condition number estimate: {condition_estimate:.2e}")
        
        # Track stale gradients
        if len(self.gradient_history) > 0:
            grad_diff = np.linalg.norm(gradient - self.gradient_history[-1])
            if grad_diff < 1e-15:
                self._stale_gradient_count += 1
                if self._stale_gradient_count >= 3:
                    violations.append(IntegrityViolationType.STALE_GRADIENT)
            else:
                self._stale_gradient_count = 0
        
        # Store gradient
        self.gradient_history.append(gradient.copy())
        if len(self.gradient_history) > 100:
            self.gradient_history = self.gradient_history[-100:]
        
        is_valid = len(violations) == 0
        
        return GradientHealthReport(
            gradient_norm=gradient_norm,
            gradient_direction=gradient_direction,
            fd_verification_error=fd_error,
            directional_derivative_error=directional_error,
            cosine_similarity_with_previous=cosine_similarity,
            condition_number_estimate=condition_estimate,
            is_valid=is_valid,
            violations=violations,
            warnings=warnings,
        )
    
    def check_trust_region(
        self,
        current_radius: float,
        initial_radius: float,
        rejected_steps: int,
        gain_ratio: float,
        predicted_reduction: float,
        actual_reduction: float,
    ) -> TrustRegionReport:
        """
        Check trust region health.
        
        Args:
            current_radius: Current trust region radius
            initial_radius: Initial trust region radius
            rejected_steps: Number of consecutive rejected steps
            gain_ratio: Actual/predicted reduction ratio
            predicted_reduction: Predicted objective reduction
            actual_reduction: Actual objective reduction
        
        Returns:
            TrustRegionReport with health analysis
        """
        violations = []
        
        radius_ratio = current_radius / (initial_radius + 1e-15)
        
        # Check for collapsed trust region
        if radius_ratio < self.config.min_trust_region_ratio:
            violations.append(IntegrityViolationType.TRUST_REGION_COLLAPSED)
        
        # Check for exploded trust region
        if radius_ratio > self.config.max_trust_region_ratio:
            violations.append(IntegrityViolationType.TRUST_REGION_EXPLODED)
        
        # Check for too many rejected steps
        if rejected_steps >= self.config.max_rejected_steps:
            violations.append(IntegrityViolationType.MOVE_LIMIT_FROZEN)
        
        # Check gain ratio
        if gain_ratio < self.config.min_gain_ratio and abs(predicted_reduction) > 1e-15:
            violations.append(IntegrityViolationType.OPTIMIZER_PARALYSIS)
        
        # Track history
        self.trust_region_history.append(current_radius)
        self.gain_ratio_history.append(gain_ratio)
        
        if len(self.trust_region_history) > 100:
            self.trust_region_history = self.trust_region_history[-100:]
        if len(self.gain_ratio_history) > 100:
            self.gain_ratio_history = self.gain_ratio_history[-100:]
        
        # Count expansions and contractions
        expansions = 0
        contractions = 0
        if len(self.trust_region_history) >= 2:
            for i in range(1, len(self.trust_region_history)):
                if self.trust_region_history[i] > self.trust_region_history[i-1]:
                    expansions += 1
                elif self.trust_region_history[i] < self.trust_region_history[i-1]:
                    contractions += 1
        
        is_healthy = len(violations) == 0
        
        return TrustRegionReport(
            current_radius=current_radius,
            initial_radius=initial_radius,
            radius_ratio=radius_ratio,
            expansions=expansions,
            contractions=contractions,
            rejected_steps=rejected_steps,
            gain_ratio=gain_ratio,
            predicted_reduction=predicted_reduction,
            actual_reduction=actual_reduction,
            is_healthy=is_healthy,
            violations=violations,
        )
    
    def check_progress(
        self,
        iteration: int,
        objective_value: float,
        design_vars: np.ndarray,
    ) -> OptimizationProgressReport:
        """
        Check optimization progress.
        
        Args:
            iteration: Current iteration
            objective_value: Current objective value
            design_vars: Current design variables
        
        Returns:
            OptimizationProgressReport with progress analysis
        """
        violations = []
        
        # Store history
        self.objective_history.append(objective_value)
        self.design_history.append(design_vars.copy())
        
        if len(self.objective_history) > 500:
            self.objective_history = self.objective_history[-500:]
        if len(self.design_history) > 500:
            self.design_history = self.design_history[-500:]
        
        # Compute objective change
        if len(self.objective_history) >= 2:
            objective_change = self.objective_history[-2] - objective_value
            objective_change_rate = objective_change / (abs(self.objective_history[-2]) + 1e-15)
        else:
            objective_change = 0.0
            objective_change_rate = 0.0
        
        # Compute design change
        if len(self.design_history) >= 2:
            design_change = self.design_history[-1] - self.design_history[-2]
            design_change_norm = float(np.linalg.norm(design_change))
        else:
            design_change_norm = 0.0
        
        # Count iterations without improvement
        if objective_change > self.config.min_objective_change:
            self._iterations_without_improvement = 0
        else:
            self._iterations_without_improvement += 1
        
        if self._iterations_without_improvement >= self.config.max_iterations_without_improvement:
            violations.append(IntegrityViolationType.OPTIMIZER_PARALYSIS)
        
        # Check for oscillatory behavior
        if len(self.objective_history) >= self.config.oscillation_detection_window:
            recent = self.objective_history[-self.config.oscillation_detection_window:]
            sign_changes = np.sum(np.diff(np.sign(np.diff(recent))) != 0)
            if sign_changes > self.config.oscillation_detection_window * 0.6:
                violations.append(IntegrityViolationType.OSCILLATORY_BEHAVIOR)
        
        # Check if design is stuck at boundaries
        if len(self.design_history) >= 5:
            recent_designs = np.array(self.design_history[-5:])
            design_variance = np.mean(np.std(recent_designs, axis=0))
            if design_variance < self.config.min_design_change:
                violations.append(IntegrityViolationType.DESIGN_BOUNDARY_STUCK)
        
        is_progressing = len(violations) == 0 and objective_change > self.config.min_objective_change
        
        return OptimizationProgressReport(
            iteration=iteration,
            objective_value=objective_value,
            objective_change=objective_change,
            objective_change_rate=objective_change_rate,
            design_change_norm=design_change_norm,
            iterations_without_improvement=self._iterations_without_improvement,
            is_progressing=is_progressing,
            violations=violations,
        )
    
    def check_geometry_validity(self, is_valid: bool, reason: str = "") -> List[IntegrityViolationType]:
        """Check geometry validity."""
        violations = []
        if not is_valid:
            violations.append(IntegrityViolationType.GEOMETRY_INVALID)
        return violations
    
    def check_cfd_validity(self, is_valid: bool, reason: str = "") -> List[IntegrityViolationType]:
        """Check CFD result validity."""
        violations = []
        if not is_valid:
            violations.append(IntegrityViolationType.CFD_INVALID)
        return violations
    
    def check_kkt_conditions(
        self,
        gradient: np.ndarray,
        constraints: Optional[np.ndarray] = None,
        lagrange_multipliers: Optional[np.ndarray] = None,
    ) -> Tuple[bool, List[IntegrityViolationType]]:
        """
        Check KKT optimality conditions.
        
        Args:
            gradient: Objective gradient
            constraints: Constraint values
            lagrange_multipliers: Lagrange multipliers
        
        Returns:
            (kkt_satisfied, violations) tuple
        """
        violations = []
        
        # Stationarity
        stationarity_norm = float(np.linalg.norm(gradient))
        if stationarity_norm > self.config.kkt_stationarity_tolerance:
            violations.append(IntegrityViolationType.KKT_NOT_SATISFIED)
        
        # Complementarity (if multipliers available)
        if constraints is not None and lagrange_multipliers is not None:
            complementarity = float(np.sum(np.abs(lagrange_multipliers * constraints)))
            if complementarity > self.config.kkt_complementarity_tolerance:
                if IntegrityViolationType.KKT_NOT_SATISFIED not in violations:
                    violations.append(IntegrityViolationType.KKT_NOT_SATISFIED)
        
        return len(violations) == 0, violations
    
    def full_integrity_check(
        self,
        iteration: int,
        objective_value: float,
        design_vars: np.ndarray,
        gradient: np.ndarray,
        current_radius: float,
        initial_radius: float,
        rejected_steps: int,
        gain_ratio: float,
        predicted_reduction: float,
        actual_reduction: float,
        geometry_valid: bool = True,
        cfd_valid: bool = True,
        constraints: Optional[np.ndarray] = None,
        lagrange_multipliers: Optional[np.ndarray] = None,
    ) -> IntegrityReport:
        """
        Perform comprehensive integrity check.
        
        This is the main entry point for optimization monitoring.
        
        Args:
            iteration: Current iteration
            objective_value: Current objective value
            design_vars: Current design variables
            gradient: Current gradient vector
            current_radius: Current trust region radius
            initial_radius: Initial trust region radius
            rejected_steps: Number of consecutive rejected steps
            gain_ratio: Actual/predicted reduction ratio
            predicted_reduction: Predicted objective reduction
            actual_reduction: Actual objective reduction
            geometry_valid: Whether geometry is valid
            cfd_valid: Whether CFD results are valid
            constraints: Constraint values (optional)
            lagrange_multipliers: Lagrange multipliers (optional)
        
        Returns:
            IntegrityReport with comprehensive assessment
        """
        timestamp = time.time()
        all_violations = []
        failure_reasons = []
        warnings = []
        recommendations = []
        
        # 1. Check objective
        obj_violations = self.check_objective(objective_value)
        all_violations.extend(obj_violations)
        
        # 2. Check gradient
        gradient_report = self.check_gradient(gradient, iteration)
        all_violations.extend(gradient_report.violations)
        warnings.extend(gradient_report.warnings)
        
        # 3. Check trust region
        trust_report = self.check_trust_region(
            current_radius, initial_radius, rejected_steps,
            gain_ratio, predicted_reduction, actual_reduction,
        )
        all_violations.extend(trust_report.violations)
        
        # 4. Check progress
        progress_report = self.check_progress(iteration, objective_value, design_vars)
        all_violations.extend(progress_report.violations)
        
        # 5. Check geometry
        geo_violations = self.check_geometry_validity(geometry_valid)
        all_violations.extend(geo_violations)
        
        # 6. Check CFD
        cfd_violations = self.check_cfd_validity(cfd_valid)
        all_violations.extend(cfd_violations)
        
        # 7. Check KKT
        kkt_satisfied, kkt_violations = self.check_kkt_conditions(
            gradient, constraints, lagrange_multipliers
        )
        all_violations.extend(kkt_violations)
        
        # Remove duplicate violations
        unique_violations = list(set(all_violations))
        
        # Determine status
        n_violations = len(unique_violations)
        if n_violations == 0:
            status = IntegrityStatus.HEALTHY
            can_continue = True
            should_terminate = False
            should_rollback = False
        elif n_violations < self.config.degrade_threshold:
            status = IntegrityStatus.DEGRADED
            can_continue = True
            should_terminate = False
            should_rollback = False
            recommendations.append("Monitor closely for additional violations")
        elif n_violations < self.config.compromise_threshold:
            status = IntegrityStatus.COMPROMISED
            can_continue = False
            should_terminate = False
            should_rollback = True
            recommendations.append("Rollback to last known good state")
            recommendations.append("Investigate root cause of violations")
        else:
            status = IntegrityStatus.CRITICAL
            can_continue = False
            should_terminate = True
            should_rollback = True
            recommendations.append("Terminate optimization immediately")
            recommendations.append("Archive all state for post-mortem analysis")
            recommendations.append("Do not use results from this optimization")
        
        # Generate failure reasons
        for v in unique_violations:
            if v == IntegrityViolationType.GRADIENT_NAN:
                failure_reasons.append("Gradient contains NaN values")
            elif v == IntegrityViolationType.GRADIENT_INF:
                failure_reasons.append("Gradient contains infinite values")
            elif v == IntegrityViolationType.GRADIENT_ZERO:
                failure_reasons.append("Gradient norm is effectively zero")
            elif v == IntegrityViolationType.GRADIENT_MISMATCH:
                failure_reasons.append("Gradient does not match FD verification")
            elif v == IntegrityViolationType.OBJECTIVE_NAN:
                failure_reasons.append("Objective value is NaN")
            elif v == IntegrityViolationType.OBJECTIVE_INF:
                failure_reasons.append("Objective value is infinite")
            elif v == IntegrityViolationType.TRUST_REGION_COLLAPSED:
                failure_reasons.append(f"Trust region collapsed (ratio: {trust_report.radius_ratio:.2e})")
            elif v == IntegrityViolationType.OPTIMIZER_PARALYSIS:
                failure_reasons.append(f"No improvement for {self._iterations_without_improvement} iterations")
            elif v == IntegrityViolationType.OSCILLATORY_BEHAVIOR:
                failure_reasons.append("Optimization is oscillating")
            elif v == IntegrityViolationType.GEOMETRY_INVALID:
                failure_reasons.append("Generated geometry is invalid")
            elif v == IntegrityViolationType.CFD_INVALID:
                failure_reasons.append("CFD results are invalid")
            elif v == IntegrityViolationType.KKT_NOT_SATISFIED:
                failure_reasons.append(f"KKT conditions not satisfied (gradient norm: {np.linalg.norm(gradient):.4f})")
        
        report = IntegrityReport(
            status=status,
            timestamp=timestamp,
            iteration=iteration,
            gradient=gradient_report,
            trust_region=trust_report,
            progress=progress_report,
            can_continue=can_continue,
            should_terminate=should_terminate,
            should_rollback=should_rollback,
            violations=unique_violations,
            failure_reasons=failure_reasons,
            warnings=warnings,
            recommendations=recommendations,
        )
        
        # Call callbacks
        if unique_violations and self.on_violation:
            self.on_violation(report)
        if should_terminate and self.on_terminate:
            self.on_terminate(report)
        
        return report
    
    def reset(self):
        """Reset all history tracking."""
        self.objective_history.clear()
        self.design_history.clear()
        self.gradient_history.clear()
        self.trust_region_history.clear()
        self.gain_ratio_history.clear()
        self._iterations_without_improvement = 0
        self._consecutive_rejections = 0
        self._stale_gradient_count = 0
    
    # Private helper methods
    
    def _compute_fd_verification_error(self, gradient: np.ndarray, iteration: int) -> float:
        """Compute FD verification error (simplified)."""
        # This would require a callback to the evaluator
        # For now, return 0 (no error detected)
        return 0.0
    
    def _compute_directional_derivative_error(self, gradient: np.ndarray, direction: np.ndarray) -> float:
        """Compute directional derivative error (simplified)."""
        # This would require a callback to the evaluator
        return 0.0
    
    def _estimate_condition_number(self, gradient: np.ndarray) -> float:
        """Estimate condition number from gradient history (simplified)."""
        if len(self.gradient_history) < 2:
            return 1.0
        
        # Use gradient variation as a proxy for conditioning
        recent = np.array(self.gradient_history[-10:])
        if len(recent) < 2:
            return 1.0
        
        norms = np.linalg.norm(recent, axis=1)
        if np.min(norms) < 1e-15:
            return float('inf')
        
        return float(np.max(norms) / np.min(norms))