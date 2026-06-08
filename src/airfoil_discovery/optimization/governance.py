"""
Optimizer governance for PDE-constrained aerodynamic shape optimization.

Implements comprehensive monitoring and governance of the optimization
process to detect and handle optimizer pathologies such as:
- Optimizer paralysis (no progress despite valid gradients)
- Stale gradients (gradients not updating)
- Move-limit freezing (design variables stuck at bounds)
- Trust-region deadlock (trust region collapsing)
- Oscillatory convergence (cycling between designs)
- Gain ratio collapse (model prediction vs actual mismatch)

The governance system ensures robust optimization behavior and provides
recovery mechanisms when issues are detected.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import warnings


class OptimizerHealthStatus(Enum):
    """Overall optimizer health status."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class OptimizerViolationType(Enum):
    """Types of optimizer violations."""
    NONE = "NONE"
    NO_PROGRESS = "NO_PROGRESS"
    STALE_GRADIENTS = "STALE_GRADIENTS"
    MOVE_LIMIT_FROZEN = "MOVE_LIMIT_FROZEN"
    TRUST_REGION_COLLAPSED = "TRUST_REGION_COLLAPSED"
    OSCILLATORY_BEHAVIOR = "OSCILLATORY_BEHAVIOR"
    GAIN_RATIO_COLLAPSE = "GAIN_RATIO_COLLAPSE"
    KKT_STAGNATION = "KKT_STAGNATION"
    CONSTRAINT_VIOLATION = "CONSTRAINT_VIOLATION"
    DESIGN_VARIABLE_BOUNDARY = "DESIGN_VARIABLE_BOUNDARY"
    STEP_REJECTED = "STEP_REJECTED"


@dataclass
class ProgressMetrics:
    """Optimization progress metrics."""
    
    # Objective progress
    objective_value: float
    objective_improvement: float
    objective_improvement_rate: float
    
    # Design variable progress
    design_change_norm: float
    design_change_direction: np.ndarray
    
    # Iteration metrics
    iterations_without_improvement: int
    total_iterations: int
    
    # Violations
    violations: List[OptimizerViolationType] = field(default_factory=list)


@dataclass
class GradientHealthMetrics:
    """Gradient health metrics."""
    
    # Gradient magnitude
    gradient_norm: float
    gradient_norm_history: List[float]
    
    # Gradient consistency
    gradient_variance: float
    gradient_direction_change: float
    
    # Gradient staleness
    iterations_without_gradient_update: int
    gradient_stale: bool
    
    # Violations
    violations: List[OptimizerViolationType] = field(default_factory=list)


@dataclass
class TrustRegionMetrics:
    """Trust region health metrics."""
    
    # Trust region size
    current_radius: float
    initial_radius: float
    radius_ratio: float
    
    # Trust region updates
    expansions: int
    contractions: int
    rejected_steps: int
    
    # Trust region health
    collapsed: bool
    too_large: bool
    
    # Violations
    violations: List[OptimizerViolationType] = field(default_factory=list)


@dataclass
class GainRatioMetrics:
    """Gain ratio (actual vs predicted reduction) metrics."""
    
    # Current gain ratio
    current_gain_ratio: float
    gain_ratio_history: List[float]
    
    # Statistics
    mean_gain_ratio: float
    gain_ratio_variance: float
    
    # Model quality
    model_quality: str  # "excellent", "good", "acceptable", "poor"
    
    # Violations
    violations: List[OptimizerViolationType] = field(default_factory=list)


@dataclass
class KKTMetrics:
    """Karush-Kuhn-Tucker optimality condition metrics."""
    
    # Stationarity
    stationarity_norm: float
    stationarity_tolerance: float
    
    # Complementarity
    complementarity: float
    complementarity_tolerance: float
    
    # Feasibility
    primal_feasibility: float
    dual_feasibility: float
    feasibility_tolerance: float
    
    # KKT status
    kkt_satisfied: bool
    
    # Violations
    violations: List[OptimizerViolationType] = field(default_factory=list)


@dataclass
class OscillationMetrics:
    """Oscillation detection metrics."""
    
    # Design oscillation
    design_oscillation_detected: bool
    design_oscillation_amplitude: float
    design_oscillation_frequency: float
    
    # Objective oscillation
    objective_oscillation_detected: bool
    objective_oscillation_amplitude: float
    
    # Cycle detection
    cycle_detected: bool
    cycle_length: int
    
    # Violations
    violations: List[OptimizerViolationType] = field(default_factory=list)


@dataclass
class OptimizerGovernanceReport:
    """Comprehensive optimizer governance report."""
    
    # Overall status
    status: OptimizerHealthStatus
    
    # Component metrics
    progress: Optional[ProgressMetrics] = None
    gradient_health: Optional[GradientHealthMetrics] = None
    trust_region: Optional[TrustRegionMetrics] = None
    gain_ratio: Optional[GainRatioMetrics] = None
    kkt: Optional[KKTMetrics] = None
    oscillation: Optional[OscillationMetrics] = None
    
    # Overall assessment
    is_healthy: bool
    can_continue: bool
    
    # Violations
    violations: List[OptimizerViolationType] = field(default_factory=list)
    failure_reasons: List[str] = field(default_factory=list)
    
    # Recommendations
    recommended_actions: List[str] = field(default_factory=list)
    recovery_strategies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "is_healthy": self.is_healthy,
            "can_continue": self.can_continue,
            "violations": [v.value for v in self.violations],
            "failure_reasons": self.failure_reasons,
            "recommended_actions": self.recommended_actions,
            "progress": {
                "objective_value": self.progress.objective_value if self.progress else None,
                "objective_improvement": self.progress.objective_improvement if self.progress else None,
                "iterations_without_improvement": self.progress.iterations_without_improvement if self.progress else None,
            } if self.progress else None,
            "trust_region": {
                "radius_ratio": self.trust_region.radius_ratio if self.trust_region else None,
                "collapsed": self.trust_region.collapsed if self.trust_region else None,
            } if self.trust_region else None,
            "gain_ratio": {
                "current_gain_ratio": self.gain_ratio.current_gain_ratio if self.gain_ratio else None,
                "model_quality": self.gain_ratio.model_quality if self.gain_ratio else None,
            } if self.gain_ratio else None,
        }


@dataclass
class OptimizerGovernanceConfig:
    """Configuration for optimizer governance."""
    
    # Progress thresholds
    min_objective_improvement: float = 1e-6
    max_iterations_without_improvement: int = 10
    min_design_change: float = 1e-8
    
    # Gradient health thresholds
    gradient_stale_iterations: int = 5
    gradient_variance_threshold: float = 0.01
    gradient_direction_change_threshold: float = 0.1
    
    # Trust region thresholds
    min_trust_region_ratio: float = 1e-6
    max_trust_region_ratio: float = 10.0
    max_rejected_steps: int = 5
    
    # Gain ratio thresholds
    excellent_gain_ratio: float = 0.9
    good_gain_ratio: float = 0.7
    acceptable_gain_ratio: float = 0.3
    poor_gain_ratio: float = 0.1
    
    # KKT thresholds
    stationarity_tolerance: float = 1e-4
    complementarity_tolerance: float = 1e-4
    feasibility_tolerance: float = 1e-6
    
    # Oscillation thresholds
    oscillation_amplitude_threshold: float = 0.01
    min_cycle_length: int = 3
    
    # Governance policy
    strict_mode: bool = True
    early_termination_on_critical: bool = True


class OptimizerGovernor:
    """
    Comprehensive optimizer governance for PDE-constrained optimization.
    
    This class monitors the optimization process and detects various
    pathologies that can prevent convergence or lead to poor solutions.
    It provides:
    
    - Progress monitoring and stagnation detection
    - Gradient health assessment
    - Trust region health monitoring
    - Gain ratio analysis
    - KKT condition verification
    - Oscillation and cycle detection
    - Recovery strategies and recommendations
    """
    
    def __init__(self, config: Optional[OptimizerGovernanceConfig] = None):
        """
        Initialize optimizer governor.
        
        Args:
            config: Governance configuration. Uses defaults if None.
        """
        self.config = config or OptimizerGovernanceConfig()
        
        # History tracking
        self.objective_history: List[float] = []
        self.design_history: List[np.ndarray] = []
        self.gradient_history: List[np.ndarray] = []
        self.gain_ratio_history: List[float] = []
        self.trust_region_history: List[float] = []
    
    def analyze_progress(
        self,
        objective_value: float,
        design_vars: np.ndarray,
        iteration: int,
    ) -> ProgressMetrics:
        """
        Analyze optimization progress.
        
        Args:
            objective_value: Current objective function value
            design_vars: Current design variables
            iteration: Current iteration number
        
        Returns:
            ProgressMetrics with progress analysis
        """
        self.objective_history.append(objective_value)
        self.design_history.append(design_vars.copy())
        
        # Compute objective improvement
        if len(self.objective_history) >= 2:
            objective_improvement = self.objective_history[-2] - objective_value
            objective_improvement_rate = objective_improvement / (abs(self.objective_history[-2]) + 1e-15)
        else:
            objective_improvement = 0.0
            objective_improvement_rate = 0.0
        
        # Compute design change
        if len(self.design_history) >= 2:
            design_change = self.design_history[-1] - self.design_history[-2]
            design_change_norm = float(np.linalg.norm(design_change))
            design_change_direction = design_change / (np.linalg.norm(design_change) + 1e-15)
        else:
            design_change_norm = 0.0
            design_change_direction = np.zeros_like(design_vars)
        
        # Count iterations without improvement
        iterations_without_improvement = 0
        for i in range(len(self.objective_history) - 1, 0, -1):
            if self.objective_history[i] >= self.objective_history[i-1]:
                iterations_without_improvement += 1
            else:
                break
        
        # Detect violations
        violations = []
        if iterations_without_improvement >= self.config.max_iterations_without_improvement:
            violations.append(OptimizerViolationType.NO_PROGRESS)
        
        if design_change_norm < self.config.min_design_change and iteration > 5:
            violations.append(OptimizerViolationType.NO_PROGRESS)
        
        return ProgressMetrics(
            objective_value=objective_value,
            objective_improvement=objective_improvement,
            objective_improvement_rate=objective_improvement_rate,
            design_change_norm=design_change_norm,
            design_change_direction=design_change_direction,
            iterations_without_improvement=iterations_without_improvement,
            total_iterations=iteration,
            violations=violations,
        )
    
    def analyze_gradient_health(
        self,
        gradient: np.ndarray,
        iteration: int,
    ) -> GradientHealthMetrics:
        """
        Analyze gradient health and consistency.
        
        Args:
            gradient: Current gradient vector
            iteration: Current iteration number
        
        Returns:
            GradientHealthMetrics with gradient analysis
        """
        self.gradient_history.append(gradient.copy())
        
        gradient_norm = float(np.linalg.norm(gradient))
        
        # Gradient norm history
        gradient_norm_history = [float(np.linalg.norm(g)) for g in self.gradient_history]
        
        # Gradient variance (are gradients changing?)
        if len(self.gradient_history) >= 3:
            recent_gradients = np.array(self.gradient_history[-5:])
            mean_gradient = np.mean(recent_gradients, axis=0)
            gradient_variance = float(np.sqrt(np.mean(np.sum((recent_gradients - mean_gradient)**2, axis=1))))
        else:
            gradient_variance = 0.0
        
        # Gradient direction change
        if len(self.gradient_history) >= 2:
            prev_gradient = self.gradient_history[-2]
            norm_product = np.linalg.norm(gradient) * np.linalg.norm(prev_gradient)
            if norm_product > 1e-15:
                gradient_direction_change = 1.0 - abs(np.dot(gradient, prev_gradient)) / norm_product
            else:
                gradient_direction_change = 0.0
        else:
            gradient_direction_change = 0.0
        
        # Check for stale gradients
        gradient_stale = False
        iterations_without_update = 0
        
        if len(self.gradient_history) >= 2:
            for i in range(len(self.gradient_history) - 1, 0, -1):
                grad_diff = np.linalg.norm(self.gradient_history[i] - self.gradient_history[i-1])
                if grad_diff < 1e-15:
                    iterations_without_update += 1
                else:
                    break
        
        if iterations_without_update >= self.config.gradient_stale_iterations:
            gradient_stale = True
        
        # Detect violations
        violations = []
        if gradient_stale:
            violations.append(OptimizerViolationType.STALE_GRADIENTS)
        
        if gradient_variance < self.config.gradient_variance_threshold and iteration > 10:
            violations.append(OptimizerViolationType.STALE_GRADIENTS)
        
        return GradientHealthMetrics(
            gradient_norm=gradient_norm,
            gradient_norm_history=gradient_norm_history,
            gradient_variance=gradient_variance,
            gradient_direction_change=gradient_direction_change,
            iterations_without_gradient_update=iterations_without_update,
            gradient_stale=gradient_stale,
            violations=violations,
        )
    
    def analyze_trust_region(
        self,
        current_radius: float,
        initial_radius: float,
        rejected_steps: int,
    ) -> TrustRegionMetrics:
        """
        Analyze trust region health.
        
        Args:
            current_radius: Current trust region radius
            initial_radius: Initial trust region radius
            rejected_steps: Number of consecutive rejected steps
        
        Returns:
            TrustRegionMetrics with trust region analysis
        """
        self.trust_region_history.append(current_radius)
        
        radius_ratio = current_radius / (initial_radius + 1e-15)
        
        # Count expansions and contractions
        expansions = 0
        contractions = 0
        if len(self.trust_region_history) >= 2:
            for i in range(1, len(self.trust_region_history)):
                if self.trust_region_history[i] > self.trust_region_history[i-1]:
                    expansions += 1
                elif self.trust_region_history[i] < self.trust_region_history[i-1]:
                    contractions += 1
        
        # Check for collapse
        collapsed = radius_ratio < self.config.min_trust_region_ratio
        too_large = radius_ratio > self.config.max_trust_region_ratio
        
        # Detect violations
        violations = []
        if collapsed:
            violations.append(OptimizerViolationType.TRUST_REGION_COLLAPSED)
        if rejected_steps >= self.config.max_rejected_steps:
            violations.append(OptimizerViolationType.STEP_REJECTED)
        
        return TrustRegionMetrics(
            current_radius=current_radius,
            initial_radius=initial_radius,
            radius_ratio=radius_ratio,
            expansions=expansions,
            contractions=contractions,
            rejected_steps=rejected_steps,
            collapsed=collapsed,
            too_large=too_large,
            violations=violations,
        )
    
    def analyze_gain_ratio(
        self,
        predicted_reduction: float,
        actual_reduction: float,
    ) -> GainRatioMetrics:
        """
        Analyze gain ratio (model prediction quality).
        
        Args:
            predicted_reduction: Predicted objective reduction from model
            actual_reduction: Actual objective reduction achieved
        
        Returns:
            GainRatioMetrics with gain ratio analysis
        """
        # Compute gain ratio
        if abs(predicted_reduction) > 1e-15:
            gain_ratio = actual_reduction / predicted_reduction
        else:
            gain_ratio = 1.0 if abs(actual_reduction) < 1e-15 else float('inf')
        
        self.gain_ratio_history.append(gain_ratio)
        
        # Statistics
        mean_gain_ratio = float(np.mean(self.gain_ratio_history[-10:]))
        gain_ratio_variance = float(np.var(self.gain_ratio_history[-10:]))
        
        # Model quality assessment
        if gain_ratio > self.config.excellent_gain_ratio:
            model_quality = "excellent"
        elif gain_ratio > self.config.good_gain_ratio:
            model_quality = "good"
        elif gain_ratio > self.config.acceptable_gain_ratio:
            model_quality = "acceptable"
        elif gain_ratio > self.config.poor_gain_ratio:
            model_quality = "poor"
        else:
            model_quality = "very_poor"
        
        # Detect violations
        violations = []
        if gain_ratio < self.config.poor_gain_ratio:
            violations.append(OptimizerViolationType.GAIN_RATIO_COLLAPSE)
        
        return GainRatioMetrics(
            current_gain_ratio=gain_ratio,
            gain_ratio_history=self.gain_ratio_history.copy(),
            mean_gain_ratio=mean_gain_ratio,
            gain_ratio_variance=gain_ratio_variance,
            model_quality=model_quality,
            violations=violations,
        )
    
    def analyze_kkt_conditions(
        self,
        gradient: np.ndarray,
        constraints: Optional[np.ndarray] = None,
        lagrange_multipliers: Optional[np.ndarray] = None,
        constraint_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> KKTMetrics:
        """
        Analyze KKT optimality conditions.
        
        Args:
            gradient: Objective gradient
            constraints: Constraint values (optional)
            lagrange_multipliers: Lagrange multipliers (optional)
            constraint_bounds: (lower, upper) constraint bounds (optional)
        
        Returns:
            KKTMetrics with KKT analysis
        """
        # Stationarity (gradient should be zero at optimum)
        stationarity_norm = float(np.linalg.norm(gradient))
        stationarity_satisfied = stationarity_norm < self.config.stationarity_tolerance
        
        # Complementarity (lambda * g = 0 for active constraints)
        complementarity = 0.0
        if constraints is not None and lagrange_multipliers is not None:
            complementarity = float(np.sum(np.abs(lagrange_multipliers * constraints)))
        
        complementarity_satisfied = complementarity < self.config.complementarity_tolerance
        
        # Primal feasibility (constraints satisfied)
        primal_feasibility = 0.0
        if constraints is not None and constraint_bounds is not None:
            lower, upper = constraint_bounds
            violation_lower = np.maximum(0, lower - constraints)
            violation_upper = np.maximum(0, constraints - upper)
            primal_feasibility = float(np.max(np.concatenate([violation_lower, violation_upper])))
        
        primal_feasible = primal_feasibility < self.config.feasibility_tolerance
        
        # Dual feasibility (multipliers non-negative for inequality constraints)
        dual_feasibility = 0.0
        if lagrange_multipliers is not None:
            dual_feasibility = float(np.sum(np.abs(np.minimum(0, lagrange_multipliers))))
        
        # Overall KKT satisfaction
        kkt_satisfied = stationarity_satisfied and complementarity_satisfied and primal_feasible
        
        # Detect violations
        violations = []
        if not stationarity_satisfied:
            violations.append(OptimizerViolationType.KKT_STAGNATION)
        if not primal_feasible:
            violations.append(OptimizerViolationType.CONSTRAINT_VIOLATION)
        
        return KKTMetrics(
            stationarity_norm=stationarity_norm,
            stationarity_tolerance=self.config.stationarity_tolerance,
            complementarity=complementarity,
            complementarity_tolerance=self.config.complementarity_tolerance,
            primal_feasibility=primal_feasibility,
            dual_feasibility=dual_feasibility,
            feasibility_tolerance=self.config.feasibility_tolerance,
            kkt_satisfied=kkt_satisfied,
            violations=violations,
        )
    
    def detect_oscillations(
        self,
        design_history: Optional[List[np.ndarray]] = None,
        objective_history: Optional[List[float]] = None,
    ) -> OscillationMetrics:
        """
        Detect oscillatory behavior and cycles.
        
        Args:
            design_history: List of design variable vectors
            objective_history: List of objective values
        
        Returns:
            OscillationMetrics with oscillation analysis
        """
        # Use stored history if not provided
        if design_history is None:
            design_history = self.design_history
        if objective_history is None:
            objective_history = self.objective_history
        
        if len(design_history) < 6 or len(objective_history) < 6:
            return OscillationMetrics(
                design_oscillation_detected=False,
                design_oscillation_amplitude=0.0,
                design_oscillation_frequency=0.0,
                objective_oscillation_detected=False,
                objective_oscillation_amplitude=0.0,
                cycle_detected=False,
                cycle_length=0,
                violations=[],
            )
        
        # Design oscillation
        recent_designs = np.array(design_history[-10:])
        design_amplitude = float(np.max(np.std(recent_designs, axis=0)))
        design_oscillation_detected = design_amplitude > self.config.oscillation_amplitude_threshold
        
        # Objective oscillation
        recent_objectives = objective_history[-10:]
        objective_amplitude = float(np.std(recent_objectives))
        objective_oscillation_detected = objective_amplitude > self.config.oscillation_amplitude_threshold
        
        # Cycle detection (simple pattern matching)
        cycle_detected = False
        cycle_length = 0
        
        if len(objective_history) >= 6:
            # Check for alternating pattern
            signs = np.sign(np.diff(objective_history[-6:]))
            if len(signs) >= 4:
                alternating = all(signs[i] != signs[i+1] for i in range(len(signs)-1))
                if alternating:
                    cycle_detected = True
                    cycle_length = 2
        
        # Detect violations
        violations = []
        if design_oscillation_detected or objective_oscillation_detected:
            violations.append(OptimizerViolationType.OSCILLATORY_BEHAVIOR)
        
        return OscillationMetrics(
            design_oscillation_detected=design_oscillation_detected,
            design_oscillation_amplitude=design_amplitude,
            design_oscillation_frequency=0.5 if cycle_detected else 0.0,
            objective_oscillation_detected=objective_oscillation_detected,
            objective_oscillation_amplitude=objective_amplitude,
            cycle_detected=cycle_detected,
            cycle_length=cycle_length,
            violations=violations,
        )
    
    def compute_health_score(
        self,
        progress: ProgressMetrics,
        gradient_health: GradientHealthMetrics,
        trust_region: TrustRegionMetrics,
        gain_ratio: GainRatioMetrics,
    ) -> float:
        """
        Compute overall optimizer health score (0-1, higher = healthier).
        
        Args:
            progress: Progress metrics
            gradient_health: Gradient health metrics
            trust_region: Trust region metrics
            gain_ratio: Gain ratio metrics
        
        Returns:
            Overall health score (0-1)
        """
        score = 1.0
        
        # Penalty for no progress
        if progress.iterations_without_improvement > 0:
            score -= 0.2 * min(1.0, progress.iterations_without_improvement / 5)
        
        # Penalty for stale gradients
        if gradient_health.gradient_stale:
            score -= 0.3
        
        # Penalty for trust region issues
        if trust_region.collapsed:
            score -= 0.4
        elif trust_region.rejected_steps > 0:
            score -= 0.1 * min(1.0, trust_region.rejected_steps / 3)
        
        # Penalty for poor gain ratio
        if gain_ratio.model_quality == "poor":
            score -= 0.3
        elif gain_ratio.model_quality == "very_poor":
            score -= 0.5
        
        return max(0.0, min(1.0, score))
    
    def get_recovery_strategies(
        self,
        violations: List[OptimizerViolationType],
        health_score: float,
    ) -> List[str]:
        """
        Get recovery strategies based on detected violations.
        
        Args:
            violations: List of detected violations
            health_score: Overall health score
        
        Returns:
            List of recovery strategy recommendations
        """
        strategies = []
        
        if OptimizerViolationType.NO_PROGRESS in violations:
            strategies.append("Try random perturbation to escape local plateau")
            strategies.append("Increase exploration fraction in candidate generation")
            strategies.append("Consider restarting with different initial design")
        
        if OptimizerViolationType.STALE_GRADIENTS in violations:
            strategies.append("Force gradient recomputation with finer FD step")
            strategies.append("Check adjoint solver convergence")
            strategies.append("Verify mesh quality and resolution")
        
        if OptimizerViolationType.TRUST_REGION_COLLAPSED in violations:
            strategies.append("Reset trust region to initial size")
            strategies.append("Use more accurate model (higher fidelity)")
            strategies.append("Reduce move limits temporarily")
        
        if OptimizerViolationType.GAIN_RATIO_COLLAPSE in violations:
            strategies.append("Switch to more conservative model")
            strategies.append("Reduce trust region size")
            strategies.append("Check for discontinuous design space")
        
        if OptimizerViolationType.OSCILLATORY_BEHAVIOR in violations:
            strategies.append("Apply gradient filtering or smoothing")
            strategies.append("Reduce step size or move limits")
            strategies.append("Add damping to optimization update")
        
        if health_score < 0.3:
            strategies.append("Consider terminating optimization and restarting")
            strategies.append("Archive current state for post-mortem analysis")
        
        return strategies
    
    def govern(
        self,
        objective_value: float,
        design_vars: np.ndarray,
        gradient: np.ndarray,
        current_radius: float,
        initial_radius: float,
        rejected_steps: int,
        predicted_reduction: float,
        actual_reduction: float,
        iteration: int,
        constraints: Optional[np.ndarray] = None,
        lagrange_multipliers: Optional[np.ndarray] = None,
    ) -> OptimizerGovernanceReport:
        """
        Perform comprehensive optimizer governance.
        
        This is the main entry point for optimizer monitoring.
        
        Args:
            objective_value: Current objective function value
            design_vars: Current design variables
            gradient: Current gradient vector
            current_radius: Current trust region radius
            initial_radius: Initial trust region radius
            rejected_steps: Number of consecutive rejected steps
            predicted_reduction: Predicted objective reduction
            actual_reduction: Actual objective reduction
            iteration: Current iteration number
            constraints: Constraint values (optional)
            lagrange_multipliers: Lagrange multipliers (optional)
        
        Returns:
            OptimizerGovernanceReport with comprehensive assessment
        """
        violations = []
        failure_reasons = []
        recommended_actions = []
        
        # 1. Analyze progress
        progress = self.analyze_progress(objective_value, design_vars, iteration)
        violations.extend(progress.violations)
        
        # 2. Analyze gradient health
        gradient_health = self.analyze_gradient_health(gradient, iteration)
        violations.extend(gradient_health.violations)
        
        # 3. Analyze trust region
        trust_region = self.analyze_trust_region(current_radius, initial_radius, rejected_steps)
        violations.extend(trust_region.violations)
        
        # 4. Analyze gain ratio
        gain_ratio = self.analyze_gain_ratio(predicted_reduction, actual_reduction)
        violations.extend(gain_ratio.violations)
        
        # 5. Analyze KKT conditions
        kkt = self.analyze_kkt_conditions(gradient, constraints, lagrange_multipliers)
        violations.extend(kkt.violations)
        
        # 6. Detect oscillations
        oscillation = self.detect_oscillations()
        violations.extend(oscillation.violations)
        
        # 7. Compute health score
        health_score = self.compute_health_score(
            progress, gradient_health, trust_region, gain_ratio
        )
        
        # 8. Get recovery strategies
        recovery_strategies = self.get_recovery_strategies(violations, health_score)
        
        # 9. Determine overall status
        unique_violations = list(set(violations))
        n_violations = len(unique_violations)
        
        if n_violations == 0 and health_score > 0.7:
            status = OptimizerHealthStatus.HEALTHY
            is_healthy = True
            can_continue = True
        elif n_violations <= 1 and health_score > 0.5:
            status = OptimizerHealthStatus.DEGRADED
            is_healthy = False
            can_continue = True
        elif n_violations <= 3 and health_score > 0.3:
            status = OptimizerHealthStatus.UNHEALTHY
            is_healthy = False
            can_continue = True
        else:
            status = OptimizerHealthStatus.CRITICAL
            is_healthy = False
            can_continue = False
        
        # Add failure reasons
        for v in unique_violations:
            if v == OptimizerViolationType.NO_PROGRESS:
                failure_reasons.append(
                    f"No improvement for {progress.iterations_without_improvement} iterations"
                )
            elif v == OptimizerViolationType.STALE_GRADIENTS:
                failure_reasons.append(
                    f"Gradients stale for {gradient_health.iterations_without_gradient_update} iterations"
                )
            elif v == OptimizerViolationType.TRUST_REGION_COLLAPSED:
                failure_reasons.append(
                    f"Trust region collapsed (radius ratio: {trust_region.radius_ratio:.2e})"
                )
            elif v == OptimizerViolationType.GAIN_RATIO_COLLAPSE:
                failure_reasons.append(
                    f"Gain ratio collapsed ({gain_ratio.current_gain_ratio:.3f}, quality: {gain_ratio.model_quality})"
                )
            elif v == OptimizerViolationType.OSCILLATORY_BEHAVIOR:
                failure_reasons.append(
                    f"Oscillatory behavior detected (amplitude: {oscillation.design_oscillation_amplitude:.4f})"
                )
            elif v == OptimizerViolationType.KKT_STAGNATION:
                failure_reasons.append(
                    f"KKT conditions not satisfied (stationarity: {kkt.stationarity_norm:.4f})"
                )
        
        return OptimizerGovernanceReport(
            status=status,
            progress=progress,
            gradient_health=gradient_health,
            trust_region=trust_region,
            gain_ratio=gain_ratio,
            kkt=kkt,
            oscillation=oscillation,
            is_healthy=is_healthy,
            can_continue=can_continue,
            violations=unique_violations,
            failure_reasons=failure_reasons,
            recommended_actions=recommended_actions,
            recovery_strategies=recovery_strategies,
        )
    
    def reset(self):
        """Reset all history tracking."""
        self.objective_history.clear()
        self.design_history.clear()
        self.gradient_history.clear()
        self.gain_ratio_history.clear()
        self.trust_region_history.clear()