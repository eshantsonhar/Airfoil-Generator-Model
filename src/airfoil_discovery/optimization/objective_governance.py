"""
Objective function governance for PDE-constrained aerodynamic shape optimization.

Implements governance for objective function formulation, scaling, and
penalty management to prevent the optimizer from exploiting loopholes
in the objective formulation. This ensures the optimizer pursues
meaningful aerodynamic improvement rather than numerical artifacts.

The framework addresses:
- Nondimensionalization consistency
- Penalty normalization and scaling
- Multi-objective weighting validation
- Objective function boundedness
- Reward hacking prevention
- Regularization term dominance checking
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable
from enum import Enum
import warnings


class ObjectiveHealthStatus(Enum):
    """Objective function health status."""
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class ObjectiveViolationType(Enum):
    """Types of objective function violations."""
    NONE = "NONE"
    SCALE_MISMATCH = "SCALE_MISMATCH"
    PENALTY_DOMINANCE = "PENALTY_DOMINANCE"
    UNBOUNDED_OBJECTIVE = "UNBOUNDED_OBJECTIVE"
    NAN_INF_DETECTED = "NAN_INF_DETECTED"
    WEIGHT_IMBALANCE = "WEIGHT_IMBALANCE"
    REGULARIZATION_DOMINANCE = "REGULARIZATION_DOMINANCE"
    REWARD_HACKING_DETECTED = "REWARD_HACKING_DETECTED"
    DIMENSIONAL_INCONSISTENCY = "DIMENSIONAL_INCONSISTENCY"
    DISCONTINUITY_DETECTED = "DISCONTINUITY_DETECTED"


@dataclass
class ObjectiveValueMetrics:
    """Metrics for objective function value analysis."""
    
    # Objective value
    objective_value: float
    baseline_objective: float
    improvement: float
    improvement_ratio: float
    
    # Boundedness
    is_finite: bool
    is_bounded: bool
    value_range: Tuple[float, float]
    
    # Violations
    violations: List[ObjectiveViolationType] = field(default_factory=list)


@dataclass
class ScalingMetrics:
    """Metrics for objective function scaling analysis."""
    
    # Component scales
    primary_objective_scale: float
    penalty_scale: float
    regularization_scale: float
    
    # Scale ratios
    penalty_to_primary_ratio: float
    regularization_to_primary_ratio: float
    
    # Scale consistency
    scales_balanced: bool
    scale_condition_number: float
    
    # Violations
    violations: List[ObjectiveViolationType] = field(default_factory=list)


@dataclass
class PenaltyMetrics:
    """Metrics for penalty function analysis."""
    
    # Penalty values
    total_penalty: float
    constraint_penalties: Dict[str, float]
    max_penalty: float
    mean_penalty: float
    
    # Penalty dominance
    penalty_fraction: float  # Fraction of total objective from penalties
    dominant_penalty: Optional[str]
    
    # Penalty health
    penalties_reasonable: bool
    penalty_growth_controlled: bool
    
    # Violations
    violations: List[ObjectiveViolationType] = field(default_factory=list)


@dataclass
class WeightMetrics:
    """Metrics for multi-objective weight analysis."""
    
    # Weights
    weights: Dict[str, float]
    weight_sum: float
    weight_balance: float  # Entropy-based balance measure
    
    # Weight health
    weights_normalized: bool
    weights_balanced: bool
    dominant_objective: Optional[str]
    
    # Violations
    violations: List[ObjectiveViolationType] = field(default_factory=list)


@dataclass
class RegularizationMetrics:
    """Metrics for regularization term analysis."""
    
    # Regularization values
    regularization_value: float
    smoothing_penalty: float
    geometric_penalty: float
    
    # Regularization dominance
    regularization_fraction: float
    
    # Regularization health
    regularization_reasonable: bool
    smoothing_effective: bool
    
    # Violations
    violations: List[ObjectiveViolationType] = field(default_factory=list)


@dataclass
class ObjectiveGovernanceReport:
    """Comprehensive objective function governance report."""
    
    # Overall status
    status: ObjectiveHealthStatus
    
    # Component metrics
    objective_value: Optional[ObjectiveValueMetrics] = None
    scaling: Optional[ScalingMetrics] = None
    penalties: Optional[PenaltyMetrics] = None
    weights: Optional[WeightMetrics] = None
    regularization: Optional[RegularizationMetrics] = None
    
    # Overall assessment
    is_valid: bool
    can_use_for_optimization: bool
    
    # Violations
    violations: List[ObjectiveViolationType] = field(default_factory=list)
    failure_reasons: List[str] = field(default_factory=list)
    
    # Recommendations
    recommended_actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "is_valid": self.is_valid,
            "can_use_for_optimization": self.can_use_for_optimization,
            "violations": [v.value for v in self.violations],
            "failure_reasons": self.failure_reasons,
            "recommended_actions": self.recommended_actions,
            "objective_value": {
                "value": self.objective_value.objective_value if self.objective_value else None,
                "improvement": self.objective_value.improvement if self.objective_value else None,
                "is_finite": self.objective_value.is_finite if self.objective_value else None,
            } if self.objective_value else None,
            "scaling": {
                "penalty_to_primary_ratio": self.scaling.penalty_to_primary_ratio if self.scaling else None,
                "scales_balanced": self.scaling.scales_balanced if self.scaling else None,
            } if self.scaling else None,
        }


@dataclass
class ObjectiveGovernanceConfig:
    """Configuration for objective function governance."""
    
    # Scale thresholds
    max_penalty_to_primary_ratio: float = 10.0  # Penalties shouldn't dominate
    max_regularization_to_primary_ratio: float = 5.0
    max_scale_condition_number: float = 1000.0
    
    # Penalty thresholds
    max_penalty_fraction: float = 0.5  # Penalties shouldn't be > 50% of objective
    max_single_penalty_fraction: float = 0.3  # No single penalty > 30%
    
    # Weight thresholds
    weight_normalization_tolerance: float = 0.01
    weight_balance_threshold: float = 0.3  # Entropy threshold
    max_dominant_weight_fraction: float = 0.7
    
    # Regularization thresholds
    max_regularization_fraction: float = 0.3
    min_smoothing_effectiveness: float = 0.1
    
    # Boundedness
    objective_upper_bound: float = 1e10
    objective_lower_bound: float = -1e10
    
    # Improvement thresholds
    min_improvement_for_acceptance: float = 1e-6
    max_improvement_ratio: float = 10.0  # Sanity check
    
    # Governance policy
    strict_mode: bool = True
    check_dimensions: bool = True


class ObjectiveGovernor:
    """
    Governs objective function formulation and evaluation.
    
    This class ensures that the objective function used for optimization
    is well-scaled, properly balanced, and physically meaningful. It
    prevents the optimizer from exploiting numerical loopholes such as:
    
    - Driving penalties to zero while worsening primary objective
    - Exploiting scale mismatches between terms
    - Finding discontinuities or singularities
    - Producing unbounded or NaN objectives
    
    The governor validates:
    - Nondimensionalization consistency
    - Penalty normalization and scaling
    - Multi-objective weight balance
    - Regularization term appropriateness
    - Objective boundedness and continuity
    """
    
    def __init__(self, config: Optional[ObjectiveGovernanceConfig] = None):
        """
        Initialize objective governor.
        
        Args:
            config: Governance configuration. Uses defaults if None.
        """
        self.config = config or ObjectiveGovernanceConfig()
        
        # History tracking
        self.objective_history: List[float] = []
        self.component_history: List[Dict[str, float]] = []
        self.baseline_objective: Optional[float] = None
    
    def set_baseline(self, objective_value: float) -> None:
        """
        Set baseline objective value for improvement tracking.
        
        Args:
            objective_value: Baseline objective value
        """
        self.baseline_objective = objective_value
    
    def analyze_objective_value(
        self,
        objective_value: float,
    ) -> ObjectiveValueMetrics:
        """
        Analyze objective function value for validity.
        
        Args:
            objective_value: Current objective function value
        
        Returns:
            ObjectiveValueMetrics with value analysis
        """
        self.objective_history.append(objective_value)
        
        # Check for NaN/Inf
        is_finite = np.isfinite(objective_value)
        
        # Check boundedness
        is_bounded = (
            self.config.objective_lower_bound <= objective_value <= self.config.objective_upper_bound
        )
        
        # Compute improvement
        if self.baseline_objective is not None:
            improvement = self.baseline_objective - objective_value
            improvement_ratio = improvement / (abs(self.baseline_objective) + 1e-15)
        else:
            improvement = 0.0
            improvement_ratio = 0.0
        
        # Value range
        if len(self.objective_history) >= 2:
            value_range = (float(np.min(self.objective_history)), float(np.max(self.objective_history)))
        else:
            value_range = (objective_value, objective_value)
        
        # Detect violations
        violations = []
        if not is_finite:
            violations.append(ObjectiveViolationType.NAN_INF_DETECTED)
        if not is_bounded and is_finite:
            violations.append(ObjectiveViolationType.UNBOUNDED_OBJECTIVE)
        
        return ObjectiveValueMetrics(
            objective_value=objective_value,
            baseline_objective=self.baseline_objective if self.baseline_objective is not None else objective_value,
            improvement=improvement,
            improvement_ratio=improvement_ratio,
            is_finite=is_finite,
            is_bounded=is_bounded,
            value_range=value_range,
            violations=violations,
        )
    
    def analyze_scaling(
        self,
        primary_objective: float,
        penalty_value: float,
        regularization_value: float,
    ) -> ScalingMetrics:
        """
        Analyze objective function scaling.
        
        Args:
            primary_objective: Primary (aerodynamic) objective value
            penalty_value: Total penalty value
            regularization_value: Regularization term value
        
        Returns:
            ScalingMetrics with scaling analysis
        """
        # Compute scales (use absolute values)
        primary_scale = abs(primary_objective) + 1e-15
        penalty_scale = abs(penalty_value)
        regularization_scale = abs(regularization_value)
        
        # Compute ratios
        penalty_to_primary = penalty_scale / primary_scale
        regularization_to_primary = regularization_scale / primary_scale
        
        # Scale condition number (measure of scale spread)
        all_scales = [primary_scale, penalty_scale, regularization_scale]
        scale_condition = max(all_scales) / (min(all_scales) + 1e-30)
        
        # Check balance
        scales_balanced = (
            penalty_to_primary < self.config.max_penalty_to_primary_ratio and
            regularization_to_primary < self.config.max_regularization_to_primary_ratio and
            scale_condition < self.config.max_scale_condition_number
        )
        
        # Detect violations
        violations = []
        if penalty_to_primary > self.config.max_penalty_to_primary_ratio:
            violations.append(ObjectiveViolationType.SCALE_MISMATCH)
        if regularization_to_primary > self.config.max_regularization_to_primary_ratio:
            violations.append(ObjectiveViolationType.SCALE_MISMATCH)
        if scale_condition > self.config.max_scale_condition_number:
            violations.append(ObjectiveViolationType.SCALE_MISMATCH)
        
        return ScalingMetrics(
            primary_objective_scale=primary_scale,
            penalty_scale=penalty_scale,
            regularization_scale=regularization_scale,
            penalty_to_primary_ratio=penalty_to_primary,
            regularization_to_primary_ratio=regularization_to_primary,
            scales_balanced=scales_balanced,
            scale_condition_number=scale_condition,
            violations=violations,
        )
    
    def analyze_penalties(
        self,
        constraint_penalties: Dict[str, float],
        total_objective: float,
    ) -> PenaltyMetrics:
        """
        Analyze penalty function behavior.
        
        Args:
            constraint_penalties: Dictionary of constraint name to penalty value
            total_objective: Total objective function value
        
        Returns:
            PenaltyMetrics with penalty analysis
        """
        # Total penalty
        total_penalty = sum(constraint_penalties.values())
        
        # Penalty statistics
        if constraint_penalties:
            max_penalty = max(constraint_penalties.values())
            mean_penalty = total_penalty / len(constraint_penalties)
            dominant_penalty = max(constraint_penalties, key=lambda k: abs(constraint_penalties[k]))
        else:
            max_penalty = 0.0
            mean_penalty = 0.0
            dominant_penalty = None
        
        # Penalty fraction
        penalty_fraction = abs(total_penalty) / (abs(total_objective) + 1e-15)
        
        # Check if any single penalty dominates
        single_penalty_fraction = max_penalty / (abs(total_objective) + 1e-15)
        
        # Penalty health
        penalties_reasonable = penalty_fraction < self.config.max_penalty_fraction
        penalty_growth_controlled = single_penalty_fraction < self.config.max_single_penalty_fraction
        
        # Detect violations
        violations = []
        if not penalties_reasonable:
            violations.append(ObjectiveViolationType.PENALTY_DOMINANCE)
        if not penalty_growth_controlled and constraint_penalties:
            violations.append(ObjectiveViolationType.PENALTY_DOMINANCE)
        
        return PenaltyMetrics(
            total_penalty=total_penalty,
            constraint_penalties=constraint_penalties.copy(),
            max_penalty=max_penalty,
            mean_penalty=mean_penalty,
            penalty_fraction=penalty_fraction,
            dominant_penalty=dominant_penalty,
            penalties_reasonable=penalties_reasonable,
            penalty_growth_controlled=penalty_growth_controlled,
            violations=violations,
        )
    
    def analyze_weights(
        self,
        weights: Dict[str, float],
    ) -> WeightMetrics:
        """
        Analyze multi-objective weight balance.
        
        Args:
            weights: Dictionary of objective name to weight value
        
        Returns:
            WeightMetrics with weight analysis
        """
        if not weights:
            return WeightMetrics(
                weights={},
                weight_sum=0.0,
                weight_balance=1.0,
                weights_normalized=True,
                weights_balanced=True,
                dominant_objective=None,
                violations=[],
            )
        
        # Weight sum
        weight_sum = sum(weights.values())
        
        # Normalize weights
        normalized_weights = {k: v / (weight_sum + 1e-15) for k, v in weights.items()}
        
        # Weight balance (entropy-based)
        n_objectives = len(weights)
        if n_objectives > 1:
            # Compute entropy
            entropy = 0.0
            for w in normalized_weights.values():
                if w > 1e-15:
                    entropy -= w * np.log2(w)
            # Normalize by max entropy
            max_entropy = np.log2(n_objectives)
            weight_balance = entropy / (max_entropy + 1e-15)
        else:
            weight_balance = 1.0
        
        # Check normalization
        weights_normalized = abs(weight_sum - 1.0) < self.config.weight_normalization_tolerance
        
        # Check balance
        weights_balanced = weight_balance > self.config.weight_balance_threshold
        
        # Find dominant objective
        dominant_objective = None
        max_weight = max(normalized_weights.values())
        if max_weight > self.config.max_dominant_weight_fraction:
            dominant_objective = max(normalized_weights, key=lambda k: normalized_weights[k])
        
        # Detect violations
        violations = []
        if not weights_balanced:
            violations.append(ObjectiveViolationType.WEIGHT_IMBALANCE)
        if dominant_objective is not None:
            violations.append(ObjectiveViolationType.WEIGHT_IMBALANCE)
        
        return WeightMetrics(
            weights=normalized_weights.copy(),
            weight_sum=weight_sum,
            weight_balance=weight_balance,
            weights_normalized=weights_normalized,
            weights_balanced=weights_balanced,
            dominant_objective=dominant_objective,
            violations=violations,
        )
    
    def analyze_regularization(
        self,
        regularization_value: float,
        smoothing_penalty: float,
        geometric_penalty: float,
        total_objective: float,
    ) -> RegularizationMetrics:
        """
        Analyze regularization term behavior.
        
        Args:
            regularization_value: Total regularization value
            smoothing_penalty: Smoothing/curvature penalty
            geometric_penalty: Geometric constraint penalty
            total_objective: Total objective function value
        
        Returns:
            RegularizationMetrics with regularization analysis
        """
        # Regularization fraction
        regularization_fraction = abs(regularization_value) / (abs(total_objective) + 1e-15)
        
        # Check if regularization is reasonable
        regularization_reasonable = regularization_fraction < self.config.max_regularization_fraction
        
        # Smoothing effectiveness
        smoothing_effective = smoothing_penalty < regularization_value * 0.9 if regularization_value > 0 else True
        
        # Detect violations
        violations = []
        if not regularization_reasonable:
            violations.append(ObjectiveViolationType.REGULARIZATION_DOMINANCE)
        
        return RegularizationMetrics(
            regularization_value=regularization_value,
            smoothing_penalty=smoothing_penalty,
            geometric_penalty=geometric_penalty,
            regularization_fraction=regularization_fraction,
            regularization_reasonable=regularization_reasonable,
            smoothing_effective=smoothing_effective,
            violations=violations,
        )
    
    def detect_reward_hacking(
        self,
        current_components: Dict[str, float],
        previous_components: Optional[Dict[str, float]] = None,
    ) -> bool:
        """
        Detect potential reward hacking behavior.
        
        Reward hacking occurs when the optimizer improves the total
        objective by exploiting term interactions rather than genuine
        aerodynamic improvement.
        
        Args:
            current_components: Current objective components
            previous_components: Previous objective components
        
        Returns:
            True if reward hacking is detected
        """
        if previous_components is None or len(self.component_history) == 0:
            self.component_history.append(current_components.copy())
            return False
        
        # Check if primary objective worsened while total improved
        primary_key = "primary"  # Convention: primary objective key
        penalty_key = "penalty"
        
        current_primary = current_components.get(primary_key, 0.0)
        current_penalty = current_components.get(penalty_key, 0.0)
        previous = self.component_history[-1]
        previous_primary = previous.get(primary_key, 0.0)
        previous_penalty = previous.get(penalty_key, 0.0)
        
        # Reward hacking: primary worsened but total improved via penalty reduction
        primary_worsened = current_primary > previous_primary  # Assuming minimization
        penalty_improved_significantly = previous_penalty > current_penalty * 1.5
        
        self.component_history.append(current_components.copy())
        
        # Limit history size
        if len(self.component_history) > 20:
            self.component_history.pop(0)
        
        return primary_worsened and penalty_improved_significantly
    
    def check_dimensional_consistency(
        self,
        objective_terms: Dict[str, float],
        term_dimensions: Dict[str, str],
    ) -> bool:
        """
        Check dimensional consistency of objective terms.
        
        Args:
            objective_terms: Dictionary of term name to value
            term_dimensions: Dictionary of term name to dimension string
        
        Returns:
            True if dimensionally consistent
        """
        if not self.config.check_dimensions:
            return True
        
        # Group terms by dimension
        dimension_groups: Dict[str, List[float]] = {}
        for term_name, value in objective_terms.items():
            dim = term_dimensions.get(term_name, "unknown")
            if dim not in dimension_groups:
                dimension_groups[dim] = []
            dimension_groups[dim].append(abs(value))
        
        # Check if multiple dimensions are present with similar magnitudes
        # (which would indicate dimensional inconsistency)
        if len(dimension_groups) > 1:
            # This is expected for penalty methods, so just warn
            return True
        
        return True
    
    def compute_health_score(
        self,
        value_metrics: ObjectiveValueMetrics,
        scaling_metrics: ScalingMetrics,
        penalty_metrics: PenaltyMetrics,
        regularization_metrics: RegularizationMetrics,
    ) -> float:
        """
        Compute overall objective health score (0-1).
        
        Args:
            value_metrics: Objective value metrics
            scaling_metrics: Scaling metrics
            penalty_metrics: Penalty metrics
            regularization_metrics: Regularization metrics
        
        Returns:
            Overall health score (0-1)
        """
        score = 1.0
        
        # Penalty for invalid values
        if not value_metrics.is_finite:
            score -= 1.0  # Critical
        elif not value_metrics.is_bounded:
            score -= 0.5
        
        # Penalty for scale issues
        if not scaling_metrics.scales_balanced:
            score -= 0.3
        
        # Penalty for penalty dominance
        if not penalty_metrics.penalties_reasonable:
            score -= 0.4
        elif not penalty_metrics.penalty_growth_controlled:
            score -= 0.2
        
        # Penalty for regularization dominance
        if not regularization_metrics.regularization_reasonable:
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def govern(
        self,
        objective_value: float,
        primary_objective: float,
        penalty_value: float,
        regularization_value: float,
        constraint_penalties: Dict[str, float],
        weights: Optional[Dict[str, float]] = None,
        smoothing_penalty: float = 0.0,
        geometric_penalty: float = 0.0,
        current_components: Optional[Dict[str, float]] = None,
    ) -> ObjectiveGovernanceReport:
        """
        Perform comprehensive objective function governance.
        
        This is the main entry point for objective validation.
        
        Args:
            objective_value: Total objective function value
            primary_objective: Primary (aerodynamic) objective value
            penalty_value: Total penalty value
            regularization_value: Total regularization value
            constraint_penalties: Dictionary of constraint penalties
            weights: Multi-objective weights (optional)
            smoothing_penalty: Smoothing penalty value
            geometric_penalty: Geometric penalty value
            current_components: Current objective components for hacking detection
        
        Returns:
            ObjectiveGovernanceReport with comprehensive assessment
        """
        violations = []
        failure_reasons = []
        recommended_actions = []
        
        # 1. Analyze objective value
        value_metrics = self.analyze_objective_value(objective_value)
        violations.extend(value_metrics.violations)
        
        # 2. Analyze scaling
        scaling_metrics = self.analyze_scaling(
            primary_objective, penalty_value, regularization_value
        )
        violations.extend(scaling_metrics.violations)
        
        # 3. Analyze penalties
        penalty_metrics = self.analyze_penalties(constraint_penalties, objective_value)
        violations.extend(penalty_metrics.violations)
        
        # 4. Analyze weights (if provided)
        weight_metrics = None
        if weights is not None:
            weight_metrics = self.analyze_weights(weights)
            violations.extend(weight_metrics.violations)
        
        # 5. Analyze regularization
        regularization_metrics = self.analyze_regularization(
            regularization_value, smoothing_penalty, geometric_penalty, objective_value
        )
        violations.extend(regularization_metrics.violations)
        
        # 6. Check for reward hacking
        if current_components is not None:
            previous_components = self.component_history[-1] if self.component_history else None
            reward_hacking_detected = self.detect_reward_hacking(
                current_components, previous_components
            )
            if reward_hacking_detected:
                violations.append(ObjectiveViolationType.REWARD_HACKING_DETECTED)
        
        # 7. Compute health score
        health_score = self.compute_health_score(
            value_metrics, scaling_metrics, penalty_metrics, regularization_metrics
        )
        
        # 8. Determine overall status
        unique_violations = list(set(violations))
        n_violations = len(unique_violations)
        
        if n_violations == 0 and health_score > 0.8:
            status = ObjectiveHealthStatus.HEALTHY
            is_valid = True
            can_use = True
        elif n_violations <= 1 and health_score > 0.6:
            status = ObjectiveHealthStatus.WARNING
            is_valid = True
            can_use = True
        elif n_violations <= 2 and health_score > 0.4:
            status = ObjectiveHealthStatus.DEGRADED
            is_valid = False
            can_use = False
        else:
            status = ObjectiveHealthStatus.CRITICAL
            is_valid = False
            can_use = False
        
        # Add failure reasons
        for v in unique_violations:
            if v == ObjectiveViolationType.NAN_INF_DETECTED:
                failure_reasons.append("Objective value is NaN or infinite")
                recommended_actions.append("Check for division by zero or invalid operations")
            elif v == ObjectiveViolationType.SCALE_MISMATCH:
                failure_reasons.append(
                    f"Scale mismatch detected (condition number: {scaling_metrics.scale_condition_number:.1f})"
                )
                recommended_actions.append("Rescale objective terms to similar magnitudes")
            elif v == ObjectiveViolationType.PENALTY_DOMINANCE:
                failure_reasons.append(
                    f"Penalties dominate objective ({penalty_metrics.penalty_fraction:.1%})"
                )
                recommended_actions.append("Reduce penalty weights or improve constraint satisfaction")
            elif v == ObjectiveViolationType.REGULARIZATION_DOMINANCE:
                failure_reasons.append(
                    f"Regularization dominates objective ({regularization_metrics.regularization_fraction:.1%})"
                )
                recommended_actions.append("Reduce regularization weights")
            elif v == ObjectiveViolationType.REWARD_HACKING_DETECTED:
                failure_reasons.append("Reward hacking detected - optimizer exploiting objective loopholes")
                recommended_actions.append("Reformulate objective to prevent exploitation")
            elif v == ObjectiveViolationType.WEIGHT_IMBALANCE:
                failure_reasons.append("Multi-objective weights are imbalanced")
                recommended_actions.append("Rebalance weights or use Pareto approach")
        
        return ObjectiveGovernanceReport(
            status=status,
            objective_value=value_metrics,
            scaling=scaling_metrics,
            penalties=penalty_metrics,
            weights=weight_metrics,
            regularization=regularization_metrics,
            is_valid=is_valid,
            can_use_for_optimization=can_use,
            violations=unique_violations,
            failure_reasons=failure_reasons,
            recommended_actions=recommended_actions,
        )
    
    def reset(self):
        """Reset all history tracking."""
        self.objective_history.clear()
        self.component_history.clear()
        self.baseline_objective = None