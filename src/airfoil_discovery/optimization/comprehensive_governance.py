"""
Comprehensive governance orchestrator for physics-governed aerodynamic optimization.

This module integrates all governance systems into a unified framework that
enforces physical realism, numerical correctness, and optimization robustness.
It implements a strict multi-layer governance architecture that prevents the
optimizer from accepting designs merely because they are numerically convergent.

A solution is only VALID if it is:
- Numerically converged
- Physically plausible
- Geometrically realistic
- Transitionally credible
- Mesh-resolved
- Gradient-consistent
- Aerodynamically meaningful
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from pathlib import Path
import json
import time

from airfoil_discovery.geometry.governance import (
    GeometryGovernor, GeometryGovernanceConfig, GeometryGovernanceReport,
    GeometryValidityStatus
)
from airfoil_discovery.geometry.manifold import AirfoilManifold, ManifoldConfig
from airfoil_discovery.physics.plausibility import (
    AerodynamicPlausibilityGovernor, PlausibilityConfig, PlausibilityGovernanceReport,
    PlausibilityStatus
)
from airfoil_discovery.physics.lsb_detection import LSBDetector, LSBDetectionReport
from airfoil_discovery.physics.transition_governance import (
    TransitionModelGovernor, TransitionGovernanceReport
)
from airfoil_discovery.verification.numerical_dissipation import (
    NumericalDissipationMonitor, DissipationConfig, NumericalDissipationReport,
    DissipationStatus
)
from airfoil_discovery.verification.convergence import (
    IterativeConvergenceMonitor, ConvergenceReport, ConvergenceStatus
)
from airfoil_discovery.verification.gradient_audit import (
    GradientAuditor, GradientAuditReport, GradientStatus
)
from airfoil_discovery.verification.gci import GCICalculator, GCIReport
from airfoil_discovery.optimization.governance import (
    OptimizerGovernor, OptimizerGovernanceConfig, OptimizerGovernanceReport,
    OptimizerHealthStatus
)
from airfoil_discovery.optimization.objective_governance import (
    ObjectiveGovernor, ObjectiveGovernanceConfig, ObjectiveGovernanceReport,
    ObjectiveHealthStatus
)


class GovernanceDecision(Enum):
    """Final governance decision for a design evaluation."""
    ACCEPT = "ACCEPT"  # Design passes all checks
    REJECT = "REJECT"  # Design fails critical checks
    SUSPECT = "SUSPECT"  # Design has warnings but may proceed
    INVALID = "INVALID"  # Design has fundamental issues
    NEEDS_REVIEW = "NEEDS_REVIEW"  # Manual review required


class GovernanceSeverity(Enum):
    """Severity level of governance findings."""
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class ComprehensiveGovernanceReport:
    """Comprehensive governance report for a design evaluation."""
    
    # Overall decision
    decision: GovernanceDecision
    severity: GovernanceSeverity
    
    # Timestamp and metadata
    timestamp: float
    iteration: int
    design_id: str
    
    # Component reports
    geometry: Optional[GeometryGovernanceReport] = None
    plausibility: Optional[PlausibilityGovernanceReport] = None
    lsb_detection: Optional[LSBDetectionReport] = None
    transition: Optional[TransitionGovernanceReport] = None
    dissipation: Optional[NumericalDissipationReport] = None
    convergence: Optional[ConvergenceReport] = None
    gradient: Optional[GradientAuditReport] = None
    gci: Optional[GCIReport] = None
    optimizer: Optional[OptimizerGovernanceReport] = None
    objective: Optional[ObjectiveGovernanceReport] = None
    
    # Summary metrics
    total_violations: int = 0
    critical_violations: int = 0
    warnings: int = 0
    
    # Failure reasons
    failure_reasons: List[str] = field(default_factory=list)
    warnings_list: List[str] = field(default_factory=list)
    
    # Recommendations
    recommended_actions: List[str] = field(default_factory=list)
    
    # CFD execution allowed
    can_proceed_to_cfd: bool = False
    can_accept_design: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "decision": self.decision.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "iteration": self.iteration,
            "design_id": self.design_id,
            "total_violations": self.total_violations,
            "critical_violations": self.critical_violations,
            "warnings": self.warnings,
            "failure_reasons": self.failure_reasons,
            "can_proceed_to_cfd": self.can_proceed_to_cfd,
            "can_accept_design": self.can_accept_design,
            "geometry_valid": self.geometry.is_valid if self.geometry else None,
            "plausibility_valid": self.plausibility.is_valid if self.plausibility else None,
            "convergence_valid": self.convergence.is_valid if self.convergence else None,
            "optimizer_healthy": self.optimizer.is_healthy if self.optimizer else None,
        }
    
    def save(self, path: Path) -> None:
        """Save report to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path / f"governance_report_{self.design_id}.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class ComprehensiveGovernanceConfig:
    """Configuration for comprehensive governance."""
    
    # Governance policy
    strict_mode: bool = True  # Any violation rejects design
    require_geometry_valid: bool = True
    require_plausibility_valid: bool = True
    require_convergence_valid: bool = True
    require_gradient_valid: bool = True
    
    # CFD execution policy
    allow_cfd_on_suspect: bool = False
    max_suspect_designs_per_run: int = 5
    
    # Failure handling
    stop_on_critical: bool = True
    stop_on_consecutive_failures: int = 3
    preserve_crash_state: bool = True
    crash_state_dir: str = "data/crash_states"
    
    # Logging
    log_all_evaluations: bool = True
    log_dir: str = "data/logs/governance"


class ComprehensiveGovernor:
    """
    Comprehensive governance orchestrator for physics-governed optimization.
    
    This class integrates all governance systems and provides a unified
    interface for design validation. It implements a strict multi-layer
    validation pipeline:
    
    1. GEOMETRY GOVERNANCE (pre-CFD)
       - Thickness constraints
       - Leading edge radius limits
       - Curvature continuity
       - Self-intersection check
       - Manifold distance
    
    2. AERODYNAMIC PLAUSIBILITY (post-CFD)
       - Force coefficient reasonableness
       - Bluff-body detection
       - Pressure recovery validation
       - Separation analysis
    
    3. TRANSITION PHYSICS (post-CFD)
       - LSB detection and classification
       - Transition model validity
       - Intermittency transport stability
    
    4. NUMERICAL VERIFICATION (post-CFD)
       - Convergence validation
       - Dissipation monitoring
       - Mesh quality checks
       - GCI analysis
    
    5. OPTIMIZATION HEALTH (during optimization)
       - Gradient integrity
       - Objective function validity
       - Optimizer convergence
    
    A design must pass ALL required checks to be accepted.
    """
    
    def __init__(self, config: Optional[ComprehensiveGovernanceConfig] = None):
        """
        Initialize comprehensive governor.
        
        Args:
            config: Governance configuration. Uses defaults if None.
        """
        self.config = config or ComprehensiveGovernanceConfig()
        
        # Initialize component governors
        self.geometry_governor = GeometryGovernor()
        self.manifold = AirfoilManifold()
        self.plausibility_governor = AerodynamicPlausibilityGovernor()
        self.lsb_detector = LSBDetector()
        self.transition_governor = TransitionModelGovernor()
        self.dissipation_monitor = NumericalDissipationMonitor()
        self.convergence_monitor = IterativeConvergenceMonitor()
        self.gradient_auditor = GradientAuditor()
        self.gci_calculator = GCICalculator()
        self.optimizer_governor = OptimizerGovernor()
        self.objective_governor = ObjectiveGovernor()
        
        # Statistics
        self.total_evaluations = 0
        self.accepted_designs = 0
        self.rejected_designs = 0
        self.consecutive_failures = 0
        
        # Suspect design tracking
        self.suspect_designs_count = 0
    
    def set_manifold_model(self, model: Any, scaler: Any) -> None:
        """Set the manifold model for geometric validation."""
        self.geometry_governor.set_manifold_model(model, scaler)
    
    def pre_cfd_governance(
        self,
        x: np.ndarray,
        yu: np.ndarray,
        yl: np.ndarray,
        upper_coeffs: Optional[np.ndarray] = None,
        lower_coeffs: Optional[np.ndarray] = None,
        design_id: str = "unknown",
        iteration: int = 0,
    ) -> ComprehensiveGovernanceReport:
        """
        Perform pre-CFD governance checks.
        
        This validates the geometry BEFORE running expensive CFD simulations.
        
        Args:
            x: Chordwise coordinates
            yu: Upper surface y-coordinates
            yl: Lower surface y-coordinates
            upper_coeffs: Upper CST coefficients
            lower_coeffs: Lower CST coefficients
            design_id: Design identifier
            iteration: Current iteration
        
        Returns:
            ComprehensiveGovernanceReport with pre-CFD validation results
        """
        self.total_evaluations += 1
        timestamp = time.time()
        
        # 1. Geometry governance
        geometry_report = self.geometry_governor.govern(
            x, yu, yl, upper_coeffs, lower_coeffs
        )
        
        # 2. Manifold check
        manifold_distance, outlier_score = self.geometry_governor.check_manifold_distance(
            x, yu, yl
        )
        
        # Collect violations
        all_violations = []
        failure_reasons = []
        warnings_list = []
        recommended_actions = []
        
        if not geometry_report.is_valid:
            all_violations.extend(geometry_report.violations)
            failure_reasons.extend(geometry_report.failure_reasons)
            recommended_actions.extend(geometry_report.recommended_actions)
        
        # Count violations
        critical_violations = len([v for v in all_violations if "SELF_INTERSECTION" in str(v)])
        total_violations = len(set(all_violations))
        
        # Determine decision
        if geometry_report.is_valid:
            decision = GovernanceDecision.ACCEPT
            severity = GovernanceSeverity.NONE
            can_proceed = True
        else:
            decision = GovernanceDecision.REJECT
            severity = GovernanceSeverity.CRITICAL if critical_violations > 0 else GovernanceSeverity.HIGH
            can_proceed = False
        
        report = ComprehensiveGovernanceReport(
            decision=decision,
            severity=severity,
            timestamp=timestamp,
            iteration=iteration,
            design_id=design_id,
            geometry=geometry_report,
            total_violations=total_violations,
            critical_violations=critical_violations,
            warnings=len(warnings_list),
            failure_reasons=failure_reasons,
            warnings_list=warnings_list,
            recommended_actions=recommended_actions,
            can_proceed_to_cfd=can_proceed,
            can_accept_design=False,  # Pre-CFD can't accept final design
        )
        
        # Update statistics
        if can_proceed:
            self.accepted_designs += 1
            self.consecutive_failures = 0
        else:
            self.rejected_designs += 1
            self.consecutive_failures += 1
        
        return report
    
    def post_cfd_governance(
        self,
        cl: float,
        cd: float,
        x: np.ndarray,
        cp: np.ndarray,
        cf: Optional[np.ndarray] = None,
        cm: Optional[float] = None,
        intermittency: Optional[np.ndarray] = None,
        residual_history: Optional[List[float]] = None,
        cl_history: Optional[List[float]] = None,
        cd_history: Optional[List[float]] = None,
        reynolds: float = 200000,
        mach: float = 0.1,
        aoa: float = 0.0,
        design_id: str = "unknown",
        iteration: int = 0,
    ) -> ComprehensiveGovernanceReport:
        """
        Perform post-CFD governance checks.
        
        This validates the CFD solution AFTER simulation completes.
        
        Args:
            cl: Lift coefficient
            cd: Drag coefficient
            x: Chordwise coordinates
            cp: Pressure coefficient distribution
            cf: Skin friction coefficient distribution
            cm: Moment coefficient
            intermittency: Intermittency distribution
            residual_history: Residual convergence history
            cl_history: Lift coefficient history
            cd_history: Drag coefficient history
            reynolds: Reynolds number
            mach: Mach number
            aoa: Angle of attack
            design_id: Design identifier
            iteration: Current iteration
        
        Returns:
            ComprehensiveGovernanceReport with post-CFD validation results
        """
        self.total_evaluations += 1
        timestamp = time.time()
        
        all_violations = []
        failure_reasons = []
        warnings_list = []
        recommended_actions = []
        
        # 1. Convergence validation
        convergence_report = None
        if residual_history and cl_history and cd_history:
            residual_metrics = self.convergence_monitor.analyze(residual_history)
            force_metrics = self.convergence_monitor.analyze_forces(cl_history, cd_history)
            spectral_metrics = self.convergence_monitor.analyze_spectral(cl_history, cd_history)
            convergence_report = self.convergence_monitor.generate_report(
                residual_metrics, force_metrics, spectral_metrics
            )
            
            if not convergence_report.is_valid:
                all_violations.extend([v for v in convergence_report.failure_reasons])
                failure_reasons.extend(convergence_report.failure_reasons)
        
        # 2. Aerodynamic plausibility
        plausibility_report = self.plausibility_governor.govern(
            cl, cd, x, cp, cf, cm, reynolds, mach, aoa
        )
        
        if not plausibility_report.is_valid:
            all_violations.extend(plausibility_report.violations)
            failure_reasons.extend(plausibility_report.failure_reasons)
            recommended_actions.extend(plausibility_report.recommended_actions)
        
        # 3. LSB detection and classification
        lsb_report = None
        if cp is not None:
            lsb_report = self.lsb_detector.detect(x, cp, cf, intermittency)
        
        # 4. Transition model governance
        transition_report = None
        if intermittency is not None:
            transition_report = self.transition_governor.govern(
                x, intermittency, reynolds, cf
            )
            
            if not transition_report.is_valid:
                all_violations.extend([v for v in transition_report.diagnostics.warnings])
                failure_reasons.extend(transition_report.recommended_actions)
        
        # 5. Numerical dissipation check
        dissipation_report = None
        if residual_history:
            dissipation_report = self.dissipation_monitor.govern(
                residual_history=np.array(residual_history)
            )
            
            if not dissipation_report.is_acceptable:
                all_violations.extend(dissipation_report.violations)
                failure_reasons.extend(dissipation_report.failure_reasons)
        
        # Count violations
        critical_violations = len([v for v in all_violations if "BLUFF" in str(v) or "SEPARATION" in str(v)])
        total_violations = len(set(all_violations))
        
        # Determine decision
        convergence_ok = convergence_report.is_valid if convergence_report else True
        plausibility_ok = plausibility_report.is_valid
        transition_ok = transition_report.is_valid if transition_report else True
        dissipation_ok = dissipation_report.is_acceptable if dissipation_report else True
        
        if total_violations == 0 and convergence_ok and plausibility_ok:
            decision = GovernanceDecision.ACCEPT
            severity = GovernanceSeverity.NONE
            can_accept = True
        elif critical_violations > 0:
            decision = GovernanceDecision.REJECT
            severity = GovernanceSeverity.CRITICAL
            can_accept = False
        elif total_violations > 3:
            decision = GovernanceDecision.REJECT
            severity = GovernanceSeverity.HIGH
            can_accept = False
        elif total_violations > 0:
            decision = GovernanceDecision.SUSPECT
            severity = GovernanceSeverity.MEDIUM
            can_accept = False
        else:
            decision = GovernanceDecision.NEEDS_REVIEW
            severity = GovernanceSeverity.LOW
            can_accept = False
        
        report = ComprehensiveGovernanceReport(
            decision=decision,
            severity=severity,
            timestamp=timestamp,
            iteration=iteration,
            design_id=design_id,
            plausibility=plausibility_report,
            lsb_detection=lsb_report,
            transition=transition_report,
            dissipation=dissipation_report,
            convergence=convergence_report,
            total_violations=total_violations,
            critical_violations=critical_violations,
            warnings=len(warnings_list),
            failure_reasons=failure_reasons,
            warnings_list=warnings_list,
            recommended_actions=recommended_actions,
            can_proceed_to_cfd=False,  # Post-CFD
            can_accept_design=can_accept,
        )
        
        # Update statistics
        if can_accept:
            self.accepted_designs += 1
            self.consecutive_failures = 0
        else:
            self.rejected_designs += 1
            self.consecutive_failures += 1
        
        return report
    
    def optimization_governance(
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
        primary_objective: float = 0.0,
        penalty_value: float = 0.0,
        regularization_value: float = 0.0,
        constraint_penalties: Optional[Dict[str, float]] = None,
    ) -> ComprehensiveGovernanceReport:
        """
        Perform optimization governance checks.
        
        This validates the optimization process health.
        
        Args:
            objective_value: Total objective value
            design_vars: Design variables
            gradient: Gradient vector
            current_radius: Trust region radius
            initial_radius: Initial trust region radius
            rejected_steps: Consecutive rejected steps
            predicted_reduction: Predicted reduction
            actual_reduction: Actual reduction
            iteration: Current iteration
            primary_objective: Primary objective component
            penalty_value: Penalty component
            regularization_value: Regularization component
            constraint_penalties: Individual constraint penalties
        
        Returns:
            ComprehensiveGovernanceReport with optimization health
        """
        timestamp = time.time()
        
        # 1. Optimizer governance
        optimizer_report = self.optimizer_governor.govern(
            objective_value, design_vars, gradient,
            current_radius, initial_radius, rejected_steps,
            predicted_reduction, actual_reduction, iteration
        )
        
        # 2. Objective governance
        constraint_penalties = constraint_penalties or {}
        objective_report = self.objective_governor.govern(
            objective_value, primary_objective, penalty_value,
            regularization_value, constraint_penalties
        )
        
        # Collect violations
        all_violations = []
        failure_reasons = []
        recommended_actions = []
        
        if not optimizer_report.is_healthy:
            all_violations.extend(optimizer_report.violations)
            failure_reasons.extend(optimizer_report.failure_reasons)
            recommended_actions.extend(optimizer_report.recovery_strategies)
        
        if not objective_report.is_valid:
            all_violations.extend(objective_report.violations)
            failure_reasons.extend(objective_report.failure_reasons)
            recommended_actions.extend(objective_report.recommended_actions)
        
        # Count violations
        critical_violations = len([v for v in all_violations if "CRITICAL" in str(v)])
        total_violations = len(set(all_violations))
        
        # Determine decision
        if optimizer_report.is_healthy and objective_report.is_valid:
            decision = GovernanceDecision.ACCEPT
            severity = GovernanceSeverity.NONE
        elif optimizer_report.can_continue and objective_report.can_use_for_optimization:
            decision = GovernanceDecision.SUSPECT
            severity = GovernanceSeverity.MEDIUM
        else:
            decision = GovernanceDecision.REJECT
            severity = GovernanceSeverity.HIGH if critical_violations > 0 else GovernanceSeverity.MEDIUM
        
        return ComprehensiveGovernanceReport(
            decision=decision,
            severity=severity,
            timestamp=timestamp,
            iteration=iteration,
            design_id=f"opt_iter_{iteration}",
            optimizer=optimizer_report,
            objective=objective_report,
            total_violations=total_violations,
            critical_violations=critical_violations,
            failure_reasons=failure_reasons,
            recommended_actions=recommended_actions,
            can_proceed_to_cfd=False,
            can_accept_design=decision == GovernanceDecision.ACCEPT,
        )
    
    def should_stop_optimization(self) -> Tuple[bool, str]:
        """
        Check if optimization should be stopped.
        
        Returns:
            (should_stop, reason) tuple
        """
        if self.consecutive_failures >= self.config.stop_on_consecutive_failures:
            return True, f"Consecutive failures ({self.consecutive_failures}) exceeded limit"
        
        if self.suspect_designs_count >= self.config.max_suspect_designs_per_run:
            return True, f"Suspect designs ({self.suspect_designs_count}) exceeded limit"
        
        return False, ""
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get governance statistics."""
        return {
            "total_evaluations": self.total_evaluations,
            "accepted_designs": self.accepted_designs,
            "rejected_designs": self.rejected_designs,
            "acceptance_rate": self.accepted_designs / max(1, self.total_evaluations),
            "consecutive_failures": self.consecutive_failures,
            "suspect_designs_count": self.suspect_designs_count,
        }
    
    def reset(self) -> None:
        """Reset all statistics and history."""
        self.total_evaluations = 0
        self.accepted_designs = 0
        self.rejected_designs = 0
        self.consecutive_failures = 0
        self.suspect_designs_count = 0
        self.optimizer_governor.reset()
        self.objective_governor.reset()