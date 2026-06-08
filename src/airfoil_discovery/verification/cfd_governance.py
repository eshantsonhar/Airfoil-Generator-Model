"""
CFD governance model with validation checkpoints.

The optimizer MUST NOT automatically trust CFD outputs. Every iteration
must pass numerical convergence, transition validity, gradient integrity,
and physical plausibility checks before being marked VALID.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from enum import Enum

from .convergence import ConvergenceReport, ConvergenceStatus
from .gradient_audit import GradientAuditReport, GradientStatus
from .mesh_verification import MeshVerificationReport, MeshStatus
from .numerical_dissipation import DissipationDiagnosticsReport
from ..physics.lsb_detection import LSBDetectionReport
from ..physics.transition_governance import TransitionGovernanceReport


class GovernanceStatus(Enum):
    """CFD governance status."""
    VALID = "VALID"
    INVALID = "INVALID"
    NUMERICAL_FAILURE = "NUMERICAL_FAILURE"
    TRANSITION_FAILURE = "TRANSITION_FAILURE"
    GRADIENT_FAILURE = "GRADIENT_FAILURE"
    PHYSICAL_FAILURE = "PHYSICAL_FAILURE"
    MESH_FAILURE = "MESH_FAILURE"
    DISSIPATION_FAILURE = "DISSIPATION_FAILURE"


@dataclass
class GovernanceCheckResult:
    """Result of a single governance check."""
    
    check_name: str
    passed: bool
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class CFDGovernanceReport:
    """Comprehensive CFD governance report."""
    
    # Overall status
    status: GovernanceStatus
    is_valid: bool
    
    # Component reports
    convergence: Optional[ConvergenceReport] = None
    gradient: Optional[GradientAuditReport] = None
    mesh: Optional[MeshVerificationReport] = None
    dissipation: Optional[DissipationDiagnosticsReport] = None
    lsb: Optional[LSBDetectionReport] = None
    transition: Optional[TransitionGovernanceReport] = None
    
    # Individual check results
    check_results: List[GovernanceCheckResult] = field(default_factory=list)
    
    # Failure analysis
    failure_reasons: List[str] = field(default_factory=list)
    
    # Recommendations
    recommended_actions: List[str] = field(default_factory=list)
    
    # Iteration metadata
    iteration_number: Optional[int] = None
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "is_valid": self.is_valid,
            "iteration_number": self.iteration_number,
            "timestamp": self.timestamp,
            "failure_reasons": self.failure_reasons,
            "recommended_actions": self.recommended_actions,
            "check_results": [
                {
                    "check_name": r.check_name,
                    "passed": r.passed,
                    "status": r.status,
                    "message": r.message,
                }
                for r in self.check_results
            ],
        }


class CFDGovernanceModel:
    """
    Comprehensive CFD governance model.
    
    Enforces that every CFD iteration must pass:
    A. Numerical Convergence Checks
    B. Transition Validity Checks
    C. Gradient Integrity Checks
    D. Physical Plausibility Checks
    
    Only then can a solution be marked VALID.
    """
    
    def __init__(
        self,
        require_convergence: bool = True,
        require_gradient_validity: bool = True,
        require_mesh_validity: bool = True,
        require_transition_validity: bool = True,
        require_physical_plausibility: bool = True,
        allow_numerical_dissipation_warning: bool = False,
    ):
        """
        Initialize CFD governance model.
        
        Args:
            require_convergence: Require numerical convergence
            require_gradient_validity: Require gradient validity
            require_mesh_validity: Require mesh validity
            require_transition_validity: Require transition model validity
            require_physical_plausibility: Require physical plausibility
            allow_numerical_dissipation_warning: Allow warnings for dissipation
        """
        self.require_convergence = require_convergence
        self.require_gradient_validity = require_gradient_validity
        self.require_mesh_validity = require_mesh_validity
        self.require_transition_validity = require_transition_validity
        self.require_physical_plausibility = require_physical_plausibility
        self.allow_numerical_dissipation_warning = allow_numerical_dissipation_warning
    
    def check_numerical_convergence(
        self,
        convergence_report: ConvergenceReport,
    ) -> GovernanceCheckResult:
        """
        Check numerical convergence.
        
        Args:
            convergence_report: Convergence analysis report
        
        Returns:
            GovernanceCheckResult for convergence check
        """
        if convergence_report is None:
            return GovernanceCheckResult(
                check_name="numerical_convergence",
                passed=False,
                status="NO_DATA",
                message="No convergence data available",
            )
        
        if not convergence_report.is_valid:
            return GovernanceCheckResult(
                check_name="numerical_convergence",
                passed=False,
                status=convergence_report.status.value,
                message=f"Convergence failed: {', '.join(convergence_report.failure_reasons)}",
                details={
                    "status": convergence_report.status.value,
                    "failure_reasons": convergence_report.failure_reasons,
                },
            )
        
        return GovernanceCheckResult(
            check_name="numerical_convergence",
            passed=True,
            status="CONVERGED",
            message="Numerical convergence verified",
            details={
                "status": convergence_report.status.value,
            },
        )
    
    def check_gradient_integrity(
        self,
        gradient_report: GradientAuditReport,
    ) -> GovernanceCheckResult:
        """
        Check gradient integrity.
        
        Args:
            gradient_report: Gradient audit report
        
        Returns:
            GovernanceCheckResult for gradient check
        """
        if gradient_report is None:
            return GovernanceCheckResult(
                check_name="gradient_integrity",
                passed=False,
                status="NO_DATA",
                message="No gradient data available",
            )
        
        if not gradient_report.is_valid:
            return GovernanceCheckResult(
                check_name="gradient_integrity",
                passed=False,
                status=gradient_report.status.value,
                message=f"Gradient check failed: {', '.join(gradient_report.failure_reasons)}",
                details={
                    "status": gradient_report.status.value,
                    "failure_reasons": gradient_report.failure_reasons,
                    "cosine_similarity": gradient_report.cosine_history[-1] if gradient_report.cosine_history else None,
                },
            )
        
        return GovernanceCheckResult(
            check_name="gradient_integrity",
            passed=True,
            status="VALID",
            message="Gradient integrity verified",
            details={
                "status": gradient_report.status.value,
            },
        )
    
    def check_mesh_validity(
        self,
        mesh_report: MeshVerificationReport,
    ) -> GovernanceCheckResult:
        """
        Check mesh validity.
        
        Args:
            mesh_report: Mesh verification report
        
        Returns:
            GovernanceCheckResult for mesh check
        """
        if mesh_report is None:
            return GovernanceCheckResult(
                check_name="mesh_validity",
                passed=False,
                status="NO_DATA",
                message="No mesh data available",
            )
        
        if not mesh_report.is_valid:
            return GovernanceCheckResult(
                check_name="mesh_validity",
                passed=False,
                status=mesh_report.status.value,
                message=f"Mesh check failed: {', '.join(mesh_report.failure_reasons)}",
                details={
                    "status": mesh_report.status.value,
                    "failure_reasons": mesh_report.failure_reasons,
                },
            )
        
        return GovernanceCheckResult(
            check_name="mesh_validity",
            passed=True,
            status="VALID",
            message="Mesh validity verified",
            details={
                "status": mesh_report.status.value,
            },
        )
    
    def check_transition_validity(
        self,
        transition_report: TransitionGovernanceReport,
    ) -> GovernanceCheckResult:
        """
        Check transition model validity.
        
        Args:
            transition_report: Transition governance report
        
        Returns:
            GovernanceCheckResult for transition check
        """
        if transition_report is None:
            return GovernanceCheckResult(
                check_name="transition_validity",
                passed=False,
                status="NO_DATA",
                message="No transition data available",
            )
        
        if not transition_report.is_valid:
            return GovernanceCheckResult(
                check_name="transition_validity",
                passed=False,
                status="TRANSITION_INVALID",
                message=f"Transition model check failed: {', '.join(transition_report.recommended_actions)}",
                details={
                    "can_trust_transition": transition_report.can_trust_transition,
                    "model_confidence": transition_report.diagnostics.model_confidence,
                },
            )
        
        if not transition_report.can_trust_transition:
            return GovernanceCheckResult(
                check_name="transition_validity",
                passed=False,
                status="TRANSITION_UNCERTAIN",
                message="Transition model uncertain - low confidence",
                details={
                    "can_trust_transition": transition_report.can_trust_transition,
                    "model_confidence": transition_report.diagnostics.model_confidence,
                },
            )
        
        return GovernanceCheckResult(
            check_name="transition_validity",
            passed=True,
            status="VALID",
            message="Transition model validity verified",
            details={
                "model_confidence": transition_report.diagnostics.model_confidence,
            },
        )
    
    def check_physical_plausibility(
        self,
        lsb_report: LSBDetectionReport,
        dissipation_report: Optional[DissipationDiagnosticsReport] = None,
    ) -> GovernanceCheckResult:
        """
        Check physical plausibility of solution.
        
        Args:
            lsb_report: LSB detection report
            dissipation_report: Optional numerical dissipation report
        
        Returns:
            GovernanceCheckResult for physical plausibility check
        """
        if lsb_report is None:
            return GovernanceCheckResult(
                check_name="physical_plausibility",
                passed=False,
                status="NO_DATA",
                message="No LSB data available",
            )
        
        issues = []
        
        # Check LSB physical consistency
        if not lsb_report.is_valid:
            issues.append("LSB metrics physically inconsistent")
        
        # Check for numerical LSB suppression
        if dissipation_report and not dissipation_report.physics_trustworthy:
            issues.append("Numerical dissipation may be suppressing LSB physics")
        
        # Check transition warnings
        if lsb_report.warnings:
            issues.extend(lsb_report.warnings)
        
        if issues:
            return GovernanceCheckResult(
                check_name="physical_plausibility",
                passed=False,
                status="PHYSICALLY_IMPLAUSIBLE",
                message=f"Physical plausibility check failed: {', '.join(issues)}",
                details={
                    "issues": issues,
                    "lsb_detected": lsb_report.metrics.lsb_detected,
                },
            )
        
        return GovernanceCheckResult(
            check_name="physical_plausibility",
            passed=True,
            status="PHYSICALLY_PLAUSIBLE",
            message="Physical plausibility verified",
            details={
                "lsb_detected": lsb_report.metrics.lsb_detected,
            },
        )
    
    def check_numerical_dissipation(
        self,
        dissipation_report: DissipationDiagnosticsReport,
    ) -> GovernanceCheckResult:
        """
        Check numerical dissipation levels.
        
        Args:
            dissipation_report: Numerical dissipation diagnostics report
        
        Returns:
            GovernanceCheckResult for dissipation check
        """
        if dissipation_report is None:
            return GovernanceCheckResult(
                check_name="numerical_dissipation",
                passed=False,
                status="NO_DATA",
                message="No dissipation data available",
            )
        
        if not dissipation_report.is_valid:
            if self.allow_numerical_dissipation_warning:
                return GovernanceCheckResult(
                    check_name="numerical_dissipation",
                    passed=True,
                    status="WARNING",
                    message=f"Numerical dissipation warning: {', '.join(dissipation_report.recommended_actions)}",
                    details={
                        "dissipation_level": dissipation_report.metrics.numerical_dissipation_level,
                        "physics_trustworthy": dissipation_report.physics_trustworthy,
                    },
                )
            else:
                return GovernanceCheckResult(
                    check_name="numerical_dissipation",
                    passed=False,
                    status="DISSIPATION_HIGH",
                    message=f"Numerical dissipation too high: {', '.join(dissipation_report.recommended_actions)}",
                    details={
                        "dissipation_level": dissipation_report.metrics.numerical_dissipation_level,
                    },
                )
        
        return GovernanceCheckResult(
            check_name="numerical_dissipation",
            passed=True,
            status="ACCEPTABLE",
            message="Numerical dissipation within acceptable limits",
            details={
                "dissipation_level": dissipation_report.metrics.numerical_dissipation_level,
            },
        )
    
    def govern(
        self,
        convergence_report: Optional[ConvergenceReport] = None,
        gradient_report: Optional[GradientAuditReport] = None,
        mesh_report: Optional[MeshVerificationReport] = None,
        dissipation_report: Optional[DissipationDiagnosticsReport] = None,
        lsb_report: Optional[LSBDetectionReport] = None,
        transition_report: Optional[TransitionGovernanceReport] = None,
        iteration_number: Optional[int] = None,
    ) -> CFDGovernanceReport:
        """
        Perform comprehensive CFD governance.
        
        Args:
            convergence_report: Convergence analysis report
            gradient_report: Gradient audit report
            mesh_report: Mesh verification report
            dissipation_report: Numerical dissipation report
            lsb_report: LSB detection report
            transition_report: Transition governance report
            iteration_number: Iteration number
        
        Returns:
            CFDGovernanceReport with comprehensive governance assessment
        """
        from datetime import datetime, timezone
        
        check_results = []
        failure_reasons = []
        recommended_actions = []
        
        # A. Numerical Convergence Checks
        if self.require_convergence:
            convergence_check = self.check_numerical_convergence(convergence_report)
            check_results.append(convergence_check)
            if not convergence_check.passed:
                failure_reasons.append(convergence_check.message)
                recommended_actions.extend(convergence_report.recommended_actions if convergence_report else [])
        
        # B. Gradient Integrity Checks
        if self.require_gradient_validity:
            gradient_check = self.check_gradient_integrity(gradient_report)
            check_results.append(gradient_check)
            if not gradient_check.passed:
                failure_reasons.append(gradient_check.message)
                recommended_actions.extend(gradient_report.recommended_actions if gradient_report else [])
        
        # C. Mesh Validity Checks
        if self.require_mesh_validity:
            mesh_check = self.check_mesh_validity(mesh_report)
            check_results.append(mesh_check)
            if not mesh_check.passed:
                failure_reasons.append(mesh_check.message)
                recommended_actions.extend(mesh_report.recommended_actions if mesh_report else [])
        
        # D. Transition Validity Checks
        if self.require_transition_validity:
            transition_check = self.check_transition_validity(transition_report)
            check_results.append(transition_check)
            if not transition_check.passed:
                failure_reasons.append(transition_check.message)
                recommended_actions.extend(transition_report.recommended_actions if transition_report else [])
        
        # E. Physical Plausibility Checks
        if self.require_physical_plausibility:
            physical_check = self.check_physical_plausibility(lsb_report, dissipation_report)
            check_results.append(physical_check)
            if not physical_check.passed:
                failure_reasons.append(physical_check.message)
        
        # F. Numerical Dissipation Checks
        dissipation_check = self.check_numerical_dissipation(dissipation_report)
        check_results.append(dissipation_check)
        if not dissipation_check.passed:
            failure_reasons.append(dissipation_check.message)
            recommended_actions.extend(dissipation_report.recommended_actions if dissipation_report else [])
        
        # Determine overall status
        all_passed = all(check.passed for check in check_results)
        
        if not all_passed:
            # Determine specific failure type
            if not check_results[0].passed and convergence_report:
                status = GovernanceStatus.NUMERICAL_FAILURE
            elif not check_results[1].passed and gradient_report:
                status = GovernanceStatus.GRADIENT_FAILURE
            elif not check_results[2].passed and mesh_report:
                status = GovernanceStatus.MESH_FAILURE
            elif not check_results[3].passed and transition_report:
                status = GovernanceStatus.TRANSITION_FAILURE
            elif not check_results[4].passed and lsb_report:
                status = GovernanceStatus.PHYSICAL_FAILURE
            elif not check_results[5].passed and dissipation_report:
                status = GovernanceStatus.DISSIPATION_FAILURE
            else:
                status = GovernanceStatus.INVALID
        else:
            status = GovernanceStatus.VALID
        
        return CFDGovernanceReport(
            status=status,
            is_valid=all_passed,
            convergence=convergence_report,
            gradient=gradient_report,
            mesh=mesh_report,
            dissipation=dissipation_report,
            lsb=lsb_report,
            transition=transition_report,
            check_results=check_results,
            failure_reasons=failure_reasons,
            recommended_actions=recommended_actions,
            iteration_number=iteration_number,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
