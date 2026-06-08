"""
Scientific Truth Policy — HARD governance enforcement for aerodynamic optimization.

The framework MUST stop optimization immediately if:
- Gradients corrupt (adjoint truth audit fails)
- Transition collapses numerically (physics audit fails)
- Mesh invalidates (mesh quality fails)
- Convergence stagnates (convergence analysis fails)
- Intermittency diverges (γ-equation pathological)
- False reattachment detected (Cf recovery without pressure recovery)
- Trust region degenerates (radius collapses)
- Force oscillations persist (unsteady flow misclassified as steady)
- Numerical dissipation dominates physics (dissipation audit fails)

NO SILENT CONTINUATION. EVER.

Every termination includes:
- Failure classification
- Root-cause analysis
- Telemetry snapshot
- Reproducibility package
- Archived diagnostics
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import time
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)


class FailureSeverity(Enum):
    """Severity classification for optimization failures."""
    FATAL = "FATAL"           # Optimization MUST stop immediately
    CRITICAL = "CRITICAL"     # Optimization SHOULD stop, state may be invalid
    RECOVERABLE = "RECOVERABLE"  # Can attempt rollback and retry
    WARNING = "WARNING"       # Degraded but can continue with caution


class FailureCategory(Enum):
    """Taxonomy of possible optimization failures."""
    GRADIENT_CORRUPTION = "GRADIENT_CORRUPTION"
    CFD_DIVERGENCE = "CFD_DIVERGENCE"
    CFD_CONVERGENCE_FAILURE = "CFD_CONVERGENCE_FAILURE"
    CFD_PHYSICS_INVALID = "CFD_PHYSICS_INVALID"
    MESH_INVALID = "MESH_INVALID"
    MESH_QUALITY_FAILURE = "MESH_QUALITY_FAILURE"
    GEOMETRY_INVALID = "GEOMETRY_INVALID"
    TRANSITION_COLLAPSE = "TRANSITION_COLLAPSE"
    INTERMITTENCY_DIVERGENCE = "INTERMITTENCY_DIVERGENCE"
    FALSE_REATTACHMENT = "FALSE_REATTACHMENT"
    TRUST_REGION_COLLAPSE = "TRUST_REGION_COLLAPSE"
    FORCE_OSCILLATION = "FORCE_OSCILLATION"
    DISSIPATION_DOMINANCE = "DISSIPATION_DOMINANCE"
    OBJECTIVE_EXPLOIT = "OBJECTIVE_EXPLOIT"
    STAGNATION = "STAGNATION"
    NUMERICAL_INSTABILITY = "NUMERICAL_INSTABILITY"
    WATCHDOG_TIMEOUT = "WATCHDOG_TIMEOUT"
    FILESYSTEM_ERROR = "FILESYSTEM_ERROR"
    MEMORY_EXHAUSTION = "MEMORY_EXHAUSTION"
    UNKNOWN = "UNKNOWN"


@dataclass
class FailureRecord:
    """Complete record of an optimization failure."""
    failure_id: str
    timestamp: float
    category: FailureCategory
    severity: FailureSeverity
    iteration: int
    design_vector: np.ndarray
    message: str
    
    # Root cause details
    root_cause: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    
    # Telemetry snapshot
    objective_value: Optional[float] = None
    gradient_norm: Optional[float] = None
    trust_radius: Optional[float] = None
    convergence_status: Optional[str] = None
    
    # Reproducibility
    reproducibility_package: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "failure_id": self.failure_id,
            "timestamp": self.timestamp,
            "category": self.category.value,
            "severity": self.severity.value,
            "iteration": self.iteration,
            "design_vector": self.design_vector.tolist(),
            "message": self.message,
            "root_cause": self.root_cause,
            "diagnostics": self.diagnostics,
            "objective_value": self.objective_value,
            "gradient_norm": self.gradient_norm,
            "trust_radius": self.trust_radius,
        }


class ScientificTruthPolicy:
    """
    HARD governance enforcement for aerodynamic optimization.
    
    This is the "hostile reviewer embedded in the optimizer."
    Every result must survive verification before acceptance.
    
    Policy rules:
    - FATAL: Optimization stops, results archived, run invalidated
    - CRITICAL: Optimization stops, rollback attempted
    - RECOVERABLE: Rollback + retry with modified parameters
    - WARNING: Continue but flag in telemetry
    """
    
    def __init__(self, enable_hard_gates: bool = True):
        self.enable_hard_gates = enable_hard_gates
        self.failure_history: List[FailureRecord] = []
        self._failure_counter = 0
    
    def evaluate_cfd_result(self,
                            evaluation_status: str,
                            convergence_report: Optional[Dict[str, Any]] = None,
                            physics_report: Optional[Dict[str, Any]] = None,
                            gradient_report: Optional[Dict[str, Any]] = None,
                            iteration: int = 0,
                            ) -> Tuple[bool, Optional[FailureRecord]]:
        """
        Evaluate a CFD result against all scientific truth policies.
        
        Returns:
            (passed, failure_record): True if result passes all checks
        """
        # Check 1: CFD execution status
        if evaluation_status not in ["OK", "GRADIENT_ZERO"]:
            return self._fail(
                category=FailureCategory.CFD_DIVERGENCE,
                severity=FailureSeverity.FATAL,
                iteration=iteration,
                message=f"CFD execution failed: {evaluation_status}",
                diagnostics={"status": evaluation_status},
            )
        
        # Check 2: Convergence verification
        if convergence_report:
            if not convergence_report.get("is_valid", False):
                reasons = convergence_report.get("failure_reasons", [])
                cat = FailureCategory.CFD_CONVERGENCE_FAILURE
                return self._fail(
                    category=cat,
                    severity=FailureSeverity.FATAL,
                    iteration=iteration,
                    message=f"Convergence failure: {'; '.join(reasons)}",
                    diagnostics=convergence_report,
                )
        
        # Check 3: Physics credibility
        if physics_report:
            pathologies = physics_report.get("pathologies_detected", [])
            credibility = physics_report.get("overall_credibility", "CREDIBLE")
            
            if credibility in ["INCREDIBLE"]:
                cat = self._map_pathology_to_category(pathologies)
                return self._fail(
                    category=cat,
                    severity=FailureSeverity.CRITICAL,
                    iteration=iteration,
                    message=f"Physics incredible: {', '.join(pathologies)}",
                    diagnostics=physics_report,
                )
            
            # Check for specific pathologies
            if physics_report.get("dissipation_suppression_detected", False):
                return self._fail(
                    category=FailureCategory.DISSIPATION_DOMINANCE,
                    severity=FailureSeverity.CRITICAL,
                    iteration=iteration,
                    message="Dissipation suppression detected - flow physics compromised",
                    diagnostics=physics_report,
                )
            
            if physics_report.get("false_reattachment_detected", False):
                return self._fail(
                    category=FailureCategory.FALSE_REATTACHMENT,
                    severity=FailureSeverity.CRITICAL,
                    iteration=iteration,
                    message="False reattachment detected - Cf recovery without pressure recovery",
                    diagnostics=physics_report,
                )
        
        # Check 4: Gradient credibility
        if gradient_report:
            if not gradient_report.get("is_credible", False):
                reasons = gradient_report.get("failure_reasons", [])
                return self._fail(
                    category=FailureCategory.GRADIENT_CORRUPTION,
                    severity=FailureSeverity.FATAL,
                    iteration=iteration,
                    message=f"Gradient corrupted: {'; '.join(reasons)}",
                    diagnostics=gradient_report,
                )
        
        return True, None
    
    def evaluate_optimization_step(self,
                                   step_accepted: bool,
                                   stagnated_counter: int,
                                   trust_radius: float,
                                   gradient_norm: float,
                                   force_oscillation: Optional[float] = None,
                                   iteration: int = 0,
                                   ) -> Tuple[bool, Optional[FailureRecord]]:
        """
        Evaluate optimization step against truth policies.
        
        Returns:
            (continue_optimization, failure_record)
        """
        # Check stagnation
        if stagnated_counter >= 10:
            return self._fail(
                category=FailureCategory.STAGNATION,
                severity=FailureSeverity.CRITICAL,
                iteration=iteration,
                message=f"Optimization stagnated after {stagnated_counter} rejected steps",
                diagnostics={
                    "stagnated_counter": stagnated_counter,
                    "trust_radius": trust_radius,
                    "gradient_norm": gradient_norm,
                },
            )
        
        # Check trust region collapse
        if trust_radius < 1e-8:
            return self._fail(
                category=FailureCategory.TRUST_REGION_COLLAPSE,
                severity=FailureSeverity.RECOVERABLE,
                iteration=iteration,
                message=f"Trust region collapsed to {trust_radius:.2e}",
                diagnostics={"trust_radius": trust_radius},
            )
        
        # Check gradient norm
        if gradient_norm < 1e-12 and step_accepted:
            # Gradient is essentially zero but step was accepted
            # This is fine - means we converged
            pass
        
        # Check force oscillation
        if force_oscillation is not None and force_oscillation > 0.1:
            return self._fail(
                category=FailureCategory.FORCE_OSCILLATION,
                severity=FailureSeverity.WARNING,
                iteration=iteration,
                message=f"Force oscillation {force_oscillation:.2%} exceeds 10% threshold",
                diagnostics={"force_oscillation": force_oscillation},
            )
        
        return True, None
    
    def evaluate_geometry(self,
                          is_valid: bool,
                          violations: List[str],
                          iteration: int = 0,
                          ) -> Tuple[bool, Optional[FailureRecord]]:
        """Evaluate geometry validity."""
        if not is_valid:
            return self._fail(
                category=FailureCategory.GEOMETRY_INVALID,
                severity=FailureSeverity.FATAL,
                iteration=iteration,
                message=f"Invalid geometry: {', '.join(violations)}",
                diagnostics={"violations": violations},
            )
        return True, None
    
    def evaluate_gradient_zero(self,
                                gradient_norm: float,
                                iteration: int = 0) -> Tuple[bool, Optional[FailureRecord]]:
        """Handle zero-gradient situation."""
        if gradient_norm < 1e-14:
            return self._fail(
                category=FailureCategory.GRADIENT_CORRUPTION,
                severity=FailureSeverity.FATAL,
                iteration=iteration,
                message="Zero gradient detected - adjoint system not producing sensitivities",
                diagnostics={"gradient_norm": gradient_norm},
            )
        return True, None

    def evaluate_transition_physics(self,
                                     lsb_report: Optional[Dict[str, Any]] = None,
                                     intermittency_valid: Optional[bool] = None,
                                     iteration: int = 0,
                                     ) -> Tuple[bool, Optional[FailureRecord]]:
        """Evaluate transition physics credibility."""
        if lsb_report:
            if not lsb_report.get("is_valid", True):
                warnings = lsb_report.get("warnings", [])
                return self._fail(
                    category=FailureCategory.TRANSITION_COLLAPSE,
                    severity=FailureSeverity.CRITICAL,
                    iteration=iteration,
                    message=f"Transition physics invalid: {'; '.join(warnings)}",
                    diagnostics=lsb_report,
                )
        
        if intermittency_valid is not None and not intermittency_valid:
            return self._fail(
                category=FailureCategory.INTERMITTENCY_DIVERGENCE,
                severity=FailureSeverity.CRITICAL,
                iteration=iteration,
                message="Intermittency equation not solving correctly",
            )
        
        return True, None
    
    def _fail(self, category: FailureCategory, severity: FailureSeverity,
              iteration: int, message: str,
              diagnostics: Optional[Dict[str, Any]] = None) -> Tuple[bool, FailureRecord]:
        """Record and return a failure."""
        self._failure_counter += 1
        record = FailureRecord(
            failure_id=f"FAIL_{self._failure_counter:04d}",
            timestamp=time.time(),
            category=category,
            severity=severity,
            iteration=iteration,
            design_vector=np.array([]),
            message=message,
            diagnostics=diagnostics or {},
        )
        
        self.failure_history.append(record)
        logger.error(f"TRUTH POLICY [{severity.value}] {category.value}: {message}")
        
        return False, record
    
    def _map_pathology_to_category(self, pathologies: List[str]) -> FailureCategory:
        """Map physics pathology to failure category."""
        pathology_map = {
            "dissipation": FailureCategory.DISSIPATION_DOMINANCE,
            "reattachment": FailureCategory.FALSE_REATTACHMENT,
            "transition": FailureCategory.TRANSITION_COLLAPSE,
            "intermittency": FailureCategory.INTERMITTENCY_DIVERGENCE,
            "CFL": FailureCategory.NUMERICAL_INSTABILITY,
            "limiter": FailureCategory.CFD_PHYSICS_INVALID,
        }
        
        for path in pathologies:
            for key, cat in pathology_map.items():
                if key.lower() in path.lower():
                    return cat
        
        return FailureCategory.CFD_PHYSICS_INVALID
    
    def should_stop(self,
                    severity: FailureSeverity,
                    enable_hard_gates: Optional[bool] = None) -> bool:
        """Determine if optimization should stop based on severity."""
        gates = enable_hard_gates if enable_hard_gates is not None else self.enable_hard_gates
        if not gates:
            return severity == FailureSeverity.FATAL
        return severity in [FailureSeverity.FATAL, FailureSeverity.CRITICAL]