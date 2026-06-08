"""
Scientific failure policies and crash preservation.

Scientific honesty overrides optimization continuation. If gradients are
corrupted, transition becomes unstable, mesh validity collapses, convergence
becomes unreliable, or physical plausibility fails, the framework MUST stop,
archive diagnostics, preserve crash states, and mark the run INVALID.
"""

from __future__ import annotations

import traceback
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone


class FailureType(Enum):
    """Types of scientific failures."""
    GRADIENT_CORRUPTION = "GRADIENT_CORRUPTION"
    TRANSITION_INSTABILITY = "TRANSITION_INSTABILITY"
    MESH_VALIDITY_COLLAPSE = "MESH_VALIDITY_COLLAPSE"
    CONVERGENCE_UNRELIABLE = "CONVERGENCE_UNRELIABLE"
    PHYSICAL_PLAUSIBILITY_FAILURE = "PHYSICAL_PLAUSIBILITY_FAILURE"
    SOLVER_CRASH = "SOLVER_CRASH"
    MESH_GENERATION_FAILURE = "MESH_GENERATION_FAILURE"
    ADJOINT_FAILURE = "ADJOINT_FAILURE"
    NUMERICAL_INSTABILITY = "NUMERICAL_INSTABILITY"
    UNKNOWN = "UNKNOWN"


class FailureSeverity(Enum):
    """Severity of failure."""
    CRITICAL = "CRITICAL"  # Must stop immediately
    SEVERE = "SEVERE"  # Should stop after current iteration
    MODERATE = "MODERATE"  # Can continue with caution
    WARNING = "WARNING"  # Log and continue


@dataclass
class FailureRecord:
    """Record of a failure event."""
    
    # Failure information
    failure_type: FailureType
    severity: FailureSeverity
    message: str
    
    # Context
    iteration: Optional[int] = None
    component: Optional[str] = None
    
    # Diagnostics
    traceback: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamp
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "failure_type": self.failure_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "iteration": self.iteration,
            "component": self.component,
            "traceback": self.traceback,
            "diagnostics": self.diagnostics,
            "timestamp": self.timestamp,
        }


@dataclass
class CrashState:
    """Preserved crash state for analysis."""
    
    # Run information
    run_id: str
    iteration: Optional[int] = None
    
    # Failure information
    failure_record: Optional[FailureRecord] = None
    
    # State at crash
    design_vector: Optional[List[float]] = None
    objective_value: Optional[float] = None
    gradient: Optional[List[float]] = None
    
    # Solver state
    solver_state: Optional[Dict[str, Any]] = None
    mesh_state: Optional[Dict[str, Any]] = None
    
    # Verification state
    verification_results: Optional[Dict[str, Any]] = None
    
    # Environment
    environment_snapshot: Optional[Dict[str, Any]] = None
    
    # Timestamp
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "iteration": self.iteration,
            "failure_record": self.failure_record.to_dict() if self.failure_record else None,
            "design_vector": self.design_vector,
            "objective_value": self.objective_value,
            "gradient": self.gradient,
            "solver_state": self.solver_state,
            "mesh_state": self.mesh_state,
            "verification_results": self.verification_results,
            "environment_snapshot": self.environment_snapshot,
            "timestamp": self.timestamp,
        }


class ScientificFailurePolicy:
    """
    Enforces scientific failure policies.
    
    Scientific honesty overrides optimization continuation. The framework
    MUST stop, archive diagnostics, preserve crash states, and mark the
    run INVALID when critical failures occur.
    """
    
    def __init__(
        self,
        crash_dir: Path,
        stop_on_critical: bool = True,
        stop_on_severe: bool = True,
        stop_on_moderate: bool = False,
    ):
        """
        Initialize scientific failure policy.
        
        Args:
            crash_dir: Directory for crash state preservation
            stop_on_critical: Stop on critical failures
            stop_on_severe: Stop on severe failures
            stop_on_moderate: Stop on moderate failures
        """
        self.crash_dir = crash_dir
        self.crash_dir.mkdir(parents=True, exist_ok=True)
        
        self.stop_on_critical = stop_on_critical
        self.stop_on_severe = stop_on_severe
        self.stop_on_moderate = stop_on_moderate
        
        self.failure_history: List[FailureRecord] = []
        self.run_invalid = False
    
    def record_failure(
        self,
        failure_type: FailureType,
        severity: FailureSeverity,
        message: str,
        iteration: Optional[int] = None,
        component: Optional[str] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> FailureRecord:
        """
        Record a failure event.
        
        Args:
            failure_type: Type of failure
            severity: Severity of failure
            message: Failure message
            iteration: Iteration number
            component: Component that failed
            diagnostics: Diagnostic information
        
        Returns:
            FailureRecord
        """
        # Capture traceback if available
        tb_str = None
        if severity in [FailureSeverity.CRITICAL, FailureSeverity.SEVERE]:
            tb_str = traceback.format_exc()
        
        record = FailureRecord(
            failure_type=failure_type,
            severity=severity,
            message=message,
            iteration=iteration,
            component=component,
            traceback=tb_str,
            diagnostics=diagnostics or {},
        )
        
        self.failure_history.append(record)
        
        # Mark run invalid for critical or severe failures
        if severity in [FailureSeverity.CRITICAL, FailureSeverity.SEVERE]:
            self.run_invalid = True
        
        return record
    
    def should_stop(self, severity: FailureSeverity) -> bool:
        """
        Determine if optimization should stop based on severity.
        
        Args:
            severity: Severity of failure
        
        Returns:
            True if should stop
        """
        if severity == FailureSeverity.CRITICAL:
            return self.stop_on_critical
        elif severity == FailureSeverity.SEVERE:
            return self.stop_on_severe
        elif severity == FailureSeverity.MODERATE:
            return self.stop_on_moderate
        else:
            return False
    
    def preserve_crash_state(
        self,
        run_id: str,
        iteration: Optional[int] = None,
        design_vector: Optional[List[float]] = None,
        objective_value: Optional[float] = None,
        gradient: Optional[List[float]] = None,
        solver_state: Optional[Dict[str, Any]] = None,
        mesh_state: Optional[Dict[str, Any]] = None,
        verification_results: Optional[Dict[str, Any]] = None,
        environment_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Preserve crash state for analysis.
        
        Args:
            run_id: Run identifier
            iteration: Iteration number
            design_vector: Design vector at crash
            objective_value: Objective value at crash
            gradient: Gradient at crash
            solver_state: Solver state
            mesh_state: Mesh state
            verification_results: Verification results
            environment_snapshot: Environment snapshot
        
        Returns:
            Path to crash state file
        """
        # Get most recent failure record
        failure_record = self.failure_history[-1] if self.failure_history else None
        
        crash_state = CrashState(
            run_id=run_id,
            iteration=iteration,
            failure_record=failure_record,
            design_vector=design_vector,
            objective_value=objective_value,
            gradient=gradient,
            solver_state=solver_state,
            mesh_state=mesh_state,
            verification_results=verification_results,
            environment_snapshot=environment_snapshot,
        )
        
        # Save crash state
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"crash_{run_id}_{timestamp}.json"
        filepath = self.crash_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(crash_state.to_dict(), f, indent=2, default=str)
        
        return filepath
    
    def archive_diagnostics(
        self,
        run_id: str,
        diagnostics: Dict[str, Any],
    ) -> Path:
        """
        Archive diagnostic information.
        
        Args:
            run_id: Run identifier
            diagnostics: Diagnostic information
        
        Returns:
            Path to diagnostics file
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"diagnostics_{run_id}_{timestamp}.json"
        filepath = self.crash_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(diagnostics, f, indent=2, default=str)
        
        return filepath
    
    def get_failure_summary(self) -> Dict[str, Any]:
        """
        Get summary of all failures.
        
        Returns:
            Dictionary with failure summary
        """
        summary = {
            "total_failures": len(self.failure_history),
            "run_invalid": self.run_invalid,
            "failures_by_type": {},
            "failures_by_severity": {},
            "recent_failures": [],
        }
        
        # Count by type and severity
        for record in self.failure_history:
            ftype = record.failure_type.value
            severity = record.severity.value
            
            summary["failures_by_type"][ftype] = summary["failures_by_type"].get(ftype, 0) + 1
            summary["failures_by_severity"][severity] = summary["failures_by_severity"].get(severity, 0) + 1
        
        # Recent failures (last 10)
        summary["recent_failures"] = [r.to_dict() for r in self.failure_history[-10:]]
        
        return summary
    
    def save_failure_history(self, run_id: str) -> Path:
        """
        Save failure history to file.
        
        Args:
            run_id: Run identifier
        
        Returns:
            Path to failure history file
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"failure_history_{run_id}_{timestamp}.json"
        filepath = self.crash_dir / filename
        
        summary = self.get_failure_summary()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return filepath
    
    def reset(self):
        """Reset failure policy state."""
        self.failure_history.clear()
        self.run_invalid = False


def handle_critical_failure(
    policy: ScientificFailurePolicy,
    failure_type: FailureType,
    message: str,
    run_id: str,
    iteration: Optional[int] = None,
    component: Optional[str] = None,
    **kwargs
) -> bool:
    """
    Handle a critical failure event.
    
    Args:
        policy: Scientific failure policy
        failure_type: Type of failure
        message: Failure message
        run_id: Run identifier
        iteration: Iteration number
        component: Component that failed
        **kwargs: Additional arguments for crash state preservation
    
    Returns:
        True if should stop optimization
    """
    # Record failure
    policy.record_failure(
        failure_type=failure_type,
        severity=FailureSeverity.CRITICAL,
        message=message,
        iteration=iteration,
        component=component,
    )
    
    # Preserve crash state
    policy.preserve_crash_state(
        run_id=run_id,
        iteration=iteration,
        **kwargs
    )
    
    # Save failure history
    policy.save_failure_history(run_id)
    
    # Determine if should stop
    should_stop = policy.should_stop(FailureSeverity.CRITICAL)
    
    return should_stop
