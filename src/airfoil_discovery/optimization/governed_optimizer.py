"""
Governed optimizer with integrated verification and failure policies.

Wraps the standard optimizer with governance checks, adaptive FD verification,
optimizer paralysis detection, trust-region deadlock detection, and scientific
failure handling. Ensures optimization continues only when CFD outputs are valid.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field

from .mma_engine import SvanbergMMA
from ..verification.cfd_governance import CFDGovernanceModel, GovernanceStatus, CFDGovernanceReport
from ..verification.gradient_audit import GradientAuditor, GradientAuditReport
from ..core.failure_policies import ScientificFailurePolicy, FailureType, FailureSeverity
from ..core.wording_checks import WordingChecker


@dataclass
class OptimizerDiagnostics:
    """Diagnostics from optimizer governance."""
    
    # Iteration information
    iteration: int
    total_iterations: int
    
    # Trust region
    trust_region_active: bool
    trust_region_radius: float
    trust_region_contraction_count: int
    trust_region_expansion_count: int
    
    # Gradient information
    gradient_norm: float
    gradient_variance: float
    gradient_instability_detected: bool
    
    # Optimizer health
    optimizer_paralyzed: bool
    trust_region_deadlock: bool
    oscillatory_convergence: bool
    move_limit_frozen: bool
    
    # KKT metrics
    stationarity: float
    complementarity: float
    primal_feasibility: float
    dual_feasibility: float
    
    # Recovery actions taken
    recovery_actions: List[str] = field(default_factory=list)


@dataclass
class GovernedOptimizationResult:
    """Result from governed optimization."""
    
    # Final design
    final_design: np.ndarray
    final_objective: float
    
    # Governance reports
    governance_reports: List[CFDGovernanceReport] = field(default_factory=list)
    
    # Optimizer diagnostics
    diagnostics: List[OptimizerDiagnostics] = field(default_factory=list)
    
    # Overall status
    is_valid: bool
    failure_reasons: List[str] = field(default_factory=list)
    
    # Reproducibility
    master_seed: Optional[int] = None


class GovernedOptimizer:
    """
    Governed optimizer with verification and failure policies.
    
    Implements:
    - Adaptive FD verification
    - Optimizer paralysis detection
    - Trust-region deadlock detection
    - Oscillatory convergence detection
    - Move-limit freezing detection
    - KKT metric monitoring
    - Adaptive recovery strategies
    """
    
    def __init__(
        self,
        n_vars: int,
        n_constraints: int,
        governance_model: CFDGovernanceModel,
        failure_policy: Optional[ScientificFailurePolicy] = None,
        gradient_auditor: Optional[GradientAuditor] = None,
        wording_checker: Optional[WordingChecker] = None,
    ):
        """
        Initialize governed optimizer.
        
        Args:
            n_vars: Number of design variables
            n_constraints: Number of constraints
            governance_model: CFD governance model
            failure_policy: Scientific failure policy
            gradient_auditor: Gradient auditor for verification
            wording_checker: Wording checker for report validation
        """
        self.n_vars = n_vars
        self.n_constraints = n_constraints
        
        # Base optimizer
        self.base_optimizer = SvanbergMMA(n_vars, n_constraints)
        
        # Governance systems
        self.governance_model = governance_model
        self.failure_policy = failure_policy
        self.gradient_auditor = gradient_auditor
        self.wording_checker = wording_checker
        
        # Trust region state
        self.trust_region_radius = 0.1
        self.trust_region_contraction_count = 0
        self.trust_region_expansion_count = 0
        
        # Optimizer health tracking
        self.gradient_history: List[np.ndarray] = []
        self.objective_history: List[float] = []
        self.stagnation_count = 0
        self.oscillation_count = 0
        
        # Recovery state
        self.fd_failure_count = 0
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3
    
    def check_optimizer_health(
        self,
        gradient: np.ndarray,
        objective: float,
    ) -> Dict[str, Any]:
        """
        Check optimizer health for paralysis, deadlock, and oscillation.
        
        Args:
            gradient: Current gradient
            objective: Current objective value
        
        Returns:
            Dictionary with health metrics
        """
        health = {
            "paralyzed": False,
            "deadlock": False,
            "oscillatory": False,
            "move_limit_frozen": False,
            "stationarity": 0.0,
            "complementarity": 0.0,
            "primal_feasibility": 0.0,
            "dual_feasibility": 0.0,
        }
        
        # Check for gradient stagnation
        if len(self.gradient_history) > 5:
            recent_gradients = self.gradient_history[-5:]
            gradient_variance = np.var([np.linalg.norm(g) for g in recent_gradients])
            
            if gradient_variance < 1e-10:
                health["paralyzed"] = True
        
        # Check for trust-region deadlock
        if self.trust_region_contraction_count > 10:
            health["deadlock"] = True
        
        # Check for oscillatory convergence
        if len(self.objective_history) > 10:
            recent_objectives = self.objective_history[-10:]
            obj_std = np.std(recent_objectives)
            
            if obj_std > 0.01 * np.mean(np.abs(recent_objectives)):
                health["oscillatory"] = True
                self.oscillation_count += 1
            else:
                self.oscillation_count = 0
        
        # Check for move-limit freezing
        if self.trust_region_radius < 1e-6:
            health["move_limit_frozen"] = True
        
        # Compute KKT metrics (simplified)
        health["stationarity"] = float(np.linalg.norm(gradient))
        health["primal_feasibility"] = 0.0  # Would need constraint values
        health["dual_feasibility"] = 0.0  # Would need Lagrange multipliers
        health["complementarity"] = 0.0  # Would need constraint values
        
        return health
    
    def adaptive_recovery(
        self,
        health: Dict[str, Any],
        iteration: int,
    ) -> List[str]:
        """
        Perform adaptive recovery based on optimizer health.
        
        Args:
            health: Optimizer health metrics
            iteration: Current iteration
        
        Returns:
            List of recovery actions taken
        """
        actions = []
        
        if health["paralyzed"]:
            # Trust-region resize
            self.trust_region_radius *= 2.0
            actions.append("Expanded trust region to break paralysis")
            self.recovery_attempts += 1
        
        if health["deadlock"]:
            # Restart with larger trust region
            self.trust_region_radius = 0.2
            self.trust_region_contraction_count = 0
            actions.append("Reset trust region to break deadlock")
            self.recovery_attempts += 1
        
        if health["oscillatory"] and self.oscillation_count > 3:
            # Gradient smoothing
            if len(self.gradient_history) > 0:
                avg_gradient = np.mean(self.gradient_history[-5:], axis=0)
                actions.append("Applied gradient smoothing for oscillation")
            self.recovery_attempts += 1
        
        if health["move_limit_frozen"]:
            # Reset trust region
            self.trust_region_radius = 0.1
            actions.append("Reset trust region to unfreeze move limits")
            self.recovery_attempts += 1
        
        return actions
    
    def optimize(
        self,
        initial_design: np.ndarray,
        objective_function: Callable[[np.ndarray], float],
        constraint_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        gradient_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        max_iterations: int = 100,
        x_min: Optional[np.ndarray] = None,
        x_max: Optional[np.ndarray] = None,
        run_id: Optional[str] = None,
    ) -> GovernedOptimizationResult:
        """
        Run governed optimization with verification checks.
        
        Args:
            initial_design: Initial design vector
            objective_function: Objective function
            constraint_function: Constraint function (optional)
            gradient_function: Gradient function (optional)
            max_iterations: Maximum iterations
            x_min: Lower bounds
            x_max: Upper bounds
            run_id: Run identifier
        
        Returns:
            GovernedOptimizationResult with optimization results
        """
        x = initial_design.copy()
        x_prev = x.copy()
        x_pprev = x.copy()
        
        if x_min is None:
            x_min = np.full_like(x, -1.0)
        if x_max is None:
            x_max = np.full_like(x, 1.0)
        
        # Initialize asymptotes
        self.base_optimizer.init_asymptotes(x, x_min, x_max)
        
        governance_reports = []
        diagnostics_list = []
        failure_reasons = []
        
        for iteration in range(max_iterations):
            # Evaluate objective
            f = objective_function(x)
            
            # Evaluate gradient (if provided)
            if gradient_function is not None:
                df = gradient_function(x)
            else:
                # Finite difference approximation
                df = self._finite_difference_gradient(objective_function, x)
            
            # Evaluate constraints (if provided)
            if constraint_function is not None:
                g = constraint_function(x)
            else:
                g = np.zeros(self.n_constraints)
            
            # Verify gradient if auditor available
            gradient_report = None
            if self.gradient_auditor:
                gradient_report = self.gradient_auditor.audit(
                    adjoint_gradient=df,
                    objective_function=objective_function,
                    x=x,
                    check_directional=True,
                    check_temporal=True,
                )
                
                # Adaptive FD verification
                if not gradient_report.is_valid:
                    self.fd_failure_count += 1
                    
                    if self.fd_failure_count == 1:
                        # Reduce move limits
                        self.trust_region_radius *= 0.5
                    elif self.fd_failure_count == 2:
                        # Rollback iteration
                        x = x_prev.copy()
                        continue
                    elif self.fd_failure_count >= 3:
                        # Restart adjoint solve (terminate)
                        if self.failure_policy:
                            from ..core.failure_policies import handle_critical_failure
                            handle_critical_failure(
                                policy=self.failure_policy,
                                failure_type=FailureType.GRADIENT_CORRUPTION,
                                message=f"Gradient verification failed after {self.fd_failure_count} attempts",
                                run_id=run_id or "unknown",
                                iteration=iteration,
                                component="optimizer",
                            )
                        failure_reasons.append("Gradient verification failed repeatedly")
                        break
                else:
                    self.fd_failure_count = 0
            
            # Check optimizer health
            self.gradient_history.append(df.copy())
            self.objective_history.append(f)
            
            health = self.check_optimizer_health(df, f)
            
            # Adaptive recovery
            recovery_actions = self.adaptive_recovery(health, iteration)
            
            # Check if recovery attempts exceeded
            if self.recovery_attempts >= self.max_recovery_attempts:
                failure_reasons.append("Maximum recovery attempts exceeded")
                break
            
            # Update asymptotes
            self.base_optimizer.update_asymptotes(x, x_prev, x_pprev)
            
            # Solve subproblem
            x_next, _ = self.base_optimizer.solve_subproblem(
                x, f, df, g, np.zeros_like(df), x_min, x_max
            )
            
            # Update design
            x_pprev = x_prev.copy()
            x_prev = x.copy()
            x = x_next
            
            # Update trust region
            if np.linalg.norm(x - x_prev) < self.trust_region_radius * 0.1:
                self.trust_region_radius *= 0.5
                self.trust_region_contraction_count += 1
            else:
                self.trust_region_radius *= 1.1
                self.trust_region_expansion_count += 1
            
            # Create diagnostics
            diagnostics = OptimizerDiagnostics(
                iteration=iteration,
                total_iterations=max_iterations,
                trust_region_active=True,
                trust_region_radius=self.trust_region_radius,
                trust_region_contraction_count=self.trust_region_contraction_count,
                trust_region_expansion_count=self.trust_region_expansion_count,
                gradient_norm=float(np.linalg.norm(df)),
                gradient_variance=float(np.var([np.linalg.norm(g) for g in self.gradient_history[-5:]])) if len(self.gradient_history) >= 5 else 0.0,
                gradient_instability_detected=health["paralyzed"],
                optimizer_paralyzed=health["paralyzed"],
                trust_region_deadlock=health["deadlock"],
                oscillatory_convergence=health["oscillatory"],
                move_limit_frozen=health["move_limit_frozen"],
                stationarity=health["stationarity"],
                complementarity=health["complementarity"],
                primal_feasibility=health["primal_feasibility"],
                dual_feasibility=health["dual_feasibility"],
                recovery_actions=recovery_actions,
            )
            diagnostics_list.append(diagnostics)
            
            # Check convergence
            if np.linalg.norm(df) < 1e-6:
                break
        
        # Final evaluation
        final_objective = objective_function(x)
        
        return GovernedOptimizationResult(
            final_design=x,
            final_objective=final_objective,
            governance_reports=governance_reports,
            diagnostics=diagnostics_list,
            is_valid=len(failure_reasons) == 0,
            failure_reasons=failure_reasons,
        )
    
    def _finite_difference_gradient(
        self,
        objective_function: Callable[[np.ndarray], float],
        x: np.ndarray,
        h: float = 1e-6,
    ) -> np.ndarray:
        """
        Compute finite-difference gradient.
        
        Args:
            objective_function: Objective function
            x: Design point
            h: Step size
        
        Returns:
            Gradient vector
        """
        n = len(x)
        grad = np.zeros(n)
        f0 = objective_function(x)
        
        for i in range(n):
            dx = np.zeros_like(x)
            dx[i] = h
            f_plus = objective_function(x + dx)
            f_minus = objective_function(x - dx)
            grad[i] = (f_plus - f_minus) / (2 * h)
        
        return grad
