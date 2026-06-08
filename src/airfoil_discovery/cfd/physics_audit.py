"""
Physics Fidelity Audit for Transitional RANS Solutions.

Verifies that SU2 CFD solutions are physically credible by detecting:
- False reattachment (numerical-only recovery)
- Artificial stabilization (excessive dissipation suppressing physics)
- Dissipation-induced bubble suppression
- Nonphysical Cp smoothing
- Unphysical wall shear recovery
- Transition wandering (metastable behavior)
- Intermittency collapse
- Separation suppression due to numerical diffusion
- CFL instability artifacts
- Limiter-induced flow modification

Every CFD result receives a PHYSICS CREDIBILITY SCORE before acceptance.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum


class PhysicsCredibility(Enum):
    """Physics credibility classification."""
    CREDIBLE = "CREDIBLE"
    QUESTIONABLE = "QUESTIONABLE"  
    INCREDIBLE = "INCREDIBLE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class TransitionCredibility(Enum):
    """Transition model credibility."""
    PHYSICAL = "PHYSICAL"
    NUMERICAL_ARTIFACT = "NUMERICAL_ARTIFACT"
    DISSIPATION_DOMINATED = "DISSIPATION_DOMINATED"
    UNVERIFIABLE = "UNVERIFIABLE"


@dataclass
class PhysicsAuditReport:
    """Comprehensive physics credibility audit."""
    
    # Overall credibility
    overall_credibility: PhysicsCredibility
    transition_credibility: TransitionCredibility
    
    # Component scores (0-1, higher = more credible)
    separation_confidence: float
    reattachment_confidence: float
    transition_confidence: float
    force_confidence: float
    dissipation_contamination: float  # Higher = more contaminated
    
    # Detected pathologies
    pathologies_detected: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Verification flags
    false_reattachment_detected: bool = False
    dissipation_suppression_detected: bool = False
    transition_wandering_detected: bool = False
    intermittency_collapse_detected: bool = False
    cfl_instability_detected: bool = False
    limiter_induced_distortion: bool = False
    
    # Quantitative metrics
    cp_plateau_strength: Optional[float] = None
    apg_severity_index: Optional[float] = None
    wall_shear_recovery_rate: Optional[float] = None
    transition_gradient_sharpness: Optional[float] = None
    
    def is_acceptable(self) -> bool:
        """Check if solution is acceptable for optimization."""
        return (self.overall_credibility in [PhysicsCredibility.CREDIBLE, PhysicsCredibility.QUESTIONABLE]
                and not self.pathologies_detected)


class PhysicsAuditor:
    """
    Physics fidelity auditor for transitional RANS solutions.
    
    Detects numerical pathologies that compromise physical credibility:
    - False reattachment: Cf recovers due to numerical mixing, not physics
    - Dissipation suppression: Excessive numerical diffusion eliminates LSB
    - Transition wandering: Intermittency oscillates without convergence
    - Limiter distortion: Slope limiters modify separation behavior
    """
    
    def __init__(self, 
                 cp_plateau_threshold: float = 0.3,
                 cf_recovery_threshold: float = 0.01,
                 intermittency_stability_threshold: float = 0.1):
        self.cp_plateau_threshold = cp_plateau_threshold
        self.cf_recovery_threshold = cf_recovery_threshold
        self.intermittency_stability_threshold = intermittency_stability_threshold
    
    def audit(self,
              residual_history: Optional[List[float]] = None,
              cl_history: Optional[List[float]] = None,
              cd_history: Optional[List[float]] = None,
              cp_surface: Optional[np.ndarray] = None,
              cf_surface: Optional[np.ndarray] = None,
              intermittency_surface: Optional[np.ndarray] = None,
              x_surface: Optional[np.ndarray] = None,
              convergence_report: Optional[Dict[str, Any]] = None,
              lsb_report: Optional[Dict[str, Any]] = None,
              ) -> PhysicsAuditReport:
        """
        Perform complete physics fidelity audit.
        
        Args:
            residual_history: RMS residual history
            cl_history: Lift coefficient history
            cd_history: Drag coefficient history
            cp_surface: Pressure coefficient along surface (N,)
            cf_surface: Skin friction along surface (N,)  
            intermittency_surface: Intermittency along surface (N,)
            x_surface: Surface x/c coordinates (N,)
            convergence_report: From convergence analysis
            lsb_report: From LSB detection
            
        Returns:
            PhysicsAuditReport with credibility assessment
        """
        pathologies = []
        warnings = []
        
        # 1. Check dissipation-induced bubble suppression
        diss_suppression = self._check_dissipation_suppression(
            residual_history, cp_surface, x_surface, lsb_report
        )
        if diss_suppression:
            pathologies.append("Dissipation-induced LSB suppression detected")
            self.dissipation_suppression_detected = True
        
        # 2. Check for false reattachment
        false_reattach = self._check_false_reattachment(
            cf_surface, cp_surface, x_surface, lsb_report
        )
        if false_reattach:
            pathologies.append("False reattachment detected - Cf recovery without pressure recovery")
            self.false_reattachment_detected = True
        
        # 3. Check for transition wandering (non-converged transition)
        transition_wandering = self._check_transition_wandering(
            intermittency_surface, cl_history, cd_history
        )
        if transition_wandering:
            pathologies.append("Transition wandering detected - oscillatory transition behavior")
            self.transition_wandering_detected = True
        
        # 4. Check for intermittency collapse
        intermit_collapse = self._check_intermittency_collapse(
            intermittency_surface
        )
        if intermit_collapse:
            pathologies.append("Intermittency collapse detected - γ-equation not solving")
            self.intermittency_collapse_detected = True
        
        # 5. Check for CFL instability artifacts
        cfl_issue = self._check_cfl_instability(residual_history)
        if cfl_issue:
            pathologies.append("CFL instability artifacts in residual history")
            self.cfl_instability_detected = True
        
        # 6. Check for limiter-induced distortion
        limiter_issue = self._check_limiter_distortion(
            cp_surface, x_surface
        )
        if limiter_issue:
            pathologies.append("Limiter-induced Cp distortion detected")
            self.limiter_induced_distortion = True
        
        # 7. Compute credibility scores
        sep_conf = self._compute_separation_confidence(
            cp_surface, x_surface, lsb_report
        )
        reattach_conf = self._compute_reattachment_confidence(
            cf_surface, cp_surface, x_surface, lsb_report
        )
        trans_conf = self._compute_transition_confidence(
            intermittency_surface, lsb_report
        )
        force_conf = self._compute_force_confidence(
            cl_history, cd_history, convergence_report
        )
        diss_contamination = self._compute_dissipation_contamination(
            cp_surface, x_surface, residual_history
        )
        
        # 8. Overall credibility
        if pathologies:
            overall = PhysicsCredibility.INCREDIBLE
        elif diss_contamination > 0.3:
            overall = PhysicsCredibility.QUESTIONABLE
        else:
            overall = PhysicsCredibility.CREDIBLE
        
        # Transition credibility
        if self.intermittency_collapse_detected:
            trans_cred = TransitionCredibility.NUMERICAL_ARTIFACT
        elif diss_contamination > 0.5:
            trans_cred = TransitionCredibility.DISSIPATION_DOMINATED
        elif transition_wandering:
            trans_cred = TransitionCredibility.UNVERIFIABLE
        else:
            trans_cred = TransitionCredibility.PHYSICAL
        
        # Compute metrics
        cp_plateau = None
        if cp_surface is not None and x_surface is not None and len(cp_surface) > 10:
            dcp = np.gradient(cp_surface, x_surface)
            plateau_mask = np.abs(dcp) < self.cp_plateau_threshold
            if np.any(plateau_mask):
                cp_plateau = float(np.mean(np.abs(dcp[plateau_mask])))
        
        return PhysicsAuditReport(
            overall_credibility=overall,
            transition_credibility=trans_cred,
            separation_confidence=sep_conf,
            reattachment_confidence=reattach_conf,
            transition_confidence=trans_conf,
            force_confidence=force_conf,
            dissipation_contamination=diss_contamination,
            pathologies_detected=pathologies,
            warnings=warnings,
            false_reattachment_detected=self.false_reattachment_detected,
            dissipation_suppression_detected=self.dissipation_suppression_detected,
            transition_wandering_detected=self.transition_wandering_detected,
            intermittency_collapse_detected=self.intermittency_collapse_detected,
            cfl_instability_detected=self.cfl_instability_detected,
            limiter_induced_distortion=self.limiter_induced_distortion,
            cp_plateau_strength=cp_plateau,
        )
    
    def _check_dissipation_suppression(self,
                                       residual_history: Optional[List[float]],
                                       cp_surface: Optional[np.ndarray],
                                       x_surface: Optional[np.ndarray],
                                       lsb_report: Optional[Dict[str, Any]]) -> bool:
        """
        Detect if numerical dissipation is suppressing LSB physics.
        
        Indicators:
        - Very smooth Cp (no plateau) despite separation indicators
        - Residual converges too quickly (< 10 iterations to 1e-6)
        - No LSB detected but conditions favor separation
        """
        if residual_history and len(residual_history) > 10:
            residuals = np.array(residual_history)
            # Very rapid convergence suggests dissipation dominance
            if np.min(residuals[-10:]) < 1e-8 and len(residual_history) < 30:
                return True
        
        if cp_surface is not None and x_surface is not None:
            dcp = np.gradient(cp_surface, x_surface)
            # Suspiciously smooth Cp
            if np.std(dcp) < 0.1:
                # Check if plateau expected but suppressed
                if lsb_report and lsb_report.get("lsb_detected") == False:
                    cp_range = np.max(cp_surface) - np.min(cp_surface)
                    if cp_range > 0.5:
                        return True
        
        return False
    
    def _check_false_reattachment(self,
                                  cf_surface: Optional[np.ndarray],
                                  cp_surface: Optional[np.ndarray],
                                  x_surface: Optional[np.ndarray],
                                  lsb_report: Optional[Dict[str, Any]]) -> bool:
        """
        Detect false reattachment: Cf recovers but pressure doesn't.
        
        Physical reattachment: Cf > 0 AND strong pressure recovery (dCp/dx << 0)
        False reattachment: Cf > 0 but Cp is flat or still adverse
        """
        if cf_surface is None or cp_surface is None or x_surface is None:
            return False
        
        if len(cf_surface) < 10 or len(cp_surface) < 10:
            return False
        
        dcp = np.gradient(cp_surface, x_surface)
        
        # Find separation and reattachment points from Cf
        sep_idx = None
        reattach_idx = None
        for i in range(1, len(cf_surface)):
            if cf_surface[i-1] >= 0 and cf_surface[i] < 0:
                sep_idx = i
            if cf_surface[i-1] <= 0 and cf_surface[i] > 0:
                reattach_idx = i
        
        if sep_idx is not None and reattach_idx is not None and reattach_idx > sep_idx:
            # Check pressure recovery at reattachment
            pressure_recovery = dcp[reattach_idx]
            if pressure_recovery > -0.5:  # Weak or no pressure recovery
                return True
        
        return False
    
    def _check_transition_wandering(self,
                                    intermittency: Optional[np.ndarray],
                                    cl_history: Optional[List[float]],
                                    cd_history: Optional[List[float]]) -> bool:
        """Detect oscillatory transition behavior indicating non-convergence."""
        if cl_history and len(cl_history) > 30:
            cl = np.array(cl_history)
            recent_cl = cl[-30:]
            cl_std = np.std(recent_cl) / max(abs(np.mean(recent_cl)), 1e-10)
            if cl_std > 0.05:  # >5% oscillation in final iterations
                return True
        
        return False
    
    def _check_intermittency_collapse(self, intermittency: Optional[np.ndarray]) -> bool:
        """Detect if intermittency equation is not solving (all zeros or all ones)."""
        if intermittency is None or len(intermittency) < 5:
            return False
        
        intermit = np.array(intermittency)
        intermit_range = np.max(intermit) - np.min(intermit)
        
        # Collapse: essentially constant (all 0 or all 1)
        if intermit_range < 0.05:
            return True
        
        # No intermittency production
        if np.all(intermit < 0.05):
            return True
        
        if np.all(intermit > 0.95):
            return True
        
        return False
    
    def _check_cfl_instability(self, residual_history: Optional[List[float]]) -> bool:
        """Detect CFL-induced oscillations in residual history."""
        if residual_history is None or len(residual_history) < 20:
            return False
        
        residuals = np.array(residual_history)
        recent = residuals[-20:]
        
        # Count sign changes in residual difference
        diffs = np.diff(np.log10(np.abs(recent) + 1e-15))
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        
        if sign_changes > 8:  # Excessive oscillation
            return True
        
        # Check for CFL spikes (sudden jumps of >2 orders)
        if len(residuals) > 5:
            log_res = np.log10(np.abs(residuals) + 1e-15)
            spikes = np.sum(np.abs(np.diff(log_res)) > 2.0)
            if spikes > 3:
                return True
        
        return False
    
    def _check_limiter_distortion(self,
                                  cp_surface: Optional[np.ndarray],
                                  x_surface: Optional[np.ndarray]) -> bool:
        """Detect limiter-induced distortion of pressure distribution."""
        if cp_surface is None or x_surface is None or len(cp_surface) < 10:
            return False
        
        dcp = np.gradient(cp_surface, x_surface)
        d2cp = np.gradient(dcp, x_surface)
        
        # Limiter artifacts: sharp transitions in d2Cp
        if len(d2cp) > 5:
            d2cp_std = np.std(d2cp)
            if d2cp_std > 10.0 * np.median(np.abs(d2cp) + 1e-10):
                return True
        
        return False
    
    def _compute_separation_confidence(self, cp, x, lsb_report) -> float:
        """Compute confidence in separation detection (0-1)."""
        if lsb_report and lsb_report.get("separation_location") is not None:
            return 0.9
        if cp is not None and x is not None and len(cp) > 10:
            dcp = np.gradient(cp, x)
            if np.any(dcp > 0):  # Adverse pressure gradient present
                return 0.6
        return 0.3
    
    def _compute_reattachment_confidence(self, cf, cp, x, lsb_report) -> float:
        """Compute confidence in reattachment detection (0-1)."""
        if lsb_report and lsb_report.get("reattachment_location") is not None:
            if cp is not None and x is not None:
                dcp = np.gradient(cp, x)
                reattach_x = lsb_report["reattachment_location"]
                idx = np.argmin(np.abs(x - reattach_x))
                if idx < len(dcp) and dcp[idx] < -1.0:
                    return 0.95
            return 0.6
        return 0.3
    
    def _compute_transition_confidence(self, intermittency, lsb_report) -> float:
        """Compute confidence in transition detection (0-1)."""
        if intermittency is not None and len(intermittency) > 5:
            if np.any((intermittency > 0.1) & (intermittency < 0.9)):
                return 0.9  # Transition zone well-resolved
            return 0.5
        if lsb_report and lsb_report.get("transition_onset") is not None:
            return 0.4
        return 0.2
    
    def _compute_force_confidence(self, cl_hist, cd_hist, convergence) -> float:
        """Compute confidence in force coefficients (0-1)."""
        if convergence and convergence.get("is_valid"):
            if cl_hist and cd_hist and len(cl_hist) > 50:
                cl = np.array(cl_hist)
                cd = np.array(cd_hist)
                cl_osc = np.std(cl[-30:]) / max(abs(np.mean(cl[-30:])), 1e-10)
                cd_osc = np.std(cd[-30:]) / max(abs(np.mean(cd[-30:])), 1e-10)
                if cl_osc < 0.01 and cd_osc < 0.01:
                    return 0.95
            return 0.7
        return 0.3
    
    def _compute_dissipation_contamination(self, cp, x, residual_history) -> float:
        """
        Estimate numerical dissipation contamination level.
        
        Uses Cp smoothness and residual convergence rate as proxies.
        Returns value 0-1 (higher = more contamination).
        """
        contamination = 0.0
        
        if cp is not None and x is not None and len(cp) > 10:
            dcp = np.gradient(cp, x)
            d2cp = np.gradient(dcp, x)
            
            # Excessively smooth Cp suggests dissipation
            cp_roughness = np.std(d2cp) / max(np.std(cp), 1e-10)
            if cp_roughness < 0.01:
                contamination += 0.3
        
        if residual_history and len(residual_history) > 5:
            residuals = np.array(residual_history)
            conv_rate = (np.log10(np.abs(residuals[-1]) + 1e-15) - 
                        np.log10(np.abs(residuals[0]) + 1e-15)) / len(residuals)
            # Very fast convergence suggests dissipation dominance
            if conv_rate < -0.5:
                contamination += 0.3
        
        return min(contamination, 1.0)