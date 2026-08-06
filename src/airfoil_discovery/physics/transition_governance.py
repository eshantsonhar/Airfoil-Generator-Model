"""
Transition model governance for γ-Reθ transition model.

Monitors intermittency transport stability, transition Reynolds evolution,
separated-flow transition sensitivity, and detects model limitations.
Provides warnings for transition uncertainty and false reattachment risk.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum


class TransitionWarning(Enum):
    """Transition model warning types."""
    NONE = "NONE"
    TRANSITION_UNCERTAIN = "TRANSITION_UNCERTAIN"
    FALSE_REATTACHMENT_RISK = "FALSE_REATTACHMENT_RISK"
    INTERMITTENCY_BREAKDOWN = "INTERMITTENCY_BREAKDOWN"
    TRANSITION_OSCILLATION = "TRANSITION_OSCILLATION"
    SEPARATED_FLOW_SENSITIVITY = "SEPARATED_FLOW_SENSITIVITY"
    REYNOLDS_OUT_OF_RANGE = "REYNOLDS_OUT_OF_RANGE"
    MODEL_LIMITATION = "MODEL_LIMITATION"


@dataclass
class TransitionDiagnostics:
    """Diagnostics from transition model analysis."""
    
    # Intermittency statistics
    mean_intermittency: float
    max_intermittency: float
    min_intermittency: float
    intermittency_std: float
    
    # Intermittency gradient
    max_intermittency_gradient: float
    intermittency_gradient_location: Optional[float]
    
    # Transition locations
    transition_onset: Optional[float]
    transition_completion: Optional[float]
    transition_length: Optional[float]
    
    # Intermittency transport stability
    transport_stable: bool
    transport_oscillation_detected: bool
    transport_oscillation_amplitude: float
    
    # Separated flow transition
    separated_flow_transition: bool
    separation_induced_transition: bool
    
    # Reynolds number checks
    reynolds_number: float
    reynolds_in_valid_range: bool
    
    # Model limitations
    gamma_re_theta_limit_exceeded: bool
    correlation_valid: bool
    
    # Overall assessment
    model_confidence: float  # 0-1 scale
    
    # Warnings
    warnings: List[TransitionWarning] = field(default_factory=list)


@dataclass
class TransitionGovernanceReport:
    """Comprehensive transition governance report."""
    
    # Diagnostics
    diagnostics: TransitionDiagnostics
    
    # Overall status
    is_valid: bool
    can_trust_transition: bool
    
    # Recommendations
    recommended_actions: List[str] = field(default_factory=list)
    
    # Mitigation strategies
    mitigation_strategies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "is_valid": self.is_valid,
            "can_trust_transition": self.can_trust_transition,
            "model_confidence": self.diagnostics.model_confidence,
            "warnings": [w.value for w in self.diagnostics.warnings],
            "recommended_actions": self.recommended_actions,
            "mitigation_strategies": self.mitigation_strategies,
        }


class TransitionModelGovernor:
    """
    Governs γ-Reθ transition model usage and monitors model reliability.
    
    The γ-Reθ model is not infallible. This system monitors:
    - Intermittency transport stability
    - Transition Reynolds evolution
    - Separated-flow transition sensitivity
    - Model correlation validity
    """
    
    def __init__(
        self,
        min_reynolds: float = 1e4,
        max_reynolds: float = 1e6,
        min_intermittency: float = 0.0,
        max_intermittency: float = 1.0,
        transport_stability_threshold: float = 0.1,
    ):
        """
        Initialize transition model governor.
        
        Args:
            min_reynolds: Minimum valid Reynolds number for model
            max_reynolds: Maximum valid Reynolds number for model
            min_intermittency: Expected minimum intermittency
            max_intermittency: Expected maximum intermittency
            transport_stability_threshold: Threshold for transport stability
        """
        self.min_reynolds = min_reynolds
        self.max_reynolds = max_reynolds
        self.min_intermittency = min_intermittency
        self.max_intermittency = max_intermittency
        self.transport_stability_threshold = transport_stability_threshold
    
    def check_reynolds_range(self, reynolds: float) -> Tuple[bool, List[TransitionWarning]]:
        """
        Check if Reynolds number is within valid model range.
        
        Args:
            reynolds: Reynolds number
        
        Returns:
            (in_range, warnings)
        """
        warnings = []
        
        if reynolds < self.min_reynolds:
            warnings.append(TransitionWarning.REYNOLDS_OUT_OF_RANGE)
            return False, warnings
        
        if reynolds > self.max_reynolds:
            warnings.append(TransitionWarning.REYNOLDS_OUT_OF_RANGE)
            return False, warnings
        
        return True, warnings
    
    def analyze_intermittency(
        self,
        x: np.ndarray,
        intermittency: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Analyze intermittency distribution for stability and consistency.
        
        Args:
            x: Surface coordinates (x/c)
            intermittency: Intermittency distribution
        
        Returns:
            Dictionary with intermittency analysis results
        """
        if len(intermittency) == 0:
            return {
                "mean": 0.0,
                "max": 0.0,
                "min": 0.0,
                "std": 0.0,
                "max_gradient": 0.0,
                "gradient_location": None,
                "transport_stable": True,
                "oscillation_detected": False,
                "oscillation_amplitude": 0.0,
            }
        
        # Statistics
        mean_gamma = float(np.mean(intermittency))
        max_gamma = float(np.max(intermittency))
        min_gamma = float(np.min(intermittency))
        std_gamma = float(np.std(intermittency))
        
        # Gradient analysis
        dgamma_dx = np.gradient(intermittency, x)
        max_gradient = float(np.max(np.abs(dgamma_dx)))
        max_gradient_idx = np.argmax(np.abs(dgamma_dx))
        gradient_location = float(x[max_gradient_idx]) if len(x) > 0 else None
        
        # Transport stability (check for oscillations)
        # Look for sign changes in gradient
        gradient_sign_changes = 0
        for i in range(1, len(dgamma_dx)):
            if dgamma_dx[i] * dgamma_dx[i-1] < 0:
                gradient_sign_changes += 1
        
        transport_stable = gradient_sign_changes < 3
        oscillation_detected = gradient_sign_changes >= 3
        
        # Oscillation amplitude
        if oscillation_detected:
            oscillation_amplitude = float(np.std(dgamma_dx))
        else:
            oscillation_amplitude = 0.0
        
        return {
            "mean": mean_gamma,
            "max": max_gamma,
            "min": min_gamma,
            "std": std_gamma,
            "max_gradient": max_gradient,
            "gradient_location": gradient_location,
            "transport_stable": transport_stable,
            "oscillation_detected": oscillation_detected,
            "oscillation_amplitude": oscillation_amplitude,
        }
    
    def detect_transition_locations(
        self,
        x: np.ndarray,
        intermittency: np.ndarray,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Detect transition onset and completion from intermittency.
        
        Args:
            x: Surface coordinates (x/c)
            intermittency: Intermittency distribution
        
        Returns:
            (onset, completion, length)
        """
        if len(intermittency) == 0:
            return None, None, None
        
        # Transition onset: intermittency > 0.1
        onset = None
        for i in range(len(intermittency)):
            if intermittency[i] > 0.1:
                onset = float(x[i])
                break
        
        # Transition completion: intermittency > 0.9
        completion = None
        for i in range(len(intermittency)):
            if intermittency[i] > 0.9:
                completion = float(x[i])
                break
        
        # Transition length
        length = None
        if onset is not None and completion is not None:
            length = completion - onset
        
        return onset, completion, length
    
    def check_separated_flow_transition(
        self,
        x: np.ndarray,
        intermittency: np.ndarray,
        cf: Optional[np.ndarray] = None,
    ) -> Tuple[bool, bool]:
        """
        Check for separated-flow transition.
        
        Args:
            x: Surface coordinates (x/c)
            intermittency: Intermittency distribution
            cf: Skin friction distribution (optional)
        
        Returns:
            (separated_flow_transition, separation_induced)
        """
        separated_flow = False
        separation_induced = False
        
        if cf is not None:
            # Check if transition occurs after separation
            separation_idx = None
            for i in range(1, len(cf)):
                if cf[i] < 0 and cf[i-1] >= 0:
                    separation_idx = i
                    break
            
            if separation_idx is not None:
                # Check if intermittency rises after separation
                for i in range(separation_idx, len(intermittency)):
                    if intermittency[i] > 0.5:
                        separated_flow = True
                        separation_induced = True
                        break
        
        return separated_flow, separation_induced
    
    def check_correlation_validity(
        self,
        reynolds: float,
        transition_onset: Optional[float],
    ) -> bool:
        """
        Check if γ-Reθ correlation is valid for this case.
        
        Args:
            reynolds: Reynolds number
            transition_onset: Transition onset location (x/c)
        
        Returns:
            True if correlation is valid
        """
        if transition_onset is None:
            return True
        
        # γ-Reθ correlations have limitations
        # Very early or very late transition may be outside correlation range
        if transition_onset < 0.05:
            return False  # Too early for correlation
        
        if transition_onset > 0.7:
            return False  # Too late for correlation
        
        # Very low Reynolds may be outside correlation range
        if reynolds < 5e4:
            return False
        
        return True
    
    def govern(
        self,
        x: np.ndarray,
        intermittency: np.ndarray,
        reynolds: float,
        cf: Optional[np.ndarray] = None,
    ) -> TransitionGovernanceReport:
        """
        Perform comprehensive transition model governance.
        
        Args:
            x: Surface coordinates (x/c)
            intermittency: Intermittency distribution
            reynolds: Reynolds number
            cf: Skin friction distribution (optional)
        
        Returns:
            TransitionGovernanceReport with governance assessment
        """
        warnings = []
        recommended_actions = []
        mitigation_strategies = []
        
        # Check Reynolds range
        reynolds_valid, reynolds_warnings = self.check_reynolds_range(reynolds)
        warnings.extend(reynolds_warnings)
        
        if not reynolds_valid:
            recommended_actions.append(
                f"Reynolds number {reynolds:.0f} is outside valid model range "
                f"[{self.min_reynolds:.0f}, {self.max_reynolds:.0f}]"
            )
            mitigation_strategies.append("Consider using different transition model or empirical correction")
        
        # Analyze intermittency
        gamma_analysis = self.analyze_intermittency(x, intermittency)
        
        # Check intermittency bounds
        if gamma_analysis["max"] > self.max_intermittency + 0.1:
            warnings.append(TransitionWarning.INTERMITTENCY_BREAKDOWN)
            recommended_actions.append("Intermittency exceeds physical bounds")
            mitigation_strategies.append("Check solver stability and transition model parameters")
        
        if gamma_analysis["min"] < self.min_intermittency - 0.1:
            warnings.append(TransitionWarning.INTERMITTENCY_BREAKDOWN)
            recommended_actions.append("Intermittency below physical bounds")
        
        # Check transport stability
        if not gamma_analysis["transport_stable"]:
            warnings.append(TransitionWarning.TRANSITION_OSCILLATION)
            recommended_actions.append("Intermittency transport oscillation detected")
            mitigation_strategies.append("Reduce CFL or check numerical dissipation")
        
        if gamma_analysis["oscillation_amplitude"] > self.transport_stability_threshold:
            warnings.append(TransitionWarning.TRANSITION_OSCILLATION)
            recommended_actions.append(
                f"High intermittency oscillation amplitude: {gamma_analysis['oscillation_amplitude']:.3f}"
            )
        
        # Detect transition locations
        onset, completion, length = self.detect_transition_locations(x, intermittency)
        
        # Check separated flow transition
        separated_flow, separation_induced = self.check_separated_flow_transition(
            x, intermittency, cf
        )
        
        if separated_flow:
            warnings.append(TransitionWarning.SEPARATED_FLOW_SENSITIVITY)
            recommended_actions.append("Separated-flow transition detected")
            mitigation_strategies.append("Model may have reduced accuracy in separated regions")
        
        # Check correlation validity
        correlation_valid = self.check_correlation_validity(reynolds, onset)
        
        if not correlation_valid:
            warnings.append(TransitionWarning.MODEL_LIMITATION)
            recommended_actions.append("γ-Reθ correlation may be invalid for this case")
            mitigation_strategies.append("Consider alternative transition modeling approach")
        
        # Check for false reattachment risk
        # This occurs when intermittency rises rapidly without proper pressure recovery
        if onset is not None and completion is not None:
            if length < 0.05 and separated_flow:
                warnings.append(TransitionWarning.FALSE_REATTACHMENT_RISK)
                recommended_actions.append("Rapid transition in separated flow may indicate false reattachment")
                mitigation_strategies.append("Verify with skin friction and pressure recovery")
        
        # Compute model confidence
        confidence = 1.0
        
        if not reynolds_valid:
            confidence -= 0.3
        
        if not gamma_analysis["transport_stable"]:
            confidence -= 0.2
        
        if not correlation_valid:
            confidence -= 0.2
        
        if separated_flow:
            confidence -= 0.1
        
        if TransitionWarning.FALSE_REATTACHMENT_RISK in warnings:
            confidence -= 0.2
        
        confidence = max(0.0, min(1.0, confidence))
        
        # Build diagnostics
        diagnostics = TransitionDiagnostics(
            mean_intermittency=gamma_analysis["mean"],
            max_intermittency=gamma_analysis["max"],
            min_intermittency=gamma_analysis["min"],
            intermittency_std=gamma_analysis["std"],
            max_intermittency_gradient=gamma_analysis["max_gradient"],
            intermittency_gradient_location=gamma_analysis["gradient_location"],
            transition_onset=onset,
            transition_completion=completion,
            transition_length=length,
            transport_stable=gamma_analysis["transport_stable"],
            transport_oscillation_detected=gamma_analysis["oscillation_detected"],
            transport_oscillation_amplitude=gamma_analysis["oscillation_amplitude"],
            separated_flow_transition=separated_flow,
            separation_induced_transition=separation_induced,
            reynolds_number=reynolds,
            reynolds_in_valid_range=reynolds_valid,
            gamma_re_theta_limit_exceeded=False,  # To be implemented with Reθ data
            correlation_valid=correlation_valid,
            warnings=warnings,
            model_confidence=confidence,
        )
        
        # Determine overall validity
        is_valid = (
            reynolds_valid and
            gamma_analysis["transport_stable"] and
            confidence > 0.5
        )
        
        can_trust_transition = is_valid and confidence > 0.7
        
        return TransitionGovernanceReport(
            diagnostics=diagnostics,
            is_valid=is_valid,
            can_trust_transition=can_trust_transition,
            recommended_actions=recommended_actions,
            mitigation_strategies=mitigation_strategies,
        )
