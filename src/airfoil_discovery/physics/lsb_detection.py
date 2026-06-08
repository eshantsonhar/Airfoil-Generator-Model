"""
Laminar Separation Bubble (LSB) detection and classification.

Implements formal LSB detection, classification, and tracking for
low-Reynolds-number transitional flows. Distinguishes between short
and long bubbles, tracks evolution, and detects bursting risk.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum


class LSBType(Enum):
    """LSB classification types."""
    NO_BUBBLE = "NO_BUBBLE"
    SHORT_BUBBLE = "SHORT_BUBBLE"
    LONG_BUBBLE = "LONG_BUBBLE"
    BURST_BUBBLE = "BURST_BUBBLE"
    UNCLASSIFIED = "UNCLASSIFIED"


@dataclass
class LSBMetrics:
    """Comprehensive LSB metrics from surface distributions."""
    
    # Detection flags
    lsb_detected: bool
    
    # Location metrics (x/c)
    separation_location: Optional[float]
    transition_onset: Optional[float]
    transition_completion: Optional[float]
    reattachment_location: Optional[float]
    
    # Bubble geometry
    bubble_length: Optional[float]
    bubble_height_proxy: Optional[float]
    bubble_area_proxy: Optional[float]
    
    # Pressure plateau
    plateau_start: Optional[float]
    plateau_end: Optional[float]
    plateau_length: Optional[float]
    plateau_pressure_level: Optional[float]
    
    # Skin friction
    cf_reversal_location: Optional[float]
    cf_recovery_location: Optional[float]
    min_cf: Optional[float]
    
    # Intermittency
    intermittency_onset: Optional[float]
    intermittency_completion: Optional[float]
    intermittency_growth_rate: Optional[float]
    
    # Adverse pressure gradient
    apg_severity: float
    apg_region_start: Optional[float]
    apg_region_end: Optional[float]
    
    # Wall shear
    wall_shear_collapse_detected: bool
    wall_shear_collapse_location: Optional[float]
    
    # Reattachment strength
    reattachment_strength: Optional[float]
    
    # Validation checks
    physically_consistent: bool
    consistency_flags: List[str] = field(default_factory=list)


@dataclass
class LSBClassification:
    """LSB classification with risk assessment."""
    
    # Classification
    bubble_type: LSBType
    
    # Risk metrics
    bursting_risk_score: float  # 0-1 scale
    hysteresis_index: float  # 0-1 scale
    
    # Evolution metrics
    bubble_growth_rate: Optional[float]
    movement_rate: Optional[float]
    
    # Stability metrics
    stability_indicator: float  # 0-1 scale, higher = more stable
    
    # Drag impact
    drag_amplification_factor: Optional[float]
    
    # Effective camber distortion
    effective_camber_distortion: Optional[float]
    
    # Recommendations
    suppression_mechanisms: List[str] = field(default_factory=list)
    critical_regions: List[str] = field(default_factory=list)


@dataclass
class LSBDetectionReport:
    """Comprehensive LSB detection report."""
    
    # Metrics
    metrics: LSBMetrics
    classification: LSBClassification
    
    # Overall assessment
    is_valid: bool
    confidence: float
    
    # Diagnostics
    detection_method: str
    warnings: List[str] = field(default_factory=list)
    
    # Surface data used
    x_coordinates: np.ndarray
    cp_upper: np.ndarray
    cf_upper: Optional[np.ndarray] = None
    intermittency_upper: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "lsb_detected": self.metrics.lsb_detected,
            "bubble_type": self.classification.bubble_type.value,
            "bursting_risk_score": self.classification.bursting_risk_score,
            "hysteresis_index": self.classification.hysteresis_index,
            "separation_location": self.metrics.separation_location,
            "transition_onset": self.metrics.transition_onset,
            "reattachment_location": self.metrics.reattachment_location,
            "bubble_length": self.metrics.bubble_length,
            "bubble_height_proxy": self.metrics.bubble_height_proxy,
            "plateau_length": self.metrics.plateau_length,
            "apg_severity": self.metrics.apg_severity,
            "is_valid": self.is_valid,
            "confidence": self.confidence,
            "warnings": self.warnings,
        }


class LSBDetector:
    """
    Detects and classifies laminar separation bubbles.
    
    Uses surface pressure, skin friction, and intermittency distributions
    to identify LSB characteristics and classify bubble type.
    """
    
    def __init__(
        self,
        plateau_threshold: float = 0.5,
        min_bubble_length: float = 0.02,
        short_long_threshold: float = 0.15,
    ):
        """
        Initialize LSB detector.
        
        Args:
            plateau_threshold: dCp/dx threshold for plateau detection
            min_bubble_length: Minimum bubble length (x/c) for detection
            short_long_threshold: Bubble length threshold for short vs long
        """
        self.plateau_threshold = plateau_threshold
        self.min_bubble_length = min_bubble_length
        self.short_long_threshold = short_long_threshold
    
    def detect_pressure_plateau(
        self,
        x: np.ndarray,
        cp: np.ndarray,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Detect pressure plateau characteristic of LSB.
        
        Args:
            x: Surface coordinates (x/c)
            cp: Pressure coefficient distribution
        
        Returns:
            (plateau_start, plateau_end, plateau_pressure_level)
        """
        if len(x) < 10 or len(cp) < 10:
            return None, None, None
        
        # Compute pressure gradient
        dcp_dx = np.gradient(cp, x)
        
        # Find regions with small pressure gradient (plateau)
        plateau_mask = np.abs(dcp_dx) < self.plateau_threshold
        
        # Find contiguous plateau regions
        plateau_regions = []
        in_plateau = False
        start_idx = None
        
        for i, is_plateau in enumerate(plateau_mask):
            if is_plateau and not in_plateau:
                in_plateau = True
                start_idx = i
            elif not is_plateau and in_plateau:
                in_plateau = False
                end_idx = i
                length = x[end_idx] - x[start_idx]
                if length >= self.min_bubble_length:
                    plateau_regions.append((start_idx, end_idx, length))
        
        # Select the most prominent plateau (longest)
        if plateau_regions:
            best_region = max(plateau_regions, key=lambda r: r[2])
            start_idx, end_idx, length = best_region
            
            plateau_start = float(x[start_idx])
            plateau_end = float(x[end_idx])
            plateau_pressure = float(np.mean(cp[start_idx:end_idx]))
            
            return plateau_start, plateau_end, plateau_pressure
        
        return None, None, None
    
    def detect_separation(
        self,
        x: np.ndarray,
        cf: Optional[np.ndarray] = None,
        cp: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        """
        Detect laminar separation location.
        
        Args:
            x: Surface coordinates (x/c)
            cf: Skin friction distribution (optional)
            cp: Pressure coefficient distribution (optional)
        
        Returns:
            Separation location (x/c) or None
        """
        if cf is not None:
            # Detect separation as skin friction reversal
            for i in range(1, len(cf)):
                if cf[i] < 0 and cf[i-1] >= 0:
                    return float(x[i])
        
        if cp is not None:
            # Detect separation as pressure gradient minimum
            dcp_dx = np.gradient(cp, x)
            d2cp_dx2 = np.gradient(dcp_dx, x)
            
            # Find inflection point where pressure gradient becomes adverse
            for i in range(1, len(dcp_dx)):
                if dcp_dx[i] > 0 and d2cp_dx2[i] > 0:
                    return float(x[i])
        
        return None
    
    def detect_reattachment(
        self,
        x: np.ndarray,
        cf: Optional[np.ndarray] = None,
        cp: Optional[np.ndarray] = None,
        separation_location: Optional[float] = None,
    ) -> Optional[float]:
        """
        Detect turbulent reattachment location.
        
        Args:
            x: Surface coordinates (x/c)
            cf: Skin friction distribution (optional)
            cp: Pressure coefficient distribution (optional)
            separation_location: Known separation location (optional)
        
        Returns:
            Reattachment location (x/c) or None
        """
        if cf is not None and separation_location is not None:
            # Find where skin friction becomes positive after separation
            sep_idx = np.argmin(np.abs(x - separation_location))
            
            for i in range(sep_idx + 1, len(cf)):
                if cf[i] > 0 and cf[i-1] <= 0:
                    return float(x[i])
        
        if cp is not None and separation_location is not None:
            # Detect reattachment as pressure recovery
            dcp_dx = np.gradient(cp, x)
            sep_idx = np.argmin(np.abs(x - separation_location))
            
            for i in range(sep_idx + 1, len(dcp_dx)):
                if dcp_dx[i] < -1.0:  # Strong pressure recovery
                    return float(x[i])
        
        return None
    
    def detect_transition(
        self,
        x: np.ndarray,
        intermittency: Optional[np.ndarray] = None,
        cp: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Detect transition onset and completion.
        
        Args:
            x: Surface coordinates (x/c)
            intermittency: Intermittency distribution (optional)
            cp: Pressure coefficient distribution (optional)
        
        Returns:
            (transition_onset, transition_completion)
        """
        if intermittency is not None:
            # Transition onset: intermittency > 0.1
            for i in range(len(intermittency)):
                if intermittency[i] > 0.1:
                    onset = float(x[i])
                    break
            else:
                onset = None
            
            # Transition completion: intermittency > 0.9
            for i in range(len(intermittency)):
                if intermittency[i] > 0.9:
                    completion = float(x[i])
                    break
            else:
                completion = None
            
            return onset, completion
        
        if cp is not None:
            # Estimate transition from pressure recovery
            dcp_dx = np.gradient(cp, x)
            
            # Transition onset: start of pressure recovery
            for i in range(1, len(dcp_dx)):
                if dcp_dx[i] < -0.5 and dcp_dx[i-1] >= -0.5:
                    onset = float(x[i])
                    break
            else:
                onset = None
            
            # Transition completion: strong pressure recovery
            for i in range(1, len(dcp_dx)):
                if dcp_dx[i] < -2.0:
                    completion = float(x[i])
                    break
            else:
                completion = None
            
            return onset, completion
        
        return None, None
    
    def compute_apg_severity(
        self,
        x: np.ndarray,
        cp: np.ndarray,
    ) -> Tuple[float, Optional[float], Optional[float]]:
        """
        Compute adverse pressure gradient severity.
        
        Args:
            x: Surface coordinates (x/c)
            cp: Pressure coefficient distribution
        
        Returns:
            (severity, region_start, region_end)
        """
        dcp_dx = np.gradient(cp, x)
        
        # Find regions with adverse pressure gradient (dCp/dx > 0)
        apg_mask = dcp_dx > 0
        
        if not np.any(apg_mask):
            return 0.0, None, None
        
        # Compute severity as integral of APG
        severity = float(np.trapz(dcp_dx[apg_mask], x[apg_mask]))
        
        # Find APG region extent
        apg_indices = np.where(apg_mask)[0]
        region_start = float(x[apg_indices[0]])
        region_end = float(x[apg_indices[-1]])
        
        return severity, region_start, region_end
    
    def classify_bubble(
        self,
        metrics: LSBMetrics,
    ) -> LSBClassification:
        """
        Classify LSB type and assess risk.
        
        Args:
            metrics: LSB metrics from detection
        
        Returns:
            LSBClassification with type and risk assessment
        """
        if not metrics.lsb_detected:
            return LSBClassification(
                bubble_type=LSBType.NO_BUBBLE,
                bursting_risk_score=0.0,
                hysteresis_index=0.0,
                bubble_growth_rate=None,
                movement_rate=None,
                stability_indicator=1.0,
                drag_amplification_factor=None,
                effective_camber_distortion=None,
            )
        
        # Determine bubble type
        if metrics.bubble_length is None:
            bubble_type = LSBType.UNCLASSIFIED
        elif metrics.bubble_length < self.short_long_threshold:
            bubble_type = LSBType.SHORT_BUBBLE
        else:
            bubble_type = LSBType.LONG_BUBBLE
        
        # Compute bursting risk
        bursting_risk = 0.0
        if metrics.bubble_length is not None:
            # Longer bubbles have higher bursting risk
            bursting_risk += min(1.0, metrics.bubble_length / 0.3)
        
        if metrics.apg_severity > 0:
            # Higher APG severity increases bursting risk
            bursting_risk += min(0.5, metrics.apg_severity / 10.0)
        
        bursting_risk = min(1.0, bursting_risk)
        
        # Compute hysteresis index
        hysteresis_index = 0.0
        if bubble_type == LSBType.LONG_BUBBLE:
            hysteresis_index = 0.7
        elif bubble_type == LSBType.SHORT_BUBBLE:
            hysteresis_index = 0.3
        
        # Stability indicator
        stability = 1.0 - bursting_risk
        
        # Drag amplification
        drag_amplification = None
        if metrics.bubble_length is not None:
            drag_amplification = 1.0 + 2.0 * metrics.bubble_length
        
        # Effective camber distortion
        camber_distortion = None
        if metrics.plateau_length is not None:
            camber_distortion = 0.5 * metrics.plateau_length
        
        # Suppression mechanisms
        suppression_mechanisms = []
        critical_regions = []
        
        if metrics.separation_location is not None:
            critical_regions.append(f"separation_at_x={metrics.separation_location:.3f}")
            suppression_mechanisms.append("leading_edge_contouring")
        
        if metrics.apg_severity > 5.0:
            critical_regions.append("high_apg_region")
            suppression_mechanisms.append("pressure_gradient_shaping")
        
        if metrics.bubble_length is not None and metrics.bubble_length > 0.2:
            critical_regions.append("long_bubble_region")
            suppression_mechanisms.append("transition_promotion")
        
        return LSBClassification(
            bubble_type=bubble_type,
            bursting_risk_score=bursting_risk,
            hysteresis_index=hysteresis_index,
            bubble_growth_rate=None,  # Requires temporal data
            movement_rate=None,  # Requires temporal data
            stability_indicator=stability,
            drag_amplification_factor=drag_amplification,
            effective_camber_distortion=camber_distortion,
            suppression_mechanisms=suppression_mechanisms,
            critical_regions=critical_regions,
        )
    
    def detect(
        self,
        x: np.ndarray,
        cp_upper: np.ndarray,
        cf_upper: Optional[np.ndarray] = None,
        intermittency_upper: Optional[np.ndarray] = None,
    ) -> LSBDetectionReport:
        """
        Perform comprehensive LSB detection.
        
        Args:
            x: Surface coordinates (x/c)
            cp_upper: Upper surface pressure coefficient
            cf_upper: Upper surface skin friction (optional)
            intermittency_upper: Upper surface intermittency (optional)
        
        Returns:
            LSBDetectionReport with comprehensive analysis
        """
        warnings = []
        
        # Detect pressure plateau
        plateau_start, plateau_end, plateau_pressure = self.detect_pressure_plateau(
            x, cp_upper
        )
        
        if plateau_start is not None:
            plateau_length = plateau_end - plateau_start
        else:
            plateau_length = None
        
        # Detect separation
        separation = self.detect_separation(x, cf_upper, cp_upper)
        
        # Detect reattachment
        reattachment = self.detect_reattachment(x, cf_upper, cp_upper, separation)
        
        # Detect transition
        transition_onset, transition_completion = self.detect_transition(
            x, intermittency_upper, cp_upper
        )
        
        # Compute bubble length
        bubble_length = None
        if separation is not None and reattachment is not None:
            bubble_length = reattachment - separation
        
        # Compute bubble height proxy (from plateau length)
        bubble_height = None
        if plateau_length is not None:
            bubble_height = 0.1 * plateau_length  # Simplified proxy
        
        # Compute APG severity
        apg_severity, apg_start, apg_end = self.compute_apg_severity(x, cp_upper)
        
        # Check physical consistency
        consistency_flags = []
        physically_consistent = True
        
        if separation is not None and transition_onset is not None:
            if not (separation < transition_onset):
                physically_consistent = False
                consistency_flags.append("transition_before_separation")
        
        if transition_onset is not None and reattachment is not None:
            if not (transition_onset < reattachment):
                physically_consistent = False
                consistency_flags.append("reattachment_before_transition")
        
        # Determine if LSB is detected
        lsb_detected = (
            plateau_length is not None and
            plateau_length >= self.min_bubble_length and
            separation is not None
        )
        
        # Build metrics
        metrics = LSBMetrics(
            lsb_detected=lsb_detected,
            separation_location=separation,
            transition_onset=transition_onset,
            transition_completion=transition_completion,
            reattachment_location=reattachment,
            bubble_length=bubble_length,
            bubble_height_proxy=bubble_height,
            bubble_area_proxy=None,  # Requires integration
            plateau_start=plateau_start,
            plateau_end=plateau_end,
            plateau_length=plateau_length,
            plateau_pressure_level=plateau_pressure,
            cf_reversal_location=separation,  # Approximation
            cf_recovery_location=reattachment,  # Approximation
            min_cf=None,  # Requires cf data
            intermittency_onset=transition_onset,  # Approximation
            intermittency_completion=transition_completion,  # Approximation
            intermittency_growth_rate=None,  # Requires temporal data
            apg_severity=apg_severity,
            apg_region_start=apg_start,
            apg_region_end=apg_end,
            wall_shear_collapse_detected=(cf_upper is not None and np.any(cf_upper < 0)),
            wall_shear_collapse_location=separation,
            reattachment_strength=None,  # Requires detailed analysis
            physically_consistent=physically_consistent,
            consistency_flags=consistency_flags,
        )
        
        # Classify bubble
        classification = self.classify_bubble(metrics)
        
        # Compute confidence
        confidence = 0.5
        if lsb_detected:
            confidence = 0.8
            if intermittency_upper is not None:
                confidence = 0.9
            if cf_upper is not None:
                confidence = 0.95
        
        # Add warnings
        if not physically_consistent:
            warnings.append("Physical inconsistency detected in LSB metrics")
        
        if lsb_detected and bubble_length is None:
            warnings.append("LSB detected but bubble length could not be computed")
        
        return LSBDetectionReport(
            metrics=metrics,
            classification=classification,
            is_valid=physically_consistent,
            confidence=confidence,
            detection_method="pressure_plateau_cf_intermittency",
            warnings=warnings,
            x_coordinates=x,
            cp_upper=cp_upper,
            cf_upper=cf_upper,
            intermittency_upper=intermittency_upper,
        )
