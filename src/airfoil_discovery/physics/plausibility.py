"""
Aerodynamic plausibility governance for CFD-based optimization.

Implements hard aerodynamic filters and plausibility checks to reject
solutions that are numerically converged but physically meaningless.
This is a critical governance layer that prevents the optimizer from
exploiting numerical loopholes to produce absurd aerodynamic results.

The framework validates:
- Drag coefficient reasonableness
- Lift-to-drag ratio plausibility
- Bluff-body detection
- Pressure recovery governance
- Wall shear and separation analysis
- Aerodynamic feasibility scoring
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import warnings


class PlausibilityStatus(Enum):
    """Aerodynamic plausibility status."""
    PHYSICALLY_PLAUSIBLE = "PHYSICALLY_PLAUSIBLE"
    AERODYNAMICALLY_SUSPECT = "AERODYNAMICALLY_SUSPECT"
    PHYSICALLY_IMPOSSIBLE = "PHYSICALLY_IMPOSSIBLE"


class PlausibilityViolationType(Enum):
    """Types of aerodynamic plausibility violations."""
    NONE = "NONE"
    DRAG_TOO_HIGH = "DRAG_TOO_HIGH"
    DRAG_TOO_LOW = "DRAG_TOO_LOW"
    LIFT_DRAG_RATIO_TOO_LOW = "LIFT_DRAG_RATIO_TOO_LOW"
    LIFT_SIGN_INCORRECT = "LIFT_SIGN_INCORRECT"
    LIFT_TOO_HIGH = "LIFT_TOO_HIGH"
    LIFT_TOO_LOW = "LIFT_TOO_LOW"
    BLUFF_BODY_DETECTED = "BLUFF_BODY_DETECTED"
    PRESSURE_RECOVERY_PATHOLOGICAL = "PRESSURE_RECOVERY_PATHOLOGICAL"
    SEPARATION_FULL_CHORD = "SEPARATION_FULL_CHORD"
    CATASTROPHIC_DRAG_RISE = "CATASTROPHIC_DRAG_RISE"
    MOMENT_COEFFICIENT_INVALID = "MOMENT_COEFFICIENT_INVALID"
    REYNOLDS_NUMBER_MISMATCH = "REYNOLDS_NUMBER_MISMATCH"
    MACH_NUMBER_INVALID = "MACH_NUMBER_INVALID"


@dataclass
class ForceCoefficientMetrics:
    """Force coefficient analysis metrics."""
    
    # Primary coefficients
    cl: float
    cd: float
    cm: Optional[float]
    
    # Lift-to-drag ratio
    ld_ratio: float
    
    # Expected ranges for operating condition
    expected_cd_min: float
    expected_cd_max: float
    expected_cl_min: float
    expected_cl_max: float
    expected_ld_min: float
    
    # Constraint compliance
    cd_within_bounds: bool
    cl_within_bounds: bool
    ld_within_bounds: bool
    
    # Violations
    violations: List[PlausibilityViolationType] = field(default_factory=list)


@dataclass
class PressureRecoveryMetrics:
    """Pressure recovery analysis metrics."""
    
    # Pressure distribution statistics
    cp_min: float
    cp_max: float
    cp_range: float
    
    # Suction peak
    suction_peak_location: float
    suction_peak_magnitude: float
    
    # Pressure recovery
    recovery_start: Optional[float]
    recovery_end: Optional[float]
    recovery_slope: float
    recovery_smoothness: float  # 0-1 scale
    
    # Adverse pressure gradient
    apg_severity: float
    apg_length: float
    
    # Violations
    violations: List[PlausibilityViolationType] = field(default_factory=list)


@dataclass
class SeparationMetrics:
    """Flow separation analysis metrics."""
    
    # Separation detection
    separation_detected: bool
    separation_location: Optional[float]
    reattachment_location: Optional[float]
    
    # Separation extent
    separated_length_fraction: float
    full_chord_separation: bool
    
    # Wall shear analysis
    cf_min: float
    cf_reversal_detected: bool
    cf_reversal_locations: List[float]
    
    # Recirculation
    recirculation_detected: bool
    recirculation_strength: float
    
    # Violations
    violations: List[PlausibilityViolationType] = field(default_factory=list)


@dataclass
class BluffBodyMetrics:
    """Bluff body detection metrics."""
    
    # Drag characteristics
    pressure_drag_fraction: float
    viscous_drag_fraction: float
    
    # Wake characteristics
    wake_width_proxy: float
    wake_strength_proxy: float
    
    # Base pressure
    base_cp: Optional[float]
    base_pressure_coefficient: float
    
    # Bluff body indicators
    drag_crisis_indicator: float  # 0-1 scale, higher = more bluff-like
    pressure_drag_dominance: bool
    
    # Assessment
    is_bluff_like: bool
    bluff_confidence: float  # 0-1 scale
    
    # Violations
    violations: List[PlausibilityViolationType] = field(default_factory=list)


@dataclass
class PlausibilityGovernanceReport:
    """Comprehensive aerodynamic plausibility report."""
    
    # Overall status
    status: PlausibilityStatus
    
    # Component metrics
    forces: Optional[ForceCoefficientMetrics] = None
    pressure_recovery: Optional[PressureRecoveryMetrics] = None
    separation: Optional[SeparationMetrics] = None
    bluff_body: Optional[BluffBodyMetrics] = None
    
    # Overall assessment
    is_valid: bool
    can_accept_solution: bool
    
    # Violations
    violations: List[PlausibilityViolationType] = field(default_factory=list)
    failure_reasons: List[str] = field(default_factory=list)
    
    # Recommendations
    recommended_actions: List[str] = field(default_factory=list)
    
    # Operating condition context
    reynolds_number: float = 0.0
    mach_number: float = 0.0
    angle_of_attack: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "is_valid": self.is_valid,
            "can_accept_solution": self.can_accept_solution,
            "violations": [v.value for v in self.violations],
            "failure_reasons": self.failure_reasons,
            "recommended_actions": self.recommended_actions,
            "forces": {
                "cl": self.forces.cl if self.forces else None,
                "cd": self.forces.cd if self.forces else None,
                "ld_ratio": self.forces.ld_ratio if self.forces else None,
                "cd_within_bounds": self.forces.cd_within_bounds if self.forces else None,
            } if self.forces else None,
            "reynolds_number": self.reynolds_number,
            "mach_number": self.mach_number,
            "angle_of_attack": self.angle_of_attack,
        }


@dataclass
class PlausibilityConfig:
    """Configuration for aerodynamic plausibility governance."""
    
    # Force coefficient bounds for low-Re (Re = 50k-500k)
    # These are configurable based on operating condition
    
    # Drag coefficient bounds
    cd_max_cruise: float = 0.05  # Maximum for efficient cruise
    cd_max_stall: float = 0.15   # Maximum for stalled condition
    cd_max_absolute: float = 0.6  # Absolute maximum (bluff body territory)
    cd_min: float = 0.001        # Minimum (can't be zero or negative)
    
    # Lift coefficient bounds
    cl_max_attached: float = 1.8  # Maximum for attached flow
    cl_max_absolute: float = 2.5  # Absolute maximum
    cl_min: float = -0.5         # Minimum (slight negative OK for some cases)
    
    # Lift-to-drag ratio bounds
    ld_min_cruise: float = 5.0   # Minimum for efficient cruise
    ld_min_acceptable: float = 2.0  # Absolute minimum
    ld_max: float = 200.0        # Maximum (can't be infinite)
    
    # Moment coefficient bounds
    cm_min: float = -0.5
    cm_max: float = 0.1
    
    # Pressure recovery thresholds
    max_pressure_recovery_slope: float = 5.0
    min_pressure_recovery_smoothness: float = 0.3
    max_apg_severity: float = 10.0
    
    # Separation thresholds
    max_separated_fraction: float = 0.5  # Maximum separated chord fraction
    full_chord_separation_threshold: float = 0.95
    
    # Bluff body detection thresholds
    bluff_pressure_drag_fraction: float = 0.7  # Pressure drag > 70% suggests bluff
    bluff_drag_threshold: float = 0.2          # Cd > 0.2 is suspicious
    bluff_wake_width_threshold: float = 0.3    # Wide wake suggests bluff
    
    # Operating condition ranges
    reynolds_min: float = 10000   # Minimum valid Re
    reynolds_max: float = 1000000  # Maximum valid Re
    mach_max_incompressible: float = 0.3  # Max for incompressible assumption
    
    # Governance policy
    strict_mode: bool = True
    suspect_threshold: int = 2


class AerodynamicPlausibilityGovernor:
    """
    Governs aerodynamic plausibility for CFD-based optimization.
    
    This class implements hard aerodynamic filters that reject solutions
    which are numerically converged but physically meaningless. It is
    designed to prevent:
    
    - Accepting Cd ≈ 0.6 at Re=200k (bluff body behavior)
    - Accepting Cl/Cd ≈ 0.24 (catastrophically inefficient)
    - Accepting pathological pressure recovery
    - Accepting full-chord separation as "optimal"
    - Accepting solutions that exploit transition model loopholes
    
    The governor uses literature-informed bounds for low-Re airfoils
    and adapts thresholds based on operating conditions.
    """
    
    def __init__(self, config: Optional[PlausibilityConfig] = None):
        """
        Initialize aerodynamic plausibility governor.
        
        Args:
            config: Governance configuration. Uses defaults if None.
        """
        self.config = config or PlausibilityConfig()
    
    def analyze_force_coefficients(
        self,
        cl: float,
        cd: float,
        cm: Optional[float] = None,
        reynolds: float = 200000,
        mach: float = 0.1,
        aoa: float = 0.0,
    ) -> ForceCoefficientMetrics:
        """
        Analyze force coefficients for plausibility.
        
        Args:
            cl: Lift coefficient
            cd: Drag coefficient
            cm: Moment coefficient (optional)
            reynolds: Reynolds number
            mach: Mach number
            aoa: Angle of attack (degrees)
        
        Returns:
            ForceCoefficientMetrics with analysis
        """
        # Compute lift-to-drag ratio
        ld_ratio = cl / cd if abs(cd) > 1e-15 else float('inf')
        
        # Adjust expected ranges based on operating condition
        # For low-Re airfoils, typical Cd ranges from 0.005 to 0.05 in cruise
        if reynolds < 100000:
            expected_cd_max = self.config.cd_max_stall
        elif reynolds < 500000:
            expected_cd_max = self.config.cd_max_cruise * 2
        else:
            expected_cd_max = self.config.cd_max_cruise
        
        expected_cd_min = self.config.cd_min
        expected_cl_min = self.config.cl_min
        expected_cl_max = self.config.cl_max_attached if abs(aoa) < 15 else self.config.cl_max_absolute
        
        # For cruise conditions (low aoa), expect reasonable L/D
        if abs(aoa) < 10:
            expected_ld_min = self.config.ld_min_cruise
        else:
            expected_ld_min = self.config.ld_min_acceptable
        
        # Check constraint compliance
        cd_within_bounds = expected_cd_min <= cd <= expected_cd_max
        cl_within_bounds = expected_cl_min <= cl <= expected_cl_max
        ld_within_bounds = ld_ratio >= expected_ld_min if cd > 1e-15 else False
        
        # Detect violations
        violations = []
        
        if cd > self.config.cd_max_absolute:
            violations.append(PlausibilityViolationType.DRAG_TOO_HIGH)
        elif cd > expected_cd_max:
            violations.append(PlausibilityViolationType.DRAG_TOO_HIGH)
        
        if cd < self.config.cd_min:
            violations.append(PlausibilityViolationType.DRAG_TOO_LOW)
        
        if cl < expected_cl_min:
            violations.append(PlausibilityViolationType.LIFT_TOO_LOW)
        elif cl > expected_cl_max:
            violations.append(PlausibilityViolationType.LIFT_TOO_HIGH)
        
        if cl * aoa < 0 and abs(aoa) > 1:  # Lift sign inconsistent with aoa
            violations.append(PlausibilityViolationType.LIFT_SIGN_INCORRECT)
        
        if ld_ratio < expected_ld_min and cd > 1e-15:
            violations.append(PlausibilityViolationType.LIFT_DRAG_RATIO_TOO_LOW)
        
        # Check moment coefficient if provided
        if cm is not None:
            if cm < self.config.cm_min or cm > self.config.cm_max:
                violations.append(PlausibilityViolationType.MOMENT_COEFFICIENT_INVALID)
        
        return ForceCoefficientMetrics(
            cl=cl,
            cd=cd,
            cm=cm,
            ld_ratio=ld_ratio,
            expected_cd_min=expected_cd_min,
            expected_cd_max=expected_cd_max,
            expected_cl_min=expected_cl_min,
            expected_cl_max=expected_cl_max,
            expected_ld_min=expected_ld_min,
            cd_within_bounds=cd_within_bounds,
            cl_within_bounds=cl_within_bounds,
            ld_within_bounds=ld_within_bounds,
            violations=violations,
        )
    
    def analyze_pressure_recovery(
        self,
        x: np.ndarray,
        cp: np.ndarray,
    ) -> PressureRecoveryMetrics:
        """
        Analyze pressure recovery for plausibility.
        
        Args:
            x: Chordwise coordinates (normalized 0-1)
            cp: Pressure coefficient distribution
        
        Returns:
            PressureRecoveryMetrics with analysis
        """
        if len(x) < 5 or len(cp) < 5:
            return PressureRecoveryMetrics(
                cp_min=0.0,
                cp_max=0.0,
                cp_range=0.0,
                suction_peak_location=0.0,
                suction_peak_magnitude=0.0,
                recovery_start=None,
                recovery_end=None,
                recovery_slope=0.0,
                recovery_smoothness=0.0,
                apg_severity=0.0,
                apg_length=0.0,
                violations=[PlausibilityViolationType.PRESSURE_RECOVERY_PATHOLOGICAL],
            )
        
        # Basic statistics
        cp_min = float(np.min(cp))
        cp_max = float(np.max(cp))
        cp_range = cp_max - cp_min
        
        # Suction peak
        suction_peak_idx = int(np.argmin(cp))
        suction_peak_location = float(x[suction_peak_idx])
        suction_peak_magnitude = float(cp_min)
        
        # Pressure recovery analysis
        # Find where pressure starts recovering (after suction peak)
        dcp_dx = np.gradient(cp, x)
        
        recovery_start = None
        recovery_end = None
        
        # Recovery starts after suction peak where dCp/dx becomes positive
        for i in range(suction_peak_idx + 1, len(dcp_dx)):
            if dcp_dx[i] > 0.1:
                recovery_start = float(x[i])
                break
        
        # Recovery ends near trailing edge or where dCp/dx becomes negative again
        if recovery_start is not None:
            start_idx = np.argmin(np.abs(x - recovery_start))
            for i in range(start_idx + 1, len(dcp_dx)):
                if dcp_dx[i] < 0 or i == len(dcp_dx) - 1:
                    recovery_end = float(x[i])
                    break
        
        # Recovery slope
        if recovery_start is not None and recovery_end is not None:
            start_idx = np.argmin(np.abs(x - recovery_start))
            end_idx = np.argmin(np.abs(x - recovery_end))
            if end_idx > start_idx:
                recovery_slope = (cp[end_idx] - cp[start_idx]) / (x[end_idx] - x[start_idx])
            else:
                recovery_slope = 0.0
        else:
            recovery_slope = 0.0
        
        # Recovery smoothness (based on dCp/dx consistency)
        if recovery_start is not None and recovery_end is not None:
            start_idx = np.argmin(np.abs(x - recovery_start))
            end_idx = np.argmin(np.abs(x - recovery_end))
            if end_idx > start_idx + 2:
                recovery_dcp_dx = dcp_dx[start_idx:end_idx]
                smoothness = float(np.exp(-np.std(recovery_dcp_dx)))
            else:
                smoothness = 0.5
        else:
            smoothness = 0.0
        
        # Adverse pressure gradient severity
        apg_mask = dcp_dx > 0
        if np.any(apg_mask):
            apg_severity = float(np.max(dcp_dx[apg_mask]))
            apg_indices = np.where(apg_mask)[0]
            apg_length = float(x[apg_indices[-1]] - x[apg_indices[0]])
        else:
            apg_severity = 0.0
            apg_length = 0.0
        
        # Detect violations
        violations = []
        
        if recovery_slope > self.config.max_pressure_recovery_slope:
            violations.append(PlausibilityViolationType.PRESSURE_RECOVERY_PATHOLOGICAL)
        
        if smoothness < self.config.min_pressure_recovery_smoothness:
            violations.append(PlausibilityViolationType.PRESSURE_RECOVERY_PATHOLOGICAL)
        
        if apg_severity > self.config.max_apg_severity:
            violations.append(PlausibilityViolationType.PRESSURE_RECOVERY_PATHOLOGICAL)
        
        # Check for pressure plateau (possible LSB indicator)
        if recovery_start is not None:
            plateau_mask = np.abs(dcp_dx) < 0.1
            plateau_regions = []
            in_plateau = False
            for i, is_plateau in enumerate(plateau_mask):
                if is_plateau and not in_plateau:
                    in_plateau = True
                    start = i
                elif not is_plateau and in_plateau:
                    in_plateau = False
                    if x[i] - x[start] > 0.05:  # Significant plateau
                        plateau_regions.append((x[start], x[i]))
            
            if plateau_regions:
                # Plateau itself is not a violation, but note it
                pass
        
        return PressureRecoveryMetrics(
            cp_min=cp_min,
            cp_max=cp_max,
            cp_range=cp_range,
            suction_peak_location=suction_peak_location,
            suction_peak_magnitude=suction_peak_magnitude,
            recovery_start=recovery_start,
            recovery_end=recovery_end,
            recovery_slope=recovery_slope,
            recovery_smoothness=smoothness,
            apg_severity=apg_severity,
            apg_length=apg_length,
            violations=violations,
        )
    
    def analyze_separation(
        self,
        x: np.ndarray,
        cf: np.ndarray,
        cp: Optional[np.ndarray] = None,
    ) -> SeparationMetrics:
        """
        Analyze flow separation for plausibility.
        
        Args:
            x: Chordwise coordinates (normalized 0-1)
            cf: Skin friction coefficient distribution
            cp: Pressure coefficient distribution (optional)
        
        Returns:
            SeparationMetrics with analysis
        """
        if len(x) < 5 or len(cf) < 5:
            return SeparationMetrics(
                separation_detected=False,
                separation_location=None,
                reattachment_location=None,
                separated_length_fraction=0.0,
                full_chord_separation=False,
                cf_min=0.0,
                cf_reversal_detected=False,
                cf_reversal_locations=[],
                recirculation_detected=False,
                recirculation_strength=0.0,
                violations=[],
            )
        
        # Detect Cf reversal (separation)
        cf_reversal_locations = []
        for i in range(1, len(cf)):
            if cf[i] < 0 and cf[i-1] >= 0:
                cf_reversal_locations.append(float(x[i]))
        
        cf_reversal_detected = len(cf_reversal_locations) > 0
        cf_min = float(np.min(cf))
        
        # Determine separation and reattachment
        separation_location = None
        reattachment_location = None
        separated_regions = []
        
        if cf_reversal_detected:
            # Find separated regions (where Cf < 0)
            separated_mask = cf < 0
            
            # Find contiguous separated regions
            in_separated = False
            sep_start = None
            
            for i, is_separated in enumerate(separated_mask):
                if is_separated and not in_separated:
                    in_separated = True
                    sep_start = i
                elif not is_separated and in_separated:
                    in_separated = False
                    sep_end = i
                    
                    if separation_location is None:
                        separation_location = float(x[sep_start])
                        reattachment_location = float(x[sep_end])
                    
                    separated_regions.append((x[sep_start], x[sep_end]))
            
            # Handle case where separation extends to trailing edge
            if in_separated:
                if separation_location is None:
                    separation_location = float(x[sep_start])
                separated_regions.append((x[sep_start], x[-1]))
        
        # Compute separated length fraction
        total_separated_length = sum(r[1] - r[0] for r in separated_regions)
        separated_length_fraction = total_separated_length / (x[-1] - x[0])
        
        # Full chord separation check
        full_chord_separation = separated_length_fraction > self.config.full_chord_separation_threshold
        
        # Recirculation strength (based on minimum Cf)
        recirculation_strength = abs(cf_min) if cf_min < 0 else 0.0
        recirculation_detected = cf_min < 0
        
        # Detect violations
        violations = []
        
        if full_chord_separation:
            violations.append(PlausibilityViolationType.SEPARATION_FULL_CHORD)
        
        if separated_length_fraction > self.config.max_separated_fraction:
            violations.append(PlausibilityViolationType.SEPARATION_FULL_CHORD)
        
        return SeparationMetrics(
            separation_detected=cf_reversal_detected,
            separation_location=separation_location,
            reattachment_location=reattachment_location,
            separated_length_fraction=separated_length_fraction,
            full_chord_separation=full_chord_separation,
            cf_min=cf_min,
            cf_reversal_detected=cf_reversal_detected,
            cf_reversal_locations=cf_reversal_locations,
            recirculation_detected=recirculation_detected,
            recirculation_strength=recirculation_strength,
            violations=violations,
        )
    
    def detect_bluff_body(
        self,
        cl: float,
        cd: float,
        cp: np.ndarray,
        x: np.ndarray,
        base_cp: Optional[float] = None,
    ) -> BluffBodyMetrics:
        """
        Detect bluff-body-like aerodynamic behavior.
        
        Args:
            cl: Lift coefficient
            cd: Drag coefficient
            cp: Pressure coefficient distribution
            x: Chordwise coordinates
            base_cp: Base pressure coefficient (optional)
        
        Returns:
            BluffBodyMetrics with bluff body assessment
        """
        # Estimate pressure drag fraction
        # For streamlined bodies, pressure drag is typically < 30% of total
        # For bluff bodies, pressure drag can be > 70%
        
        # Simple estimate: if Cp distribution shows large base suction,
        # pressure drag is significant
        if base_cp is not None:
            base_pressure_coefficient = base_cp
        else:
            # Estimate from trailing edge Cp
            base_pressure_coefficient = float(cp[-1])
        
        # Pressure drag proxy (based on base suction)
        # More negative base Cp = higher pressure drag
        pressure_drag_proxy = max(0, -base_pressure_coefficient)
        
        # Normalize by dynamic pressure effect
        pressure_drag_fraction = min(1.0, pressure_drag_proxy / (cd + 1e-15))
        
        # Viscous drag fraction
        viscous_drag_fraction = 1.0 - pressure_drag_fraction
        
        # Wake width proxy (based on pressure recovery location)
        dcp_dx = np.gradient(cp, x)
        recovery_idx = np.argmax(dcp_dx) if len(dcp_dx) > 0 else 0
        wake_width_proxy = 1.0 - float(x[recovery_idx]) if len(x) > 0 else 0.0
        
        # Wake strength proxy
        wake_strength_proxy = float(np.max(dcp_dx)) if len(dcp_dx) > 0 else 0.0
        
        # Bluff body indicators
        drag_crisis_indicator = 0.0
        
        if cd > self.config.bluff_drag_threshold:
            drag_crisis_indicator += 0.3
        
        if pressure_drag_fraction > self.config.bluff_pressure_drag_fraction:
            drag_crisis_indicator += 0.3
        
        if wake_width_proxy > self.config.bluff_wake_width_threshold:
            drag_crisis_indicator += 0.2
        
        if wake_strength_proxy > 5.0:
            drag_crisis_indicator += 0.2
        
        drag_crisis_indicator = min(1.0, drag_crisis_indicator)
        
        pressure_drag_dominance = pressure_drag_fraction > self.config.bluff_pressure_drag_fraction
        
        # Overall bluff body assessment
        is_bluff_like = (
            cd > self.config.bluff_drag_threshold or
            drag_crisis_indicator > 0.5 or
            pressure_drag_dominance
        )
        
        # Confidence in assessment
        bluff_confidence = 0.5
        if cd > self.config.cd_max_absolute:
            bluff_confidence = 0.95
        elif cd > 0.3:
            bluff_confidence = 0.8
        elif is_bluff_like:
            bluff_confidence = 0.7
        
        # Detect violations
        violations = []
        if is_bluff_like:
            violations.append(PlausibilityViolationType.BLUFF_BODY_DETECTED)
        
        if cd > self.config.cd_max_absolute:
            violations.append(PlausibilityViolationType.DRAG_TOO_HIGH)
        
        return BluffBodyMetrics(
            pressure_drag_fraction=pressure_drag_fraction,
            viscous_drag_fraction=viscous_drag_fraction,
            wake_width_proxy=wake_width_proxy,
            wake_strength_proxy=wake_strength_proxy,
            base_cp=base_cp,
            base_pressure_coefficient=base_pressure_coefficient,
            drag_crisis_indicator=drag_crisis_indicator,
            pressure_drag_dominance=pressure_drag_dominance,
            is_bluff_like=is_bluff_like,
            bluff_confidence=bluff_confidence,
            violations=violations,
        )
    
    def govern(
        self,
        cl: float,
        cd: float,
        x: np.ndarray,
        cp: np.ndarray,
        cf: Optional[np.ndarray] = None,
        cm: Optional[float] = None,
        reynolds: float = 200000,
        mach: float = 0.1,
        aoa: float = 0.0,
        base_cp: Optional[float] = None,
    ) -> PlausibilityGovernanceReport:
        """
        Perform comprehensive aerodynamic plausibility governance.
        
        This is the main entry point for plausibility validation. It performs
        all checks and returns a comprehensive report.
        
        Args:
            cl: Lift coefficient
            cd: Drag coefficient
            x: Chordwise coordinates (normalized 0-1)
            cp: Pressure coefficient distribution
            cf: Skin friction coefficient distribution (optional)
            cm: Moment coefficient (optional)
            reynolds: Reynolds number
            mach: Mach number
            aoa: Angle of attack (degrees)
            base_cp: Base pressure coefficient (optional)
        
        Returns:
            PlausibilityGovernanceReport with comprehensive assessment
        """
        violations = []
        failure_reasons = []
        recommended_actions = []
        
        # 1. Check Reynolds and Mach number validity
        if reynolds < self.config.reynolds_min or reynolds > self.config.reynolds_max:
            violations.append(PlausibilityViolationType.REYNOLDS_NUMBER_MISMATCH)
            failure_reasons.append(
                f"Reynolds number {reynolds:.0f} outside valid range "
                f"[{self.config.reynolds_min:.0f}, {self.config.reynolds_max:.0f}]"
            )
        
        if mach > self.config.mach_max_incompressible:
            violations.append(PlausibilityViolationType.MACH_NUMBER_INVALID)
            failure_reasons.append(
                f"Mach number {mach:.3f} exceeds incompressible limit "
                f"{self.config.mach_max_incompressible}"
            )
        
        # 2. Analyze force coefficients
        force_metrics = self.analyze_force_coefficients(cl, cd, cm, reynolds, mach, aoa)
        violations.extend(force_metrics.violations)
        
        if force_metrics.violations:
            for v in force_metrics.violations:
                if v == PlausibilityViolationType.DRAG_TOO_HIGH:
                    failure_reasons.append(
                        f"Drag coefficient {cd:.4f} is unphysically high for low-Re airfoil"
                    )
                    recommended_actions.append(
                        "Reject design - likely bluff-body behavior or numerical artifact"
                    )
                elif v == PlausibilityViolationType.LIFT_DRAG_RATIO_TOO_LOW:
                    failure_reasons.append(
                        f"Lift-to-drag ratio {force_metrics.ld_ratio:.2f} is catastrophically low"
                    )
                    recommended_actions.append(
                        "Reject design - aerodynamically inefficient beyond physical limits"
                    )
                elif v == PlausibilityViolationType.LIFT_SIGN_INCORRECT:
                    failure_reasons.append(
                        f"Lift coefficient sign inconsistent with angle of attack {aoa:.1f}°"
                    )
        
        # 3. Analyze pressure recovery
        pressure_metrics = self.analyze_pressure_recovery(x, cp)
        violations.extend(pressure_metrics.violations)
        
        if pressure_metrics.violations:
            failure_reasons.append("Pathological pressure recovery detected")
            recommended_actions.append("Check for numerical issues or invalid geometry")
        
        # 4. Analyze separation (if Cf available)
        separation_metrics = None
        if cf is not None:
            separation_metrics = self.analyze_separation(x, cf, cp)
            violations.extend(separation_metrics.violations)
            
            if separation_metrics.violations:
                for v in separation_metrics.violations:
                    if v == PlausibilityViolationType.SEPARATION_FULL_CHORD:
                        failure_reasons.append(
                            f"Full-chord separation detected "
                            f"({separation_metrics.separated_length_fraction:.1%} separated)"
                        )
                        recommended_actions.append(
                            "Reject design - massive separation indicates invalid solution"
                        )
        
        # 5. Detect bluff body behavior
        bluff_metrics = self.detect_bluff_body(cl, cd, cp, x, base_cp)
        violations.extend(bluff_metrics.violations)
        
        if bluff_metrics.violations:
            failure_reasons.append(
                f"Bluff-body behavior detected (Cd={cd:.4f}, "
                f"pressure drag fraction={bluff_metrics.pressure_drag_fraction:.1%})"
            )
            recommended_actions.append(
                "Reject design - optimizer exploiting bluff-body drag mechanisms"
            )
        
        # Determine overall status
        unique_violations = list(set(violations))
        n_violations = len(unique_violations)
        
        if n_violations == 0:
            status = PlausibilityStatus.PHYSICALLY_PLAUSIBLE
            is_valid = True
            can_accept_solution = True
        elif n_violations >= self.config.suspect_threshold or self.config.strict_mode:
            status = PlausibilityStatus.PHYSICALLY_IMPOSSIBLE
            is_valid = False
            can_accept_solution = False
        else:
            status = PlausibilityStatus.AERODYNAMICALLY_SUSPECT
            is_valid = False
            can_accept_solution = False
        
        return PlausibilityGovernanceReport(
            status=status,
            forces=force_metrics,
            pressure_recovery=pressure_metrics,
            separation=separation_metrics,
            bluff_body=bluff_metrics,
            is_valid=is_valid,
            can_accept_solution=can_accept_solution,
            violations=unique_violations,
            failure_reasons=failure_reasons,
            recommended_actions=recommended_actions,
            reynolds_number=reynolds,
            mach_number=mach,
            angle_of_attack=aoa,
        )
    
    def quick_check(
        self,
        cl: float,
        cd: float,
        reynolds: float = 200000,
        aoa: float = 0.0,
    ) -> Tuple[bool, str]:
        """
        Perform a quick plausibility check without full analysis.
        
        This is useful for early rejection before expensive post-processing.
        
        Args:
            cl: Lift coefficient
            cd: Drag coefficient
            reynolds: Reynolds number
            aoa: Angle of attack (degrees)
        
        Returns:
            (is_plausible, reason) tuple
        """
        # Hard limits that should never be exceeded
        if cd > self.config.cd_max_absolute:
            return False, f"Cd={cd:.4f} exceeds absolute maximum {self.config.cd_max_absolute}"
        
        if cd < self.config.cd_min:
            return False, f"Cd={cd:.4f} below minimum {self.config.cd_min}"
        
        if abs(cl) > self.config.cl_max_absolute:
            return False, f"|Cl|={abs(cl):.2f} exceeds maximum {self.config.cl_max_absolute}"
        
        # L/D check for cruise conditions
        if abs(aoa) < 10 and cd > 1e-15:
            ld = cl / cd
            if ld < self.config.ld_min_acceptable:
                return False, f"L/D={ld:.2f} below minimum {self.config.ld_min_acceptable}"
        
        # Reynolds number check
        if reynolds < self.config.reynolds_min or reynolds > self.config.reynolds_max:
            return False, f"Re={reynolds:.0f} outside valid range"
        
        return True, "Quick check passed"