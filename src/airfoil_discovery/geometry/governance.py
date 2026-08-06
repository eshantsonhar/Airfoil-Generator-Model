"""
Comprehensive geometric governance framework for airfoil shape optimization.

Implements strict multi-layer geometry validation to prevent the optimizer
from producing physically absurd or non-manufacturing geometries. This is
a critical governance layer that must reject invalid geometries BEFORE
CFD execution.

The framework validates:
- Thickness constraints (absolute and relative)
- Leading edge radius limits
- Curvature continuity and smoothness
- Surface angle constraints
- Self-intersection detection
- CST coefficient reasonableness
- Geometric feasibility scoring
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import warnings


class GeometryValidityStatus(Enum):
    """Geometry validity classification."""
    GEOMETRIC_VALID = "GEOMETRIC_VALID"
    GEOMETRIC_SUSPECT = "GEOMETRIC_SUSPECT"
    GEOMETRIC_INVALID = "GEOMETRIC_INVALID"


class GeometryViolationType(Enum):
    """Types of geometry violations."""
    NONE = "NONE"
    THICKNESS_MIN_VIOLATION = "THICKNESS_MIN_VIOLATION"
    THICKNESS_MAX_VIOLATION = "THICKNESS_MAX_VIOLATION"
    THICKNESS_LOCATION_VIOLATION = "THICKNESS_LOCATION_VIOLATION"
    LE_RADIUS_TOO_LARGE = "LE_RADIUS_TOO_LARGE"
    LE_RADIUS_TOO_SMALL = "LE_RADIUS_TOO_SMALL"
    CURVATURE_SPIKE = "CURVATURE_SPIKE"
    CURVATURE_OSCILLATION = "CURVATURE_OSCILLATION"
    SURFACE_WAVINESS = "SURFACE_WAVINESS"
    SELF_INTERSECTION = "SELF_INTERSECTION"
    NEGATIVE_THICKNESS = "NEGATIVE_THICKNESS"
    CST_COEFFICIENT_OSCILLATION = "CST_COEFFICIENT_OSCILLATION"
    SURFACE_ANGLE_VIOLATION = "SURFACE_ANGLE_VIOLATION"
    BLUNT_AFT_BODY = "BLUNT_AFT_BODY"
    CUSP_LIKE_STRUCTURE = "CUSP_LIKE_STRUCTURE"
    MANIFOLD_DISTANCE_EXCEEDED = "MANIFOLD_DISTANCE_EXCEEDED"


@dataclass
class ThicknessMetrics:
    """Detailed thickness analysis metrics."""
    
    # Basic thickness
    max_thickness: float
    max_thickness_location: float
    min_thickness: float
    mean_thickness: float
    thickness_distribution: np.ndarray
    
    # Thickness gradients
    max_thickness_gradient: float
    thickness_gradient_rms: float
    
    # Thickness smoothness
    thickness_smoothness: float  # 0-1 scale
    
    # Constraint compliance
    within_absolute_bounds: bool
    within_relative_bounds: bool
    
    # Violations (must come after non-default fields)
    violations: List[GeometryViolationType] = field(default_factory=list)


@dataclass
class LeadingEdgeMetrics:
    """Leading edge geometry metrics."""
    
    # LE radius
    le_radius: float
    le_radius_estimate_method: str
    
    # LE curvature
    le_curvature: float
    le_curvature_max: float
    
    # LE wedge angle (for non-parabolic LE)
    le_wedge_angle: Optional[float]
    
    # LE curvature continuity
    le_curvature_continuity: float  # 0-1 scale
    
    # Constraint compliance
    within_bounds: bool
    
    # Violations
    violations: List[GeometryViolationType] = field(default_factory=list)


@dataclass
class CurvatureMetrics:
    """Surface curvature analysis metrics."""
    
    # Curvature statistics
    max_curvature: float
    mean_curvature: float
    rms_curvature: float
    curvature_std: float
    
    # Curvature derivatives
    max_curvature_derivative: float
    curvature_derivative_rms: float
    
    # Curvature spectral analysis
    curvature_spectral_energy: float
    high_frequency_energy_ratio: float
    
    # Smoothness metrics
    smoothness_score: float  # 0-1 scale
    waviness_detected: bool
    oscillation_detected: bool
    
    # Violations
    violations: List[GeometryViolationType] = field(default_factory=list)


@dataclass
class SurfaceAngleMetrics:
    """Surface angle and slope analysis metrics."""
    
    # Surface angles
    max_surface_angle: float
    mean_surface_angle: float
    
    # Slope analysis
    max_slope: float
    max_slope_location: float
    
    # Monotonicity
    chordwise_monotonic: bool
    monotonicity_violations: int
    
    # Signed area consistency
    signed_area: float
    area_consistency: bool
    
    # Violations
    violations: List[GeometryViolationType] = field(default_factory=list)


@dataclass
class CSTCoefficientMetrics:
    """CST coefficient analysis metrics."""
    
    # Coefficient statistics
    upper_coefficients: np.ndarray
    lower_coefficients: np.ndarray
    
    # Coefficient smoothness
    coefficient_oscillation_index: float
    coefficient_decay_rate: float
    
    # Coefficient reasonableness
    coefficients_bounded: bool
    coefficient_pattern_valid: bool
    
    # Violations
    violations: List[GeometryViolationType] = field(default_factory=list)


@dataclass
class GeometryGovernanceReport:
    """Comprehensive geometry governance report."""
    
    # Overall status
    status: GeometryValidityStatus
    
    # Overall assessment (required fields must come before optional)
    is_valid: bool
    can_proceed_to_cfd: bool
    
    # Component metrics
    thickness: Optional[ThicknessMetrics] = None
    leading_edge: Optional[LeadingEdgeMetrics] = None
    curvature: Optional[CurvatureMetrics] = None
    surface_angles: Optional[SurfaceAngleMetrics] = None
    cst_coefficients: Optional[CSTCoefficientMetrics] = None
    
    # Manifold analysis
    manifold_distance: Optional[float] = None
    manifold_outlier_score: Optional[float] = None
    
    # Violations
    violations: List[GeometryViolationType] = field(default_factory=list)
    failure_reasons: List[str] = field(default_factory=list)
    
    # Recommendations
    recommended_actions: List[str] = field(default_factory=list)
    
    # Design variable diagnostics
    design_variable_analysis: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "is_valid": self.is_valid,
            "can_proceed_to_cfd": self.can_proceed_to_cfd,
            "violations": [v.value for v in self.violations],
            "failure_reasons": self.failure_reasons,
            "recommended_actions": self.recommended_actions,
            "thickness": {
                "max_thickness": self.thickness.max_thickness if self.thickness else None,
                "max_thickness_location": self.thickness.max_thickness_location if self.thickness else None,
                "within_bounds": self.thickness.within_absolute_bounds if self.thickness else None,
            } if self.thickness else None,
            "leading_edge": {
                "le_radius": self.leading_edge.le_radius if self.leading_edge else None,
                "within_bounds": self.leading_edge.within_bounds if self.leading_edge else None,
            } if self.leading_edge else None,
            "manifold_distance": self.manifold_distance,
            "manifold_outlier_score": self.manifold_outlier_score,
        }


@dataclass
class GeometryGovernanceConfig:
    """Configuration for geometry governance."""
    
    # Thickness constraints (low-Re airfoils)
    thickness_min: float = 0.06  # 6% minimum
    thickness_max: float = 0.18  # 18% maximum
    thickness_location_min: float = 0.2  # Max thickness should be after 20% chord
    thickness_location_max: float = 0.5  # Max thickness should be before 50% chord
    thickness_gradient_max: float = 0.5  # Maximum thickness gradient
    
    # Leading edge radius constraints
    le_radius_min: float = 0.005  # Minimum LE radius (sharp but not singular)
    le_radius_max: float = 0.04   # Maximum LE radius (not bluff-body-like)
    
    # Curvature constraints
    max_curvature: float = 50.0  # Maximum absolute curvature
    curvature_derivative_max: float = 100.0  # Maximum curvature derivative
    high_frequency_energy_max: float = 0.1  # Maximum high-frequency energy ratio
    
    # Surface angle constraints
    max_surface_angle_deg: float = 85.0  # Maximum surface angle (degrees)
    min_monotonicity_fraction: float = 0.95  # Minimum fraction of monotonic progression
    
    # CST coefficient constraints
    max_coefficient_magnitude: float = 2.0  # Maximum CST coefficient magnitude
    coefficient_oscillation_max: float = 0.3  # Maximum coefficient oscillation index
    
    # Manifold constraints
    manifold_distance_threshold: float = 3.0  # Maximum distance from realistic manifold
    manifold_outlier_threshold: float = 0.95  # Outlier score threshold
    
    # Smoothness constraints
    min_thickness_smoothness: float = 0.5  # Minimum thickness smoothness score
    min_curvature_smoothness: float = 0.5  # Minimum curvature smoothness score
    
    # Governance policy
    strict_mode: bool = True  # If True, any violation rejects geometry
    suspect_threshold: int = 2  # Number of suspect violations before rejection


class GeometryGovernor:
    """
    Comprehensive geometry governance for airfoil optimization.
    
    This class implements a strict multi-layer validation system that
    prevents the optimizer from producing physically absurd geometries.
    All geometries must pass validation BEFORE CFD execution.
    
    The governance system is designed to prevent:
    - 28% thickness low-Re airfoils
    - Enormous leading-edge radii
    - Bluff-body-like geometries
    - Pathological pressure recovery
    - Self-intersecting surfaces
    - Oscillatory/wavy surfaces
    - Non-manufacturable shapes
    """
    
    def __init__(self, config: Optional[GeometryGovernanceConfig] = None):
        """
        Initialize geometry governor.
        
        Args:
            config: Governance configuration. Uses defaults if None.
        """
        self.config = config or GeometryGovernanceConfig()
        
        # Manifold model (to be initialized with airfoil data)
        self._manifold_model = None
        self._manifold_scaler = None
        
    def set_manifold_model(self, model: Any, scaler: Any) -> None:
        """
        Set the manifold model for realistic airfoil validation.
        
        Args:
            model: PCA or similar model trained on realistic airfoils
            scaler: Scaler for the model
        """
        self._manifold_model = model
        self._manifold_scaler = scaler
    
    def analyze_thickness(
        self,
        x: np.ndarray,
        yu: np.ndarray,
        yl: np.ndarray,
    ) -> ThicknessMetrics:
        """
        Analyze thickness distribution.
        
        Args:
            x: Chordwise coordinates (normalized 0-1)
            yu: Upper surface y-coordinates
            yl: Lower surface y-coordinates
        
        Returns:
            ThicknessMetrics with detailed analysis
        """
        thickness = yu - yl
        
        # Basic statistics
        max_thickness = float(np.max(thickness))
        max_thickness_idx = int(np.argmax(thickness))
        max_thickness_location = float(x[max_thickness_idx])
        min_thickness = float(np.min(thickness))
        mean_thickness = float(np.mean(thickness))
        
        # Thickness gradients
        thickness_gradient = np.gradient(thickness, x)
        max_thickness_gradient = float(np.max(np.abs(thickness_gradient)))
        thickness_gradient_rms = float(np.sqrt(np.mean(thickness_gradient**2)))
        
        # Thickness smoothness (based on gradient consistency)
        if len(thickness_gradient) > 2:
            gradient_variation = np.std(np.diff(thickness_gradient))
            thickness_smoothness = float(np.exp(-gradient_variation))
        else:
            thickness_smoothness = 1.0
        
        # Constraint compliance
        within_absolute = (
            self.config.thickness_min <= max_thickness <= self.config.thickness_max
        )
        within_relative = (
            self.config.thickness_location_min <= max_thickness_location <= 
            self.config.thickness_location_max
        )
        
        # Detect violations
        violations = []
        if max_thickness < self.config.thickness_min:
            violations.append(GeometryViolationType.THICKNESS_MIN_VIOLATION)
        if max_thickness > self.config.thickness_max:
            violations.append(GeometryViolationType.THICKNESS_MAX_VIOLATION)
        if not within_relative:
            violations.append(GeometryViolationType.THICKNESS_LOCATION_VIOLATION)
        if min_thickness <= 0:
            violations.append(GeometryViolationType.NEGATIVE_THICKNESS)
        
        return ThicknessMetrics(
            max_thickness=max_thickness,
            max_thickness_location=max_thickness_location,
            min_thickness=min_thickness,
            mean_thickness=mean_thickness,
            thickness_distribution=thickness,
            max_thickness_gradient=max_thickness_gradient,
            thickness_gradient_rms=thickness_gradient_rms,
            thickness_smoothness=thickness_smoothness,
            within_absolute_bounds=within_absolute,
            within_relative_bounds=within_relative,
            violations=violations,
        )
    
    def analyze_leading_edge(
        self,
        x: np.ndarray,
        yu: np.ndarray,
        yl: np.ndarray,
    ) -> LeadingEdgeMetrics:
        """
        Analyze leading edge geometry.
        
        Uses the parabolic LE behavior of CST airfoils:
        y_t ~ sqrt(2 * R * x) near the leading edge.
        
        Args:
            x: Chordwise coordinates (normalized 0-1)
            yu: Upper surface y-coordinates
            yl: Lower surface y-coordinates
        
        Returns:
            LeadingEdgeMetrics with detailed analysis
        """
        half_thickness = 0.5 * (yu - yl)
        
        # Use first few points for LE radius estimation
        sample_end = min(8, len(x))
        x_local = np.clip(x[1:sample_end], 1e-8, None)
        y_local = np.clip(half_thickness[1:sample_end], 0.0, None)
        
        if len(x_local) == 0 or np.all(y_local == 0):
            return LeadingEdgeMetrics(
                le_radius=0.0,
                le_radius_estimate_method="none",
                le_curvature=0.0,
                le_curvature_max=0.0,
                le_wedge_angle=None,
                le_curvature_continuity=0.0,
                within_bounds=False,
                violations=[GeometryViolationType.LE_RADIUS_TOO_SMALL],
            )
        
        # Estimate radius from y^2 / (2x) relationship
        radii = (y_local**2) / (2.0 * x_local)
        le_radius = float(np.mean(radii))
        
        # Alternative: quadratic fit
        try:
            coeffs = np.polyfit(x_local, y_local**2, 1)
            le_radius_fit = float(coeffs[0] / 2.0)
            if le_radius_fit > 0:
                le_radius = le_radius_fit
        except (np.linalg.LinAlgError, ValueError) as e:
            warnings.warn(f"LE radius quadratic fit failed, using y^2/(2x) estimate: {e}")
        
        # LE curvature (inverse of radius for circular approximation)
        le_curvature = 1.0 / (le_radius + 1e-15)
        
        # Maximum curvature near LE
        dx = np.diff(x[:5])
        dy = np.diff(yu[:5])
        if len(dx) > 0 and np.any(dx > 0):
            slopes = dy / (dx + 1e-15)
            ds = np.sqrt(dx**2 + dy**2)
            d_slope = np.diff(slopes)
            if len(d_slope) > 0 and len(ds) > 1:
                curvature_vals = np.abs(d_slope) / (ds[:-1] + 1e-15)
                le_curvature_max = float(np.max(curvature_vals))
            else:
                le_curvature_max = le_curvature
        else:
            le_curvature_max = le_curvature
        
        # LE curvature continuity (how well the LE blends with the rest)
        # Compare LE curvature with mid-chord curvature
        mid_idx = len(x) // 2
        if mid_idx > 2:
            mid_dx = np.diff(x[mid_idx-2:mid_idx+2])
            mid_dy = np.diff(yu[mid_idx-2:mid_idx+2])
            if len(mid_dx) > 0 and np.any(mid_dx > 0):
                mid_slopes = mid_dy / (mid_dx + 1e-15)
                mid_ds = np.sqrt(mid_dx**2 + mid_dy**2)
                mid_d_slope = np.diff(mid_slopes)
                if len(mid_d_slope) > 0 and len(mid_ds) > 1:
                    mid_curvature = float(np.mean(np.abs(mid_d_slope) / (mid_ds[:-1] + 1e-15)))
                    curvature_ratio = min(le_curvature_max, mid_curvature) / (max(le_curvature_max, mid_curvature) + 1e-15)
                    le_curvature_continuity = curvature_ratio
                else:
                    le_curvature_continuity = 0.5
            else:
                le_curvature_continuity = 0.5
        else:
            le_curvature_continuity = 0.5
        
        # Check bounds
        within_bounds = self.config.le_radius_min <= le_radius <= self.config.le_radius_max
        
        # Detect violations
        violations = []
        if le_radius > self.config.le_radius_max:
            violations.append(GeometryViolationType.LE_RADIUS_TOO_LARGE)
        if le_radius < self.config.le_radius_min:
            violations.append(GeometryViolationType.LE_RADIUS_TOO_SMALL)
        
        return LeadingEdgeMetrics(
            le_radius=le_radius,
            le_radius_estimate_method="parabolic_fit",
            le_curvature=le_curvature,
            le_curvature_max=le_curvature_max,
            le_wedge_angle=None,  # Would require more detailed analysis
            le_curvature_continuity=le_curvature_continuity,
            within_bounds=within_bounds,
            violations=violations,
        )
    
    def analyze_curvature(
        self,
        x: np.ndarray,
        yu: np.ndarray,
        yl: np.ndarray,
    ) -> CurvatureMetrics:
        """
        Analyze surface curvature for smoothness and continuity.
        
        Args:
            x: Chordwise coordinates (normalized 0-1)
            yu: Upper surface y-coordinates
            yl: Lower surface y-coordinates
        
        Returns:
            CurvatureMetrics with detailed analysis
        """
        # Compute curvature for upper surface
        dy_dx = np.gradient(yu, x, edge_order=2)
        d2y_dx2 = np.gradient(dy_dx, x, edge_order=2)
        curvature_upper = np.abs(d2y_dx2) / np.power(1.0 + dy_dx**2, 1.5)
        
        # Curvature statistics
        max_curvature = float(np.max(curvature_upper))
        mean_curvature = float(np.mean(curvature_upper))
        rms_curvature = float(np.sqrt(np.mean(curvature_upper**2)))
        curvature_std = float(np.std(curvature_upper))
        
        # Curvature derivatives
        curvature_derivative = np.gradient(curvature_upper, x)
        max_curvature_derivative = float(np.max(np.abs(curvature_derivative)))
        curvature_derivative_rms = float(np.sqrt(np.mean(curvature_derivative**2)))
        
        # Spectral analysis (detect high-frequency oscillations)
        if len(curvature_upper) > 8:
            # FFT of curvature
            curvature_fft = np.fft.fft(curvature_upper - np.mean(curvature_upper))
            power_spectrum = np.abs(curvature_fft[:len(curvature_fft)//2])**2
            total_energy = float(np.sum(power_spectrum[1:]))  # Exclude DC
            
            if total_energy > 1e-15:
                # High-frequency energy (upper 25% of spectrum)
                high_freq_start = len(power_spectrum) * 3 // 4
                high_freq_energy = float(np.sum(power_spectrum[high_freq_start:]))
                high_frequency_energy_ratio = high_freq_energy / total_energy
            else:
                high_frequency_energy_ratio = 0.0
        else:
            high_frequency_energy_ratio = 0.0
        
        # Smoothness score (based on curvature variation)
        if len(curvature_upper) > 2:
            curvature_variation = np.std(np.diff(curvature_upper))
            smoothness_score = float(np.exp(-curvature_variation))
        else:
            smoothness_score = 1.0
        
        # Detect waviness and oscillations
        waviness_detected = high_frequency_energy_ratio > self.config.high_frequency_energy_max
        oscillation_detected = False
        
        # Check for curvature sign changes (oscillations)
        if len(curvature_derivative) > 2:
            sign_changes = np.sum(np.diff(np.sign(curvature_derivative)) != 0)
            if sign_changes > len(curvature_derivative) * 0.3:
                oscillation_detected = True
        
        # Detect violations
        violations = []
        if max_curvature > self.config.max_curvature:
            violations.append(GeometryViolationType.CURVATURE_SPIKE)
        if oscillation_detected:
            violations.append(GeometryViolationType.CURVATURE_OSCILLATION)
        if waviness_detected:
            violations.append(GeometryViolationType.SURFACE_WAVINESS)
        
        return CurvatureMetrics(
            max_curvature=max_curvature,
            mean_curvature=mean_curvature,
            rms_curvature=rms_curvature,
            curvature_std=curvature_std,
            max_curvature_derivative=max_curvature_derivative,
            curvature_derivative_rms=curvature_derivative_rms,
            curvature_spectral_energy=float(np.sum(power_spectrum[1:])) if len(curvature_upper) > 8 else 0.0,
            high_frequency_energy_ratio=high_frequency_energy_ratio,
            smoothness_score=smoothness_score,
            waviness_detected=waviness_detected,
            oscillation_detected=oscillation_detected,
            violations=violations,
        )
    
    def analyze_surface_angles(
        self,
        x: np.ndarray,
        yu: np.ndarray,
        yl: np.ndarray,
    ) -> SurfaceAngleMetrics:
        """
        Analyze surface angles and slope constraints.
        
        Args:
            x: Chordwise coordinates (normalized 0-1)
            yu: Upper surface y-coordinates
            yl: Lower surface y-coordinates
        
        Returns:
            SurfaceAngleMetrics with detailed analysis
        """
        # Compute slopes
        dy_dx = np.gradient(yu, x)
        slopes = np.arctan(dy_dx)
        surface_angles_deg = np.degrees(slopes)
        
        max_surface_angle = float(np.max(np.abs(surface_angles_deg)))
        mean_surface_angle = float(np.mean(np.abs(surface_angles_deg)))
        
        max_slope = float(np.max(np.abs(dy_dx)))
        max_slope_idx = int(np.argmax(np.abs(dy_dx)))
        max_slope_location = float(x[max_slope_idx])
        
        # Check chordwise monotonicity
        dx = np.diff(x)
        monotonic_violations = int(np.sum(dx <= 0))
        chordwise_monotonic = monotonic_violations == 0
        monotonicity_fraction = 1.0 - monotonic_violations / max(len(dx), 1)
        
        # Signed area consistency (check for self-intersection tendencies)
        # Upper surface should have positive area, lower should have negative
        # Use np.trapezoid for NumPy 2.0+ compatibility, fallback to np.trapz
        if hasattr(np, 'trapezoid'):
            upper_area = float(np.trapezoid(yu, x))
            lower_area = float(np.trapezoid(yl, x))
        else:
            upper_area = float(np.trapz(yu, x))
            lower_area = float(np.trapz(yl, x))
        signed_area = upper_area - lower_area
        area_consistency = signed_area > 0  # Upper should be above lower
        
        # Detect violations
        violations = []
        if max_surface_angle > self.config.max_surface_angle_deg:
            violations.append(GeometryViolationType.SURFACE_ANGLE_VIOLATION)
        if not chordwise_monotonic or monotonicity_fraction < self.config.min_monotonicity_fraction:
            violations.append(GeometryViolationType.SURFACE_ANGLE_VIOLATION)
        
        # Check for blunt aft body (sudden change in slope near trailing edge)
        aft_idx = int(0.8 * len(x))
        if aft_idx < len(x) - 2:
            aft_slopes = dy_dx[aft_idx:]
            aft_slope_variation = float(np.std(aft_slopes))
            if aft_slope_variation > 2.0:
                violations.append(GeometryViolationType.BLUNT_AFT_BODY)
        
        return SurfaceAngleMetrics(
            max_surface_angle=max_surface_angle,
            mean_surface_angle=mean_surface_angle,
            max_slope=max_slope,
            max_slope_location=max_slope_location,
            chordwise_monotonic=chordwise_monotonic,
            monotonicity_violations=monotonic_violations,
            signed_area=signed_area,
            area_consistency=area_consistency,
            violations=violations,
        )
    
    def analyze_cst_coefficients(
        self,
        upper_coeffs: np.ndarray,
        lower_coeffs: np.ndarray,
    ) -> CSTCoefficientMetrics:
        """
        Analyze CST coefficients for reasonableness.
        
        Args:
            upper_coeffs: Upper surface CST coefficients
            lower_coeffs: Lower surface CST coefficients
        
        Returns:
            CSTCoefficientMetrics with detailed analysis
        """
        # Check coefficient magnitudes
        all_coeffs = np.concatenate([upper_coeffs, lower_coeffs])
        coefficients_bounded = bool(np.all(np.abs(all_coeffs) <= self.config.max_coefficient_magnitude))
        
        # Coefficient oscillation index
        # Measure how much coefficients oscillate (high oscillation = bad)
        if len(upper_coeffs) > 2:
            upper_diff = np.diff(upper_coeffs)
            upper_sign_changes = np.sum(np.diff(np.sign(upper_diff)) != 0)
            upper_oscillation = upper_sign_changes / (len(upper_coeffs) - 2)
        else:
            upper_oscillation = 0.0
        
        if len(lower_coeffs) > 2:
            lower_diff = np.diff(lower_coeffs)
            lower_sign_changes = np.sum(np.diff(np.sign(lower_diff)) != 0)
            lower_oscillation = lower_sign_changes / (len(lower_coeffs) - 2)
        else:
            lower_oscillation = 0.0
        
        coefficient_oscillation_index = max(upper_oscillation, lower_oscillation)
        
        # Coefficient decay rate (higher-order coefficients should generally be smaller)
        if len(upper_coeffs) > 3:
            # Compare first half magnitude with second half
            mid = len(upper_coeffs) // 2
            first_half_mag = np.mean(np.abs(upper_coeffs[:mid]))
            second_half_mag = np.mean(np.abs(upper_coeffs[mid:]))
            if first_half_mag > 1e-15:
                coefficient_decay_rate = second_half_mag / first_half_mag
            else:
                coefficient_decay_rate = 0.0
        else:
            coefficient_decay_rate = 0.0
        
        # Pattern validity (coefficients should follow a reasonable pattern)
        coefficient_pattern_valid = (
            coefficient_oscillation_index < self.config.coefficient_oscillation_max and
            coefficients_bounded
        )
        
        # Detect violations
        violations = []
        if not coefficients_bounded:
            violations.append(GeometryViolationType.CST_COEFFICIENT_OSCILLATION)
        if coefficient_oscillation_index > self.config.coefficient_oscillation_max:
            violations.append(GeometryViolationType.CST_COEFFICIENT_OSCILLATION)
        
        return CSTCoefficientMetrics(
            upper_coefficients=upper_coeffs,
            lower_coefficients=lower_coeffs,
            coefficient_oscillation_index=coefficient_oscillation_index,
            coefficient_decay_rate=coefficient_decay_rate,
            coefficients_bounded=coefficients_bounded,
            coefficient_pattern_valid=coefficient_pattern_valid,
            violations=violations,
        )
    
    def check_manifold_distance(
        self,
        x: np.ndarray,
        yu: np.ndarray,
        yl: np.ndarray,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Check distance from realistic airfoil manifold.
        
        Args:
            x: Chordwise coordinates (normalized 0-1)
            yu: Upper surface y-coordinates
            yl: Lower surface y-coordinates
        
        Returns:
            (manifold_distance, outlier_score) or (None, None) if model unavailable
        """
        if self._manifold_model is None or self._manifold_scaler is None:
            return None, None
        
        try:
            # Construct airfoil feature vector
            # Use both upper and lower surface coordinates
            features = np.concatenate([yu, yl])
            
            # Scale features
            features_scaled = self._manifold_scaler.transform(features.reshape(1, -1))
            
            # Transform to latent space
            latent = self._manifold_model.transform(features_scaled)
            
            # Reconstruct
            reconstructed = self._manifold_model.inverse_transform(latent)
            
            # Compute reconstruction error (distance from manifold)
            reconstruction_error = float(np.sqrt(np.mean((features_scaled - reconstructed)**2)))
            
            # Compute outlier score (e.g., based on latent space distance from mean)
            outlier_score = float(np.sqrt(np.sum(latent**2)))
            
            return reconstruction_error, outlier_score
            
        except Exception as e:
            warnings.warn(f"Manifold distance computation failed: {e}")
            return None, None
    
    def check_self_intersection(
        self,
        x: np.ndarray,
        yu: np.ndarray,
        yl: np.ndarray,
    ) -> bool:
        """
        Check for self-intersection of upper and lower surfaces.
        
        Args:
            x: Chordwise coordinates (normalized 0-1)
            yu: Upper surface y-coordinates
            yl: Lower surface y-coordinates
        
        Returns:
            True if self-intersection detected
        """
        # Handle case where yu and yl have different lengths
        # Interpolate to common x-grid for comparison
        if len(yu) != len(yl):
            # Use the original full arrays for thickness check
            # The caller should pass consistent arrays
            # For safety, we'll check using the minimum length
            min_len = min(len(yu), len(yl))
            yu_check = yu[:min_len]
            yl_check = yl[:min_len]
            thickness = yu_check - yl_check
            return bool(np.any(thickness <= 0))
        
        # Upper surface should always be above lower surface
        thickness = yu - yl
        return bool(np.any(thickness <= 0))
    
    def govern(
        self,
        x: np.ndarray,
        yu: np.ndarray,
        yl: np.ndarray,
        upper_coeffs: Optional[np.ndarray] = None,
        lower_coeffs: Optional[np.ndarray] = None,
    ) -> GeometryGovernanceReport:
        """
        Perform comprehensive geometry governance.
        
        This is the main entry point for geometry validation. It performs
        all checks and returns a comprehensive report.
        
        Args:
            x: Chordwise coordinates (normalized 0-1)
            yu: Upper surface y-coordinates
            yl: Lower surface y-coordinates
            upper_coeffs: Upper surface CST coefficients (optional)
            lower_coeffs: Lower surface CST coefficients (optional)
        
        Returns:
            GeometryGovernanceReport with comprehensive assessment
        """
        violations = []
        failure_reasons = []
        recommended_actions = []
        
        # 1. Check self-intersection (hard fail)
        if self.check_self_intersection(x, yu, yl):
            violations.append(GeometryViolationType.SELF_INTERSECTION)
            failure_reasons.append("Self-intersection detected: upper and lower surfaces cross")
            recommended_actions.append("Reduce CST coefficient magnitudes or adjust thickness distribution")
        
        # 2. Analyze thickness
        thickness_metrics = self.analyze_thickness(x, yu, yl)
        violations.extend(thickness_metrics.violations)
        
        if thickness_metrics.violations:
            for v in thickness_metrics.violations:
                if v == GeometryViolationType.THICKNESS_MAX_VIOLATION:
                    failure_reasons.append(
                        f"Maximum thickness {thickness_metrics.max_thickness:.3f} exceeds limit "
                        f"{self.config.thickness_max:.3f}"
                    )
                elif v == GeometryViolationType.THICKNESS_MIN_VIOLATION:
                    failure_reasons.append(
                        f"Maximum thickness {thickness_metrics.max_thickness:.3f} below minimum "
                        f"{self.config.thickness_min:.3f}"
                    )
                elif v == GeometryViolationType.NEGATIVE_THICKNESS:
                    failure_reasons.append("Negative thickness detected (self-intersection)")
            
            recommended_actions.append("Adjust thickness distribution constraints")
        
        # 3. Analyze leading edge
        le_metrics = self.analyze_leading_edge(x, yu, yl)
        violations.extend(le_metrics.violations)
        
        if le_metrics.violations:
            for v in le_metrics.violations:
                if v == GeometryViolationType.LE_RADIUS_TOO_LARGE:
                    failure_reasons.append(
                        f"Leading edge radius {le_metrics.le_radius:.4f} exceeds maximum "
                        f"{self.config.le_radius_max:.4f} (bluff-body-like)"
                    )
                elif v == GeometryViolationType.LE_RADIUS_TOO_SMALL:
                    failure_reasons.append(
                        f"Leading edge radius {le_metrics.le_radius:.4f} below minimum "
                        f"{self.config.le_radius_min:.4f} (near-singular)"
                    )
            recommended_actions.append("Adjust leading edge CST coefficients (typically first 2-3)")
        
        # 4. Analyze curvature
        curvature_metrics = self.analyze_curvature(x, yu, yl)
        violations.extend(curvature_metrics.violations)
        
        if curvature_metrics.violations:
            for v in curvature_metrics.violations:
                if v == GeometryViolationType.CURVATURE_SPIKE:
                    failure_reasons.append(
                        f"Excessive curvature {curvature_metrics.max_curvature:.1f} detected"
                    )
                elif v == GeometryViolationType.CURVATURE_OSCILLATION:
                    failure_reasons.append("Curvature oscillation detected (wavy surface)")
                elif v == GeometryViolationType.SURFACE_WAVINESS:
                    failure_reasons.append("High-frequency surface waviness detected")
            recommended_actions.append("Smooth CST coefficients or reduce high-order terms")
        
        # 5. Analyze surface angles
        angle_metrics = self.analyze_surface_angles(x, yu, yl)
        violations.extend(angle_metrics.violations)
        
        if angle_metrics.violations:
            for v in angle_metrics.violations:
                if v == GeometryViolationType.SURFACE_ANGLE_VIOLATION:
                    failure_reasons.append(
                        f"Surface angle {angle_metrics.max_surface_angle:.1f}° exceeds limit "
                        f"{self.config.max_surface_angle_deg:.1f}°"
                    )
                elif v == GeometryViolationType.BLUNT_AFT_BODY:
                    failure_reasons.append("Blunt aft body detected (excessive slope variation)")
            recommended_actions.append("Adjust aft section CST coefficients")
        
        # 6. Analyze CST coefficients (if provided)
        cst_metrics = None
        if upper_coeffs is not None and lower_coeffs is not None:
            cst_metrics = self.analyze_cst_coefficients(upper_coeffs, lower_coeffs)
            violations.extend(cst_metrics.violations)
            
            if cst_metrics.violations:
                failure_reasons.append("CST coefficient pattern is unreasonable")
                recommended_actions.append("Apply coefficient bounds or smoothing constraints")
        
        # 7. Check manifold distance
        manifold_distance, outlier_score = self.check_manifold_distance(x, yu, yl)
        
        if manifold_distance is not None and manifold_distance > self.config.manifold_distance_threshold:
            violations.append(GeometryViolationType.MANIFOLD_DISTANCE_EXCEEDED)
            failure_reasons.append(
                f"Geometry is too far from realistic airfoil manifold "
                f"(distance: {manifold_distance:.2f})"
            )
            recommended_actions.append("Move design closer to known airfoil shapes")
        
        # Determine overall status
        unique_violations = list(set(violations))
        n_violations = len(unique_violations)
        
        if n_violations == 0:
            status = GeometryValidityStatus.GEOMETRIC_VALID
            is_valid = True
            can_proceed_to_cfd = True
        elif n_violations >= self.config.suspect_threshold or self.config.strict_mode:
            status = GeometryValidityStatus.GEOMETRIC_INVALID
            is_valid = False
            can_proceed_to_cfd = False
        else:
            status = GeometryValidityStatus.GEOMETRIC_SUSPECT
            is_valid = False
            can_proceed_to_cfd = False  # Suspect geometries also don't proceed
        
        return GeometryGovernanceReport(
            status=status,
            thickness=thickness_metrics,
            leading_edge=le_metrics,
            curvature=curvature_metrics,
            surface_angles=angle_metrics,
            cst_coefficients=cst_metrics,
            manifold_distance=manifold_distance,
            manifold_outlier_score=outlier_score,
            is_valid=is_valid,
            can_proceed_to_cfd=can_proceed_to_cfd,
            violations=unique_violations,
            failure_reasons=failure_reasons,
            recommended_actions=recommended_actions,
            design_variable_analysis={
                "x_range": [float(x[0]), float(x[-1])],
                "yu_range": [float(np.min(yu)), float(np.max(yu))],
                "yl_range": [float(np.min(yl)), float(np.max(yl))],
                "n_points": len(x),
            },
        )
    
    def validate_design_variables(
        self,
        design_vars: np.ndarray,
        cst_airfoil: Any,
        params: Any,
    ) -> GeometryGovernanceReport:
        """
        Validate geometry from design variables using CST parameterization.
        
        This is a convenience method that generates coordinates from design
        variables and then performs governance checks.
        
        Args:
            design_vars: Design variables (CST coefficients)
            cst_airfoil: CSTAirfoil instance
            params: CSTParameters instance
        
        Returns:
            GeometryGovernanceReport with comprehensive assessment
        """
        # Generate coordinates
        x, yu, yl = cst_airfoil.coordinates(params)
        
        # Split design variables into upper and lower coefficients
        n_coeffs = len(design_vars) // 2
        upper_coeffs = design_vars[:n_coeffs]
        lower_coeffs = design_vars[n_coeffs:]
        
        # Perform governance
        return self.govern(x, yu, yl, upper_coeffs, lower_coeffs)