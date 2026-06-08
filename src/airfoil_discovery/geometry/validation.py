"""
Comprehensive airfoil geometry validation system.

Provides rigorous validation of airfoil geometries to prevent:
- Self-intersecting surfaces
- Negative thickness regions
- Invalid leading edge radii
- Curvature discontinuities
- Non-manifold geometries
- Oscillatory surfaces

This module enforces strict geometric constraints to ensure all
airfoils sent to CFD are physically realizable and numerically stable.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
from pathlib import Path


class GeometryValidationStatus(Enum):
    """Status of geometry validation."""
    VALID = "VALID"
    INVALID = "INVALID"
    WARNING = "WARNING"
    UNCHECKED = "UNCHECKED"


class GeometryViolationType(Enum):
    """Types of geometry violations."""
    NONE = "NONE"
    SELF_INTERSECTION = "SELF_INTERSECTION"
    NEGATIVE_THICKNESS = "NEGATIVE_THICKNESS"
    ZERO_THICKNESS = "ZERO_THICKNESS"
    INVALID_LE_RADIUS = "INVALID_LE_RADIUS"
    INVALID_TE_THICKNESS = "INVALID_TE_THICKNESS"
    CURVATURE_DISCONTINUITY = "CURVATURE_DISCONTINUITY"
    CURVATURE_SPIKE = "CURVATURE_SPIKE"
    NON_MONOTONIC_X = "NON_MONOTONIC_X"
    SURFACE_CROSSING = "SURFACE_CROSSING"
    INFINITE_COORDINATES = "INFINITE_COORDINATES"
    NaN_COORDINATES = "NaN_COORDINATES"
    INSUFFICIENT_POINTS = "INSUFFICIENT_POINTS"
    DUPLICATE_POINTS = "DUPLICATE_POINTS"
    THICKNESS_OUT_OF_BOUNDS = "THICKNESS_OUT_OF_BOUNDS"
    CAMBER_OUT_OF_BOUNDS = "CAMBER_OUT_OF_BOUNDS"
    OSCILLATORY_SURFACE = "OSCILLATORY_SURFACE"
    CUSP_EXPLOSION = "CUSP_EXPLOSION"
    FOLDED_GEOMETRY = "FOLDED_GEOMETRY"
    HOOK_GEOMETRY = "HOOK_GEOMETRY"
    RAZOR_LE = "RAZOR_LE"
    DEGENERATE_TE = "DEGENERATE_TE"


@dataclass
class GeometryValidationResult:
    """Result of geometry validation."""
    
    # Overall status
    status: GeometryValidationStatus
    
    # Coordinates
    coordinates: np.ndarray  # Full airfoil coordinates (N, 2)
    
    # Violations detected
    violations: List[GeometryViolationType] = field(default_factory=list)
    
    # Detailed metrics
    min_thickness: float = 0.0
    max_thickness: float = 0.0
    mean_thickness: float = 0.0
    thickness_distribution: Optional[np.ndarray] = None
    
    # Leading edge
    le_radius: float = 0.0
    le_location: Tuple[float, float] = (0.0, 0.0)
    
    # Trailing edge
    te_thickness: float = 0.0
    te_upper: Tuple[float, float] = (1.0, 0.0)
    te_lower: Tuple[float, float] = (1.0, 0.0)
    
    # Camber
    max_camber: float = 0.0
    max_camber_location: float = 0.0
    
    # Curvature
    max_curvature: float = 0.0
    max_curvature_location: float = 0.0
    curvature_std: float = 0.0
    
    # Messages
    failure_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Validity
    is_valid: bool = False
    can_proceed_to_cfd: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "is_valid": self.is_valid,
            "can_proceed_to_cfd": self.can_proceed_to_cfd,
            "violations": [v.value for v in self.violations],
            "failure_reasons": self.failure_reasons,
            "warnings": self.warnings,
            "min_thickness": self.min_thickness,
            "max_thickness": self.max_thickness,
            "le_radius": self.le_radius,
            "te_thickness": self.te_thickness,
            "max_camber": self.max_camber,
            "max_curvature": self.max_curvature,
        }


@dataclass
class GeometryValidationConfig:
    """Configuration for geometry validation."""
    
    # Thickness constraints
    min_thickness: float = 1e-6  # Minimum local thickness (relaxed for low-Re)
    min_thickness_ratio: float = 0.06  # Minimum thickness/chord
    max_thickness_ratio: float = 0.25  # Maximum thickness/chord
    
    # Leading edge constraints (relaxed for low-Re CST-represented airfoils)
    # Low-Re airfoils (Re < 500k) often have sharper LEs (radius ~0.001-0.003)
    # CST parameterization can also produce physically valid but sharper LEs
    min_le_radius: float = 0.0005  # Minimum LE radius (reduced from 0.005 for low-Re)
    max_le_radius: float = 0.06  # Maximum LE radius
    
    # Trailing edge constraints
    min_te_thickness: float = 1e-6  # Minimum TE thickness (relaxed for CST)
    max_te_thickness: float = 0.03  # Maximum TE thickness
    max_te_gap: float = 0.02  # Maximum gap between upper/lower TE
    
    # Curvature constraints
    max_curvature: float = 200.0  # Maximum allowed curvature
    max_curvature_ratio: float = 5.0  # Max ratio of max/min curvature
    
    # Surface quality
    max_oscillation_amplitude: float = 0.01  # Max surface oscillation
    min_points_per_surface: int = 20  # Minimum points per surface
    
    # Coordinate constraints
    max_coordinate_value: float = 10.0  # Maximum absolute coordinate
    min_point_spacing: float = 1e-8  # Minimum distance between points
    
    # Camber constraints
    max_camber: float = 0.15  # Maximum camber/chord
    max_camber_derivative: float = 0.5  # Maximum d(camber)/d(x)


class AirfoilGeometryValidator:
    """
    Comprehensive airfoil geometry validator.
    
    Performs rigorous checks on airfoil geometries to ensure they are:
    - Physically realizable
    - Numerically stable for meshing
    - Aerodynamically reasonable
    - Free of self-intersections and degeneracies
    """
    
    def __init__(self, config: Optional[GeometryValidationConfig] = None):
        """
        Initialize validator.
        
        Args:
            config: Validation configuration. Uses defaults if None.
        """
        self.config = config or GeometryValidationConfig()
    
    def validate_coordinates(self, coordinates: np.ndarray) -> GeometryValidationResult:
        """
        Validate airfoil coordinates.
        
        Args:
            coordinates: Airfoil coordinates as (N, 2) array
        
        Returns:
            GeometryValidationResult with validation status and details
        """
        violations = []
        failure_reasons = []
        warnings = []
        
        # Input validation
        if coordinates is None:
            return GeometryValidationResult(
                status=GeometryValidationStatus.INVALID,
                coordinates=np.array([]),
                violations=[GeometryViolationType.NONE],
                failure_reasons=["Coordinates are None"],
                is_valid=False,
                can_proceed_to_cfd=False,
            )
        
        coords = np.asarray(coordinates, dtype=float)
        
        if coords.ndim != 2 or coords.shape[1] != 2:
            return GeometryValidationResult(
                status=GeometryValidationStatus.INVALID,
                coordinates=coords,
                violations=[GeometryViolationType.NONE],
                failure_reasons=[f"Invalid coordinate shape: {coords.shape}, expected (N, 2)"],
                is_valid=False,
                can_proceed_to_cfd=False,
            )
        
        if len(coords) < 2 * self.config.min_points_per_surface:
            violations.append(GeometryViolationType.INSUFFICIENT_POINTS)
            failure_reasons.append(
                f"Insufficient points: {len(coords)} < {2 * self.config.min_points_per_surface}"
            )
        
        # Check for NaN/Inf
        if np.any(np.isnan(coords)):
            violations.append(GeometryViolationType.NaN_COORDINATES)
            failure_reasons.append("Coordinates contain NaN values")
        
        if np.any(np.isinf(coords)):
            violations.append(GeometryViolationType.INFINITE_COORDINATES)
            failure_reasons.append("Coordinates contain infinite values")
        
        if violations:
            return GeometryValidationResult(
                status=GeometryValidationStatus.INVALID,
                coordinates=coords,
                violations=violations,
                failure_reasons=failure_reasons,
                is_valid=False,
                can_proceed_to_cfd=False,
            )
        
        # Check coordinate bounds
        if np.max(np.abs(coords)) > self.config.max_coordinate_value:
            warnings.append(
                f"Large coordinates detected: max |coord| = {np.max(np.abs(coords)):.4f}"
            )
        
        # Split into upper and lower surfaces
        upper, lower = self._split_surfaces(coords)
        
        if upper is None or lower is None:
            violations.append(GeometryViolationType.INSUFFICIENT_POINTS)
            failure_reasons.append("Could not split into upper and lower surfaces")
            return GeometryValidationResult(
                status=GeometryValidationStatus.INVALID,
                coordinates=coords,
                violations=violations,
                failure_reasons=failure_reasons,
                is_valid=False,
                can_proceed_to_cfd=False,
            )
        
        # Check for duplicate points
        dup_violations = self._check_duplicate_points(coords)
        violations.extend(dup_violations)
        
        # Check x-coordinate monotonicity on each surface
        mono_violations = self._check_monotonicity(upper, lower)
        violations.extend(mono_violations)
        
        # Compute thickness distribution
        thickness_dist, x_locs = self._compute_thickness_distribution(upper, lower)
        
        # Check thickness
        thick_violations, thick_warnings = self._check_thickness(thickness_dist, x_locs)
        violations.extend(thick_violations)
        warnings.extend(thick_warnings)
        
        # Check for self-intersection (upper below lower)
        intersect_violations = self._check_self_intersection(upper, lower, x_locs)
        violations.extend(intersect_violations)
        
        # Compute and check leading edge radius
        le_radius, le_loc = self._compute_le_radius(upper, lower)
        le_violations = self._check_le_radius(le_radius, le_loc)
        violations.extend(le_violations)
        
        # Compute and check trailing edge
        te_thickness, te_gap = self._compute_te_properties(upper, lower)
        te_violations = self._check_te_properties(te_thickness, te_gap)
        violations.extend(te_violations)
        
        # Compute camber and check
        camber_dist = self._compute_camber_distribution(upper, lower, x_locs)
        camber_violations, camber_warnings = self._check_camber(camber_dist, x_locs)
        violations.extend(camber_violations)
        warnings.extend(camber_warnings)
        
        # Compute curvature and check
        curvature_upper = self._compute_curvature(upper)
        curvature_lower = self._compute_curvature(lower)
        max_curvature = max(np.max(curvature_upper), np.max(curvature_lower))
        curvature_violations = self._check_curvature(
            curvature_upper, curvature_lower, x_locs
        )
        violations.extend(curvature_violations)
        
        # Check for oscillatory surfaces
        osc_violations = self._check_oscillations(upper, lower)
        violations.extend(osc_violations)
        
        # Check for hook/folded geometries
        hook_violations = self._check_hook_geometry(upper, lower)
        violations.extend(hook_violations)
        
        # Determine overall status
        if violations:
            status = GeometryValidationStatus.INVALID
            is_valid = False
            can_proceed = False
        elif warnings:
            status = GeometryValidationStatus.WARNING
            is_valid = True
            can_proceed = True
        else:
            status = GeometryValidationStatus.VALID
            is_valid = True
            can_proceed = True
        
        # Compute summary metrics
        min_thick = float(np.min(thickness_dist)) if len(thickness_dist) > 0 else 0.0
        max_thick = float(np.max(thickness_dist)) if len(thickness_dist) > 0 else 0.0
        mean_thick = float(np.mean(thickness_dist)) if len(thickness_dist) > 0 else 0.0
        max_camber_val = float(np.max(np.abs(camber_dist))) if len(camber_dist) > 0 else 0.0
        max_camber_loc = float(x_locs[np.argmax(np.abs(camber_dist))]) if len(camber_dist) > 0 else 0.0
        
        return GeometryValidationResult(
            status=status,
            coordinates=coords,
            violations=list(set(violations)),  # Remove duplicates
            failure_reasons=failure_reasons,
            warnings=warnings,
            min_thickness=min_thick,
            max_thickness=max_thick,
            mean_thickness=mean_thick,
            thickness_distribution=thickness_dist,
            le_radius=le_radius,
            le_location=le_loc,
            te_thickness=te_thickness,
            te_upper=(float(upper[0, 0]), float(upper[0, 1])),
            te_lower=(float(lower[-1, 0]), float(lower[-1, 1])),
            max_camber=max_camber_val,
            max_camber_location=max_camber_loc,
            max_curvature=max_curvature,
            curvature_std=float(np.std(curvature_upper)),
            is_valid=is_valid,
            can_proceed_to_cfd=can_proceed,
        )
    
    def _split_surfaces(self, coords: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Split airfoil coordinates into upper and lower surfaces."""
        if len(coords) < 4:
            return None, None
        
        # Find the leading edge (minimum x)
        x_values = coords[:, 0]
        le_idx = np.argmin(x_values)
        
        # Upper surface: from TE to LE (reverse order in airfoil dat files)
        # Lower surface: from LE to TE
        upper = coords[:le_idx + 1]
        lower = coords[le_idx:]
        
        # Ensure upper surface goes from LE to TE (increasing x)
        if upper[0, 0] > upper[-1, 0]:
            upper = upper[::-1]
        
        # Ensure lower surface goes from LE to TE (increasing x)
        if lower[0, 0] > lower[-1, 0]:
            lower = lower[::-1]
        
        return upper, lower
    
    def _check_duplicate_points(self, coords: np.ndarray) -> List[GeometryViolationType]:
        """Check for duplicate or very close points."""
        violations = []
        
        if len(coords) < 2:
            return violations
        
        # Compute distances between consecutive points
        diffs = np.diff(coords, axis=0)
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        
        close_points = np.sum(distances < self.config.min_point_spacing)
        if close_points > 0:
            violations.append(GeometryViolationType.DUPLICATE_POINTS)
        
        return violations
    
    def _check_monotonicity(self, upper: np.ndarray, lower: np.ndarray) -> List[GeometryViolationType]:
        """Check that x-coordinates are monotonic on each surface."""
        violations = []
        
        if len(upper) > 1:
            x_diffs_upper = np.diff(upper[:, 0])
            if np.any(x_diffs_upper < -self.config.min_point_spacing):
                violations.append(GeometryViolationType.NON_MONOTONIC_X)
        
        if len(lower) > 1:
            x_diffs_lower = np.diff(lower[:, 0])
            if np.any(x_diffs_lower < -self.config.min_point_spacing):
                violations.append(GeometryViolationType.NON_MONOTONIC_X)
        
        return violations
    
    def _compute_thickness_distribution(
        self, upper: np.ndarray, lower: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute thickness distribution by interpolating to common x-locations."""
        # Find common x-range
        x_upper = upper[:, 0]
        x_lower = lower[:, 0]
        x_min = max(x_upper[0], x_lower[0])
        x_max = min(x_upper[-1], x_lower[-1])
        
        if x_max <= x_min:
            return np.array([]), np.array([])
        
        # Create common x-locations
        n_points = min(len(upper), len(lower), 200)
        x_locs = np.linspace(x_min, x_max, n_points)
        
        # Interpolate y-values
        y_upper = np.interp(x_locs, x_upper, upper[:, 1])
        y_lower = np.interp(x_locs, x_lower, lower[:, 1])
        
        # Thickness is upper y - lower y
        thickness = y_upper - y_lower
        
        return thickness, x_locs
    
    def _check_thickness(
        self, thickness: np.ndarray, x_locs: np.ndarray
    ) -> Tuple[List[GeometryViolationType], List[str]]:
        """Check thickness constraints."""
        violations = []
        warnings = []
        
        if len(thickness) == 0:
            return [GeometryViolationType.NEGATIVE_THICKNESS], ["No thickness data"]
        
        # Check for negative thickness (truly negative, not just close to zero)
        if np.any(thickness < -1e-8):
            violations.append(GeometryViolationType.NEGATIVE_THICKNESS)
        
        # Check for zero thickness at interior points only (ignore TE convergence)
        if len(thickness) > 4:
            interior = thickness[1:-1]  # exclude endpoints where surfaces converge
            if np.any(interior < 1e-6):
                violations.append(GeometryViolationType.ZERO_THICKNESS)
        
        # Check thickness bounds
        max_thick = np.max(thickness)
        if max_thick > self.config.max_thickness_ratio:
            violations.append(GeometryViolationType.THICKNESS_OUT_OF_BOUNDS)
            warnings.append(f"Maximum thickness {max_thick:.4f} exceeds limit {self.config.max_thickness_ratio}")
        
        if max_thick < self.config.min_thickness_ratio:
            warnings.append(f"Maximum thickness {max_thick:.4f} is below minimum {self.config.min_thickness_ratio}")
        
        return violations, warnings
    
    def _check_self_intersection(
        self, upper: np.ndarray, lower: np.ndarray, x_locs: np.ndarray
    ) -> List[GeometryViolationType]:
        """Check if upper and lower surfaces cross."""
        violations = []
        
        if len(x_locs) < 2:
            return violations
        
        # Interpolate to common x-locations
        y_upper = np.interp(x_locs, upper[:, 0], upper[:, 1])
        y_lower = np.interp(x_locs, lower[:, 0], lower[:, 1])
        
        # Check if upper is ever below lower (excluding TE region)
        interior_mask = (x_locs > 0.01) & (x_locs < 0.99)
        if np.any(y_upper[interior_mask] < y_lower[interior_mask]):
            violations.append(GeometryViolationType.SELF_INTERSECTION)
            violations.append(GeometryViolationType.SURFACE_CROSSING)
        
        return violations
    
    def _compute_le_radius(
        self, upper: np.ndarray, lower: np.ndarray
    ) -> Tuple[float, Tuple[float, float]]:
        """Estimate leading edge radius."""
        # Find LE point (minimum x)
        le_idx_upper = np.argmin(upper[:, 0])
        le_idx_lower = np.argmin(lower[:, 0])
        
        le_x = (upper[le_idx_upper, 0] + lower[le_idx_lower, 0]) / 2
        le_y = (upper[le_idx_upper, 1] + lower[le_idx_lower, 1]) / 2
        
        # Use the y^2/(2x) method near the LE
        # Take points near the LE on the lower surface
        le_region = lower[:min(10, len(lower))]
        x_local = le_region[:, 0] - le_x
        y_local = le_region[:, 1] - le_y
        
        # Only use points with x > le_x (downstream of LE)
        mask = x_local > 1e-10
        if np.sum(mask) < 3:
            return 0.0, (le_x, le_y)
        
        x_local = x_local[mask]
        y_local = y_local[mask]
        
        # Fit y^2 = 2*R*x to estimate radius
        # R = y^2 / (2*x)
        radii = (y_local**2) / (2 * x_local)
        radius = float(np.median(radii[radii > 0])) if np.any(radii > 0) else 0.0
        
        return radius, (le_x, le_y)
    
    def _check_le_radius(self, radius: float, location: Tuple[float, float]) -> List[GeometryViolationType]:
        """Check leading edge radius constraints."""
        violations = []
        
        if radius < self.config.min_le_radius:
            violations.append(GeometryViolationType.INVALID_LE_RADIUS)
            violations.append(GeometryViolationType.RAZOR_LE)
        
        if radius > self.config.max_le_radius:
            violations.append(GeometryViolationType.INVALID_LE_RADIUS)
        
        return violations
    
    def _compute_te_properties(
        self, upper: np.ndarray, lower: np.ndarray
    ) -> Tuple[float, float]:
        """Compute trailing edge thickness and gap."""
        # TE is at maximum x
        te_upper_y = upper[-1, 1]
        te_lower_y = lower[-1, 1]
        
        te_thickness = abs(te_upper_y - te_lower_y)
        te_gap = abs(upper[-1, 0] - lower[-1, 0])
        
        return te_thickness, te_gap
    
    def _check_te_properties(self, te_thickness: float, te_gap: float) -> List[GeometryViolationType]:
        """Check trailing edge properties."""
        violations = []
        
        if te_thickness < self.config.min_te_thickness:
            violations.append(GeometryViolationType.INVALID_TE_THICKNESS)
            violations.append(GeometryViolationType.DEGENERATE_TE)
        
        if te_thickness > self.config.max_te_thickness:
            violations.append(GeometryViolationType.INVALID_TE_THICKNESS)
        
        if te_gap > self.config.max_te_gap:
            violations.append(GeometryViolationType.DEGENERATE_TE)
            violations.append(GeometryViolationType.CUSP_EXPLOSION)
        
        return violations
    
    def _compute_camber_distribution(
        self, upper: np.ndarray, lower: np.ndarray, x_locs: np.ndarray
    ) -> np.ndarray:
        """Compute camber line."""
        if len(x_locs) == 0:
            return np.array([])
        
        y_upper = np.interp(x_locs, upper[:, 0], upper[:, 1])
        y_lower = np.interp(x_locs, lower[:, 0], lower[:, 1])
        
        return (y_upper + y_lower) / 2
    
    def _check_camber(
        self, camber: np.ndarray, x_locs: np.ndarray
    ) -> Tuple[List[GeometryViolationType], List[str]]:
        """Check camber constraints."""
        violations = []
        warnings = []
        
        if len(camber) == 0:
            return violations, warnings
        
        max_camber = np.max(np.abs(camber))
        
        if max_camber > self.config.max_camber:
            violations.append(GeometryViolationType.CAMBER_OUT_OF_BOUNDS)
            warnings.append(f"Maximum camber {max_camber:.4f} exceeds limit {self.config.max_camber}")
        
        # Check camber derivative
        if len(camber) > 1:
            dcamber_dx = np.gradient(camber, x_locs)
            if np.any(np.abs(dcamber_dx) > self.config.max_camber_derivative):
                warnings.append("High camber gradient detected")
        
        return violations, warnings
    
    def _compute_curvature(self, curve: np.ndarray) -> np.ndarray:
        """Compute curvature along a curve."""
        if len(curve) < 3:
            return np.zeros(len(curve))
        
        x = curve[:, 0]
        y = curve[:, 1]
        
        # First derivatives
        dx = np.gradient(x)
        dy = np.gradient(y)
        
        # Second derivatives
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        # Curvature formula: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = (dx**2 + dy**2) ** 1.5
        
        # Avoid division by zero
        mask = denominator > 1e-15
        curvature = np.zeros_like(numerator)
        curvature[mask] = numerator[mask] / denominator[mask]
        
        return curvature
    
    def _check_curvature(
        self, curvature_upper: np.ndarray, curvature_lower: np.ndarray, x_locs: np.ndarray
    ) -> List[GeometryViolationType]:
        """Check curvature constraints."""
        violations = []
        
        max_curv = max(np.max(curvature_upper), np.max(curvature_lower))
        
        if max_curv > self.config.max_curvature:
            violations.append(GeometryViolationType.CURVATURE_SPIKE)
            violations.append(GeometryViolationType.CURVATURE_DISCONTINUITY)
        
        return violations
    
    def _check_oscillations(self, upper: np.ndarray, lower: np.ndarray) -> List[GeometryViolationType]:
        """Check for oscillatory surface behavior."""
        violations = []
        
        for surface in [upper, lower]:
            if len(surface) < 5:
                continue
            
            y = surface[:, 1]
            
            # Count sign changes in the second derivative
            d2y = np.diff(y, n=2)
            sign_changes = np.sum(np.diff(np.sign(d2y)) != 0)
            
            # Too many inflection points indicate oscillation
            if sign_changes > len(surface) // 3:
                violations.append(GeometryViolationType.OSCILLATORY_SURFACE)
                break
        
        return violations
    
    def _check_hook_geometry(self, upper: np.ndarray, lower: np.ndarray) -> List[GeometryViolationType]:
        """Check for hook or folded geometries."""
        violations = []
        
        # Check if any point on upper surface has y < lower surface at same x
        # This would indicate a "hook" shape
        if len(upper) > 2 and len(lower) > 2:
            # Check for non-physical LE hook
            le_upper = upper[0]  # Should be near TE
            le_lower = lower[0]  # Should be near LE
            
            # Check if surface doubles back on itself
            for surface in [upper, lower]:
                y_values = surface[:, 1]
                if len(y_values) > 3:
                    # Check for large local variations
                    y_range = np.max(y_values) - np.min(y_values)
                    local_max_var = np.max(np.abs(np.diff(y_values, n=2)))
                    
                    if local_max_var > 0.1 and y_range > 0.05:
                        violations.append(GeometryViolationType.HOOK_GEOMETRY)
                        violations.append(GeometryViolationType.FOLDED_GEOMETRY)
                        break
        
        return violations
    
    def validate_cst_parameters(
        self,
        upper_coeffs: np.ndarray,
        lower_coeffs: np.ndarray,
        te_thickness: float,
    ) -> GeometryValidationResult:
        """
        Validate CST parameters before geometry generation.
        
        This provides early rejection of parameter combinations that
        are likely to produce invalid geometries.
        
        Args:
            upper_coeffs: CST coefficients for upper surface
            lower_coeffs: CST coefficients for lower surface
            te_thickness: Trailing edge thickness
        
        Returns:
            GeometryValidationResult with pre-generation validation
        """
        violations = []
        warnings = []
        failure_reasons = []
        
        # Check coefficient bounds
        if np.any(np.abs(upper_coeffs) > 1.0):
            warnings.append(f"Large upper CST coefficients: max |coeff| = {np.max(np.abs(upper_coeffs)):.4f}")
        
        if np.any(np.abs(lower_coeffs) > 1.0):
            warnings.append(f"Large lower CST coefficients: max |coeff| = {np.max(np.abs(lower_coeffs)):.4f}")
        
        # Check for reasonable TE thickness
        if te_thickness < 0.0:
            violations.append(GeometryViolationType.INVALID_TE_THICKNESS)
            failure_reasons.append(f"Negative TE thickness: {te_thickness}")
        
        if te_thickness > 0.05:
            warnings.append(f"Large TE thickness: {te_thickness:.4f}")
        
        # Check coefficient smoothness (second differences)
        if len(upper_coeffs) >= 3:
            d2_upper = np.diff(upper_coeffs, n=2)
            if np.max(np.abs(d2_upper)) > 0.5:
                warnings.append("Upper surface CST coefficients show high variation")
        
        if len(lower_coeffs) >= 3:
            d2_lower = np.diff(lower_coeffs, n=2)
            if np.max(np.abs(d2_lower)) > 0.5:
                warnings.append("Lower surface CST coefficients show high variation")
        
        if violations:
            status = GeometryValidationStatus.INVALID
            is_valid = False
        elif warnings:
            status = GeometryValidationStatus.WARNING
            is_valid = True
        else:
            status = GeometryValidationStatus.VALID
            is_valid = True
        
        return GeometryValidationResult(
            status=status,
            coordinates=np.array([]),
            violations=violations,
            failure_reasons=failure_reasons,
            warnings=warnings,
            is_valid=is_valid,
            can_proceed_to_cfd=is_valid and not warnings,
        )


class GeometryValidationSuite:
    """
    Comprehensive test suite for geometry validation.
    
    Generates synthetic test cases including:
    - Valid airfoils
    - Adversarial geometries
    - Edge cases
    - Stress tests
    """
    
    def __init__(self):
        self.validator = AirfoilGeometryValidator()
        self.test_results: List[Dict[str, Any]] = []
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all validation tests."""
        self.test_results = []
        
        # Test valid airfoil
        self._test_valid_airfoil()
        
        # Test self-intersecting airfoil
        self._test_self_intersection()
        
        # Test negative thickness
        self._test_negative_thickness()
        
        # Test NaN coordinates
        self._test_nan_coordinates()
        
        # Test insufficient points
        self._test_insufficient_points()
        
        # Test oscillatory surface
        self._test_oscillatory_surface()
        
        # Test hook geometry
        self._test_hook_geometry()
        
        # Test CST parameter validation
        self._test_cst_parameter_validation()
        
        # Summary
        passed = sum(1 for r in self.test_results if r["passed"])
        total = len(self.test_results)
        
        return {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "results": self.test_results,
        }
    
    def _record_result(self, name: str, passed: bool, details: str = ""):
        """Record a test result."""
        self.test_results.append({
            "name": name,
            "passed": passed,
            "details": details,
        })
    
    def _test_valid_airfoil(self):
        """Test that a valid NACA-like airfoil passes validation."""
        # Simple symmetric airfoil
        n_points = 100
        x = np.linspace(0, 1, n_points)
        y_thickness = 0.12 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 
                               0.2843 * x**3 - 0.1015 * x**4)
        
        upper = np.column_stack([x[::-1], y_thickness[::-1]])
        lower = np.column_stack([x[1:], -y_thickness[1:]])
        coords = np.vstack([upper, lower])
        
        result = self.validator.validate_coordinates(coords)
        self._record_result(
            "valid_airfoil",
            result.is_valid,
            f"Status: {result.status.value}, Violations: {len(result.violations)}"
        )
    
    def _test_self_intersection(self):
        """Test detection of self-intersecting airfoil."""
        # Create airfoil where lower surface crosses above upper
        n_points = 50
        x = np.linspace(0, 1, n_points)
        y_upper = 0.05 * (1 - x)
        y_lower = 0.1 * (1 - x)  # Lower is above upper!
        
        upper = np.column_stack([x[::-1], y_upper[::-1]])
        lower = np.column_stack([x[1:], y_lower[1:]])
        coords = np.vstack([upper, lower])
        
        result = self.validator.validate_coordinates(coords)
        self._record_result(
            "self_intersection",
            GeometryViolationType.SELF_INTERSECTION in result.violations,
            f"Violations: {[v.value for v in result.violations]}"
        )
    
    def _test_negative_thickness(self):
        """Test detection of negative thickness."""
        n_points = 50
        x = np.linspace(0, 1, n_points)
        y_upper = 0.02 * (1 - x)
        y_lower = 0.08 * (1 - x)  # Creates negative thickness
        
        upper = np.column_stack([x[::-1], y_upper[::-1]])
        lower = np.column_stack([x[1:], y_lower[1:]])
        coords = np.vstack([upper, lower])
        
        result = self.validator.validate_coordinates(coords)
        self._record_result(
            "negative_thickness",
            GeometryViolationType.NEGATIVE_THICKNESS in result.violations,
            f"Violations: {[v.value for v in result.violations]}"
        )
    
    def _test_nan_coordinates(self):
        """Test detection of NaN coordinates."""
        coords = np.array([
            [1.0, 0.0],
            [0.5, np.nan],
            [0.0, 0.0],
            [0.5, -0.05],
            [1.0, -0.01],
        ])
        
        result = self.validator.validate_coordinates(coords)
        self._record_result(
            "nan_coordinates",
            GeometryViolationType.NaN_COORDINATES in result.violations,
            f"Violations: {[v.value for v in result.violations]}"
        )
    
    def _test_insufficient_points(self):
        """Test detection of insufficient points."""
        coords = np.array([
            [1.0, 0.0],
            [0.5, 0.05],
            [0.0, 0.0],
            [0.5, -0.05],
        ])
        
        result = self.validator.validate_coordinates(coords)
        self._record_result(
            "insufficient_points",
            GeometryViolationType.INSUFFICIENT_POINTS in result.violations,
            f"Violations: {[v.value for v in result.violations]}"
        )
    
    def _test_oscillatory_surface(self):
        """Test detection of oscillatory surface."""
        n_points = 50
        x = np.linspace(0, 1, n_points)
        # Create oscillatory surface
        y_upper = 0.05 * np.sin(20 * np.pi * x) * (1 - x)
        y_lower = -0.05 * np.ones_like(x)
        
        upper = np.column_stack([x[::-1], y_upper[::-1]])
        lower = np.column_stack([x[1:], y_lower[1:]])
        coords = np.vstack([upper, lower])
        
        result = self.validator.validate_coordinates(coords)
        self._record_result(
            "oscillatory_surface",
            GeometryViolationType.OSCILLATORY_SURFACE in result.violations,
            f"Violations: {[v.value for v in result.violations]}"
        )
    
    def _test_hook_geometry(self):
        """Test detection of hook geometry."""
        # Create a hook-like shape
        coords = np.array([
            [1.0, 0.0],
            [0.8, 0.1],
            [0.6, 0.3],  # Hook goes way up
            [0.4, 0.1],
            [0.2, 0.05],
            [0.0, 0.0],
            [0.2, -0.05],
            [0.4, -0.08],
            [0.6, -0.05],
            [0.8, -0.03],
            [1.0, -0.01],
        ])
        
        result = self.validator.validate_coordinates(coords)
        self._record_result(
            "hook_geometry",
            (GeometryViolationType.HOOK_GEOMETRY in result.violations or
             GeometryViolationType.FOLDED_GEOMETRY in result.violations),
            f"Violations: {[v.value for v in result.violations]}"
        )
    
    def _test_cst_parameter_validation(self):
        """Test CST parameter pre-validation."""
        # Valid coefficients
        upper_valid = np.array([0.18, 0.05, 0.34, 0.10])
        lower_valid = np.array([-0.19, 0.05, -0.09, 0.03])
        
        result = self.validator.validate_cst_parameters(upper_valid, lower_valid, 0.004)
        self._record_result(
            "cst_valid_params",
            result.is_valid,
            f"Status: {result.status.value}"
        )
        
        # Invalid coefficients (too large)
        upper_invalid = np.array([2.0, 3.0, -1.0, 0.5])
        lower_invalid = np.array([-2.0, 1.0, 0.5, -0.3])
        
        result = self.validator.validate_cst_parameters(upper_invalid, lower_invalid, 0.004)
        self._record_result(
            "cst_invalid_params",
            len(result.warnings) > 0,
            f"Warnings: {len(result.warnings)}"
        )
        
        # Negative TE thickness
        result = self.validator.validate_cst_parameters(upper_valid, lower_valid, -0.01)
        self._record_result(
            "cst_negative_te",
            GeometryViolationType.INVALID_TE_THICKNESS in result.violations,
            f"Violations: {[v.value for v in result.violations]}"
        )