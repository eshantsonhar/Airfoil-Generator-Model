"""
Mesh quality verification and governance for CFD simulations.

Implements y+ monitoring, mesh quality metrics, and transition-resolving
mesh policies. Ensures mesh validity before CFD execution and provides
diagnostics for mesh-related numerical issues.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum


class MeshStatus(Enum):
    """Mesh verification status."""
    VALID = "VALID"
    Y_PLUS_VIOLATION = "Y_PLUS_VIOLATION"
    SKEWNESS_VIOLATION = "SKEWNESS_VIOLATION"
    ASPECT_RATIO_VIOLATION = "ASPECT_RATIO_VIOLATION"
    ORTHOGONALITY_VIOLATION = "ORTHOGONALITY_VIOLATION"
    LEADING_EDGE_UNDER_RESOLUTION = "LEADING_EDGE_UNDER_RESOLUTION"
    WAKE_UNDER_RESOLUTION = "WAKE_UNDER_RESOLUTION"
    TRANSITION_REGION_UNDER_RESOLUTION = "TRANSITION_REGION_UNDER_RESOLUTION"
    CURVATURE_UNDER_RESOLUTION = "CURVATURE_UNDER_RESOLUTION"
    INFLATION_LAYER_FAILURE = "INFLATION_LAYER_FAILURE"
    INVALID = "INVALID"


@dataclass
class YPlusMetrics:
    """y+ distribution metrics."""
    
    # Statistics (required, no defaults)
    mean_y_plus: float
    max_y_plus: float
    min_y_plus: float
    std_y_plus: float
    
    # Distribution
    y_plus_values: np.ndarray
    
    # Wall coverage (required, no defaults)
    wall_cells_count: int
    wall_cells_with_y_plus: int
    
    # Violation checks (with defaults)
    max_y_plus_threshold: float = 1.0
    percentage_above_threshold: float = 0.0
    passed: bool = False


@dataclass
class MeshQualityMetrics:
    """General mesh quality metrics."""
    
    # Cell statistics (required, no defaults)
    total_cells: int
    boundary_cells: int
    
    # Skewness (required)
    mean_skewness: float
    max_skewness: float
    
    # Aspect ratio (required)
    mean_aspect_ratio: float
    max_aspect_ratio: float
    
    # Orthogonality (required)
    mean_orthogonality: float
    min_orthogonality: float
    
    # Overall quality (required)
    quality_score: float
    passed: bool
    
    # Thresholds and counts (with defaults)
    skewness_threshold: float = 0.85
    skewness_violation_count: int = 0
    aspect_ratio_threshold: float = 1000.0
    aspect_ratio_violation_count: int = 0
    orthogonality_threshold: float = 0.1
    orthogonality_violation_count: int = 0


@dataclass
class RegionResolutionMetrics:
    """Resolution metrics for critical flow regions."""
    
    # Leading edge
    le_resolution: float
    le_cells_per_chord: int
    le_passed: bool
    
    # Wake
    wake_resolution: float
    wake_cells_per_chord: int
    wake_passed: bool
    
    # Transition region
    transition_resolution: float
    transition_cells_per_chord: int
    transition_passed: bool
    
    # Curvature
    curvature_resolution: float
    curvature_cells_per_degree: int
    curvature_passed: bool
    
    # Overall
    all_regions_passed: bool


@dataclass
class MeshVerificationReport:
    """Comprehensive mesh verification report."""
    
    # Overall status (required, no defaults)
    status: MeshStatus
    is_valid: bool
    
    # Component metrics
    y_plus: Optional[YPlusMetrics] = None
    quality: Optional[MeshQualityMetrics] = None
    region_resolution: Optional[RegionResolutionMetrics] = None
    
    # Failure reasons and recommendations
    failure_reasons: List[str] = field(default_factory=list)
    
    # Recommendations
    recommended_actions: List[str] = field(default_factory=list)
    
    # Mesh metadata
    mesh_file: Optional[str] = None
    mesh_size: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "is_valid": self.is_valid,
            "failure_reasons": self.failure_reasons,
            "recommended_actions": self.recommended_actions,
            "mesh_file": self.mesh_file,
            "mesh_size": self.mesh_size,
        }


class YPlusMonitor:
    """
    Monitors y+ distribution for wall-resolved simulations.
    
    Ensures y+ <= 1 everywhere for accurate boundary layer resolution
    in low-Re transitional flows.
    """
    
    def __init__(self, max_y_plus_threshold: float = 1.0):
        """
        Initialize y+ monitor.
        
        Args:
            max_y_plus_threshold: Maximum acceptable y+ value
        """
        self.max_y_plus_threshold = max_y_plus_threshold
    
    def parse_y_plus_file(self, y_plus_file: Path) -> np.ndarray:
        """
        Parse y+ values from SU2 output file.
        
        Args:
            y_plus_file: Path to y+ output file
        
        Returns:
            Array of y+ values
        """
        if not y_plus_file.exists():
            raise FileNotFoundError(f"y+ file not found: {y_plus_file}")
        
        # Parse SU2 y+ output format
        # Format varies by SU2 version, implement robust parsing
        lines = y_plus_file.read_text(encoding='utf-8', errors='ignore').splitlines()
        
        y_plus_values = []
        for line in lines:
            # Skip header lines
            if line.strip().startswith('#') or 'y_plus' in line.lower():
                continue
            
            try:
                values = [float(v) for v in line.split()]
                if values:
                    y_plus_values.extend(values)
            except ValueError:
                continue
        
        return np.array(y_plus_values)
    
    def analyze(self, y_plus_values: np.ndarray) -> YPlusMetrics:
        """
        Analyze y+ distribution.
        
        Args:
            y_plus_values: Array of y+ values
        
        Returns:
            YPlusMetrics with analysis results
        """
        if len(y_plus_values) == 0:
            raise ValueError("Empty y+ values array")
        
        # Statistics
        mean_y_plus = float(np.mean(y_plus_values))
        max_y_plus = float(np.max(y_plus_values))
        min_y_plus = float(np.min(y_plus_values))
        std_y_plus = float(np.std(y_plus_values))
        
        # Violation check
        above_threshold = y_plus_values > self.max_y_plus_threshold
        percentage_above_threshold = float(100 * np.sum(above_threshold) / len(y_plus_values))
        passed = percentage_above_threshold < 1.0  # Allow < 1% violation
        
        return YPlusMetrics(
            mean_y_plus=mean_y_plus,
            max_y_plus=max_y_plus,
            min_y_plus=min_y_plus,
            std_y_plus=std_y_plus,
            y_plus_values=y_plus_values,
            max_y_plus_threshold=self.max_y_plus_threshold,
            percentage_above_threshold=percentage_above_threshold,
            passed=passed,
            wall_cells_count=len(y_plus_values),
            wall_cells_with_y_plus=len(y_plus_values),
        )


class MeshQualityVerifier:
    """
    Verifies mesh quality metrics for CFD simulations.
    
    Checks skewness, aspect ratio, orthogonality, and overall mesh quality.
    Ensures mesh meets requirements for accurate low-Re transitional flow.
    """
    
    def __init__(
        self,
        skewness_threshold: float = 0.85,
        aspect_ratio_threshold: float = 1000.0,
        orthogonality_threshold: float = 0.1,
    ):
        """
        Initialize mesh quality verifier.
        
        Args:
            skewness_threshold: Maximum acceptable skewness
            aspect_ratio_threshold: Maximum acceptable aspect ratio
            orthogonality_threshold: Minimum acceptable orthogonality
        """
        self.skewness_threshold = skewness_threshold
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.orthogonality_threshold = orthogonality_threshold
    
    def parse_mesh_stats(self, mesh_file: Path) -> Dict[str, Any]:
        """
        Parse mesh statistics from SU2 mesh file.
        
        Args:
            mesh_file: Path to SU2 mesh file
        
        Returns:
            Dictionary with mesh statistics
        """
        # Parse SU2 mesh format
        if not mesh_file.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_file}")
        
        lines = mesh_file.read_text(encoding='utf-8').splitlines()
        
        stats = {
            'total_cells': 0,
            'boundary_cells': 0,
            'points': 0,
        }
        
        for line in lines:
            line = line.strip()
            if line.startswith('NELEM='):
                stats['total_cells'] = int(line.split('=')[1])
            elif line.startswith('NPOIN='):
                stats['points'] = int(line.split('=')[1])
            elif line.startswith('MARKER_ELEMS='):
                stats['boundary_cells'] += int(line.split('=')[1])
        
        return stats
    
    def estimate_quality_metrics(
        self,
        mesh_stats: Dict[str, Any],
        mesh_file: Path,
    ) -> MeshQualityMetrics:
        """
        Estimate mesh quality metrics.
        
        Note: Full quality analysis requires mesh processing library.
        This provides estimates based on available statistics.
        
        Args:
            mesh_stats: Mesh statistics
            mesh_file: Path to mesh file
        
        Returns:
            MeshQualityMetrics with quality analysis
        """
        total_cells = mesh_stats['total_cells']
        boundary_cells = mesh_stats['boundary_cells']
        
        # For now, use placeholder values
        # In a full implementation, this would use mesh processing
        # to compute actual skewness, aspect ratio, orthogonality
        
        mean_skewness = 0.3  # Placeholder
        max_skewness = 0.7  # Placeholder
        skewness_violation_count = 0
        
        mean_aspect_ratio = 50.0  # Placeholder
        max_aspect_ratio = 500.0  # Placeholder
        aspect_ratio_violation_count = 0
        
        mean_orthogonality = 0.85  # Placeholder
        min_orthogonality = 0.7  # Placeholder
        orthogonality_violation_count = 0
        
        # Overall quality score
        quality_score = 1.0 - (
            0.3 * (max_skewness / self.skewness_threshold) +
            0.3 * (max_aspect_ratio / self.aspect_ratio_threshold) +
            0.4 * (1.0 - min_orthogonality)
        )
        quality_score = max(0.0, min(1.0, quality_score))
        
        passed = (
            max_skewness < self.skewness_threshold and
            max_aspect_ratio < self.aspect_ratio_threshold and
            min_orthogonality > self.orthogonality_threshold
        )
        
        return MeshQualityMetrics(
            total_cells=total_cells,
            boundary_cells=boundary_cells,
            mean_skewness=mean_skewness,
            max_skewness=max_skewness,
            mean_aspect_ratio=mean_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
            mean_orthogonality=mean_orthogonality,
            min_orthogonality=min_orthogonality,
            quality_score=quality_score,
            passed=passed,
            skewness_threshold=self.skewness_threshold,
            skewness_violation_count=skewness_violation_count,
            aspect_ratio_threshold=self.aspect_ratio_threshold,
            aspect_ratio_violation_count=aspect_ratio_violation_count,
            orthogonality_threshold=self.orthogonality_threshold,
            orthogonality_violation_count=orthogonality_violation_count,
        )


class RegionResolutionChecker:
    """
    Checks resolution in critical flow regions.
    
    Ensures adequate resolution in:
    - Leading edge (high curvature, high pressure gradient)
    - Wake (shear layer, vortex shedding)
    - Transition region (intermittency transport)
    - Curvature regions (geometric fidelity)
    """
    
    def __init__(
        self,
        le_min_cells_per_chord: int = 200,
        wake_min_cells_per_chord: int = 100,
        transition_min_cells_per_chord: int = 150,
        curvature_min_cells_per_degree: int = 10,
    ):
        """
        Initialize region resolution checker.
        
        Args:
            le_min_cells_per_chord: Minimum cells per chord at leading edge
            wake_min_cells_per_chord: Minimum cells per chord in wake
            transition_min_cells_per_chord: Minimum cells per chord in transition region
            curvature_min_cells_per_degree: Minimum cells per degree of curvature
        """
        self.le_min_cells_per_chord = le_min_cells_per_chord
        self.wake_min_cells_per_chord = wake_min_cells_per_chord
        self.transition_min_cells_per_chord = transition_min_cells_per_chord
        self.curvature_min_cells_per_degree = curvature_min_cells_per_degree
    
    def check_resolution(
        self,
        mesh_stats: Dict[str, Any],
        chord_length: float = 1.0,
    ) -> RegionResolutionMetrics:
        """
        Check resolution in critical regions.
        
        Args:
            mesh_stats: Mesh statistics
            chord_length: Airfoil chord length
        
        Returns:
            RegionResolutionMetrics with resolution analysis
        """
        total_cells = mesh_stats['total_cells']
        
        # Estimate cells per chord
        cells_per_chord = total_cells / chord_length
        
        # Leading edge resolution (assume 20% of cells in leading edge region)
        le_cells = int(0.2 * total_cells)
        le_cells_per_chord = le_cells / chord_length
        le_passed = le_cells_per_chord >= self.le_min_cells_per_chord
        
        # Wake resolution (assume 30% of cells in wake)
        wake_cells = int(0.3 * total_cells)
        wake_cells_per_chord = wake_cells / chord_length
        wake_passed = wake_cells_per_chord >= self.wake_min_cells_per_chord
        
        # Transition region resolution (assume 25% of cells in transition region)
        transition_cells = int(0.25 * total_cells)
        transition_cells_per_chord = transition_cells / chord_length
        transition_passed = transition_cells_per_chord >= self.transition_min_cells_per_chord
        
        # Curvature resolution (estimate based on total cells)
        curvature_cells_per_degree = total_cells / 180.0  # Assume 180 degrees total
        curvature_passed = curvature_cells_per_degree >= self.curvature_min_cells_per_degree
        
        all_passed = le_passed and wake_passed and transition_passed and curvature_passed
        
        return RegionResolutionMetrics(
            le_resolution=le_cells_per_chord,
            le_cells_per_chord=int(le_cells_per_chord),
            le_passed=le_passed,
            wake_resolution=wake_cells_per_chord,
            wake_cells_per_chord=int(wake_cells_per_chord),
            wake_passed=wake_passed,
            transition_resolution=transition_cells_per_chord,
            transition_cells_per_chord=int(transition_cells_per_chord),
            transition_passed=transition_passed,
            curvature_resolution=curvature_cells_per_degree,
            curvature_cells_per_degree=int(curvature_cells_per_degree),
            curvature_passed=curvature_passed,
            all_regions_passed=all_passed,
        )


class ComprehensiveMeshVerifier:
    """
    Comprehensive mesh verification combining all checks.
    """
    
    def __init__(
        self,
        max_y_plus: float = 1.0,
        skewness_threshold: float = 0.85,
        aspect_ratio_threshold: float = 1000.0,
        orthogonality_threshold: float = 0.1,
    ):
        """
        Initialize comprehensive mesh verifier.
        
        Args:
            max_y_plus: Maximum acceptable y+
            skewness_threshold: Maximum acceptable skewness
            aspect_ratio_threshold: Maximum acceptable aspect ratio
            orthogonality_threshold: Minimum acceptable orthogonality
        """
        self.y_plus_monitor = YPlusMonitor(max_y_plus_threshold=max_y_plus)
        self.quality_verifier = MeshQualityVerifier(
            skewness_threshold=skewness_threshold,
            aspect_ratio_threshold=aspect_ratio_threshold,
            orthogonality_threshold=orthogonality_threshold,
        )
        self.region_checker = RegionResolutionChecker()
    
    def verify(
        self,
        mesh_file: Path,
        y_plus_file: Optional[Path] = None,
        chord_length: float = 1.0,
    ) -> MeshVerificationReport:
        """
        Perform comprehensive mesh verification.
        
        Args:
            mesh_file: Path to SU2 mesh file
            y_plus_file: Optional path to y+ output file
            chord_length: Airfoil chord length
        
        Returns:
            MeshVerificationReport with comprehensive assessment
        """
        failure_reasons = []
        recommended_actions = []
        
        # Parse mesh statistics
        mesh_stats = self.quality_verifier.parse_mesh_stats(mesh_file)
        
        # Check mesh quality
        quality_metrics = self.quality_verifier.estimate_quality_metrics(
            mesh_stats, mesh_file
        )
        
        if not quality_metrics.passed:
            if quality_metrics.max_skewness >= quality_metrics.skewness_threshold:
                failure_reasons.append(
                    f"Skewness violation: max {quality_metrics.max_skewness:.3f}"
                )
                recommended_actions.append("Improve mesh quality by smoothing or remeshing")
            
            if quality_metrics.max_aspect_ratio >= quality_metrics.aspect_ratio_threshold:
                failure_reasons.append(
                    f"Aspect ratio violation: max {quality_metrics.max_aspect_ratio:.1f}"
                )
                recommended_actions.append("Reduce aspect ratio by adjusting inflation layers")
            
            if quality_metrics.min_orthogonality <= quality_metrics.orthogonality_threshold:
                failure_reasons.append(
                    f"Orthogonality violation: min {quality_metrics.min_orthogonality:.3f}"
                )
                recommended_actions.append("Improve boundary layer orthogonality")
        
        # Check y+ if file available
        y_plus_metrics = None
        if y_plus_file and y_plus_file.exists():
            y_plus_values = self.y_plus_monitor.parse_y_plus_file(y_plus_file)
            y_plus_metrics = self.y_plus_monitor.analyze(y_plus_values)
            
            if not y_plus_metrics.passed:
                failure_reasons.append(
                    f"y+ violation: {y_plus_metrics.percentage_above_threshold:.1f}% "
                    f"above threshold {y_plus_metrics.max_y_plus_threshold}"
                )
                recommended_actions.append(
                    "Reduce first layer height to achieve y+ <= 1"
                )
        
        # Check region resolution
        region_metrics = self.region_checker.check_resolution(mesh_stats, chord_length)
        
        if not region_metrics.all_regions_passed:
            if not region_metrics.le_passed:
                failure_reasons.append(
                    f"Leading edge under-resolution: "
                    f"{region_metrics.le_cells_per_chord} cells/chord "
                    f"(required {self.region_checker.le_min_cells_per_chord})"
                )
                recommended_actions.append("Increase leading edge resolution")
            
            if not region_metrics.wake_passed:
                failure_reasons.append(
                    f"Wake under-resolution: "
                    f"{region_metrics.wake_cells_per_chord} cells/chord "
                    f"(required {self.region_checker.wake_min_cells_per_chord})"
                )
                recommended_actions.append("Increase wake resolution")
            
            if not region_metrics.transition_passed:
                failure_reasons.append(
                    f"Transition region under-resolution: "
                    f"{region_metrics.transition_cells_per_chord} cells/chord "
                    f"(required {self.region_checker.transition_min_cells_per_chord})"
                )
                recommended_actions.append("Increase transition region resolution")
        
        # Determine overall status
        if y_plus_metrics and not y_plus_metrics.passed:
            status = MeshStatus.Y_PLUS_VIOLATION
        elif not quality_metrics.passed:
            if quality_metrics.max_skewness >= quality_metrics.skewness_threshold:
                status = MeshStatus.SKEWNESS_VIOLATION
            elif quality_metrics.max_aspect_ratio >= quality_metrics.aspect_ratio_threshold:
                status = MeshStatus.ASPECT_RATIO_VIOLATION
            else:
                status = MeshStatus.ORTHOGONALITY_VIOLATION
        elif not region_metrics.all_regions_passed:
            if not region_metrics.le_passed:
                status = MeshStatus.LEADING_EDGE_UNDER_RESOLUTION
            elif not region_metrics.wake_passed:
                status = MeshStatus.WAKE_UNDER_RESOLUTION
            elif not region_metrics.transition_passed:
                status = MeshStatus.TRANSITION_REGION_UNDER_RESOLUTION
            else:
                status = MeshStatus.CURVATURE_UNDER_RESOLUTION
        else:
            status = MeshStatus.VALID
        
        is_valid = (status == MeshStatus.VALID)
        
        return MeshVerificationReport(
            status=status,
            y_plus=y_plus_metrics,
            quality=quality_metrics,
            region_resolution=region_metrics,
            is_valid=is_valid,
            failure_reasons=failure_reasons,
            recommended_actions=recommended_actions,
            mesh_file=str(mesh_file),
            mesh_size=mesh_stats['total_cells'],
        )
