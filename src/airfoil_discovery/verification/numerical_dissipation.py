"""
Numerical dissipation governance for CFD-based optimization.

Implements monitoring and detection of numerical dissipation effects
that can artificially stabilize flows, suppress separation, or create
fake attached flow conditions. This is critical for distinguishing
physical stability from numerical stability.

The framework tracks:
- Artificial viscosity levels
- Limiter activation and saturation
- Total variation damping
- Flux dissipation intensity
- Numerical smoothing effects
- Residual spectral damping
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import warnings


class DissipationStatus(Enum):
    """Numerical dissipation status."""
    LOW_DISSIPATION = "LOW_DISSIPATION"
    MODERATE_DISSIPATION = "MODERATE_DISSIPATION"
    HIGH_DISSIPATION = "HIGH_DISSIPATION"
    EXCESSIVE_DISSIPATION = "EXCESSIVE_DISSIPATION"
    UNKNOWN = "UNKNOWN"


class DissipationViolationType(Enum):
    """Types of numerical dissipation violations."""
    NONE = "NONE"
    ARTIFICIAL_VISCOSITY_EXCESSIVE = "ARTIFICIAL_VISCOSITY_EXCESSIVE"
    LIMITER_SATURATED = "LIMITER_SATURATED"
    TVD_EXCESSIVE = "TVD_EXCESSIVE"
    FLUX_DISSIPATION_HIGH = "FLUX_DISSIPATION_HIGH"
    NUMERICAL_SMOOTHING_EXCESSIVE = "NUMERICAL_SMOOTHING_EXCESSIVE"
    SPECTRAL_DAMPING_EXCESSIVE = "SPECTRAL_DAMPING_EXCESSIVE"
    SEPARATION_SUPPRESSED = "SEPARATION_SUPPRESSED"
    TRANSITION_DELAYED = "TRANSITION_DELAYED"


@dataclass
class ArtificialViscosityMetrics:
    """Artificial viscosity analysis metrics."""
    
    # Viscosity levels
    max_artificial_viscosity: float
    mean_artificial_viscosity: float
    artificial_viscosity_ratio: float  # Ratio to physical viscosity
    
    # Spatial distribution
    av_concentrated_in_boundary_layer: bool
    av_concentrated_in_wake: bool
    av_uniform: bool
    
    # Violations
    violations: List[DissipationViolationType] = field(default_factory=list)


@dataclass
class LimiterMetrics:
    """Limiter activation analysis metrics."""
    
    # Limiter statistics
    limiter_activation_fraction: float  # Fraction of cells with active limiter
    max_limiter_value: float
    mean_limiter_value: float
    
    # Limiter saturation
    limiter_saturated: bool
    saturation_fraction: float  # Fraction of cells with saturated limiter
    
    # Spatial distribution
    limiter_active_in_smooth_region: bool
    limiter_active_near_shocks: bool
    
    # Violations
    violations: List[DissipationViolationType] = field(default_factory=list)


@dataclass
class TotalVariationMetrics:
    """Total variation analysis metrics."""
    
    # TVD metrics
    total_variation: float
    tvd_ratio: float  # Current TV / Initial TV
    
    # TV growth
    tv_growth_rate: float
    tv_oscillation_detected: bool
    
    # Violations
    violations: List[DissipationViolationType] = field(default_factory=list)


@dataclass
class FluxDissipationMetrics:
    """Flux dissipation analysis metrics."""
    
    # Dissipation levels
    max_flux_dissipation: float
    mean_flux_dissipation: float
    
    # Dissipation distribution
    dissipation_in_upwind: float
    dissipation_in_central: float
    
    # Energy conservation
    energy_error: float
    
    # Violations
    violations: List[DissipationViolationType] = field(default_factory=list)


@dataclass
class SpectralDampingMetrics:
    """Spectral damping analysis metrics."""
    
    # Spectral analysis
    high_frequency_damping: float
    low_frequency_damping: float
    damping_ratio: float  # HF/LF ratio
    
    # Residual spectrum
    residual_spectral_slope: float
    spectral_energy_decay: float
    
    # Violations
    violations: List[DissipationViolationType] = field(default_factory=list)


@dataclass
class NumericalDissipationReport:
    """Comprehensive numerical dissipation report."""
    
    # Overall status and assessment (required, no defaults)
    status: DissipationStatus
    is_acceptable: bool
    physical_stability_confidence: float
    
    # Component metrics (with defaults)
    artificial_viscosity: Optional[ArtificialViscosityMetrics] = None
    limiter: Optional[LimiterMetrics] = None
    total_variation: Optional[TotalVariationMetrics] = None
    flux_dissipation: Optional[FluxDissipationMetrics] = None
    spectral_damping: Optional[SpectralDampingMetrics] = None
    
    # Violations and recommendations (with defaults)
    violations: List[DissipationViolationType] = field(default_factory=list)
    failure_reasons: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "is_acceptable": self.is_acceptable,
            "physical_stability_confidence": self.physical_stability_confidence,
            "violations": [v.value for v in self.violations],
            "failure_reasons": self.failure_reasons,
            "recommended_actions": self.recommended_actions,
            "artificial_viscosity": {
                "av_ratio": self.artificial_viscosity.artificial_viscosity_ratio,
                "max_av": self.artificial_viscosity.max_artificial_viscosity,
            } if self.artificial_viscosity else None,
            "limiter": {
                "activation_fraction": self.limiter.limiter_activation_fraction,
                "saturated": self.limiter.limiter_saturated,
            } if self.limiter else None,
        }


@dataclass
class DissipationConfig:
    """Configuration for numerical dissipation governance."""
    
    # Artificial viscosity thresholds
    max_av_ratio: float = 10.0  # Max ratio of artificial to physical viscosity
    max_av_mean_ratio: float = 1.0  # Max mean ratio
    
    # Limiter thresholds
    max_limiter_activation: float = 0.3  # Max 30% cells with active limiter
    max_limiter_saturation: float = 0.1  # Max 10% cells with saturated limiter
    
    # TVD thresholds
    max_tvd_ratio: float = 2.0  # TV should not grow excessively
    max_tv_growth_rate: float = 0.1
    
    # Flux dissipation thresholds
    max_flux_dissipation: float = 0.01  # Max mean flux dissipation
    max_energy_error: float = 0.01  # Max 1% energy error
    
    # Spectral damping thresholds
    max_hf_damping: float = 0.5  # Max high-frequency damping
    min_damping_ratio: float = 2.0  # HF should damp faster than LF
    
    # Overall thresholds
    excessive_dissipation_threshold: float = 0.8  # Status threshold
    high_dissipation_threshold: float = 0.5
    moderate_dissipation_threshold: float = 0.2
    
    # Physical effects detection
    separation_suppression_threshold: float = 0.7  # Confidence threshold
    transition_delay_threshold: float = 0.5


class NumericalDissipationMonitor:
    """
    Monitors numerical dissipation in CFD simulations.
    
    This class tracks various indicators of numerical dissipation that
    can artificially affect the flow physics. High numerical dissipation
    can:
    
    - Suppress laminar separation (fake attached flow)
    - Delay transition (incorrect transition location)
    - Dampen vortex shedding (miss unsteady effects)
    - Smooth pressure gradients (incorrect pressure recovery)
    
    The monitor helps distinguish between physical stability and
    numerical stability, which is critical for low-Re transitional
    flow simulations.
    """
    
    def __init__(self, config: Optional[DissipationConfig] = None):
        """
        Initialize numerical dissipation monitor.
        
        Args:
            config: Monitor configuration. Uses defaults if None.
        """
        self.config = config or DissipationConfig()
    
    def analyze_artificial_viscosity(
        self,
        artificial_viscosity: np.ndarray,
        physical_viscosity: float,
        cell_volumes: Optional[np.ndarray] = None,
    ) -> ArtificialViscosityMetrics:
        """
        Analyze artificial viscosity levels.
        
        Args:
            artificial_viscosity: Cell-wise artificial viscosity values
            physical_viscosity: Physical dynamic viscosity
            cell_volumes: Cell volumes for weighted averaging
        
        Returns:
            ArtificialViscosityMetrics with analysis
        """
        if len(artificial_viscosity) == 0 or physical_viscosity <= 0:
            return ArtificialViscosityMetrics(
                max_artificial_viscosity=0.0,
                mean_artificial_viscosity=0.0,
                artificial_viscosity_ratio=0.0,
                av_concentrated_in_boundary_layer=False,
                av_concentrated_in_wake=False,
                av_uniform=True,
                violations=[],
            )
        
        av = np.array(artificial_viscosity)
        
        # Basic statistics
        max_av = float(np.max(av))
        
        if cell_volumes is not None and len(cell_volumes) == len(av):
            # Volume-weighted mean
            weights = cell_volumes / np.sum(cell_volumes)
            mean_av = float(np.sum(av * weights))
        else:
            mean_av = float(np.mean(av))
        
        # Ratio to physical viscosity
        av_ratio = max_av / physical_viscosity if physical_viscosity > 0 else float('inf')
        mean_av_ratio = mean_av / physical_viscosity if physical_viscosity > 0 else float('inf')
        
        # Spatial distribution analysis (simplified)
        # In practice, this would use cell location information
        av_concentrated_in_boundary_layer = False
        av_concentrated_in_wake = False
        av_uniform = np.std(av) / (np.mean(av) + 1e-15) < 0.5
        
        # Detect violations
        violations = []
        if av_ratio > self.config.max_av_ratio:
            violations.append(DissipationViolationType.ARTIFICIAL_VISCOSITY_EXCESSIVE)
        if mean_av_ratio > self.config.max_av_mean_ratio:
            violations.append(DissipationViolationType.ARTIFICIAL_VISCOSITY_EXCESSIVE)
        
        return ArtificialViscosityMetrics(
            max_artificial_viscosity=max_av,
            mean_artificial_viscosity=mean_av,
            artificial_viscosity_ratio=av_ratio,
            av_concentrated_in_boundary_layer=av_concentrated_in_boundary_layer,
            av_concentrated_in_wake=av_concentrated_in_wake,
            av_uniform=av_uniform,
            violations=violations,
        )
    
    def analyze_limiter(
        self,
        limiter_values: np.ndarray,
        max_limiter_value: float = 1.0,
    ) -> LimiterMetrics:
        """
        Analyze limiter activation and saturation.
        
        Args:
            limiter_values: Cell-wise limiter values (0 = inactive, 1 = fully active)
            max_limiter_value: Maximum possible limiter value
        
        Returns:
            LimiterMetrics with analysis
        """
        if len(limiter_values) == 0:
            return LimiterMetrics(
                limiter_activation_fraction=0.0,
                max_limiter_value=0.0,
                mean_limiter_value=0.0,
                limiter_saturated=False,
                saturation_fraction=0.0,
                limiter_active_in_smooth_region=False,
                limiter_active_near_shocks=False,
                violations=[],
            )
        
        lim = np.array(limiter_values)
        
        # Activation fraction (cells with significant limiter activity)
        activation_threshold = 0.1 * max_limiter_value
        activation_fraction = float(np.mean(lim > activation_threshold))
        
        # Statistics
        max_lim = float(np.max(lim))
        mean_lim = float(np.mean(lim))
        
        # Saturation (cells near maximum limiter value)
        saturation_threshold = 0.9 * max_limiter_value
        saturation_fraction = float(np.mean(lim > saturation_threshold))
        limiter_saturated = saturation_fraction > 0.01
        
        # Spatial distribution (simplified)
        # In practice, would use gradient information
        limiter_active_in_smooth_region = activation_fraction > 0.5
        limiter_active_near_shocks = max_lim > 0.8 * max_limiter_value
        
        # Detect violations
        violations = []
        if activation_fraction > self.config.max_limiter_activation:
            violations.append(DissipationViolationType.LIMITER_SATURATED)
        if saturation_fraction > self.config.max_limiter_saturation:
            violations.append(DissipationViolationType.LIMITER_SATURATED)
        
        return LimiterMetrics(
            limiter_activation_fraction=activation_fraction,
            max_limiter_value=max_lim,
            mean_limiter_value=mean_lim,
            limiter_saturated=limiter_saturated,
            saturation_fraction=saturation_fraction,
            limiter_active_in_smooth_region=limiter_active_in_smooth_region,
            limiter_active_near_shocks=limiter_active_near_shocks,
            violations=violations,
        )
    
    def analyze_total_variation(
        self,
        solution_history: List[np.ndarray],
    ) -> TotalVariationMetrics:
        """
        Analyze total variation behavior.
        
        Args:
            solution_history: List of solution vectors at different iterations
        
        Returns:
            TotalVariationMetrics with analysis
        """
        if len(solution_history) < 2:
            return TotalVariationMetrics(
                total_variation=0.0,
                tvd_ratio=1.0,
                tv_growth_rate=0.0,
                tv_oscillation_detected=False,
                violations=[],
            )
        
        # Compute total variation for each iteration
        tv_values = []
        for sol in solution_history:
            if len(sol) > 1:
                tv = float(np.sum(np.abs(np.diff(sol))))
                tv_values.append(tv)
        
        if len(tv_values) < 2:
            return TotalVariationMetrics(
                total_variation=tv_values[0] if tv_values else 0.0,
                tvd_ratio=1.0,
                tv_growth_rate=0.0,
                tv_oscillation_detected=False,
                violations=[],
            )
        
        tv_array = np.array(tv_values)
        
        # Current TV
        current_tv = tv_array[-1]
        
        # TVD ratio (current / initial)
        initial_tv = tv_array[0]
        tvd_ratio = current_tv / (initial_tv + 1e-15)
        
        # TV growth rate
        if len(tv_array) > 10:
            recent_tv = tv_array[-10:]
            tv_growth_rate = float((recent_tv[-1] - recent_tv[0]) / (recent_tv[0] + 1e-15))
        else:
            tv_growth_rate = 0.0
        
        # Oscillation detection
        if len(tv_array) > 5:
            tv_diff = np.diff(tv_array)
            sign_changes = np.sum(np.diff(np.sign(tv_diff)) != 0)
            tv_oscillation_detected = sign_changes > len(tv_diff) * 0.3
        else:
            tv_oscillation_detected = False
        
        # Detect violations
        violations = []
        if tvd_ratio > self.config.max_tvd_ratio:
            violations.append(DissipationViolationType.TVD_EXCESSIVE)
        if abs(tv_growth_rate) > self.config.max_tv_growth_rate:
            violations.append(DissipationViolationType.TVD_EXCESSIVE)
        
        return TotalVariationMetrics(
            total_variation=current_tv,
            tvd_ratio=tvd_ratio,
            tv_growth_rate=tv_growth_rate,
            tv_oscillation_detected=tv_oscillation_detected,
            violations=violations,
        )
    
    def analyze_flux_dissipation(
        self,
        flux_dissipation: np.ndarray,
        total_energy: float,
    ) -> FluxDissipationMetrics:
        """
        Analyze flux dissipation levels.
        
        Args:
            flux_dissipation: Cell-wise flux dissipation values
            total_energy: Total energy for normalization
        
        Returns:
            FluxDissipationMetrics with analysis
        """
        if len(flux_dissipation) == 0:
            return FluxDissipationMetrics(
                max_flux_dissipation=0.0,
                mean_flux_dissipation=0.0,
                dissipation_in_upwind=0.0,
                dissipation_in_central=0.0,
                energy_error=0.0,
                violations=[],
            )
        
        fd = np.array(flux_dissipation)
        
        # Statistics
        max_fd = float(np.max(np.abs(fd)))
        mean_fd = float(np.mean(np.abs(fd)))
        
        # Simplified upwind vs central split
        # In practice, would track flux type per cell
        dissipation_in_upwind = mean_fd * 0.7  # Estimate
        dissipation_in_central = mean_fd * 0.3
        
        # Energy error (normalized dissipation)
        energy_error = mean_fd / (total_energy + 1e-15) if total_energy > 0 else 0.0
        
        # Detect violations
        violations = []
        if mean_fd > self.config.max_flux_dissipation:
            violations.append(DissipationViolationType.FLUX_DISSIPATION_HIGH)
        if energy_error > self.config.max_energy_error:
            violations.append(DissipationViolationType.FLUX_DISSIPATION_HIGH)
        
        return FluxDissipationMetrics(
            max_flux_dissipation=max_fd,
            mean_flux_dissipation=mean_fd,
            dissipation_in_upwind=dissipation_in_upwind,
            dissipation_in_central=dissipation_in_central,
            energy_error=energy_error,
            violations=violations,
        )
    
    def analyze_spectral_damping(
        self,
        residual_history: np.ndarray,
        solution_field: Optional[np.ndarray] = None,
    ) -> SpectralDampingMetrics:
        """
        Analyze spectral damping characteristics.
        
        Args:
            residual_history: Residual history for spectral analysis
            solution_field: Solution field for spatial spectral analysis
        
        Returns:
            SpectralDampingMetrics with analysis
        """
        if len(residual_history) < 16:
            return SpectralDampingMetrics(
                high_frequency_damping=0.0,
                low_frequency_damping=0.0,
                damping_ratio=1.0,
                residual_spectral_slope=0.0,
                spectral_energy_decay=0.0,
                violations=[],
            )
        
        res = np.array(residual_history)
        
        # FFT of residual history
        res_detrended = res - np.mean(res)
        fft_values = np.fft.fft(res_detrended)
        power_spectrum = np.abs(fft_values[:len(fft_values)//2])**2
        
        # High-frequency vs low-frequency damping
        n_freq = len(power_spectrum)
        lf_end = n_freq // 4
        hf_start = n_freq * 3 // 4
        
        lf_power = float(np.sum(power_spectrum[1:lf_end])) if lf_end > 1 else 0.0
        hf_power = float(np.sum(power_spectrum[hf_start:])) if hf_start < n_freq else 0.0
        
        # Damping ratio (HF/LF)
        damping_ratio = hf_power / (lf_power + 1e-15) if lf_power > 0 else 1.0
        
        # High-frequency damping
        high_frequency_damping = hf_power / (np.sum(power_spectrum[1:]) + 1e-15)
        low_frequency_damping = lf_power / (np.sum(power_spectrum[1:]) + 1e-15)
        
        # Spectral slope (from power spectrum)
        if len(power_spectrum) > 5:
            freq_indices = np.arange(1, len(power_spectrum))
            log_freq = np.log10(freq_indices)
            log_power = np.log10(power_spectrum[1:] + 1e-30)
            
            # Linear fit to log-log spectrum
            valid = np.isfinite(log_power) & np.isfinite(log_freq)
            if np.sum(valid) > 3:
                slope, _ = np.polyfit(log_freq[valid], log_power[valid], 1)
                spectral_slope = float(slope)
            else:
                spectral_slope = 0.0
        else:
            spectral_slope = 0.0
        
        # Spectral energy decay
        if len(power_spectrum) > 5:
            total_power = np.sum(power_spectrum[1:])
            low_freq_power = np.sum(power_spectrum[1:n_freq//2])
            spectral_energy_decay = 1.0 - low_freq_power / (total_power + 1e-15)
        else:
            spectral_energy_decay = 0.0
        
        # Detect violations
        violations = []
        if high_frequency_damping > self.config.max_hf_damping:
            violations.append(DissipationViolationType.SPECTRAL_DAMPING_EXCESSIVE)
        
        return SpectralDampingMetrics(
            high_frequency_damping=high_frequency_damping,
            low_frequency_damping=low_frequency_damping,
            damping_ratio=damping_ratio,
            residual_spectral_slope=spectral_slope,
            spectral_energy_decay=spectral_energy_decay,
            violations=violations,
        )
    
    def detect_physical_effects(
        self,
        expected_separation_location: Optional[float],
        computed_separation_location: Optional[float],
        expected_transition_location: Optional[float],
        computed_transition_location: Optional[float],
    ) -> Tuple[bool, bool]:
        """
        Detect if numerical dissipation is affecting physical phenomena.
        
        Args:
            expected_separation_location: Expected separation location from theory/experiment
            computed_separation_location: Computed separation location
            expected_transition_location: Expected transition location
            computed_transition_location: Computed transition location
        
        Returns:
            (separation_suppressed, transition_delayed) tuple
        """
        separation_suppressed = False
        transition_delayed = False
        
        # Check if separation is suppressed
        if expected_separation_location is not None and computed_separation_location is None:
            # Expected separation but none computed - possibly suppressed
            separation_suppressed = True
        elif expected_separation_location is not None and computed_separation_location is not None:
            # Check if separation is significantly delayed
            delay = computed_separation_location - expected_separation_location
            if delay > 0.1:  # More than 10% chord delay
                separation_suppressed = True
        
        # Check if transition is delayed
        if expected_transition_location is not None and computed_transition_location is None:
            transition_delayed = True
        elif expected_transition_location is not None and computed_transition_location is not None:
            delay = computed_transition_location - expected_transition_location
            if delay > 0.05:  # More than 5% chord delay
                transition_delayed = True
        
        return separation_suppressed, transition_delayed
    
    def compute_dissipation_score(
        self,
        av_metrics: Optional[ArtificialViscosityMetrics],
        limiter_metrics: Optional[LimiterMetrics],
        tv_metrics: Optional[TotalVariationMetrics],
        fd_metrics: Optional[FluxDissipationMetrics],
        spectral_metrics: Optional[SpectralDampingMetrics],
    ) -> float:
        """
        Compute overall dissipation score (0-1, higher = more dissipation).
        
        Args:
            av_metrics: Artificial viscosity metrics
            limiter_metrics: Limiter metrics
            tv_metrics: Total variation metrics
            fd_metrics: Flux dissipation metrics
            spectral_metrics: Spectral damping metrics
        
        Returns:
            Overall dissipation score (0-1)
        """
        score = 0.0
        weight = 0.0
        
        if av_metrics is not None:
            av_score = min(1.0, av_metrics.artificial_viscosity_ratio / self.config.max_av_ratio)
            score += av_score * 0.25
            weight += 0.25
        
        if limiter_metrics is not None:
            lim_score = min(1.0, limiter_metrics.limiter_activation_fraction / self.config.max_limiter_activation)
            score += lim_score * 0.2
            weight += 0.2
        
        if tv_metrics is not None:
            tv_score = min(1.0, abs(tv_metrics.tv_growth_rate) / self.config.max_tv_growth_rate)
            score += tv_score * 0.15
            weight += 0.15
        
        if fd_metrics is not None:
            fd_score = min(1.0, fd_metrics.mean_flux_dissipation / self.config.max_flux_dissipation)
            score += fd_score * 0.2
            weight += 0.2
        
        if spectral_metrics is not None:
            spec_score = min(1.0, spectral_metrics.high_frequency_damping / self.config.max_hf_damping)
            score += spec_score * 0.2
            weight += 0.2
        
        return score / weight if weight > 0 else 0.0
    
    def govern(
        self,
        artificial_viscosity: Optional[np.ndarray] = None,
        physical_viscosity: float = 1e-5,
        limiter_values: Optional[np.ndarray] = None,
        solution_history: Optional[List[np.ndarray]] = None,
        flux_dissipation: Optional[np.ndarray] = None,
        total_energy: float = 1.0,
        residual_history: Optional[np.ndarray] = None,
        expected_separation: Optional[float] = None,
        computed_separation: Optional[float] = None,
        expected_transition: Optional[float] = None,
        computed_transition: Optional[float] = None,
    ) -> NumericalDissipationReport:
        """
        Perform comprehensive numerical dissipation governance.
        
        This is the main entry point for dissipation monitoring.
        
        Args:
            artificial_viscosity: Cell-wise artificial viscosity values
            physical_viscosity: Physical dynamic viscosity
            limiter_values: Cell-wise limiter values
            solution_history: List of solution vectors at different iterations
            flux_dissipation: Cell-wise flux dissipation values
            total_energy: Total energy for normalization
            residual_history: Residual history for spectral analysis
            expected_separation: Expected separation location
            computed_separation: Computed separation location
            expected_transition: Expected transition location
            computed_transition: Computed transition location
        
        Returns:
            NumericalDissipationReport with comprehensive assessment
        """
        violations = []
        failure_reasons = []
        recommended_actions = []
        
        # 1. Analyze artificial viscosity
        av_metrics = None
        if artificial_viscosity is not None:
            av_metrics = self.analyze_artificial_viscosity(
                artificial_viscosity, physical_viscosity
            )
            violations.extend(av_metrics.violations)
        
        # 2. Analyze limiter
        limiter_metrics = None
        if limiter_values is not None:
            limiter_metrics = self.analyze_limiter(limiter_values)
            violations.extend(limiter_metrics.violations)
        
        # 3. Analyze total variation
        tv_metrics = None
        if solution_history is not None and len(solution_history) >= 2:
            tv_metrics = self.analyze_total_variation(solution_history)
            violations.extend(tv_metrics.violations)
        
        # 4. Analyze flux dissipation
        fd_metrics = None
        if flux_dissipation is not None:
            fd_metrics = self.analyze_flux_dissipation(flux_dissipation, total_energy)
            violations.extend(fd_metrics.violations)
        
        # 5. Analyze spectral damping
        spectral_metrics = None
        if residual_history is not None:
            spectral_metrics = self.analyze_spectral_damping(residual_history)
            violations.extend(spectral_metrics.violations)
        
        # 6. Detect physical effects
        separation_suppressed = False
        transition_delayed = False
        if any(m is not None for m in [av_metrics, limiter_metrics, fd_metrics]):
            separation_suppressed, transition_delayed = self.detect_physical_effects(
                expected_separation, computed_separation,
                expected_transition, computed_transition,
            )
            
            if separation_suppressed:
                violations.append(DissipationViolationType.SEPARATION_SUPPRESSED)
                failure_reasons.append("Numerical dissipation may be suppressing separation")
                recommended_actions.append("Reduce artificial dissipation or refine mesh")
            
            if transition_delayed:
                violations.append(DissipationViolationType.TRANSITION_DELAYED)
                failure_reasons.append("Numerical dissipation may be delaying transition")
                recommended_actions.append("Use lower-dissipation scheme or refine mesh")
        
        # 7. Compute overall dissipation score
        dissipation_score = self.compute_dissipation_score(
            av_metrics, limiter_metrics, tv_metrics, fd_metrics, spectral_metrics
        )
        
        # 8. Determine status
        if dissipation_score > self.config.excessive_dissipation_threshold:
            status = DissipationStatus.EXCESSIVE_DISSIPATION
            is_acceptable = False
            physical_confidence = 0.1
        elif dissipation_score > self.config.high_dissipation_threshold:
            status = DissipationStatus.HIGH_DISSIPATION
            is_acceptable = False
            physical_confidence = 0.3
        elif dissipation_score > self.config.moderate_dissipation_threshold:
            status = DissipationStatus.MODERATE_DISSIPATION
            is_acceptable = True
            physical_confidence = 0.6
        else:
            status = DissipationStatus.LOW_DISSIPATION
            is_acceptable = True
            physical_confidence = 0.9
        
        # Add failure reasons based on violations
        for v in violations:
            if v == DissipationViolationType.ARTIFICIAL_VISCOSITY_EXCESSIVE:
                failure_reasons.append(
                    f"Artificial viscosity ratio {av_metrics.artificial_viscosity_ratio:.1f} exceeds limit"
                )
                recommended_actions.append("Reduce artificial dissipation settings")
            elif v == DissipationViolationType.LIMITER_SATURATED:
                failure_reasons.append(
                    f"Limiter saturation {limiter_metrics.saturation_fraction:.1%} in {limiter_metrics.saturation_fraction*100:.0f}% of cells"
                )
                recommended_actions.append("Check for flow discontinuities or reduce CFL")
            elif v == DissipationViolationType.SEPARATION_SUPPRESSED:
                recommended_actions.append("Consider using lower-dissipation numerical scheme")
        
        return NumericalDissipationReport(
            status=status,
            artificial_viscosity=av_metrics,
            limiter=limiter_metrics,
            total_variation=tv_metrics,
            flux_dissipation=fd_metrics,
            spectral_damping=spectral_metrics,
            is_acceptable=is_acceptable,
            physical_stability_confidence=physical_confidence,
            violations=list(set(violations)),
            failure_reasons=failure_reasons,
            recommended_actions=recommended_actions,
        )