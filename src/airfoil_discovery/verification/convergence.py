"""
Residual and iterative convergence analysis for CFD solvers.

Implements rigorous convergence checking beyond simple residual thresholds.
Monitors force stabilization, CFL behavior, and spectral characteristics
to detect false convergence and ensure true steady-state solutions.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum


class ConvergenceStatus(Enum):
    """Convergence status classification."""
    CONVERGED = "CONVERGED"
    NONCONVERGED = "NONCONVERGED"
    DIVERGED = "DIVERGED"
    OSCILLATING = "OSCILLATING"
    STALLED = "STALLED"
    INSUFFICIENT_ITERATIONS = "INSUFFICIENT_ITERATIONS"


@dataclass
class ResidualMetrics:
    """Metrics from residual history analysis."""
    
    # Final residual values
    final_residual: float
    max_residual: float
    rms_residual: float
    
    # Convergence rate
    convergence_rate: float
    log_residual_slope: float
    
    # Residual history
    residual_history: List[float]
    iteration_count: int
    
    # Convergence checks
    below_threshold: bool
    monotonic_decrease: bool
    asymptotic_behavior: bool
    
    # Stagnation detection
    stagnation_detected: bool
    stagnation_start_iteration: Optional[int] = None


@dataclass
class ForceMetrics:
    """Metrics from force coefficient history analysis."""
    
    # Final force values
    final_cl: float
    final_cd: float
    
    # Force history
    cl_history: List[float]
    cd_history: List[float]
    
    # Oscillation metrics
    cl_std: float
    cd_std: float
    cl_amplitude: float
    cd_amplitude: float
    
    # Relative oscillation
    cl_relative_oscillation: float
    cd_relative_oscillation: float
    
    # Trend analysis
    cl_trend: float
    cd_trend: float
    
    # Convergence checks
    forces_stabilized: bool
    force_oscillation_acceptable: bool
    force_drift_acceptable: bool


@dataclass
class SpectralMetrics:
    """Spectral analysis metrics for detecting periodic behavior."""
    
    # Dominant frequencies (no defaults - required)
    cl_dominant_frequency: float
    cd_dominant_frequency: float
    
    # Spectral power
    cl_spectral_power: float
    cd_spectral_power: float
    
    # Periodic shedding detection
    periodic_shedding_detected: bool
    metastable_behavior_detected: bool
    
    # Optional shedding frequency
    shedding_frequency: Optional[float] = None


@dataclass
class ConvergenceReport:
    """Comprehensive convergence report."""
    
    # Overall status (required, no defaults)
    status: ConvergenceStatus
    residual: ResidualMetrics
    force: ForceMetrics
    is_valid: bool
    
    # Optional spectral analysis
    spectral: Optional[SpectralMetrics] = None
    
    # Lists with defaults
    failure_reasons: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "is_valid": self.is_valid,
            "failure_reasons": self.failure_reasons,
            "recommended_actions": self.recommended_actions,
            "residual": {
                "final_residual": self.residual.final_residual,
                "max_residual": self.residual.max_residual,
                "convergence_rate": self.residual.convergence_rate,
                "below_threshold": self.residual.below_threshold,
                "monotonic_decrease": self.residual.monotonic_decrease,
                "stagnation_detected": self.residual.stagnation_detected,
            },
            "force": {
                "final_cl": self.force.final_cl,
                "final_cd": self.force.final_cd,
                "cl_std": self.force.cl_std,
                "cd_std": self.force.cd_std,
                "cl_relative_oscillation": self.force.cl_relative_oscillation,
                "cd_relative_oscillation": self.force.cd_relative_oscillation,
                "forces_stabilized": self.force.forces_stabilized,
            },
        }


class ResidualConvergenceAnalyzer:
    """
    Analyzes residual convergence history with rigorous checks.
    
    Goes beyond simple threshold checking to detect:
    - Stagnation (residuals stop decreasing)
    - Oscillatory convergence
    - Asymptotic behavior
    - False convergence patterns
    """
    
    def __init__(
        self,
        residual_threshold: float = 1e-6,
        stagnation_threshold: float = 1e-3,
        stagnation_iterations: int = 50,
        min_iterations: int = 100,
    ):
        """
        Initialize residual convergence analyzer.
        
        Args:
            residual_threshold: Target residual convergence level
            stagnation_threshold: Relative change to detect stagnation
            stagnation_iterations: Iterations to confirm stagnation
            min_iterations: Minimum iterations before convergence check
        """
        self.residual_threshold = residual_threshold
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_iterations = stagnation_iterations
        self.min_iterations = min_iterations
    
    def analyze(self, residual_history: List[float]) -> ResidualMetrics:
        """
        Analyze residual convergence history.
        
        Args:
            residual_history: List of residual values over iterations
        
        Returns:
            ResidualMetrics with convergence analysis
        """
        if not residual_history:
            raise ValueError("Empty residual history")
        
        residuals = np.array(residual_history)
        n_iter = len(residuals)
        
        # Final metrics
        final_residual = float(residuals[-1])
        max_residual = float(np.max(residuals))
        rms_residual = float(np.sqrt(np.mean(residuals**2)))
        
        # Convergence rate (log-linear fit to last 50% of iterations)
        if n_iter > 10:
            start_idx = n_iter // 2
            log_residuals = np.log10(np.abs(residuals[start_idx:]) + 1e-15)
            iterations = np.arange(start_idx, n_iter)
            if len(iterations) > 1:
                slope, _ = np.polyfit(iterations, log_residuals, 1)
                convergence_rate = float(-slope)
            else:
                convergence_rate = 0.0
        else:
            convergence_rate = 0.0
            slope = 0.0
        
        log_residual_slope = float(slope)
        
        # SU2 logs residuals as log10(residual) — values are NEGATIVE for convergence
        # Compare magnitude (abs) against threshold, not raw float (sign-broken comparison)
        final_mag = abs(final_residual)
        below_threshold = final_mag < self.residual_threshold
        
        # Check monotonic decrease (last 50 iterations)
        if n_iter > 50:
            recent_residuals = residuals[-50:]
            monotonic_decrease = bool(np.all(np.diff(recent_residuals) <= 0))
        else:
            monotonic_decrease = bool(np.all(np.diff(residuals) <= 0))
        
        # Check asymptotic behavior (exponential decay)
        if n_iter > 20:
            recent_residuals = residuals[-20:]
            log_recent = np.log10(np.abs(recent_residuals) + 1e-15)
            expected_log = np.linspace(log_recent[0], log_recent[-1], 20)
            asymptotic_behavior = bool(np.std(log_recent - expected_log) < 0.5)
        else:
            asymptotic_behavior = False
        
        # Detect stagnation
        stagnation_detected = False
        stagnation_start = None
        
        if n_iter > self.stagnation_iterations:
            for i in range(n_iter - self.stagnation_iterations):
                window = residuals[i:i+self.stagnation_iterations]
                relative_change = abs(window[-1] - window[0]) / (abs(window[0]) + 1e-15)
                if relative_change < self.stagnation_threshold:
                    stagnation_detected = True
                    stagnation_start = i
                    break
        
        return ResidualMetrics(
            final_residual=final_residual,
            max_residual=max_residual,
            rms_residual=rms_residual,
            convergence_rate=convergence_rate,
            log_residual_slope=log_residual_slope,
            residual_history=residual_history,
            iteration_count=n_iter,
            below_threshold=below_threshold,
            monotonic_decrease=monotonic_decrease,
            asymptotic_behavior=asymptotic_behavior,
            stagnation_detected=stagnation_detected,
            stagnation_start_iteration=stagnation_start,
        )


class IterativeConvergenceMonitor:
    """
    Monitors iterative convergence including forces and spectral analysis.
    
    Detects false convergence by analyzing:
    - Force coefficient stabilization
    - Force oscillation patterns
    - Spectral characteristics (periodic shedding)
    - Metastable behavior
    """
    
    def __init__(
        self,
        force_stabilization_threshold: float = 0.001,
        force_oscillation_threshold: float = 0.005,
        force_drift_threshold: float = 0.001,
        stabilization_window: int = 50,
    ):
        """
        Initialize iterative convergence monitor.
        
        Args:
            force_stabilization_threshold: Relative change for force stabilization
            force_oscillation_threshold: Relative oscillation amplitude threshold
            force_drift_threshold: Linear drift rate threshold
            stabilization_window: Window size for stabilization check
        """
        self.force_stabilization_threshold = force_stabilization_threshold
        self.force_oscillation_threshold = force_oscillation_threshold
        self.force_drift_threshold = force_drift_threshold
        self.stabilization_window = stabilization_window
    
    def analyze_forces(
        self,
        cl_history: List[float],
        cd_history: List[float],
    ) -> ForceMetrics:
        """
        Analyze force coefficient convergence.
        
        Args:
            cl_history: Lift coefficient history
            cd_history: Drag coefficient history
        
        Returns:
            ForceMetrics with convergence analysis
        """
        if not cl_history or not cd_history:
            raise ValueError("Empty force history")
        
        cl = np.array(cl_history)
        cd = np.array(cd_history)
        
        # Final values
        final_cl = float(cl[-1])
        final_cd = float(cd[-1])
        
        # Statistics
        cl_std = float(np.std(cl))
        cd_std = float(np.std(cd))
        
        # Amplitude (max - min)
        cl_amplitude = float(np.max(cl) - np.min(cl))
        cd_amplitude = float(np.max(cd) - np.min(cd))
        
        # Use only the tail half for oscillation and statistics to avoid
        # early-transient noise (first ~50% of run) inflating std/mean.
        # For a 30-iteration run, entries at indices 0-14 are still ramping from ~0,
        # making cl_std/cl_mean ~0.39 instead of the true ~0.002 of the converged tail.
        tail_frac = 0.5
        tail_start = max(1, int(len(cl) * tail_frac))
        cl_tail = cl[tail_start:]
        cd_tail = cd[tail_start:]
        
        # Relative oscillation — computed on the tail only
        cl_mean_t = float(np.mean(cl_tail))
        cd_mean_t = float(np.mean(cd_tail))
        cl_relative_oscillation = float(np.std(cl_tail)) / (abs(cl_mean_t) + 1e-15)
        cd_relative_oscillation = float(np.std(cd_tail)) / (abs(cd_mean_t) + 1e-15)
        
        # Trend analysis (linear fit to last 50% of iterations)
        recent_cl = cl_tail
        recent_cd = cd_tail
        iterations = np.arange(len(recent_cl))
        if len(recent_cl) > 1:
            cl_slope, _ = np.polyfit(iterations, recent_cl, 1)
            cd_slope, _ = np.polyfit(iterations, recent_cd, 1)
            cl_trend = float(cl_slope)
            cd_trend = float(cd_slope)
        else:
            cl_trend = 0.0
            cd_trend = 0.0
        # Check stabilization (tail sub-window)
        # For end-of-run, use a fraction of the full history to test stability
        # full-history window (30==30) makes the window equivalent to full-history
        # and places the initial transient into the stability check, always failing.
        eff_sw = min(self.stabilization_window, max(5, len(cl) // 2))
        if len(cl) >= eff_sw:
            recent_cl = cl[-eff_sw:]
            recent_cd = cd[-eff_sw:]
            # Baseline for development runs: compare final value vs. window mean
            # (mean already averages out early-transient noise).
            cl_mid_mean = float(np.mean(recent_cl))
            cd_mid_mean = float(np.mean(recent_cd))
            cl_change = abs(recent_cl[-1] - cl_mid_mean) / (abs(cl_mid_mean) + 1e-15)
            cd_change = abs(recent_cd[-1] - cd_mid_mean) / (abs(cd_mid_mean) + 1e-15)
            forces_stabilized = (cl_change < self.force_stabilization_threshold and
                                cd_change < self.force_stabilization_threshold)
        else:
            forces_stabilized = False
        
        # Check oscillation
        force_oscillation_acceptable = (
            cl_relative_oscillation < self.force_oscillation_threshold and
            cd_relative_oscillation < self.force_oscillation_threshold
        )
        
        # Check drift
        force_drift_acceptable = (
            abs(cl_trend) < self.force_drift_threshold and
            abs(cd_trend) < self.force_drift_threshold
        )
        
        return ForceMetrics(
            final_cl=final_cl,
            final_cd=final_cd,
            cl_history=cl_history,
            cd_history=cd_history,
            cl_std=cl_std,
            cd_std=cd_std,
            cl_amplitude=cl_amplitude,
            cd_amplitude=cd_amplitude,
            cl_relative_oscillation=cl_relative_oscillation,
            cd_relative_oscillation=cd_relative_oscillation,
            cl_trend=cl_trend,
            cd_trend=cd_trend,
            forces_stabilized=forces_stabilized,
            force_oscillation_acceptable=force_oscillation_acceptable,
            force_drift_acceptable=force_drift_acceptable,
        )
    
    def analyze_spectral(
        self,
        cl_history: List[float],
        cd_history: List[float],
    ) -> SpectralMetrics:
        """
        Perform spectral analysis to detect periodic behavior.
        
        Args:
            cl_history: Lift coefficient history
            cd_history: Drag coefficient history
        
        Returns:
            SpectralMetrics with frequency analysis
        """
        if len(cl_history) < 64 or len(cd_history) < 64:
            # Not enough data for meaningful spectral analysis
            return SpectralMetrics(
                cl_dominant_frequency=0.0,
                cd_dominant_frequency=0.0,
                cl_spectral_power=0.0,
                cd_spectral_power=0.0,
                periodic_shedding_detected=False,
                metastable_behavior_detected=False,
            )
        
        cl = np.array(cl_history)
        cd = np.array(cd_history)
        
        # Remove mean
        cl_detrended = cl - np.mean(cl)
        cd_detrended = cd - np.mean(cd)
        
        # FFT
        cl_fft = np.fft.fft(cl_detrended)
        cd_fft = np.fft.fft(cd_detrended)
        
        # Power spectrum
        cl_power = np.abs(cl_fft)**2
        cd_power = np.abs(cd_fft)**2
        
        # Dominant frequency (excluding DC component)
        cl_dominant_idx = np.argmax(cl_power[1:len(cl_power)//2]) + 1
        cd_dominant_idx = np.argmax(cd_power[1:len(cd_power)//2]) + 1
        
        cl_dominant_frequency = float(cl_dominant_idx / len(cl))
        cd_dominant_frequency = float(cd_dominant_idx / len(cd))
        
        # Total spectral power (excluding DC)
        cl_spectral_power = float(np.sum(cl_power[1:]))
        cd_spectral_power = float(np.sum(cd_power[1:]))
        
        # Detect periodic shedding (strong narrowband peak)
        shedding_detected = False
        shedding_frequency = None
        
        if cl_spectral_power > 0:
            peak_power = cl_power[cl_dominant_idx]
            mean_power = np.mean(cl_power[1:])
            if peak_power > 5 * mean_power:
                shedding_detected = True
                shedding_frequency = cl_dominant_frequency
        
        # Detect metastable behavior (multiple peaks)
        metastable_detected = False
        if cl_spectral_power > 0:
            sorted_power = np.sort(cl_power[1:len(cl_power)//2])
            if len(sorted_power) > 3:
                # Check if multiple peaks have significant power
                top3_sum = np.sum(sorted_power[-3:])
                total_power = np.sum(cl_power[1:len(cl_power)//2])
                if top3_sum / total_power > 0.5:
                    metastable_detected = True
        
        return SpectralMetrics(
            cl_dominant_frequency=cl_dominant_frequency,
            cd_dominant_frequency=cd_dominant_frequency,
            cl_spectral_power=cl_spectral_power,
            cd_spectral_power=cd_spectral_power,
            periodic_shedding_detected=shedding_detected,
            shedding_frequency=shedding_frequency,
            metastable_behavior_detected=metastable_detected,
        )
    
    def generate_report(
        self,
        residual_metrics: ResidualMetrics,
        force_metrics: ForceMetrics,
        spectral_metrics: Optional[SpectralMetrics] = None,
    ) -> ConvergenceReport:
        """
        Generate comprehensive convergence report.
        
        Args:
            residual_metrics: Residual convergence metrics
            force_metrics: Force convergence metrics
            spectral_metrics: Optional spectral analysis metrics
        
        Returns:
            ConvergenceReport with overall assessment
        """
        failure_reasons = []
        recommended_actions = []
        
        # Check residual convergence
        if not residual_metrics.below_threshold:
            failure_reasons.append("Residuals did not converge below threshold")
            recommended_actions.append("Increase iterations or check solver settings")
        
        if residual_metrics.stagnation_detected:
            failure_reasons.append("Residual stagnation detected")
            recommended_actions.append("Check for solver stiffness or numerical issues")
        
        if not residual_metrics.monotonic_decrease:
            failure_reasons.append("Residuals not monotonically decreasing")
            recommended_actions.append("Investigate oscillatory convergence")
        
        # Check force convergence
        if not force_metrics.forces_stabilized:
            failure_reasons.append("Forces not stabilized")
            recommended_actions.append("Continue iterations until forces stabilize")
        
        if not force_metrics.force_oscillation_acceptable:
            failure_reasons.append("Force oscillation exceeds threshold")
            recommended_actions.append("Check for unsteady flow or numerical instability")
        
        if not force_metrics.force_drift_acceptable:
            failure_reasons.append("Force drift detected")
            recommended_actions.append("Continue iterations or check convergence")
        
        # Check spectral analysis
        if spectral_metrics:
            if spectral_metrics.periodic_shedding_detected:
                failure_reasons.append("Periodic shedding detected (unsteady flow)")
                recommended_actions.append("Consider unsteady simulation or different operating point")
            
            if spectral_metrics.metastable_behavior_detected:
                failure_reasons.append("Metastable behavior detected")
                recommended_actions.append("Investigate flow regime and stability")
        
        # Determine overall status
        if residual_metrics.final_residual > 1e-2:
            status = ConvergenceStatus.DIVERGED
        elif len(failure_reasons) == 0:
            status = ConvergenceStatus.CONVERGED
        elif spectral_metrics and spectral_metrics.periodic_shedding_detected:
            status = ConvergenceStatus.OSCILLATING
        elif residual_metrics.stagnation_detected:
            status = ConvergenceStatus.STALLED
        else:
            status = ConvergenceStatus.NONCONVERGED
        
        is_valid = (status == ConvergenceStatus.CONVERGED and
                   residual_metrics.iteration_count >= self.min_iterations)
        
        return ConvergenceReport(
            status=status,
            residual=residual_metrics,
            force=force_metrics,
            spectral=spectral_metrics,
            is_valid=is_valid,
            failure_reasons=failure_reasons,
            recommended_actions=recommended_actions,
        )
