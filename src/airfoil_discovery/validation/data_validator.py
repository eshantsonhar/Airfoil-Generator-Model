"""
Automated Data Validation Filter for Aerodynamic Shape Optimization.

This module implements automated filtering to discard optimization runs that violate
physical constraints and select the best run according to specified criteria.

Rejection Criteria (Discard Run If):
- Max non-dimensional thickness t/c < 0.08
- Pressure distribution failure (Cp stays flat or max suction peak |Cp,min| < 0.5)
- Flow solver failure or non-converged residuals (ΔR > 10^-4)
- Unrealistic aerodynamic coefficients (Cl < 0.2 or Cd < 0.005)

Selection Criteria (Keep Best Run):
- Maximum reduction in Cd while satisfying Cl >= 1.0 and t/c >= 0.09
- Clean, physically smooth Cp curve demonstrating suppression/shortening of the LSB plateau
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation for a single optimization run."""
    
    # Run identification
    run_id: str
    iteration: int
    
    # Aerodynamic coefficients
    cl: float
    cd: float
    ld_ratio: float
    
    # Geometry metrics
    max_thickness: float
    max_thickness_location: float
    min_thickness: float
    
    # Pressure distribution metrics
    cp_min: float
    cp_min_location: float
    cp_suction_peak: float
    cp_smoothness: float
    
    # Convergence metrics
    residual_norm: float
    converged: bool
    
    # Validation results
    is_valid: bool
    rejection_reasons: List[str] = field(default_factory=list)
    
    # Selection score (higher is better)
    selection_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "iteration": self.iteration,
            "cl": self.cl,
            "cd": self.cd,
            "ld_ratio": self.ld_ratio,
            "max_thickness": self.max_thickness,
            "max_thickness_location": self.max_thickness_location,
            "min_thickness": self.min_thickness,
            "cp_min": self.cp_min,
            "cp_min_location": self.cp_min_location,
            "cp_suction_peak": self.cp_suction_peak,
            "cp_smoothness": self.cp_smoothness,
            "residual_norm": self.residual_norm,
            "converged": self.converged,
            "is_valid": self.is_valid,
            "rejection_reasons": self.rejection_reasons,
            "selection_score": self.selection_score,
        }


class DataValidator:
    """
    Automated data validation filter for optimization runs.
    
    This class implements validation criteria to filter out unphysical
    or corrupted optimization results and select the best run.
    """
    
    # Rejection thresholds
    MIN_THICKNESS = 0.08  # Minimum acceptable t/c
    MIN_CL = 0.2  # Minimum acceptable lift coefficient
    MIN_CD = 0.005  # Minimum acceptable drag coefficient
    MIN_CP_SUCTION = 0.5  # Minimum pressure suction peak magnitude
    MAX_RESIDUAL = 1e-4  # Maximum acceptable residual norm
    
    # Selection thresholds
    SELECTION_MIN_CL = 1.0  # Minimum CL for selection
    SELECTION_MIN_THICKNESS = 0.09  # Minimum t/c for selection
    
    def __init__(
        self,
        min_thickness: float = MIN_THICKNESS,
        min_cl: float = MIN_CL,
        min_cd: float = MIN_CD,
        min_cp_suction: float = MIN_CP_SUCTION,
        max_residual: float = MAX_RESIDUAL,
        selection_min_cl: float = SELECTION_MIN_CL,
        selection_min_thickness: float = SELECTION_MIN_THICKNESS,
    ):
        """
        Initialize the data validator.
        
        Parameters
        ----------
        min_thickness : float
            Minimum acceptable thickness ratio (t/c)
        min_cl : float
            Minimum acceptable lift coefficient
        min_cd : float
            Minimum acceptable drag coefficient
        min_cp_suction : float
            Minimum acceptable pressure suction peak magnitude
        max_residual : float
            Maximum acceptable residual norm
        selection_min_cl : float
            Minimum CL for run selection
        selection_min_thickness : float
            Minimum t/c for run selection
        """
        self.min_thickness = min_thickness
        self.min_cl = min_cl
        self.min_cd = min_cd
        self.min_cp_suction = min_cp_suction
        self.max_residual = max_residual
        self.selection_min_cl = selection_min_cl
        self.selection_min_thickness = selection_min_thickness
    
    def validate_run(
        self,
        run_id: str,
        iteration: int,
        cl: float,
        cd: float,
        max_thickness: float,
        cp_data: Optional[np.ndarray] = None,
        residual_norm: float = 0.0,
        converged: bool = True,
        **kwargs
    ) -> ValidationResult:
        """
        Validate a single optimization run.
        
        Parameters
        ----------
        run_id : str
            Identifier for the optimization run
        iteration : int
            Iteration number
        cl : float
            Lift coefficient
        cd : float
            Drag coefficient
        max_thickness : float
            Maximum thickness ratio (t/c)
        cp_data : np.ndarray, optional
            Pressure coefficient data [x, cp]
        residual_norm : float
            Residual norm from CFD solver
        converged : bool
            Whether the CFD solver converged
        **kwargs
            Additional optional parameters
            
        Returns
        -------
        ValidationResult
            Validation result with rejection reasons and selection score
        """
        rejection_reasons = []
        
        # Check thickness constraint
        if max_thickness < self.min_thickness:
            rejection_reasons.append(
                f"Thickness violation: t/c={max_thickness:.4f} < {self.min_thickness:.4f}"
            )
        
        # Check aerodynamic coefficient bounds
        if cl < self.min_cl:
            rejection_reasons.append(
                f"Lift coefficient too low: Cl={cl:.4f} < {self.min_cl:.4f}"
            )
        
        if cd < self.min_cd:
            rejection_reasons.append(
                f"Drag coefficient too low: Cd={cd:.6f} < {self.min_cd:.6f}"
            )
        
        # Check convergence
        if not converged:
            rejection_reasons.append("CFD solver did not converge")
        
        if residual_norm > self.max_residual:
            rejection_reasons.append(
                f"Residual norm too high: {residual_norm:.4e} > {self.max_residual:.4e}"
            )
        
        # Check pressure distribution if provided
        cp_min = 0.0
        cp_min_location = 0.0
        cp_suction_peak = 0.0
        cp_smoothness = 0.0
        
        if cp_data is not None and len(cp_data) > 0:
            cp_min = float(np.min(cp_data[:, 1]))
            cp_min_idx = int(np.argmin(cp_data[:, 1]))
            cp_min_location = float(cp_data[cp_min_idx, 0])
            cp_suction_peak = abs(cp_min)
            
            # Check for flat pressure distribution
            cp_range = float(np.max(cp_data[:, 1]) - np.min(cp_data[:, 1]))
            if cp_range < self.min_cp_suction:
                rejection_reasons.append(
                    f"Pressure distribution failure: Cp range={cp_range:.4f} < {self.min_cp_suction:.4f}"
                )
            
            # Check suction peak magnitude
            if cp_suction_peak < self.min_cp_suction:
                rejection_reasons.append(
                    f"Suction peak too weak: |Cp,min|={cp_suction_peak:.4f} < {self.min_cp_suction:.4f}"
                )
            
            # Compute pressure smoothness (second derivative)
            if len(cp_data) > 2:
                cp_smoothness = self._compute_cp_smoothness(cp_data)
        
        # Determine validity
        is_valid = len(rejection_reasons) == 0
        
        # Compute selection score
        selection_score = self._compute_selection_score(
            cl, cd, max_thickness, cp_suction_peak, cp_smoothness
        )
        
        return ValidationResult(
            run_id=run_id,
            iteration=iteration,
            cl=cl,
            cd=cd,
            ld_ratio=cl / cd if cd > 0 else 0.0,
            max_thickness=max_thickness,
            max_thickness_location=kwargs.get("max_thickness_location", 0.0),
            min_thickness=kwargs.get("min_thickness", 0.0),
            cp_min=cp_min,
            cp_min_location=cp_min_location,
            cp_suction_peak=cp_suction_peak,
            cp_smoothness=cp_smoothness,
            residual_norm=residual_norm,
            converged=converged,
            is_valid=is_valid,
            rejection_reasons=rejection_reasons,
            selection_score=selection_score,
        )
    
    def _compute_cp_smoothness(self, cp_data: np.ndarray) -> float:
        """
        Compute pressure distribution smoothness metric.
        
        Higher values indicate smoother pressure recovery.
        """
        if len(cp_data) < 3:
            return 0.0
        
        x = cp_data[:, 0]
        cp = cp_data[:, 1]
        
        # Compute second derivative of Cp
        d2cp = np.gradient(np.gradient(cp, x), x)
        
        # Smoothness metric: inverse of RMS second derivative
        rms_d2cp = np.sqrt(np.mean(d2cp**2))
        smoothness = 1.0 / (1.0 + rms_d2cp)
        
        return float(smoothness)
    
    def _compute_selection_score(
        self,
        cl: float,
        cd: float,
        max_thickness: float,
        cp_suction_peak: float,
        cp_smoothness: float
    ) -> float:
        """
        Compute selection score for a run.
        
        Higher scores indicate better runs according to selection criteria:
        - Maximum reduction in Cd while satisfying Cl >= 1.0 and t/c >= 0.09
        - Clean, physically smooth Cp curve
        """
        score = 0.0
        
        # Check if run meets selection criteria
        meets_cl = cl >= self.selection_min_cl
        meets_thickness = max_thickness >= self.selection_min_thickness
        
        if not (meets_cl and meets_thickness):
            # Penalize runs that don't meet selection criteria
            return -1.0 * (cd + 10.0 * (self.selection_min_cl - cl) + 10.0 * (self.selection_min_thickness - max_thickness))
        
        # Score based on drag reduction (lower Cd = higher score)
        score += 10.0 / (cd + 0.001)
        
        # Reward high L/D ratio
        ld_ratio = cl / cd if cd > 0 else 0.0
        score += ld_ratio
        
        # Reward strong suction peak
        score += cp_suction_peak
        
        # Reward smooth pressure recovery
        score += cp_smoothness
        
        return score
    
    def filter_runs(
        self,
        results: List[ValidationResult]
    ) -> Tuple[List[ValidationResult], ValidationResult]:
        """
        Filter runs and select the best one.
        
        Parameters
        ----------
        results : List[ValidationResult]
            List of validation results
            
        Returns
        -------
        valid_runs : List[ValidationResult]
            List of valid runs
        best_run : ValidationResult
            Best run according to selection criteria
        """
        # Filter valid runs
        valid_runs = [r for r in results if r.is_valid]
        
        logger.info(f"Data validation: {len(valid_runs)}/{len(results)} runs passed validation")
        
        if not valid_runs:
            logger.warning("No valid runs found. Returning best invalid run.")
            best_run = max(results, key=lambda r: r.selection_score)
            return valid_runs, best_run
        
        # Select best run
        best_run = max(valid_runs, key=lambda r: r.selection_score)
        
        logger.info(
            f"Best run: {best_run.run_id} (iteration {best_run.iteration}), "
            f"Cl={best_run.cl:.4f}, Cd={best_run.cd:.6f}, L/D={best_run.ld_ratio:.2f}, "
            f"t/c={best_run.max_thickness:.4f}, score={best_run.selection_score:.2f}"
        )
        
        return valid_runs, best_run
    
    def generate_validation_report(
        self,
        results: List[ValidationResult],
        output_path: Path
    ) -> Path:
        """
        Generate a validation report.
        
        Parameters
        ----------
        results : List[ValidationResult]
            List of validation results
        output_path : Path
            Path to save the report
            
        Returns
        -------
        Path
            Path to the generated report
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        valid_runs, best_run = self.filter_runs(results)
        
        lines = [
            "# Data Validation Report",
            f"",
            f"Total runs evaluated: {len(results)}",
            f"Valid runs: {len(valid_runs)}",
            f"Invalid runs: {len(results) - len(valid_runs)}",
            f"",
            f"## Best Run",
            f"Run ID: {best_run.run_id}",
            f"Iteration: {best_run.iteration}",
            f"Cl: {best_run.cl:.6f}",
            f"Cd: {best_run.cd:.6f}",
            f"L/D: {best_run.ld_ratio:.2f}",
            f"Max thickness: {best_run.max_thickness:.4f}",
            f"Min suction peak: {best_run.cp_suction_peak:.4f}",
            f"Selection score: {best_run.selection_score:.2f}",
            f"",
            f"## Invalid Runs",
        ]
        
        for result in results:
            if not result.is_valid:
                lines.append(
                    f"- {result.run_id} (iteration {result.iteration}): "
                    f"{', '.join(result.rejection_reasons)}"
                )
        
        lines.append("")
        lines.append("## Valid Runs Summary")
        
        for result in valid_runs:
            lines.append(
                f"- {result.run_id} (iteration {result.iteration}): "
                f"Cl={result.cl:.4f}, Cd={result.cd:.6f}, L/D={result.ld_ratio:.2f}, "
                f"t/c={result.max_thickness:.4f}, score={result.selection_score:.2f}"
            )
        
        report_text = "\n".join(lines)
        output_path.write_text(report_text, encoding="utf-8")
        
        logger.info(f"Validation report saved to {output_path}")
        
        return output_path
