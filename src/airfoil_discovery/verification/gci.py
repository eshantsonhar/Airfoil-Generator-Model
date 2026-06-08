"""
Grid Convergence Index (GCI) and Richardson Extrapolation.

Implements ASME V&V 20-2009 standard for grid convergence studies.
Provides observed order of accuracy, extrapolated values, and
uncertainty estimates for CFD solutions.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import json


@dataclass
class GCIResult:
    """Results from Grid Convergence Index analysis."""
    
    # Grid information
    grid_sizes: list[int]
    refinement_ratios: list[float]
    
    # Solution values on each grid
    fine_value: float
    medium_value: float
    coarse_value: float
    
    # Convergence metrics
    observed_order: float
    
    # Theoretical order (default 2.0 for second-order)
    theoretical_order: float = 2.0
    
    # Extrapolation (defaults for NaN when not computed)
    extrapolated_value: float = 0.0
    extrapolation_error: float = 0.0
    
    # GCI values
    gci_fine_medium: float = 0.0
    gci_medium_coarse: float = 0.0
    
    # Convergence check
    convergence_ratio: float = 0.0
    is_monotonic: bool = False
    is_asymptotic: bool = False
    
    # Uncertainty
    numerical_uncertainty: float = 0.0
    relative_uncertainty: float = 0.0
    
    # Validation
    passed_asymptotic_range: bool = False
    passed_order_check: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "grid_sizes": self.grid_sizes,
            "refinement_ratios": self.refinement_ratios,
            "fine_value": self.fine_value,
            "medium_value": self.medium_value,
            "coarse_value": self.coarse_value,
            "observed_order": self.observed_order,
            "theoretical_order": self.theoretical_order,
            "extrapolated_value": self.extrapolated_value,
            "extrapolation_error": self.extrapolation_error,
            "gci_fine_medium": self.gci_fine_medium,
            "gci_medium_coarse": self.gci_medium_coarse,
            "convergence_ratio": self.convergence_ratio,
            "is_monotonic": self.is_monotonic,
            "is_asymptotic": self.is_asymptotic,
            "numerical_uncertainty": self.numerical_uncertainty,
            "relative_uncertainty": self.relative_uncertainty,
            "passed_asymptotic_range": self.passed_asymptotic_range,
            "passed_order_check": self.passed_order_check,
        }


class GridConvergenceIndex:
    """
    Implements Grid Convergence Index (GCI) analysis per ASME V&V 20-2009.
    
    The GCI provides a measure of numerical uncertainty due to discretization.
    It requires solutions on at least three systematically refined grids.
    
    Usage:
        analyzer = GridConvergenceIndex()
        result = analyzer.compute(
            fine_value=1.234,
            medium_value=1.245,
            coarse_value=1.267,
            grid_sizes=[100000, 50000, 25000],
            safety_factor=1.25
        )
    """
    
    def __init__(self, safety_factor: float = 1.25):
        """
        Initialize GCI analyzer.
        
        Args:
            safety_factor: Fs from ASME standard (default 1.25 for 3+ grids)
        """
        self.safety_factor = safety_factor
        self.asymptotic_range_lower = 0.5
        self.asymptotic_range_upper = 1.5
    
    def compute(
        self,
        fine_value: float,
        medium_value: float,
        coarse_value: float,
        grid_sizes: list[int],
        theoretical_order: float = 2.0,
    ) -> GCIResult:
        """
        Compute GCI for three-grid convergence study.
        
        Args:
            fine_value: Solution on finest grid
            medium_value: Solution on medium grid
            coarse_value: Solution on coarsest grid
            grid_sizes: Number of cells for [fine, medium, coarse]
            theoretical_order: Expected order of accuracy (default 2.0)
        
        Returns:
            GCIResult with all convergence metrics
        """
        # Compute refinement ratios
        r12 = grid_sizes[1] / grid_sizes[0]  # medium/fine
        r23 = grid_sizes[2] / grid_sizes[1]  # coarse/medium
        
        refinement_ratios = [1.0, r12, r23]
        
        # Compute observed order of accuracy
        epsilon1 = abs(fine_value - medium_value)
        epsilon2 = abs(medium_value - coarse_value)
        
        if epsilon1 > 1e-15 and epsilon2 > 1e-15:
            p = np.log(epsilon2 / epsilon1) / np.log(r12)
        else:
            p = theoretical_order
        
        # Extrapolated value (Richardson)
        if abs(p) > 1e-10:
            extrapolated = fine_value + (fine_value - medium_value) / (r12**p - 1)
            extrapolation_error = abs(extrapolated - fine_value)
        else:
            extrapolated = fine_value
            extrapolation_error = 0.0
        
        # GCI for fine-medium
        if abs(p) > 1e-10:
            gci_fine = self.safety_factor * abs(fine_value - medium_value) / (
                abs(fine_value) * (r12**p - 1)
            )
        else:
            gci_fine = 0.0
        
        # GCI for medium-coarse
        if abs(p) > 1e-10:
            gci_coarse = self.safety_factor * abs(medium_value - coarse_value) / (
                abs(medium_value) * (r23**p - 1)
            )
        else:
            gci_coarse = 0.0
        
        # Convergence ratio (should be ~1 for asymptotic convergence)
        if gci_fine > 1e-15:
            convergence_ratio = gci_coarse / (r12**p * gci_fine)
        else:
            convergence_ratio = 0.0
        
        # Check monotonic convergence
        is_monotonic = (fine_value < medium_value < coarse_value) or \
                       (fine_value > medium_value > coarse_value)
        
        # Check asymptotic range (convergence ratio should be in [0.5, 1.5])
        is_asymptotic = (self.asymptotic_range_lower <= convergence_ratio <= 
                         self.asymptotic_range_upper)
        
        # Numerical uncertainty (use fine-medium GCI)
        numerical_uncertainty = abs(fine_value) * gci_fine
        relative_uncertainty = gci_fine
        
        # Validation checks
        passed_asymptotic = is_asymptotic
        passed_order = abs(p - theoretical_order) / theoretical_order < 0.5
        
        return GCIResult(
            grid_sizes=grid_sizes,
            refinement_ratios=refinement_ratios,
            fine_value=fine_value,
            medium_value=medium_value,
            coarse_value=coarse_value,
            observed_order=float(p),
            theoretical_order=theoretical_order,
            extrapolated_value=float(extrapolated),
            extrapolation_error=float(extrapolation_error),
            gci_fine_medium=float(gci_fine),
            gci_medium_coarse=float(gci_coarse),
            convergence_ratio=float(convergence_ratio),
            is_monotonic=is_monotonic,
            is_asymptotic=is_asymptotic,
            numerical_uncertainty=float(numerical_uncertainty),
            relative_uncertainty=float(relative_uncertainty),
            passed_asymptotic_range=passed_asymptotic,
            passed_order_check=passed_order,
        )
    
    def compute_multi_variable(
        self,
        values: Dict[str, tuple[float, float, float]],
        grid_sizes: list[int],
        theoretical_order: float = 2.0,
    ) -> Dict[str, GCIResult]:
        """
        Compute GCI for multiple variables simultaneously.
        
        Args:
            values: Dict mapping variable names to (fine, medium, coarse) tuples
            grid_sizes: Number of cells for [fine, medium, coarse]
            theoretical_order: Expected order of accuracy
        
        Returns:
            Dict mapping variable names to GCIResult objects
        """
        results = {}
        for var_name, (fine, medium, coarse) in values.items():
            results[var_name] = self.compute(
                fine_value=fine,
                medium_value=medium,
                coarse_value=coarse,
                grid_sizes=grid_sizes,
                theoretical_order=theoretical_order,
            )
        return results


class RichardsonExtrapolation:
    """
    Richardson extrapolation for grid-independent solutions.
    
    Uses the observed order of accuracy to extrapolate to zero grid spacing.
    Provides uncertainty estimates and convergence diagnostics.
    """
    
    def __init__(self):
        """Initialize Richardson extrapolation analyzer."""
        pass
    
    def extrapolate(
        self,
        fine_value: float,
        medium_value: float,
        refinement_ratio: float,
        observed_order: float,
    ) -> tuple[float, float]:
        """
        Perform Richardson extrapolation to zero grid spacing.
        
        Args:
            fine_value: Solution on fine grid
            medium_value: Solution on medium grid
            refinement_ratio: h_medium / h_fine (should be > 1)
            observed_order: Observed order of accuracy p
        
        Returns:
            (extrapolated_value, uncertainty_estimate)
        """
        if abs(observed_order) < 1e-10 or abs(refinement_ratio**observed_order - 1) < 1e-10:
            return fine_value, 0.0
        
        extrapolated = fine_value + (fine_value - medium_value) / (
            refinement_ratio**observed_order - 1
        )
        
        # Uncertainty estimate (simplified)
        uncertainty = abs(extrapolated - fine_value)
        
        return float(extrapolated), float(uncertainty)
    
    def estimate_order(
        self,
        fine_value: float,
        medium_value: float,
        coarse_value: float,
        r12: float,
        r23: float,
    ) -> float:
        """
        Estimate observed order of accuracy from three solutions.
        
        Args:
            fine_value: Solution on fine grid
            medium_value: Solution on medium grid
            coarse_value: Solution on coarse grid
            r12: Refinement ratio h2/h1
            r23: Refinement ratio h3/h2
        
        Returns:
            Observed order of accuracy p
        """
        epsilon1 = abs(fine_value - medium_value)
        epsilon2 = abs(medium_value - coarse_value)
        
        if epsilon1 > 1e-15 and epsilon2 > 1e-15:
            p = np.log(epsilon2 / epsilon1) / np.log(r12)
            return float(p)
        
        return 2.0  # Default to second order
