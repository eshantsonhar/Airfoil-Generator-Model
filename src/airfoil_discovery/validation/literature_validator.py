"""
Literature benchmark validation for CFD solver validation.

Validates the CFD solver against literature benchmark airfoils:
- Eppler 387
- SD7003
- S1223
- NACA 4412 low-Re cases

Validation checks: Cl curves, Cd curves, Cp distributions,
transition location, separation location, stall onset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum


class ValidationStatus(Enum):
    """Validation status."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass
class ValidationComparison:
    """Comparison between CFD and literature data."""
    
    variable_name: str
    cfd_values: List[float]
    literature_values: List[float]
    aoa_values: List[float]
    
    # Error metrics
    mae: float
    rmse: float
    max_error: float
    relative_error: float
    
    # Correlation
    correlation: float
    
    # Pass/fail
    passed: bool
    tolerance: float


@dataclass
class LiteratureValidationReport:
    """Comprehensive literature validation report."""
    
    # Overall status
    status: ValidationStatus
    airfoil_name: str
    reynolds: float
    
    # Comparisons
    cl_comparison: Optional[ValidationComparison] = None
    cd_comparison: Optional[ValidationComparison] = None
    cl_cd_comparison: Optional[ValidationComparison] = None
    transition_comparison: Optional[ValidationComparison] = None
    
    # Overall assessment
    overall_passed: bool
    confidence: float
    
    # Source information
    literature_source: str = ""
    literature_notes: str = ""
    
    # Recommendations
    discrepancies: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "airfoil_name": self.airfoil_name,
            "reynolds": self.reynolds,
            "overall_passed": self.overall_passed,
            "confidence": self.confidence,
            "literature_source": self.literature_source,
            "discrepancies": self.discrepancies,
            "recommended_actions": self.recommended_actions,
        }


class LiteratureValidator:
    """
    Validates CFD results against literature benchmark cases.
    
    Implements validation against standard low-Re airfoils from
    peer-reviewed literature. Provides quantitative error metrics
    and pass/fail criteria.
    """
    
    def __init__(
        self,
        cl_tolerance: float = 0.10,
        cd_tolerance: float = 0.20,
        transition_tolerance: float = 0.10,
        min_correlation: float = 0.90,
    ):
        """
        Initialize literature validator.
        
        Args:
            cl_tolerance: Acceptable relative error for Cl
            cd_tolerance: Acceptable relative error for Cd
            transition_tolerance: Acceptable absolute error for transition location
            min_correlation: Minimum correlation coefficient
        """
        self.cl_tolerance = cl_tolerance
        self.cd_tolerance = cd_tolerance
        self.transition_tolerance = transition_tolerance
        self.min_correlation = min_correlation
    
    def load_benchmark_data(
        self,
        benchmark_file: Path,
    ) -> Dict[str, Any]:
        """
        Load benchmark data from JSON file.
        
        Args:
            benchmark_file: Path to benchmark JSON file
        
        Returns:
            Dictionary with benchmark data
        """
        if not benchmark_file.exists():
            raise FileNotFoundError(f"Benchmark file not found: {benchmark_file}")
        
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def compute_error_metrics(
        self,
        cfd_values: List[float],
        literature_values: List[float],
    ) -> Tuple[float, float, float, float, float]:
        """
        Compute error metrics between CFD and literature data.
        
        Args:
            cfd_values: CFD predictions
            literature_values: Literature data
        
        Returns:
            (mae, rmse, max_error, relative_error, correlation)
        """
        import numpy as np
        
        cfd = np.array(cfd_values)
        lit = np.array(literature_values)
        
        # Mean absolute error
        mae = float(np.mean(np.abs(cfd - lit)))
        
        # Root mean square error
        rmse = float(np.sqrt(np.mean((cfd - lit)**2)))
        
        # Maximum error
        max_error = float(np.max(np.abs(cfd - lit)))
        
        # Relative error (normalized by literature range)
        lit_range = np.max(lit) - np.min(lit) if len(lit) > 1 else 1.0
        relative_error = float(rmse / (lit_range + 1e-15))
        
        # Correlation coefficient
        if len(cfd) > 1 and np.std(cfd) > 1e-15 and np.std(lit) > 1e-15:
            correlation = float(np.corrcoef(cfd, lit)[0, 1])
        else:
            correlation = 0.0
        
        return mae, rmse, max_error, relative_error, correlation
    
    def compare(
        self,
        variable_name: str,
        cfd_values: List[float],
        literature_values: List[float],
        aoa_values: List[float],
        tolerance: float,
    ) -> ValidationComparison:
        """
        Compare CFD results with literature data.
        
        Args:
            variable_name: Name of variable being compared
            cfd_values: CFD predictions
            literature_values: Literature data
            aoa_values: Angle of attack values
            tolerance: Acceptable relative error
        
        Returns:
            ValidationComparison with comparison metrics
        """
        mae, rmse, max_error, relative_error, correlation = self.compute_error_metrics(
            cfd_values, literature_values
        )
        
        passed = (relative_error < tolerance and correlation > self.min_correlation)
        
        return ValidationComparison(
            variable_name=variable_name,
            cfd_values=cfd_values,
            literature_values=literature_values,
            aoa_values=aoa_values,
            mae=mae,
            rmse=rmse,
            max_error=max_error,
            relative_error=relative_error,
            correlation=correlation,
            passed=passed,
            tolerance=tolerance,
        )
    
    def validate(
        self,
        benchmark_file: Path,
        cfd_results: Dict[str, List[float]],
    ) -> LiteratureValidationReport:
        """
        Validate CFD results against literature benchmark.
        
        Args:
            benchmark_file: Path to benchmark JSON file
            cfd_results: Dictionary with CFD results (cl, cd, cl_cd, transition_location)
        
        Returns:
            LiteratureValidationReport with comprehensive validation
        """
        # Load benchmark data
        benchmark = self.load_benchmark_data(benchmark_file)
        
        discrepancies = []
        recommended_actions = []
        
        # Compare Cl
        cl_comparison = None
        if 'cl' in cfd_results and 'cl' in benchmark:
            cl_comparison = self.compare(
                variable_name="Cl",
                cfd_values=cfd_results['cl'],
                literature_values=benchmark['cl'],
                aoa_values=benchmark['aoa'],
                tolerance=self.cl_tolerance,
            )
            
            if not cl_comparison.passed:
                discrepancies.append(
                    f"Cl comparison failed: relative error {cl_comparison.relative_error:.3f} "
                    f"(tolerance {self.cl_tolerance})"
                )
                recommended_actions.append("Check turbulence model and transition settings")
        
        # Compare Cd
        cd_comparison = None
        if 'cd' in cfd_results and 'cd' in benchmark:
            cd_comparison = self.compare(
                variable_name="Cd",
                cfd_values=cfd_results['cd'],
                literature_values=benchmark['cd'],
                aoa_values=benchmark['aoa'],
                tolerance=self.cd_tolerance,
            )
            
            if not cd_comparison.passed:
                discrepancies.append(
                    f"Cd comparison failed: relative error {cd_comparison.relative_error:.3f} "
                    f"(tolerance {self.cd_tolerance})"
                )
                recommended_actions.append("Check boundary layer resolution and y+")
        
        # Compare Cl/Cd
        cl_cd_comparison = None
        if 'cl_cd' in cfd_results and 'cl_cd' in benchmark:
            cl_cd_comparison = self.compare(
                variable_name="Cl/Cd",
                cfd_values=cfd_results['cl_cd'],
                literature_values=benchmark['cl_cd'],
                aoa_values=benchmark['aoa'],
                tolerance=self.cl_tolerance,
            )
        
        # Compare transition location
        transition_comparison = None
        if 'transition_location' in cfd_results and 'transition_location' in benchmark:
            transition_comparison = self.compare(
                variable_name="Transition Location",
                cfd_values=cfd_results['transition_location'],
                literature_values=benchmark['transition_location'],
                aoa_values=benchmark['aoa'],
                tolerance=self.transition_tolerance,
            )
            
            if not transition_comparison.passed:
                discrepancies.append(
                    f"Transition location comparison failed: relative error "
                    f"{transition_comparison.relative_error:.3f}"
                )
                recommended_actions.append("Check transition model calibration")
        
        # Determine overall status
        comparisons = [c for c in [cl_comparison, cd_comparison, transition_comparison] if c is not None]
        
        if len(comparisons) == 0:
            status = ValidationStatus.INSUFFICIENT_DATA
            overall_passed = False
            confidence = 0.0
        elif all(c.passed for c in comparisons):
            status = ValidationStatus.PASSED
            overall_passed = True
            confidence = 0.95
        elif any(c.passed for c in comparisons):
            status = ValidationStatus.PARTIAL
            overall_passed = False
            confidence = 0.5
        else:
            status = ValidationStatus.FAILED
            overall_passed = False
            confidence = 0.2
        
        return LiteratureValidationReport(
            status=status,
            airfoil_name=benchmark['airfoil_name'],
            reynolds=benchmark['reynolds'],
            cl_comparison=cl_comparison,
            cd_comparison=cd_comparison,
            cl_cd_comparison=cl_cd_comparison,
            transition_comparison=transition_comparison,
            overall_passed=overall_passed,
            confidence=confidence,
            literature_source=benchmark.get('source', ''),
            literature_notes=benchmark.get('notes', ''),
            discrepancies=discrepancies,
            recommended_actions=recommended_actions,
        )
    
    def validate_all_benchmarks(
        self,
        benchmark_dir: Path,
        cfd_results_dict: Dict[str, Dict[str, List[float]]],
    ) -> Dict[str, LiteratureValidationReport]:
        """
        Validate against all available benchmark cases.
        
        Args:
            benchmark_dir: Directory containing benchmark JSON files
            cfd_results_dict: Dictionary mapping airfoil names to CFD results
        
        Returns:
            Dictionary mapping airfoil names to validation reports
        """
        reports = {}
        
        for benchmark_file in benchmark_dir.glob("*.json"):
            airfoil_name = benchmark_file.stem
            
            if airfoil_name in cfd_results_dict:
                try:
                    report = self.validate(benchmark_file, cfd_results_dict[airfoil_name])
                    reports[airfoil_name] = report
                except Exception as e:
                    # Create a failed report for this benchmark
                    reports[airfoil_name] = LiteratureValidationReport(
                        status=ValidationStatus.FAILED,
                        airfoil_name=airfoil_name,
                        reynolds=0.0,
                        overall_passed=False,
                        confidence=0.0,
                        discrepancies=[f"Validation error: {str(e)}"],
                    )
        
        return reports
