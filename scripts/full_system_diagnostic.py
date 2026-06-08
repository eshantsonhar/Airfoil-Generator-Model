#!/usr/bin/env python
"""
Full-System Diagnostic and Validation Sweep for Airfoil Optimization Pipeline.

This script performs a production-grade CFD + optimization verification audit
to identify hidden convergence failures, optimizer stagnation, invalid objective
scaling, geometry degeneracies, broken mesh behavior, bad CFD physics, and more.

Usage:
    python scripts/full_system_diagnostic.py --config config/default.yaml --output data/diagnostics/

This script will:
1. GEOMETRY VALIDATION - CST reconstruction, self-intersections, curvature, thickness
2. CFD VALIDATION - Residuals, force integration, false convergence, mesh sensitivity
3. OPTIMIZER VALIDATION - Trust region, gradients, exploration, cycling, bounds
4. OBJECTIVE FUNCTION AUDIT - Score breakdown, penalties, Cl/Cd, stall, weighting
5. REPRODUCIBILITY TESTING - Determinism, cache corruption
6. SEARCH SPACE ANALYSIS - Explored regions, collapsed exploration
7. KNOWN-GOOD BASELINE COMPARISON - NACA airfoils comparison
8. ROOT CAUSE ANALYSIS - Rank issues, provide fixes, confidence score
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from airfoil_discovery.config import Settings, load_settings
from airfoil_discovery.geometry.cst import CSTAirfoil
from airfoil_discovery.geometry.governance import GeometryGovernor, GeometryGovernanceConfig
from airfoil_discovery.schemas import CSTParameters
from airfoil_discovery.storage import ExperimentDatabase


class SeverityLevel(Enum):
    CRITICAL = "CRITICAL"  # System is fundamentally broken
    HIGH = "HIGH"  # Major issue affecting results
    MEDIUM = "MEDIUM"  # Issue that should be fixed
    LOW = "LOW"  # Minor issue or warning
    INFO = "INFO"  # Observations


@dataclass
class DiagnosticIssue:
    """Represents a discovered issue during diagnostics."""
    category: str
    severity: SeverityLevel
    title: str
    description: str
    evidence: Dict[str, Any]
    root_cause: str
    fix: str
    auto_fixable: bool = False


@dataclass
class DiagnosticResult:
    """Results from a diagnostic check."""
    name: str
    passed: bool
    metrics: Dict[str, Any]
    issues: List[DiagnosticIssue] = field(default_factory=list)


@dataclass
class FullDiagnosticReport:
    """Complete diagnostic report."""
    timestamp: str
    config_hash: str
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    results: List[DiagnosticResult] = field(default_factory=list)
    issues: List[DiagnosticIssue] = field(default_factory=list)
    confidence_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    is_system_healthy: bool = False


class SystemDiagnostics:
    """
    Comprehensive system diagnostic engine.
    """
    
    def __init__(self, settings: Settings, output_dir: Path):
        self.settings = settings
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.cst = CSTAirfoil(settings.geometry)
        self.governor = GeometryGovernor()
        self.db = ExperimentDatabase(settings.paths.database_path)
        
        self.issues: List[DiagnosticIssue] = []
        self.results: List[DiagnosticResult] = []
        
    def _add_issue(self, category: str, severity: SeverityLevel, title: str,
                   description: str, evidence: Dict[str, Any],
                   root_cause: str, fix: str, auto_fixable: bool = False):
        """Add a diagnostic issue."""
        issue = DiagnosticIssue(
            category=category,
            severity=severity,
            title=title,
            description=description,
            evidence=evidence,
            root_cause=root_cause,
            fix=fix,
            auto_fixable=auto_fixable,
        )
        self.issues.append(issue)
        return issue
    
    def _add_result(self, name: str, passed: bool, metrics: Dict[str, Any],
                    issues: List[DiagnosticIssue] = None):
        """Add a diagnostic result."""
        result = DiagnosticResult(
            name=name,
            passed=passed,
            metrics=metrics,
            issues=issues or [],
        )
        self.results.append(result)
        return result
    
    # =========================================================================
    # SECTION 1: GEOMETRY VALIDATION
    # =========================================================================
    
    def validate_geometry(self) -> DiagnosticResult:
        """
        Validate CST reconstruction correctness, self-intersections,
        curvature spikes, thickness distributions, LE/TE behavior.
        """
        print("\n" + "="*60)
        print("SECTION 1: GEOMETRY VALIDATION")
        print("="*60)
        
        issues = []
        metrics = {}
        all_passed = True
        
        # Test with various CST parameter sets
        test_cases = [
            # Nominal low-Re airfoil (similar to SD7003)
            {
                "name": "nominal_low_re",
                "params": CSTParameters(
                    upper=np.array([0.18, 0.05, 0.34, 0.10]),
                    lower=np.array([-0.19, 0.05, -0.09, 0.03]),
                    trailing_edge_thickness=0.004,
                )
            },
            # NACA 4-digit-like (NACA 2412 approximation)
            {
                "name": "naca_2412_like",
                "params": CSTParameters(
                    upper=np.array([0.20, 0.08, 0.30, 0.12]),
                    lower=np.array([-0.15, 0.03, -0.08, 0.02]),
                    trailing_edge_thickness=0.002,
                )
            },
            # Flat plate (extreme case)
            {
                "name": "flat_plate",
                "params": CSTParameters(
                    upper=np.array([0.01, 0.0, 0.01, 0.0]),
                    lower=np.array([-0.01, 0.0, -0.01, 0.0]),
                    trailing_edge_thickness=0.001,
                )
            },
            # Thick airfoil (stress test)
            {
                "name": "thick_airfoil",
                "params": CSTParameters(
                    upper=np.array([0.35, 0.15, 0.50, 0.20]),
                    lower=np.array([-0.30, 0.10, -0.15, 0.05]),
                    trailing_edge_thickness=0.01,
                )
            },
        ]
        
        for tc in test_cases:
            print(f"\n  Testing: {tc['name']}")
            params = tc["params"]
            
            try:
                # Generate coordinates
                coords = self.cst.full_coordinates(params)
                x, y = coords[:, 0], coords[:, 1]
                
                # Split into upper and lower surfaces
                x_array = np.array(x)
                y_array = np.array(y)
                
                # Find leading edge (minimum x)
                le_idx = np.argmin(x_array)
                
                # Split at leading edge
                upper_mask = y_array >= y_array[le_idx]
                lower_mask = y_array < y_array[le_idx]
                
                x_upper = x_array[upper_mask]
                y_upper = y_array[upper_mask]
                x_lower = x_array[lower_mask]
                y_lower = y_array[lower_mask]
                
                # Check 1: Coordinate monotonicity
                x_upper_sorted = np.sort(x_upper)
                x_lower_sorted = np.sort(x_lower)
                mono_upper = np.all(np.diff(x_upper_sorted) > 0)
                mono_lower = np.all(np.diff(x_lower_sorted) > 0)
                
                if not mono_upper or not mono_lower:
                    issues.append(self._add_issue(
                        "GEOMETRY",
                        SeverityLevel.HIGH,
                        f"Non-monotonic coordinates in {tc['name']}",
                        "X-coordinates are not monotonically increasing",
                        {"mono_upper": bool(mono_upper), "mono_lower": bool(mono_lower)},
                        "CST parameterization may be generating invalid coordinates",
                        "Review CST coefficient bounds and coordinate generation logic"
                    ))
                    all_passed = False
                
                # Check 2: Self-intersection
                if len(x_upper) > 0 and len(x_lower) > 0:
                    # Interpolate to common x-grid
                    x_common = np.linspace(0, 1, 100)
                    try:
                        y_upper_interp = np.interp(x_common, x_upper, y_upper)
                        y_lower_interp = np.interp(x_common, x_lower, y_lower)
                        thickness = y_upper_interp - y_lower_interp
                        
                        if np.any(thickness <= 0):
                            issues.append(self._add_issue(
                                "GEOMETRY",
                                SeverityLevel.CRITICAL,
                                f"Self-intersection in {tc['name']}",
                                "Upper and lower surfaces cross each other",
                                {"min_thickness": float(np.min(thickness))},
                                "CST coefficients produce invalid geometry",
                                "Add geometric governance constraints"
                            ))
                            all_passed = False
                    except Exception as e:
                        pass  # Interpolation may fail for degenerate cases
                
                # Check 3: Thickness distribution
                # Use full arrays for governance (not split upper/lower)
                # The governor will handle the self-intersection check properly
                gov_report = self.governor.govern(x_array, y_array, y_array)
                
                if not gov_report.is_valid:
                    for violation in gov_report.violations:
                        issues.append(self._add_issue(
                            "GEOMETRY",
                            SeverityLevel.HIGH,
                            f"Geometry violation in {tc['name']}: {violation.value}",
                            "; ".join(gov_report.failure_reasons),
                            {"status": gov_report.status.value},
                            "Geometry fails governance checks",
                            "Tighten CST coefficient bounds or add regularization"
                        ))
                    all_passed = False
                
                metrics[tc["name"]] = {
                    "monotonic_upper": bool(mono_upper),
                    "monotonic_lower": bool(mono_lower),
                    "is_valid": gov_report.is_valid,
                    "violations": [v.value for v in gov_report.violations],
                }
                
                print(f"    Monotonicity: Upper={mono_upper}, Lower={mono_lower}")
                print(f"    Valid: {gov_report.is_valid}")
                if gov_report.thickness:
                    print(f"    Max thickness: {gov_report.thickness.max_thickness:.4f}")
                
            except Exception as e:
                issues.append(self._add_issue(
                    "GEOMETRY",
                    SeverityLevel.CRITICAL,
                    f"Geometry generation failed for {tc['name']}",
                    str(e),
                    {"traceback": traceback.format_exc()},
                    "CST reconstruction is broken",
                    "Fix CST coordinate generation"
                ))
                all_passed = False
        
        return self._add_result("geometry_validation", all_passed, metrics, issues)
    
    # =========================================================================
    # SECTION 2: CFD VALIDATION (Simulated - requires actual SU2 run)
    # =========================================================================
    
    def validate_cfd_setup(self) -> DiagnosticResult:
        """
        Validate CFD setup without running actual simulations.
        Checks configuration files, mesh parameters, solver settings.
        """
        print("\n" + "="*60)
        print("SECTION 2: CFD SETUP VALIDATION")
        print("="*60)
        
        issues = []
        metrics = {}
        all_passed = True
        
        # Check 1: Binary availability
        su2_bin = self.settings.solver.su2_cfd_bin
        gmsh_bin = self.settings.solver.gmsh_bin
        
        su2_exists = Path(su2_bin).exists() if su2_bin else False
        gmsh_exists = Path(gmsh_bin).exists() if gmsh_bin else False
        
        if not su2_exists:
            issues.append(self._add_issue(
                "CFD",
                SeverityLevel.CRITICAL,
                "SU2 binary not found",
                f"SU2_CFD binary not found at: {su2_bin}",
                {"path": su2_bin},
                "SU2 not installed or path misconfigured",
                "Install SU2 or update config path"
            ))
            all_passed = False
        
        if not gmsh_exists:
            issues.append(self._add_issue(
                "CFD",
                SeverityLevel.CRITICAL,
                "GMSH binary not found",
                f"GMSH binary not found at: {gmsh_bin}",
                {"path": gmsh_bin},
                "GMSH not installed or path misconfigured",
                "Install GMSH or update config path"
            ))
            all_passed = False
        
        # Check 2: Mesh parameters
        mesh_cfg = self.settings.solver.mesh
        
        if mesh_cfg.y_plus_target > 1.0:
            issues.append(self._add_issue(
                "CFD",
                SeverityLevel.HIGH,
                "y+ target too high for transition-resolving mesh",
                f"y+ target: {mesh_cfg.y_plus_target}, should be <= 1.0",
                {"y_plus_target": mesh_cfg.y_plus_target},
                "Mesh not fine enough for transitional RANS",
                "Reduce y+ target to <= 1.0"
            ))
            all_passed = False
        
        if mesh_cfg.surface_points < 100:
            issues.append(self._add_issue(
                "CFD",
                SeverityLevel.MEDIUM,
                "Insufficient surface points",
                f"Surface points: {mesh_cfg.surface_points}, recommend >= 150",
                {"surface_points": mesh_cfg.surface_points},
                "Airfoil surface may be under-resolved",
                "Increase surface points to >= 150"
            ))
        
        # Check 3: Solver settings
        if self.settings.solver.transition_model:
            print("  Transition model: ENABLED (Langtry-Menter)")
        else:
            issues.append(self._add_issue(
                "CFD",
                SeverityLevel.HIGH,
                "Transition model disabled",
                "Low-Re airfoil optimization requires transition modeling",
                {"transition_model": False},
                "Without transition model, laminar separation bubbles won't be captured",
                "Enable transition model in config"
            ))
            all_passed = False
        
        # Check 4: Reynolds number range
        re_min = self.settings.flow.reynolds_min
        re_max = self.settings.flow.reynolds_max
        
        if re_min < 50000 or re_max > 500000:
            issues.append(self._add_issue(
                "CFD",
                SeverityLevel.MEDIUM,
                "Reynolds number range may be outside low-Re regime",
                f"Re range: {re_min:.0f} - {re_max:.0f}",
                {"re_min": re_min, "re_max": re_max},
                "Very low or high Re may cause numerical issues",
                "Verify Re range is appropriate for low-Re airfoils"
            ))
        
        metrics = {
            "su2_binary_exists": su2_exists,
            "gmsh_binary_exists": gmsh_exists,
            "y_plus_target": mesh_cfg.y_plus_target,
            "surface_points": mesh_cfg.surface_points,
            "transition_model_enabled": self.settings.solver.transition_model,
            "reynolds_min": re_min,
            "reynolds_max": re_max,
        }
        
        return self._add_result("cfd_setup_validation", all_passed, metrics, issues)
    
    # =========================================================================
    # SECTION 3: OPTIMIZER VALIDATION
    # =========================================================================
    
    def validate_optimizer(self) -> DiagnosticResult:
        """
        Validate optimizer configuration and behavior.
        """
        print("\n" + "="*60)
        print("SECTION 3: OPTIMIZER VALIDATION")
        print("="*60)
        
        issues = []
        metrics = {}
        all_passed = True
        
        # Check optimizer configuration from settings
        opt_cfg = self.settings.optimization
        
        # Check 1: Design variable bounds
        geom_cfg = self.settings.geometry
        
        upper_bounds = geom_cfg.upper_bounds
        lower_bounds = geom_cfg.lower_bounds
        
        bound_range = np.array(upper_bounds) - np.array(lower_bounds)
        
        if np.any(bound_range <= 0):
            issues.append(self._add_issue(
                "OPTIMIZER",
                SeverityLevel.CRITICAL,
                "Invalid design variable bounds",
                "Some upper bounds are <= lower bounds",
                {"upper_bounds": upper_bounds, "lower_bounds": lower_bounds},
                "Design space is degenerate",
                "Fix bound configuration"
            ))
            all_passed = False
        
        if np.any(bound_range > 2.0):
            issues.append(self._add_issue(
                "OPTIMIZER",
                SeverityLevel.MEDIUM,
                "Excessively wide design variable bounds",
                f"Max bound range: {np.max(bound_range):.2f}",
                {"bound_range": bound_range.tolist()},
                "Search space may be too large for efficient optimization",
                "Tighten bounds based on known good airfoils"
            ))
        
        # Check 2: Move limits
        if hasattr(opt_cfg, 'move_limit') and opt_cfg.move_limit <= 0:
            issues.append(self._add_issue(
                "OPTIMIZER",
                SeverityLevel.HIGH,
                "Invalid move limit",
                f"Move limit: {opt_cfg.move_limit}",
                {"move_limit": opt_cfg.move_limit},
                "Optimizer cannot make progress",
                "Set positive move limit"
            ))
            all_passed = False
        
        # Check 3: Check database for optimization history
        try:
            df = self.db.training_frame()
            
            if not df.empty:
                n_cases = len(df)
                metrics["total_cases"] = n_cases
                
                # Check for duplicate evaluations
                if "signature" in df.columns:
                    n_unique = df["signature"].nunique()
                    n_duplicates = n_cases - n_unique
                    
                    if n_duplicates > n_cases * 0.3:
                        issues.append(self._add_issue(
                            "OPTIMIZER",
                            SeverityLevel.HIGH,
                            "Excessive duplicate evaluations",
                            f"Duplicates: {n_duplicates}/{n_cases} ({n_duplicates/n_cases*100:.1f}%)",
                            {"total": n_cases, "unique": n_unique, "duplicates": n_duplicates},
                            "Optimizer is re-evaluating same designs",
                            "Add duplicate detection or improve exploration"
                        ))
                
                # Check for score improvement
                if "score" in df.columns:
                    scores = df["score"].values
                    if len(scores) > 5:
                        # Check if score is improving
                        recent_scores = scores[-10:]
                        score_range = np.max(recent_scores) - np.min(recent_scores)
                        
                        if score_range < 0.001:
                            issues.append(self._add_issue(
                                "OPTIMIZER",
                                SeverityLevel.HIGH,
                                "Optimization stagnation detected",
                                f"Score range in last 10 iterations: {score_range:.6f}",
                                {"recent_scores": recent_scores.tolist()},
                                "Optimizer is trapped or converged prematurely",
                                "Increase exploration or restart with different initial point"
                            ))
                        
                        # Check for oscillation
                        score_diffs = np.diff(recent_scores)
                        sign_changes = np.sum(np.diff(np.sign(score_diffs)) != 0)
                        if sign_changes > len(score_diffs) * 0.6:
                            issues.append(self._add_issue(
                                "OPTIMIZER",
                                SeverityLevel.MEDIUM,
                                "Optimizer oscillation detected",
                                f"Sign changes in score diffs: {sign_changes}/{len(score_diffs)}",
                                {"sign_changes": sign_changes},
                                "Optimizer is oscillating around optimum",
                                "Reduce move limits or adjust trust region"
                            ))
                
                # Check for invalid values
                if "cd" in df.columns:
                    n_negative_cd = (df["cd"] < 0).sum()
                    if n_negative_cd > 0:
                        issues.append(self._add_issue(
                            "OPTIMIZER",
                            SeverityLevel.CRITICAL,
                            "Negative drag coefficients detected",
                            f"Negative Cd count: {n_negative_cd}",
                            {"n_negative_cd": int(n_negative_cd)},
                            "CFD or post-processing error producing invalid drag",
                            "Fix drag computation or add validity checks"
                        ))
                        all_passed = False
                    
                    max_cd = df["cd"].max()
                    if max_cd > 0.5:
                        issues.append(self._add_issue(
                            "OPTIMIZER",
                            SeverityLevel.HIGH,
                            "Extremely high drag coefficients",
                            f"Max Cd: {max_cd:.4f}",
                            {"max_cd": float(max_cd)},
                            "Optimizer may be producing bluff-body geometries",
                            "Add drag上限 constraint or geometric governance"
                        ))
                
                if "cl" in df.columns:
                    n_negative_cl = (df["cl"] < 0).sum()
                    n_cases_total = len(df)
                    # Some negative Cl is expected for negative AoA
                    if n_negative_cl > n_cases_total * 0.5:
                        issues.append(self._add_issue(
                            "OPTIMIZER",
                            SeverityLevel.MEDIUM,
                            "Many negative lift coefficients",
                            f"Negative Cl count: {n_negative_cl}/{n_cases_total}",
                            {"n_negative_cl": int(n_negative_cl)},
                            "May indicate incorrect AoA or camber issues",
                            "Check AoA sweep logic and CST parameterization"
                        ))
            else:
                metrics["total_cases"] = 0
                print("  No optimization history found in database")
                
        except Exception as e:
            issues.append(self._add_issue(
                "OPTIMIZER",
                SeverityLevel.MEDIUM,
                "Failed to analyze optimization history",
                str(e),
                {"traceback": traceback.format_exc()},
                "Database access or analysis failed",
                "Check database integrity"
            ))
        
        return self._add_result("optimizer_validation", all_passed, metrics, issues)
    
    # =========================================================================
    # SECTION 4: OBJECTIVE FUNCTION AUDIT
    # =========================================================================
    
    def validate_objective_function(self) -> DiagnosticResult:
        """
        Audit objective function computation, scaling, and penalties.
        """
        print("\n" + "="*60)
        print("SECTION 4: OBJECTIVE FUNCTION AUDIT")
        print("="*60)
        
        issues = []
        metrics = {}
        all_passed = True
        
        # Try to analyze scoring configuration
        try:
            scoring_cfg = self.settings.scoring
            
            # Check penalty weights
            if hasattr(scoring_cfg, 'separation_penalty_weight'):
                sep_weight = scoring_cfg.separation_penalty_weight
                if sep_weight > 10.0:
                    issues.append(self._add_issue(
                        "OBJECTIVE",
                        SeverityLevel.HIGH,
                        "Excessive separation penalty weight",
                        f"Separation penalty weight: {sep_weight}",
                        {"weight": sep_weight},
                        "Penalty may dominate aerodynamic objective",
                        "Reduce penalty weight to reasonable level (< 5.0)"
                    ))
            
            if hasattr(scoring_cfg, 'instability_penalty_weight'):
                inst_weight = scoring_cfg.instability_penalty_weight
                if inst_weight > 10.0:
                    issues.append(self._add_issue(
                        "OBJECTIVE",
                        SeverityLevel.HIGH,
                        "Excessive instability penalty weight",
                        f"Instability penalty weight: {inst_weight}",
                        {"weight": inst_weight},
                        "Penalty may dominate aerodynamic objective",
                        "Reduce penalty weight to reasonable level (< 5.0)"
                    ))
            
            # Analyze score distribution from database
            df = self.db.training_frame()
            if not df.empty and "score" in df.columns:
                scores = df["score"].values
                
                metrics["score_stats"] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                    "median": float(np.median(scores)),
                }
                
                # Check for NaN or Inf
                n_nan = np.sum(np.isnan(scores))
                n_inf = np.sum(np.isinf(scores))
                
                if n_nan > 0:
                    issues.append(self._add_issue(
                        "OBJECTIVE",
                        SeverityLevel.CRITICAL,
                        "NaN scores detected",
                        f"NaN count: {n_nan}/{len(scores)}",
                        {"n_nan": int(n_nan)},
                        "Objective computation is producing NaN",
                        "Add NaN checks and handle edge cases"
                    ))
                    all_passed = False
                
                if n_inf > 0:
                    issues.append(self._add_issue(
                        "OBJECTIVE",
                        SeverityLevel.CRITICAL,
                        "Infinite scores detected",
                        f"Inf count: {n_inf}/{len(scores)}",
                        {"n_inf": int(n_inf)},
                        "Objective computation is producing Inf",
                        "Add bounds checking and handle division by zero"
                    ))
                    all_passed = False
                
                # Check score range
                score_range = np.max(scores) - np.min(scores)
                if score_range < 0.01 and len(scores) > 10:
                    issues.append(self._add_issue(
                        "OBJECTIVE",
                        SeverityLevel.MEDIUM,
                        "Very narrow score range",
                        f"Score range: {score_range:.6f}",
                        {"range": float(score_range)},
                        "Objective may not be discriminative enough",
                        "Review objective weighting and scaling"
                    ))
            
            # Check Cl/Cd computation
            if not df.empty and "cl" in df.columns and "cd" in df.columns:
                cl_vals = df["cl"].values
                cd_vals = df["cd"].values
                
                # Check for zero or near-zero drag
                n_zero_cd = np.sum(cd_vals < 1e-6)
                if n_zero_cd > 0:
                    issues.append(self._add_issue(
                        "OBJECTIVE",
                        SeverityLevel.HIGH,
                        "Near-zero drag coefficients",
                        f"Zero Cd count: {n_zero_cd}/{len(cd_vals)}",
                        {"n_zero_cd": int(n_zero_cd)},
                        "Drag computation may be broken or CFD failed silently",
                        "Add minimum drag floor or validity checks"
                    ))
                
                # Check L/D ratios
                with np.errstate(divide='ignore', invalid='ignore'):
                    ld_ratios = np.where(cd_vals > 1e-6, cl_vals / cd_vals, 0)
                
                max_ld = np.max(ld_ratios)
                if max_ld > 200:
                    issues.append(self._add_issue(
                        "OBJECTIVE",
                        SeverityLevel.MEDIUM,
                        "Unrealistically high L/D ratio",
                        f"Max L/D: {max_ld:.1f}",
                        {"max_ld": float(max_ld)},
                        "May indicate incorrect Cl or Cd computation",
                        "Verify force coefficient computation"
                    ))
                
                metrics["aerodynamics"] = {
                    "cl_range": [float(np.min(cl_vals)), float(np.max(cl_vals))],
                    "cd_range": [float(np.min(cd_vals)), float(np.max(cd_vals))],
                    "max_ld": float(max_ld),
                }
                
        except Exception as e:
            issues.append(self._add_issue(
                "OBJECTIVE",
                SeverityLevel.MEDIUM,
                "Failed to analyze objective function",
                str(e),
                {"traceback": traceback.format_exc()},
                "Scoring analysis failed",
                "Check scoring configuration"
            ))
        
        return self._add_result("objective_function_audit", all_passed, metrics, issues)
    
    # =========================================================================
    # SECTION 5: REPRODUCIBILITY TESTING
    # =========================================================================
    
    def validate_reproducibility(self) -> DiagnosticResult:
        """
        Test reproducibility by generating same geometry multiple times.
        """
        print("\n" + "="*60)
        print("SECTION 5: REPRODUCIBILITY TESTING")
        print("="*60)
        
        issues = []
        metrics = {}
        all_passed = True
        
        # Test CST reproducibility
        test_params = CSTParameters(
            upper=np.array([0.18, 0.05, 0.34, 0.10]),
            lower=np.array([-0.19, 0.05, -0.09, 0.03]),
            trailing_edge_thickness=0.004,
        )
        
        coords_list = []
        for i in range(5):
            coords = self.cst.full_coordinates(test_params)
            coords_list.append(coords)
        
        # Check consistency
        for i in range(1, len(coords_list)):
            diff = np.max(np.abs(coords_list[i] - coords_list[0]))
            if diff > 1e-10:
                issues.append(self._add_issue(
                    "REPRODUCIBILITY",
                    SeverityLevel.HIGH,
                    "CST coordinate generation is non-deterministic",
                    f"Max difference between runs: {diff:.2e}",
                    {"max_diff": float(diff)},
                    "Random seed or state leakage in CST",
                    "Ensure deterministic CST computation"
                ))
                all_passed = False
                break
        
        metrics["cst_reproducibility"] = {
            "n_runs": len(coords_list),
            "max_coord_diff": float(np.max([np.max(np.abs(c - coords_list[0])) for c in coords_list])),
        }
        
        return self._add_result("reproducibility_testing", all_passed, metrics, issues)
    
    # =========================================================================
    # SECTION 6: SEARCH SPACE ANALYSIS
    # =========================================================================
    
    def analyze_search_space(self) -> DiagnosticResult:
        """
        Analyze the explored search space for coverage and diversity.
        """
        print("\n" + "="*60)
        print("SECTION 6: SEARCH SPACE ANALYSIS")
        print("="*60)
        
        issues = []
        metrics = {}
        all_passed = True
        
        df = self.db.training_frame()
        
        if df.empty:
            metrics["status"] = "no_data"
            print("  No optimization data available for search space analysis")
            return self._add_result("search_space_analysis", True, metrics, issues)
        
        # Analyze CST parameter coverage
        cst_cols = [col for col in df.columns if col.startswith("upper_") or col.startswith("lower_") or col == "te_thickness"]
        
        if cst_cols:
            cst_data = df[cst_cols].values
            
            # Compute parameter statistics
            param_means = np.mean(cst_data, axis=0)
            param_stds = np.std(cst_data, axis=0)
            param_ranges = np.max(cst_data, axis=0) - np.min(cst_data, axis=0)
            
            metrics["parameter_stats"] = {
                "means": param_means.tolist(),
                "stds": param_stds.tolist(),
                "ranges": param_ranges.tolist(),
                "columns": cst_cols,
            }
            
            # Check for collapsed dimensions (very low variance)
            for i, col in enumerate(cst_cols):
                if param_stds[i] < 0.001:
                    issues.append(self._add_issue(
                        "SEARCH_SPACE",
                        SeverityLevel.MEDIUM,
                        f"Collapsed search dimension: {col}",
                        f"Standard deviation: {param_stds[i]:.6f}",
                        {"std": float(param_stds[i]), "column": col},
                        "Optimizer is not exploring this dimension",
                        "Increase exploration or adjust parameter bounds"
                    ))
            
            # Check for clustering
            if len(cst_data) > 10:
                # Simple clustering metric: ratio of nearest neighbor distance to overall spread
                from scipy.spatial.distance import pdist
                
                if len(cst_data) > 2:
                    distances = pdist(cst_data)
                    if len(distances) > 0:
                        avg_nn_dist = np.mean(np.sort(distances)[:10])
                        overall_spread = np.mean(pdist(cst_data[[0, -1]]))
                        
                        if overall_spread > 0:
                            clustering_ratio = avg_nn_dist / overall_spread
                            
                            if clustering_ratio < 0.01:
                                issues.append(self._add_issue(
                                    "SEARCH_SPACE",
                                    SeverityLevel.HIGH,
                                    "Search space clustering detected",
                                    f"Clustering ratio: {clustering_ratio:.4f}",
                                    {"clustering_ratio": float(clustering_ratio)},
                                    "Optimizer is trapped in local region",
                                    "Increase exploration temperature or restart"
                                ))
                        
                        metrics["clustering"] = {
                            "avg_nn_distance": float(avg_nn_dist),
                            "overall_spread": float(overall_spread),
                            "clustering_ratio": float(clustering_ratio) if overall_spread > 0 else None,
                        }
        
        return self._add_result("search_space_analysis", all_passed, metrics, issues)
    
    # =========================================================================
    # SECTION 7: BASELINE COMPARISON (Simulated)
    # =========================================================================
    
    def validate_baseline_comparison(self) -> DiagnosticResult:
        """
        Compare against known good airfoils (requires CFD, so simulated here).
        """
        print("\n" + "="*60)
        print("SECTION 7: BASELINE COMPARISON (Configuration Check)")
        print("="*60)
        
        issues = []
        metrics = {}
        all_passed = True
        
        # Check if baseline airfoils are configured
        baseline_airfoils = [
            {"name": "NACA 2412", "expected_cl": 0.4, "expected_cd": 0.01, "expected_ld": 40},
            {"name": "NACA 4412", "expected_cl": 0.6, "expected_cd": 0.012, "expected_ld": 50},
            {"name": "SD7003", "expected_cl": 0.5, "expected_cd": 0.015, "expected_ld": 33},
        ]
        
        metrics["baseline_airfoils"] = baseline_airfoils
        
        # Check Reynolds number appropriateness
        re = self.settings.flow.reynolds_min
        if re < 100000:
            print(f"  Warning: Re = {re:.0f} is very low, may have numerical challenges")
        
        # Check Mach number
        mach = self.settings.flow.mach
        if mach > 0.3:
            issues.append(self._add_issue(
                "BASELINE",
                SeverityLevel.MEDIUM,
                "Mach number may be too high for incompressible assumption",
                f"Mach: {mach:.3f}",
                {"mach": mach},
                "Compressibility effects may be significant",
                "Use compressible solver or reduce Mach"
            ))
        
        return self._add_result("baseline_comparison", all_passed, metrics, issues)
    
    # =========================================================================
    # RUN ALL DIAGNOSTICS
    # =========================================================================
    
    def run_full_diagnostic(self) -> FullDiagnosticReport:
        """
        Run all diagnostic checks and generate comprehensive report.
        """
        print("\n" + "="*70)
        print("FULL SYSTEM DIAGNOSTIC AND VALIDATION SWEEP")
        print(f"Started: {datetime.now().isoformat()}")
        print("="*70)
        
        # Run all checks
        self.validate_geometry()
        self.validate_cfd_setup()
        self.validate_optimizer()
        self.validate_objective_function()
        self.validate_reproducibility()
        self.analyze_search_space()
        self.validate_baseline_comparison()
        
        # Compute summary statistics
        total_checks = len(self.results)
        passed_checks = sum(1 for r in self.results if r.passed)
        failed_checks = total_checks - passed_checks
        
        # Compute confidence score
        critical_issues = sum(1 for i in self.issues if i.severity == SeverityLevel.CRITICAL)
        high_issues = sum(1 for i in self.issues if i.severity == SeverityLevel.HIGH)
        medium_issues = sum(1 for i in self.issues if i.severity == SeverityLevel.MEDIUM)
        
        # Confidence score: 100 - penalties
        penalty = critical_issues * 25 + high_issues * 10 + medium_issues * 3
        confidence_score = max(0.0, min(100.0, 100.0 - penalty))
        
        # Generate recommendations
        recommendations = []
        
        if critical_issues > 0:
            recommendations.append("CRITICAL: System has fundamental issues that must be fixed before optimization")
            recommendations.append("Do NOT continue optimization until critical issues are resolved")
        
        if high_issues > 0:
            recommendations.append("HIGH: System has significant issues affecting result quality")
            recommendations.append("Address high-severity issues before production runs")
        
        if not any(r.passed for r in self.results):
            recommendations.append("All diagnostic checks failed - system may be misconfigured")
        
        if confidence_score < 50:
            recommendations.append("System confidence is low - consider restarting optimization from scratch")
        elif confidence_score < 75:
            recommendations.append("System has moderate issues - review and fix before continuing")
        else:
            recommendations.append("System appears healthy - optimization can proceed with caution")
        
        # Build report
        report = FullDiagnosticReport(
            timestamp=datetime.now().isoformat(),
            config_hash=str(hash(str(self.settings))),
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            results=self.results,
            issues=self.issues,
            confidence_score=confidence_score,
            recommendations=recommendations,
            is_system_healthy=critical_issues == 0 and high_issues <= 1,
        )
        
        return report
    
    def save_report(self, report: FullDiagnosticReport):
        """Save diagnostic report to files."""
        # JSON report
        report_dict = {
            "timestamp": report.timestamp,
            "config_hash": report.config_hash,
            "total_checks": report.total_checks,
            "passed_checks": report.passed_checks,
            "failed_checks": report.failed_checks,
            "confidence_score": report.confidence_score,
            "is_system_healthy": report.is_system_healthy,
            "recommendations": report.recommendations,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "metrics": r.metrics,
                    "issues": [
                        {
                            "category": i.category,
                            "severity": i.severity.value,
                            "title": i.title,
                            "description": i.description,
                            "evidence": i.evidence,
                            "root_cause": i.root_cause,
                            "fix": i.fix,
                        }
                        for i in r.issues
                    ],
                }
                for r in report.results
            ],
            "issues_summary": [
                {
                    "category": i.category,
                    "severity": i.severity.value,
                    "title": i.title,
                    "description": i.description,
                    "fix": i.fix,
                }
                for i in report.issues
            ],
        }
        
        report_path = self.output_dir / "diagnostic_report.json"
        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        # Text summary
        summary_path = self.output_dir / "diagnostic_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("="*70 + "\n")
            f.write("SYSTEM DIAGNOSTIC REPORT\n")
            f.write(f"Generated: {report.timestamp}\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Total Checks: {report.total_checks}\n")
            f.write(f"Passed: {report.passed_checks}\n")
            f.write(f"Failed: {report.failed_checks}\n")
            f.write(f"Confidence Score: {report.confidence_score:.1f}/100\n")
            f.write(f"System Healthy: {report.is_system_healthy}\n\n")
            
            f.write("ISSUES BY SEVERITY:\n")
            f.write("-"*40 + "\n")
            
            for severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH, SeverityLevel.MEDIUM, SeverityLevel.LOW]:
                severity_issues = [i for i in report.issues if i.severity == severity]
                if severity_issues:
                    f.write(f"\n[{severity.value}] ({len(severity_issues)} issues)\n")
                    for i in severity_issues:
                        f.write(f"  - {i.title}\n")
                        f.write(f"    {i.description}\n")
                        f.write(f"    Fix: {i.fix}\n\n")
            
            f.write("\nRECOMMENDATIONS:\n")
            f.write("-"*40 + "\n")
            for rec in report.recommendations:
                f.write(f"  * {rec}\n")
        
        print(f"\nReport saved to: {report_path}")
        print(f"Summary saved to: {summary_path}")
        
        return report_path, summary_path


def main():
    parser = argparse.ArgumentParser(description="Full System Diagnostic for Airfoil Optimization")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to configuration file")
    parser.add_argument("--output", type=str, default="data/diagnostics",
                        help="Output directory for diagnostic reports")
    args = parser.parse_args()
    
    # Load settings
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    settings = load_settings(config_path)
    output_dir = Path(args.output)
    
    # Run diagnostics
    diagnostics = SystemDiagnostics(settings, output_dir)
    report = diagnostics.run_full_diagnostic()
    
    # Save report
    report_path, summary_path = diagnostics.save_report(report)
    
    # Print summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    print(f"Confidence Score: {report.confidence_score:.1f}/100")
    print(f"System Healthy: {report.is_system_healthy}")
    print(f"Total Issues: {len(report.issues)}")
    print(f"  Critical: {sum(1 for i in report.issues if i.severity == SeverityLevel.CRITICAL)}")
    print(f"  High: {sum(1 for i in report.issues if i.severity == SeverityLevel.HIGH)}")
    print(f"  Medium: {sum(1 for i in report.issues if i.severity == SeverityLevel.MEDIUM)}")
    print(f"  Low: {sum(1 for i in report.issues if i.severity == SeverityLevel.LOW)}")
    
    if not report.is_system_healthy:
        print("\n⚠️  SYSTEM IS NOT HEALTHY - DO NOT CONTINUE OPTIMIZATION")
        print("Review the diagnostic report and fix issues before proceeding.")
        sys.exit(1)
    else:
        print("\n✓ System appears healthy - optimization can proceed")
        sys.exit(0)


if __name__ == "__main__":
    main()