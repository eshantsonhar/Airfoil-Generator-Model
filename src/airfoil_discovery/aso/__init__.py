"""
PDE-Constrained Aerodynamic Shape Optimization Framework for SU2.

This package implements a gradient-based optimization loop for
low-Reynolds-number airfoil design using:
  - 6th-order CST parameterization (12 design variables)
  - k-ω SST + γ-Re_θ transition modeling
  - Discrete adjoint sensitivity analysis
  - MMA / SLSQP optimization with convergence tracking
  - SU2_DEF mesh deformation

Key Components:
  - ``cst``: 6th-order Class-Shape Transformation with 12 design variables
  - ``config_primal``: SU2 primal RANS + transition config generator
  - ``config_adjoint``: SU2 discrete adjoint config generator
  - ``adjoint``: Surface sensitivity parser + CST gradient projection
  - ``mesh_deform``: SU2_DEF wrapper for mesh deformation
  - ``optimizer``: PDEOptimizer — the main optimization loop
"""

from .cst import (
    CST_ORDER,
    N_DESIGN_VARS,
    CSTBounds,
    compute_airfoil_coordinates,
    compute_surface_coordinates,
    check_geometry_validity,
    design_vector_to_surface_coefficients,
    surface_coefficients_to_design_vector,
    project_surface_sensitivity_to_cst,
    cosine_spacing,
    bernstein_basis,
    class_function,
)
from .config_primal import generate_primal_config, write_primal_config
from .config_adjoint import generate_adjoint_config, write_adjoint_config
from .adjoint import extract_adjoint_gradient, verify_adjoint_gradient, parse_surface_sensitivity_file
from .mesh_deform import deform_mesh, write_airfoil_dat, compute_mesh_displacement, generate_su2_def_config
from .diagnostics import (
    SurfaceFlowData,
    LSBResult,
    AerodynamicMetrics,
    parse_surface_flow,
    extract_lsb_from_cf,
    extract_lsb_from_cp,
    compute_aerodynamic_metrics,
    compare_baseline_optimized,
)
from .optimizer import PDEOptimizer, ConvergenceHistory, IterationRecord, CFDResult, ASOObjectiveFunction

__all__ = [
    # CST
    "CST_ORDER",
    "N_DESIGN_VARS",
    "CSTBounds",
    "compute_airfoil_coordinates",
    "compute_surface_coordinates",
    "check_geometry_validity",
    "design_vector_to_surface_coefficients",
    "surface_coefficients_to_design_vector",
    "project_surface_sensitivity_to_cst",
    "cosine_spacing",
    "bernstein_basis",
    "class_function",
    # Config
    "generate_primal_config",
    "write_primal_config",
    "generate_adjoint_config",
    "write_adjoint_config",
    # Adjoint
    "extract_adjoint_gradient",
    "verify_adjoint_gradient",
    "parse_surface_sensitivity_file",
    # Mesh deformation
    "deform_mesh",
    "write_airfoil_dat",
    "compute_mesh_displacement",
    "generate_su2_def_config",
    # Optimizer
    "PDEOptimizer",
    "ConvergenceHistory",
    "IterationRecord",
    "CFDResult",
    "ASOObjectiveFunction",
    # Diagnostics
    "SurfaceFlowData",
    "LSBResult",
    "AerodynamicMetrics",
    "parse_surface_flow",
    "extract_lsb_from_cf",
    "extract_lsb_from_cp",
    "compute_aerodynamic_metrics",
    "compare_baseline_optimized",
]
