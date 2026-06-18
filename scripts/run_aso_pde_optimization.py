#!/usr/bin/env python3
"""
PDE-Constrained Aerodynamic Shape Optimization Entry Point.

Run the full optimization cycle:
  1. Load configuration (SU2 paths, flow conditions, bounds)
  2. Initialize baseline geometry and mesh
  3. Run gradient-based optimization (MMA or SLSQP)
  4. Save convergence history and final design

Usage:
    python scripts/run_aso_pde_optimization.py [--config CONFIG] [--method {mma,slsqp}] [--max-iter N]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from airfoil_discovery.aso import (
    PDEOptimizer,
    CSTBounds,
    ConvergenceHistory,
    N_DESIGN_VARS,
)


def setup_logging(log_path: Path, verbose: bool = False) -> None:
    """Configure logging to both file and console."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PDE-Constrained Aerodynamic Shape Optimization for Low-Re Airfoils"
    )

    # Optimization method
    parser.add_argument(
        "--method",
        type=str,
        default="mma",
        choices=["mma", "slsqp"],
        help="Optimization algorithm: MMA (Svanberg 1987) or SLSQP (scipy fallback)",
    )

    # Flow conditions
    parser.add_argument("--aoa", type=float, default=4.0, help="Angle of attack (degrees)")
    parser.add_argument("--reynolds", type=float, default=1.0e5, help="Chord Reynolds number")
    parser.add_argument("--mach", type=float, default=0.1, help="Freestream Mach number")

    # Iterations
    parser.add_argument("--max-iter", type=int, default=30, help="Maximum optimization iterations")
    parser.add_argument("--n-iter-primal", type=int, default=3000, help="Primal CFD iterations")
    parser.add_argument("--n-iter-adjoint", type=int, default=500, help="Adjoint CFD iterations")

    # CFL
    parser.add_argument("--cfl-primal", type=float, default=3.0, help="Primal CFL number")
    parser.add_argument("--cfl-adjoint", type=float, default=1.0, help="Adjoint CFL number")

    # Convergence tolerance
    parser.add_argument("--tol", type=float, default=1e-4, help="Gradient norm convergence tolerance")

    # SU2 binaries
    parser.add_argument(
        "--su2-cfd",
        type=str,
        default=None,
        help="Path to SU2_CFD executable (overrides environment variable SU2_CFD_BIN)",
    )
    parser.add_argument(
        "--su2-def",
        type=str,
        default=None,
        help="Path to SU2_DEF executable (overrides environment variable SU2_DEF_BIN)",
    )

    # Mesh
    parser.add_argument("--mesh", type=str, required=True, help="Path to baseline SU2 mesh file")
    parser.add_argument("--mesh-deform", action="store_true", default=True, help="Enable mesh deformation via SU2_DEF")

    # Output
    parser.add_argument("--output", type=str, default="aso_results", help="Output directory for results")

    # Initial design (optional, overrides default NACA-like)
    parser.add_argument("--init-dv", type=str, default=None, help="Path to .npy file with initial 12 DV")

    # Transition model
    parser.add_argument("--no-transition", action="store_true", help="Disable γ-Re_θ transition model")
    parser.add_argument("--turb-intensity", type=float, default=0.001, help="Freestream turbulence intensity")
    parser.add_argument("--turb-viscosity-ratio", type=float, default=5.0, help="Freestream μ_t/μ")

    # Verbose
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output (DEBUG level)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Prepare output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_path = output_dir / "optimization.log"
    setup_logging(log_path, verbose=args.verbose)

    logger = logging.getLogger("aso_pde_optimizer")
    logger.info("=" * 60)
    logger.info("PDE-Constrained Aerodynamic Shape Optimization")
    logger.info(f"Method: {args.method}")
    logger.info(f"AoA: {args.aoa}°, Re: {args.reynolds:.1e}, Mach: {args.mach}")
    logger.info(f"Max iterations: {args.max_iter}")
    logger.info("=" * 60)

    # Resolve SU2 binaries
    su2_cfd_bin = args.su2_cfd or os.environ.get("SU2_CFD_BIN", "SU2_CFD")
    su2_def_bin = args.su2_def or os.environ.get("SU2_DEF_BIN", "SU2_DEF")

    mesh_path = Path(args.mesh).resolve()
    if not mesh_path.exists():
        logger.error(f"Mesh file not found: {mesh_path}")
        sys.exit(1)

    logger.info(f"SU2_CFD: {su2_cfd_bin}")
    logger.info(f"SU2_DEF: {su2_def_bin}")
    logger.info(f"Mesh: {mesh_path}")

    # Load initial design vector if provided
    dv_initial = None
    if args.init_dv:
        dv_initial = np.load(args.init_dv)
        if len(dv_initial) != N_DESIGN_VARS:
            logger.error(f"Initial DV has {len(dv_initial)} elements, expected {N_DESIGN_VARS}")
            sys.exit(1)
        logger.info(f"Loaded initial design from {args.init_dv}")

    # Create optimizer
    bounds = CSTBounds.default()

    optimizer = PDEOptimizer(
        su2_cfd_bin=su2_cfd_bin,
        mesh_path=mesh_path,
        work_dir=output_dir,
        dv_initial=dv_initial,
        bounds=bounds,
        aoa_deg=args.aoa,
        reynolds=args.reynolds,
        mach=args.mach,
        n_iter_primal=args.n_iter_primal,
        n_iter_adjoint=args.n_iter_adjoint,
        cfl_primal=args.cfl_primal,
        cfl_adjoint=args.cfl_adjoint,
        transition_model=not args.no_transition,
        turbulence_intensity=args.turb_intensity,
        turb_viscosity_ratio=args.turb_viscosity_ratio,
        move_limit=0.05,
        use_slsqp_fallback=(args.method == "slsqp"),
        su2_def_bin=su2_def_bin,
        use_mesh_deformation=args.mesh_deform,
        max_iterations=args.max_iter,
        convergence_tolerance=args.tol,
    )

    # Run optimization
    start_time = time.time()
    try:
        history = optimizer.run(method=args.method)
    except Exception as e:
        logger.exception(f"Optimization failed with exception: {e}")
        sys.exit(1)

    elapsed = time.time() - start_time

    # Save results
    optimizer.save_results(output_dir)

    # Print summary
    logger.info("=" * 60)
    logger.info("Optimization Complete")
    logger.info(f"Time elapsed: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    logger.info(f"Total iterations: {history.total_iterations}")
    logger.info(f"Converged: {history.converged}")

    if history.iterations:
        first = history.iterations[0]
        last = history.iterations[-1]
        logger.info(f"Initial Cd: {first.cd:.6f}, Final Cd: {last.cd:.6f} (delta: {last.cd - first.cd:+.6f})")
        logger.info(f"Initial Cl: {first.cl:.6f}, Final Cl: {last.cl:.6f} (delta: {last.cl - first.cl:+.6f})")
        logger.info(f"Initial |∇Cd|: {first.grad_norm:.6f}, Final |∇Cd|: {last.grad_norm:.6f}")

    logger.info(f"Results saved to: {output_dir.resolve()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()