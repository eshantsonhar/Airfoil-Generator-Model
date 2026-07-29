#!/usr/bin/env python3
"""
Production Batch Optimization Sweeper.

Automates parametric optimization sweeps across critical low-Reynolds
conditions and angles of attack. Saves structured results including
final geometries, aerodynamic deltas, and convergence data.

Sweep grid:
  Reynolds: [5e4, 1e5, 2e5]
  AoA:      [3.0, 4.0, 5.0]

Each optimization is run independently and results are collected
into a structured JSON archive.

Usage:
    python scripts/run_production_sweeps.py \\
        --su2-cfd /path/to/SU2_CFD \\
        --mesh /path/to/baseline_mesh.su2 \\
        --output sweep_results
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from airfoil_discovery.aso import (
    PDEOptimizer,
    CSTBounds,
    N_DESIGN_VARS,
    ConvergenceHistory,
    compute_airfoil_coordinates,
    check_geometry_validity,
    compute_aerodynamic_metrics,
    compare_baseline_optimized,
    AerodynamicMetrics,
)


@dataclass
class SweepCaseResult:
    """Result from a single sweep case."""
    reynolds: float
    aoa: float
    success: bool
    total_iterations: int
    converged: bool
    cd_initial: float
    cd_final: float
    cl_initial: float
    cl_final: float
    cd_reduction_percent: float
    efficiency_initial: float
    efficiency_final: float
    efficiency_improvement_percent: float
    final_design_vector: List[float]
    final_airfoil_file: str
    history_file: str
    elapsed_seconds: float
    lsb_characterization: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Production batch optimization sweeper for low-Re airfoils"
    )
    parser.add_argument("--su2-cfd", required=True, help="Path to SU2_CFD executable")
    parser.add_argument("--su2-def", default=None, help="Path to SU2_DEF executable")
    parser.add_argument("--mesh", required=True, help="Path to baseline SU2 mesh")
    parser.add_argument("--output", default="sweep_results", help="Output root directory")
    parser.add_argument("--reynolds", type=str, default="50000,100000,200000",
                        help="Comma-separated Reynolds numbers")
    parser.add_argument("--aoa", type=str, default="3.0,4.0,5.0",
                        help="Comma-separated angles of attack (degrees)")
    parser.add_argument("--mach", type=float, default=0.1, help="Freestream Mach")
    parser.add_argument("--max-iter", type=int, default=20, help="Max iterations per optimization")
    parser.add_argument("--n-iter-primal", type=int, default=2000, help="Primal CFD iterations")
    parser.add_argument("--n-iter-adjoint", type=int, default=300, help="Adjoint CFD iterations")
    parser.add_argument("--method", type=str, default="slsqp", choices=["mma", "slsqp"],
                        help="Optimization method")
    parser.add_argument("--design", type=str, default=None,
                        help="Path to .npy initial design vector")
    parser.add_argument("--resume", action="store_true",
                        help="Skip completed cases if output files exist")
    parser.add_argument("--no-adjoint", action="store_true", help="Skip adjoint solves and use finite-difference gradients")
    parser.add_argument("--no-mesh-deform", action="store_true", help="Disable mesh deformation")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def setup_logging(log_path: Path, verbose: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(fh)
    root.addHandler(ch)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    setup_logging(output_root / "sweep.log", verbose=args.verbose)
    logger = logging.getLogger("production_sweep")

    # Parse sweep grid
    reynolds_list = [float(r) for r in args.reynolds.split(",")]
    aoa_list = [float(a) for a in args.aoa.split(",")]

    mesh_path = Path(args.mesh).resolve()
    if not mesh_path.exists():
        logger.error(f"Mesh not found: {mesh_path}")
        sys.exit(1)

    # Load baseline design
    if args.design:
        dv_initial = np.load(args.design)
        logger.info(f"Loaded design from {args.design}")
    else:
        dv_initial = np.array([0.18, 0.28, 0.34, 0.25, 0.15, 0.08,
                               -0.19, -0.12, -0.09, -0.05, -0.02, -0.01])
        logger.info("Using default NACA-like design")

    logger.info("=" * 60)
    logger.info(f"Production Sweep: {len(reynolds_list)} Re x {len(aoa_list)} AoA = {len(reynolds_list) * len(aoa_list)} cases")
    logger.info(f"Reynolds: {reynolds_list}")
    logger.info(f"AoA: {aoa_list}")
    logger.info(f"Method: {args.method}, Max iter: {args.max_iter}")
    logger.info("=" * 60)

    all_results: List[Dict[str, Any]] = []
    total_cases = len(reynolds_list) * len(aoa_list)
    case_index = 0

    for Re in reynolds_list:
        for aoa in aoa_list:
            case_index += 1
            case_name = f"Re{Re:.0e}_AoA{aoa:.1f}".replace("+", "").replace(".", "p")
            case_dir = output_root / case_name
            case_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"\n[{case_index}/{total_cases}] Running: {case_name}")

            # Check resume
            if args.resume:
                history_file = case_dir / "convergence_history.json"
                final_file = case_dir / "final_airfoil.dat"
                if history_file.exists() and final_file.exists():
                    logger.info(f"  Already completed, skipping.")
                    with open(history_file) as f:
                        hist_data = json.load(f)
                    all_results.append({
                        "case": case_name,
                        "reynolds": Re,
                        "aoa": aoa,
                        "note": "resumed (already completed)",
                        "total_iterations": hist_data.get("total_iterations", 0),
                        "converged": hist_data.get("converged", False),
                    })
                    continue

            start_time = time.time()
            case_errors = []

            try:
                # Create optimizer
                optimizer = PDEOptimizer(
                    su2_cfd_bin=args.su2_cfd,
                    mesh_path=mesh_path,
                    work_dir=case_dir,
                    dv_initial=dv_initial.copy(),
                    bounds=CSTBounds.default(),
                    aoa_deg=aoa,
                    reynolds=Re,
                    mach=args.mach,
                    n_iter_primal=args.n_iter_primal,
                    n_iter_adjoint=args.n_iter_adjoint,
                    cfl_primal=3.0,
                    cfl_adjoint=1.0,
                    transition_model=True,
                    turbulence_intensity=0.001,
                    turb_viscosity_ratio=5.0,
                    move_limit=0.05,
                    use_slsqp_fallback=(args.method == "slsqp"),
                    su2_def_bin=args.su2_def,
                    use_mesh_deformation=(not args.no_mesh_deform),
                    max_iterations=args.max_iter,
                    convergence_tolerance=1e-4,
                    use_adjoint=not args.no_adjoint,
                )

                # Run
                history = optimizer.run(method=args.method)
                optimizer.save_results(case_dir)

                elapsed = time.time() - start_time

                # Build case result
                first_it = history.iterations[0] if history.iterations else None
                last_it = history.iterations[-1] if history.iterations else None

                cd_initial = first_it.cd if first_it else 0.0
                cd_final = last_it.cd if last_it else 0.0
                cl_initial = first_it.cl if first_it else 0.0
                cl_final = last_it.cl if last_it else 0.0

                eff_initial = cl_initial / max(cd_initial, 1e-10)
                eff_final = cl_final / max(cd_final, 1e-10)

                cd_reduction = (cd_initial - cd_final) / max(cd_initial, 1e-10) * 100
                eff_improvement = (eff_final - eff_initial) / max(eff_initial, 1e-10) * 100

                # Try to compute LSB analysis from history/surface flow
                lsb_info: Dict[str, Any] = {}
                for sf_file in case_dir.rglob("surface_flow*.csv"):
                    try:
                        metrics = compute_aerodynamic_metrics(surface_file=sf_file)
                        if metrics.lsb:
                            lsb_info = {
                                "lsb_detected": metrics.lsb.lsb_detected,
                                "separation_point": metrics.lsb.separation_point,
                                "reattachment_point": metrics.lsb.reattachment_point,
                                "bubble_length": metrics.lsb.bubble_length,
                            }
                        break
                    except Exception:
                        pass

                result = SweepCaseResult(
                    reynolds=Re,
                    aoa=aoa,
                    success=True,
                    total_iterations=history.total_iterations,
                    converged=history.converged,
                    cd_initial=cd_initial,
                    cd_final=cd_final,
                    cl_initial=cl_initial,
                    cl_final=cl_final,
                    cd_reduction_percent=cd_reduction,
                    efficiency_initial=eff_initial,
                    efficiency_final=eff_final,
                    efficiency_improvement_percent=eff_improvement,
                    final_design_vector=optimizer._current_dv.tolist(),
                    final_airfoil_file=str(case_dir / "final_airfoil.dat"),
                    history_file=str(case_dir / "convergence_history.json"),
                    elapsed_seconds=elapsed,
                    lsb_characterization=lsb_info,
                    errors=case_errors,
                )

                logger.info(
                    f"  Complete: Cd {cd_initial:.6f} -> {cd_final:.6f} "
                    f"({cd_reduction:+.1f}%), Eff {eff_initial:.1f} -> {eff_final:.1f} "
                    f"({eff_improvement:+.1f}%), {elapsed:.0f}s"
                )

            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"  FAILED: {e}", exc_info=True)
                result = SweepCaseResult(
                    reynolds=Re,
                    aoa=aoa,
                    success=False,
                    total_iterations=0,
                    converged=False,
                    cd_initial=0.0, cd_final=0.0,
                    cl_initial=0.0, cl_final=0.0,
                    cd_reduction_percent=0.0,
                    efficiency_initial=0.0, efficiency_final=0.0,
                    efficiency_improvement_percent=0.0,
                    final_design_vector=[],
                    final_airfoil_file="",
                    history_file="",
                    elapsed_seconds=elapsed,
                    errors=[str(e)],
                )

            all_results.append(asdict(result))

            # Save intermediate results after each case
            sweep_file = output_root / "sweep_all_results.json"
            with open(sweep_file, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

    # ── Summary Report ──
    print("\n" + "=" * 80)
    print("PRODUCTION SWEEP SUMMARY")
    print("=" * 80)

    success_count = sum(1 for r in all_results if r.get("success"))
    converged_count = sum(1 for r in all_results if r.get("converged"))
    print(f"Total cases: {len(all_results)}")
    print(f"Successful: {success_count}")
    print(f"Converged: {converged_count}")

    print(f"\n{'Case':>20} | {'Re':>8} | {'AoA':>5} | {'Cd_init':>10} | {'Cd_final':>10} | {'ΔCd%':>8} | {'Eff_init':>8} | {'Eff_final':>8} | {'Conv':>6}")
    print("-" * 96)
    for r in all_results:
        if r.get("success"):
            print(
                f"{r.get('reynolds', 0):>8.0e} x {r.get('aoa', 0):>3.0f}°  | "
                f"{r.get('reynolds', 0):>8.0e} | {r.get('aoa', 0):>5.1f} | "
                f"{r.get('cd_initial', 0):>10.6f} | {r.get('cd_final', 0):>10.6f} | "
                f"{r.get('cd_reduction_percent', 0):>+7.1f}% | "
                f"{r.get('efficiency_initial', 0):>8.1f} | {r.get('efficiency_final', 0):>8.1f} | "
                f"{'YES' if r.get('converged') else 'no':>6}"
            )
        else:
            print(
                f"{r.get('reynolds', 0):>8.0e} x {r.get('aoa', 0):>3.0f}°  | "
                f"{r.get('reynolds', 0):>8.0e} | {r.get('aoa', 0):>5.1f} | "
                f"{'FAILED':>10} | {'':>10} | {'':>8} | {'':>8} | {'':>8} | {'':>6}"
            )

    # Save final summary report
    summary_path = output_root / "sweep_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Production Sweep Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Reynolds: {args.reynolds}\n")
        f.write(f"AoA: {args.aoa}\n")
        f.write(f"Method: {args.method}\n")
        f.write(f"Max iterations: {args.max_iter}\n")
        f.write(f"\nTotal cases: {len(all_results)}\n")
        f.write(f"Successful: {success_count}\n")
        f.write(f"Converged: {converged_count}\n\n")
        for r in all_results:
            f.write(f"{json.dumps(r, default=str)}\n")

    logger.info(f"\nResults saved to: {output_root.resolve()}")
    logger.info(f"Summary: {summary_path}")
    logger.info(f"Full results: {sweep_file}")


if __name__ == "__main__":
    main()