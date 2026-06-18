#!/usr/bin/env python3
"""
Adjoint Gradient Verification via Taylor Series / Finite Difference Check.

Runs a rigorous gradient accuracy test comparing discrete adjoint gradients
(from SU2_CFD_ADJ + CST projection) against central finite difference (FD)
approximations across multiple perturbation step sizes.

The Taylor series expansion:
    f(x + eps*e_i) - f(x - eps*e_i)
    -------------------------------  ->  df/dx_i  as eps -> 0
                2*eps

Theory predicts:
  - Central FD error is O(eps^2) — reducing eps by 10x should reduce
    the relative error by 100x, until floating-point precision dominates.
  - Cosine similarity between adjoint and FD gradients should be > 0.95
    for a well-posed adjoint.

Usage:
    python scripts/verify_adjoint_gradients.py \\
        --su2-cfd /path/to/SU2_CFD \\
        --mesh /path/to/mesh.su2 \\
        --output verification_results
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from airfoil_discovery.aso import (
    N_DESIGN_VARS,
    CSTBounds,
    compute_airfoil_coordinates,
    check_geometry_validity,
    generate_primal_config,
    generate_adjoint_config,
    write_primal_config,
    write_adjoint_config,
    extract_adjoint_gradient,
    verify_adjoint_gradient as verify_grad,
)
from airfoil_discovery.aso.optimizer import run_primal_and_adjoint, CFDResult
from airfoil_discovery.aso.diagnostics import compute_aerodynamic_metrics


def setup_logging(log_path: Path, verbose: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(fh)
    root.addHandler(ch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adjoint Gradient Verification via Taylor Series / FD Check"
    )
    parser.add_argument("--su2-cfd", required=True, help="Path to SU2_CFD executable")
    parser.add_argument("--mesh", required=True, help="Path to baseline SU2 mesh")
    parser.add_argument("--output", default="gradient_verification", help="Output directory")
    parser.add_argument("--aoa", type=float, default=4.0, help="Angle of attack (degrees)")
    parser.add_argument("--reynolds", type=float, default=1e5, help="Chord Reynolds number")
    parser.add_argument("--mach", type=float, default=0.1, help="Freestream Mach")
    parser.add_argument("--n-iter-primal", type=int, default=2000, help="Primal iterations")
    parser.add_argument("--n-iter-adjoint", type=int, default=300, help="Adjoint iterations")
    parser.add_argument("--epsilons", type=str, default="1e-2,1e-3,1e-4,1e-5,1e-6,1e-7",
                        help="Comma-separated FD step sizes to test")
    parser.add_argument("--dv-index", type=int, default=None,
                        help="Test only one DV index (0-11). If None, test all.")
    parser.add_argument("--design", type=str, default=None,
                        help="Path to .npy file with initial 12-DV design. Defaults to NACA-like.")
    parser.add_argument("--no-adjoint", action="store_true",
                        help="Skip adjoint solve, only compute FD gradients")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def compute_fd_gradient(
    dv: np.ndarray,
    eps: float,
    su2_cfd_bin: str,
    mesh_path: Path,
    case_root: Path,
    aoa_deg: float,
    reynolds: float,
    mach: float,
    n_iter_primal: int,
    n_iter_adjoint: int,
    cfl_primal: float = 3.0,
    cfl_adjoint: float = 1.0,
    transition_model: bool = True,
    turb_intensity: float = 0.001,
    turb_viscosity_ratio: float = 5.0,
    dv_indices: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute central finite difference gradient.

    For each design variable i:
      fd_i = [f(dv + eps*e_i) - f(dv - eps*e_i)] / (2*eps)

    Parameters
    ----------
    dv : np.ndarray, shape (12,)
    eps : float
        Perturbation step size.
    dv_indices : list of int, optional
        Indices to perturb. None = all 12.

    Returns
    -------
    fd_grad : np.ndarray, shape (12,)
    cd_base : float
        Baseline drag coefficient.
    fd_errors : np.ndarray, shape (12,)
        Estimated FD error from asymmetry.
    """
    if dv_indices is None:
        dv_indices = list(range(N_DESIGN_VARS))

    # Baseline evaluation
    base_dir = case_root / f"base_fd"
    if base_dir.exists():
        import shutil
        shutil.rmtree(base_dir)

    logger = logging.getLogger(__name__)
    logger.info(f"Running baseline primal at dv={dv}")

    base_result = run_primal_and_adjoint(
        su2_cfd_bin=su2_cfd_bin,
        su2_adj_bin=su2_cfd_bin,
        mesh_path=mesh_path,
        dv=dv,
        case_dir=base_dir,
        aoa_deg=aoa_deg,
        reynolds=reynolds,
        mach=mach,
        n_iter_primal=n_iter_primal,
        n_iter_adjoint=n_iter_adjoint,
        cfl_primal=cfl_primal,
        cfl_adjoint=cfl_adjoint,
        transition_model=transition_model,
        turbulence_intensity=turb_intensity,
        turb_viscosity_ratio=turb_viscosity_ratio,
    )

    if not base_result.primal_converged:
        logger.warning(f"Baseline primal did not converge: {base_result.failure_reason}")
        cd_base = base_result.cd or 1e10
    else:
        cd_base = base_result.cd

    fd_grad = np.zeros(N_DESIGN_VARS)
    fd_errors = np.zeros(N_DESIGN_VARS)

    for idx in dv_indices:
        # Forward perturbation
        dv_plus = dv.copy()
        dv_plus[idx] += eps
        valid_plus, _ = check_geometry_validity(dv_plus)
        if not valid_plus:
            logger.warning(f"DV[{idx}]+eps invalid geometry, skipping")
            continue

        plus_dir = case_root / f"fd_plus_dv{idx}"
        res_plus = run_primal_and_adjoint(
            su2_cfd_bin=su2_cfd_bin, su2_adj_bin=su2_cfd_bin,
            mesh_path=mesh_path, dv=dv_plus, case_dir=plus_dir,
            aoa_deg=aoa_deg, reynolds=reynolds, mach=mach,
            n_iter_primal=n_iter_primal, n_iter_adjoint=n_iter_adjoint,
            cfl_primal=cfl_primal, cfl_adjoint=cfl_adjoint,
            transition_model=transition_model,
            turbulence_intensity=turb_intensity,
            turb_viscosity_ratio=turb_viscosity_ratio,
        )

        # Backward perturbation
        dv_minus = dv.copy()
        dv_minus[idx] -= eps
        valid_minus, _ = check_geometry_validity(dv_minus)
        if not valid_minus:
            logger.warning(f"DV[{idx}]-eps invalid geometry, skipping")
            continue

        minus_dir = case_root / f"fd_minus_dv{idx}"
        res_minus = run_primal_and_adjoint(
            su2_cfd_bin=su2_cfd_bin, su2_adj_bin=su2_cfd_bin,
            mesh_path=mesh_path, dv=dv_minus, case_dir=minus_dir,
            aoa_deg=aoa_deg, reynolds=reynolds, mach=mach,
            n_iter_primal=n_iter_primal, n_iter_adjoint=n_iter_adjoint,
            cfl_primal=cfl_primal, cfl_adjoint=cfl_adjoint,
            transition_model=transition_model,
            turbulence_intensity=turb_intensity,
            turb_viscosity_ratio=turb_viscosity_ratio,
        )

        cd_plus = res_plus.cd if res_plus.primal_converged else 1e10
        cd_minus = res_minus.cd if res_minus.primal_converged else 1e10

        fd_grad[idx] = (cd_plus - cd_minus) / (2.0 * eps)
        # Error estimate: asymmetry
        fd_errors[idx] = abs((cd_plus + cd_minus - 2.0 * cd_base) / (eps**2))

        logger.info(
            f"  DV[{idx:2d}]: cd_plus={cd_plus:.6f}, cd_minus={cd_minus:.6f}, "
            f"fd_grad={fd_grad[idx]:.6e}, asymmetry={fd_errors[idx]:.6e}"
        )

    return fd_grad, cd_base, fd_errors


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "verification.log", verbose=args.verbose)
    logger = logging.getLogger("gradient_verify")

    logger.info("=" * 60)
    logger.info("Adjoint Gradient Verification")
    logger.info(f"Mesh: {args.mesh}")
    logger.info(f"AoA: {args.aoa}, Re: {args.reynolds:.1e}, Mach: {args.mach}")
    logger.info("=" * 60)

    mesh_path = Path(args.mesh).resolve()
    if not mesh_path.exists():
        logger.error(f"Mesh not found: {mesh_path}")
        sys.exit(1)

    su2_cfd_bin = args.su2_cfd

    # Load or create baseline design
    if args.design:
        dv = np.load(args.design)
        logger.info(f"Loaded design from {args.design}")
    else:
        dv = np.array([0.18, 0.28, 0.34, 0.25, 0.15, 0.08,
                       -0.19, -0.12, -0.09, -0.05, -0.02, -0.01])
        logger.info("Using default NACA-like design")

    assert len(dv) == N_DESIGN_VARS, f"Design vector has {len(dv)} elements, expected {N_DESIGN_VARS}"

    # Check design validity
    valid, reason = check_geometry_validity(dv)
    if not valid:
        logger.error(f"Baseline design invalid: {reason}")
        sys.exit(1)

    epsilons = [float(e) for e in args.epsilons.split(",")]
    logger.info(f"FD step sizes: {epsilons}")

    dv_indices = [args.dv_index] if args.dv_index is not None else list(range(N_DESIGN_VARS))
    logger.info(f"Testing DV indices: {dv_indices}")

    case_root = output_dir / "cfd_cases"
    case_root.mkdir(parents=True, exist_ok=True)

    # ── 1. Compute adjoint gradient at baseline ──
    adj_grad = None
    if not args.no_adjoint:
        logger.info("Computing adjoint gradient at baseline design...")
        adj_dir = case_root / "adjoint_baseline"
        adj_result = run_primal_and_adjoint(
            su2_cfd_bin=su2_cfd_bin, su2_adj_bin=su2_cfd_bin,
            mesh_path=mesh_path, dv=dv, case_dir=adj_dir,
            aoa_deg=args.aoa, reynolds=args.reynolds, mach=args.mach,
            n_iter_primal=args.n_iter_primal, n_iter_adjoint=args.n_iter_adjoint,
            cfl_primal=3.0, cfl_adjoint=1.0,
            transition_model=True,
        )
        if adj_result.gradient_valid:
            adj_grad = adj_result.adjoint_gradient.copy()
            logger.info(f"Adjoint gradient: norm={np.linalg.norm(adj_grad):.6e}")
            for i in range(N_DESIGN_VARS):
                logger.info(f"  DV[{i:2d}]: adj={adj_grad[i]:+.6e}")
        else:
            logger.warning(f"Adjoint gradient invalid: {adj_result.failure_reason}")

    # ── 2. Compute FD gradients at each epsilon ──
    results: Dict[str, List] = {
        "metadata": {
            "aoa": args.aoa,
            "reynolds": args.reynolds,
            "mach": args.mach,
            "n_iter_primal": args.n_iter_primal,
            "n_iter_adjoint": args.n_iter_adjoint,
            "dv_indices": dv_indices,
            "baseline_dv": dv.tolist(),
        },
        "epsilons": [],
    }

    for eps in epsilons:
        logger.info(f"\n--- FD step size: eps = {eps:.1e} ---")
        fd_grad, cd_base, fd_errors = compute_fd_gradient(
            dv=dv, eps=eps,
            su2_cfd_bin=su2_cfd_bin, mesh_path=mesh_path,
            case_root=case_root / f"eps_{eps:.0e}",
            aoa_deg=args.aoa, reynolds=args.reynolds, mach=args.mach,
            n_iter_primal=args.n_iter_primal, n_iter_adjoint=args.n_iter_adjoint,
            dv_indices=dv_indices,
        )

        entry = {
            "epsilon": eps,
            "cd_base": float(cd_base),
            "fd_gradient": fd_grad.tolist(),
            "fd_errors": fd_errors.tolist(),
        }

        # Compare with adjoint
        if adj_grad is not None:
            adj_norm = float(np.linalg.norm(adj_grad))
            fd_norm = float(np.linalg.norm(fd_grad))

            entry["adjoint_gradient"] = adj_grad.tolist()
            entry["adjoint_norm"] = adj_norm
            entry["fd_norm"] = fd_norm

            if adj_norm > 0 and fd_norm > 0:
                cos_sim = float(np.dot(adj_grad, fd_grad) / (adj_norm * fd_norm))
                l2_error = float(np.linalg.norm(adj_grad - fd_grad))
                rel_error = float(l2_error / adj_norm) if adj_norm > 0 else float("inf")
                entry["cosine_similarity"] = cos_sim
                entry["l2_error"] = l2_error
                entry["relative_error"] = rel_error

                # Per-DV comparison table
                dv_table = []
                for i in dv_indices:
                    adj_i = adj_grad[i]
                    fd_i = fd_grad[i]
                    abs_err = abs(adj_i - fd_i)
                    rel_err_i = abs_err / max(abs(adj_i), 1e-30)
                    dv_table.append({
                        "index": i,
                        "adjoint": float(adj_i),
                        "fd": float(fd_i),
                        "absolute_error": float(abs_err),
                        "relative_error": float(rel_err_i),
                    })
                entry["dv_comparison"] = dv_table

                logger.info(f"  Cosine similarity: {cos_sim:.6f}")
                logger.info(f"  L2 error: {l2_error:.6e}")
                logger.info(f"  Relative error: {rel_error:.6e}")
                logger.info(f"  |adj|={adj_norm:.6e}, |fd|={fd_norm:.6e}")
            else:
                entry["cosine_similarity"] = 0.0
                entry["l2_error"] = float("inf")
                entry["relative_error"] = float("inf")

        results["epsilons"].append(entry)

    # ── 3. Save results ──
    results_path = output_dir / "gradient_verification_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    # ── 4. Print summary table ──
    print("\n" + "=" * 80)
    print("GRADIENT VERIFICATION SUMMARY")
    print("=" * 80)

    header = f"{'eps':>10} | {'|fd|':>12} | {'|adj|':>12} | {'cosine_sim':>12} | {'rel_err':>12}"
    print(header)
    print("-" * len(header))

    for entry in results["epsilons"]:
        cs = entry.get("cosine_similarity", float("nan"))
        re = entry.get("relative_error", float("nan"))
        fn = entry.get("fd_norm", 0.0)
        an = entry.get("adjoint_norm", 0.0)
        print(f"{entry['epsilon']:>10.1e} | {fn:>12.4e} | {an:>12.4e} | {cs:>12.6f} | {re:>12.4e}")

    print("-" * len(header))
    print()

    # ── 5. Per-DV table (best epsilon) ──
    if results["epsilons"]:
        best_eps = min(results["epsilons"], key=lambda e: e.get("relative_error", float("inf")))
        print(f"\nPer-DV Comparison (best eps = {best_eps['epsilon']:.1e}):")
        print(f"{'DV':>4} | {'Adjoint':>14} | {'FD':>14} | {'Abs Err':>14} | {'Rel Err':>14}")
        print("-" * 64)
        for dv_row in best_eps.get("dv_comparison", []):
            print(
                f"{dv_row['index']:>4d} | {dv_row['adjoint']:>14.6e} | "
                f"{dv_row['fd']:>14.6e} | {dv_row['absolute_error']:>14.6e} | "
                f"{dv_row['relative_error']:>14.6e}"
            )

    logger.info("Gradient verification complete.")
    print(f"\nDetailed results: {results_path.resolve()}")


if __name__ == "__main__":
    main()