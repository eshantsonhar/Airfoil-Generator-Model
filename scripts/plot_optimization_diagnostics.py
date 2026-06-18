#!/usr/bin/env python3
"""
Publication-Ready Optimization Diagnostics Visualization.

Generates multi-panel figures from ASO convergence history and
surface flow data for baseline vs. optimized comparison:

Panel 1: Convergence History (Cd, Cl, gradient norm, constraints)
Panel 2: Geometry Evolution (baseline vs. optimized airfoil, zoomed LE/TE)
Panel 3: Flow Profiles (Cp and Cf distributions showing LSB mitigation)

Usage:
    python scripts/plot_optimization_diagnostics.py \\
        --history /path/to/convergence_history.json \\
        --surface-baseline /path/to/baseline/surface_flow.csv \\
        --surface-optimized /path/to/optimized/surface_flow.csv \\
        --coords-baseline /path/to/baseline_airfoil.dat \\
        --coords-optimized /path/to/optimized_airfoil.dat \\
        --output /path/to/figures
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from airfoil_discovery.aso import (
    parse_surface_flow,
    extract_lsb_from_cf,
    compute_aerodynamic_metrics,
    compare_baseline_optimized,
    SurfaceFlowData,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publication-ready optimization diagnostics visualization"
    )
    parser.add_argument("--history", type=str, default=None, help="Path to convergence_history.json")
    parser.add_argument("--surface-baseline", type=str, default=None, help="Baseline surface_flow.csv")
    parser.add_argument("--surface-optimized", type=str, default=None, help="Optimized surface_flow.csv")
    parser.add_argument("--coords-baseline", type=str, default=None, help="Baseline airfoil .dat")
    parser.add_argument("--coords-optimized", type=str, default=None, help="Optimized airfoil .dat")
    parser.add_argument("--output", type=str, default="aso_diagnostics_figures",
                        help="Output directory for figures")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"],
                        help="Figure format")
    return parser.parse_args()


def load_airfoil_coords(path: Path) -> np.ndarray:
    """Load airfoil coordinates from .dat file."""
    lines = path.read_text().strip().split("\n")
    data = np.loadtxt(lines[1:])  # skip header
    return data


def plot_convergence_history(history_path: Path, output_dir: Path, dpi: int, fmt: str) -> Optional[Path]:
    """Plot convergence history: Cd, Cl, gradient norm, constraints."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history_path.exists():
        logging.warning(f"History file not found: {history_path}")
        return None

    with open(history_path) as f:
        data = json.load(f)

    iterations = data.get("iterations", [])
    if not iterations:
        logging.warning("No iteration data in history file")
        return None

    iters = [it["iteration"] for it in iterations]
    cd = [it["cd"] for it in iterations]
    cl = [it["cl"] for it in iterations]
    grad_norm = [it.get("grad_norm", 0) for it in iterations]
    constraint_viol = [it.get("constraint_violations", []) for it in iterations]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    # Panel 1: Cd
    ax = axes[0, 0]
    ax.plot(iters, cd, "b-o", markersize=4, linewidth=1.5, label=r"$C_D$")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$C_D$")
    ax.set_title("Drag Coefficient Convergence")
    ax.grid(True, alpha=0.3)
    if len(iters) > 1:
        ax.axhline(y=cd[-1], color="gray", linestyle="--", alpha=0.5,
                   label=f"Final: {cd[-1]:.6f}")
    ax.legend()

    # Panel 2: Cl
    ax = axes[0, 1]
    ax.plot(iters, cl, "r-s", markersize=4, linewidth=1.5, label=r"$C_L$")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$C_L$")
    ax.set_title("Lift Coefficient Convergence")
    ax.grid(True, alpha=0.3)
    if len(iters) > 1:
        ax.axhline(y=cl[-1], color="gray", linestyle="--", alpha=0.5,
                   label=f"Final: {cl[-1]:.4f}")
    ax.legend()

    # Panel 3: Gradient norm
    ax = axes[1, 0]
    ax.semilogy(iters, [max(g, 1e-12) for g in grad_norm], "g-^", markersize=4, linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\|\nabla C_D\|$")
    ax.set_title("Gradient Norm Convergence")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1e-4, color="gray", linestyle="--", alpha=0.5, label="Tolerance")
    ax.legend()

    # Panel 4: Constraint violations
    ax = axes[1, 1]
    if constraint_viol and any(cv for cv in constraint_viol):
        for i in range(len(constraint_viol[0])):
            vals = [cv[i] if i < len(cv) else 0 for cv in constraint_viol]
            ax.plot(iters, vals, "-o", markersize=3, linewidth=1,
                    label=f"Constraint {i}")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Constraint Value")
    ax.set_title("Constraint Violations")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle("Optimization Convergence History", fontsize=14, fontweight="bold")

    output_path = output_dir / f"convergence_history.{fmt}"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved: {output_path}")
    return output_path


def plot_geometry_comparison(
    coords_baseline: Optional[Path],
    coords_optimized: Optional[Path],
    output_dir: Path,
    dpi: int,
    fmt: str,
) -> Optional[Path]:
    """Plot baseline vs. optimized airfoil geometry with LE/TE zooms."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not coords_baseline or not coords_baseline.exists():
        logging.warning(f"Baseline coords not found: {coords_baseline}")
        return None
    if not coords_optimized or not coords_optimized.exists():
        logging.warning(f"Optimized coords not found: {coords_optimized}")
        return None

    base = load_airfoil_coords(coords_baseline)
    opt = load_airfoil_coords(coords_optimized)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    # Main airfoil shape
    ax = axes[0, 0]
    ax.plot(base[:, 0], base[:, 1], "b-", linewidth=1.5, alpha=0.8, label="Baseline")
    ax.plot(opt[:, 0], opt[:, 1], "r--", linewidth=1.5, alpha=0.8, label="Optimized")
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_title("Airfoil Shape Comparison")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(-0.02, 1.02)

    # Thickness comparison
    ax = axes[1, 0]
    # Interpolate to common x for thickness
    x_common = np.linspace(0, 1, 500)

    def sort_by_x(coords):
        return coords[np.argsort(coords[:, 0])]

    base_sorted = sort_by_x(base)
    opt_sorted = sort_by_x(opt)

    # Split upper/lower approximately
    base_upper = base_sorted[: len(base_sorted) // 2]
    base_lower = base_sorted[len(base_sorted) // 2 :]
    opt_upper = opt_sorted[: len(opt_sorted) // 2]
    opt_lower = opt_sorted[len(opt_sorted) // 2 :]

    base_y_u = np.interp(x_common, base_upper[:, 0], base_upper[:, 1], left=0, right=0)
    base_y_l = np.interp(x_common, base_lower[:, 0], base_lower[:, 1], left=0, right=0)
    opt_y_u = np.interp(x_common, opt_upper[:, 0], opt_upper[:, 1], left=0, right=0)
    opt_y_l = np.interp(x_common, opt_lower[:, 0], opt_lower[:, 1], left=0, right=0)

    base_t = base_y_u - base_y_l
    opt_t = opt_y_u - opt_y_l

    ax.plot(x_common, base_t, "b-", linewidth=1.5, alpha=0.8, label="Baseline")
    ax.plot(x_common, opt_t, "r--", linewidth=1.5, alpha=0.8, label="Optimized")
    ax.set_xlabel("x/c")
    ax.set_ylabel("Thickness (y_u - y_l)")
    ax.set_title("Thickness Distribution")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # LE zoom
    ax = axes[0, 1]
    ax.plot(base[:, 0], base[:, 1], "b-", linewidth=1.5, alpha=0.8)
    ax.plot(opt[:, 0], opt[:, 1], "r--", linewidth=1.5, alpha=0.8)
    ax.set_xlim(-0.005, 0.08)
    ax.set_ylim(-0.06, 0.06)
    ax.set_aspect("equal")
    ax.set_title("Leading Edge Detail")
    ax.grid(True, alpha=0.3)

    # TE zoom
    ax = axes[1, 1]
    ax.plot(base[:, 0], base[:, 1], "b-", linewidth=1.5, alpha=0.8, label="Baseline")
    ax.plot(opt[:, 0], opt[:, 1], "r--", linewidth=1.5, alpha=0.8, label="Optimized")
    ax.set_xlim(0.92, 1.01)
    ax.set_ylim(-0.02, 0.02)
    ax.set_aspect("equal")
    ax.set_title("Trailing Edge Detail")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle("Baseline vs. Optimized Airfoil Geometry", fontsize=14, fontweight="bold")

    output_path = output_dir / f"geometry_comparison.{fmt}"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved: {output_path}")
    return output_path


def plot_flow_profiles(
    surface_baseline: Optional[Path],
    surface_optimized: Optional[Path],
    output_dir: Path,
    dpi: int,
    fmt: str,
) -> Optional[Path]:
    """Plot Cp and Cf distributions showing LSB mitigation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not surface_baseline or not surface_baseline.exists():
        logging.warning(f"Baseline surface flow not found: {surface_baseline}")
        return None
    if not surface_optimized or not surface_optimized.exists():
        logging.warning(f"Optimized surface flow not found: {surface_optimized}")
        return None

    # Parse surface flow data
    sf_base = parse_surface_flow(surface_baseline)
    sf_opt = parse_surface_flow(surface_optimized)

    if not sf_base.has_upper_lower_split or not sf_opt.has_upper_lower_split:
        logging.warning("Surface flow data lacks upper/lower split")
        return None

    # LSB detection
    lsb_base = extract_lsb_from_cf(sf_base.x_upper, sf_base.cf_upper)
    lsb_opt = extract_lsb_from_cf(sf_opt.x_upper, sf_opt.cf_upper)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # Panel 1: Cp distribution
    ax = axes[0]
    ax.plot(sf_base.x_upper, sf_base.cp_upper, "b-", linewidth=1.5, alpha=0.8,
            label=f"Baseline (Cd={np.mean(np.abs(sf_base.cp)):.3f})")
    ax.plot(sf_opt.x_upper, sf_opt.cp_upper, "r--", linewidth=1.5, alpha=0.8,
            label=f"Optimized (Cd={np.mean(np.abs(sf_opt.cp)):.3f})")

    # Mark LSB plateau region on baseline
    if lsb_base.lsb_detected and lsb_base.plateau_start and lsb_base.plateau_end:
        ax.axvspan(lsb_base.plateau_start, lsb_base.plateau_end,
                   alpha=0.15, color="blue", label=f"LSB (base)")
    if lsb_opt.lsb_detected and lsb_opt.plateau_start and lsb_opt.plateau_end:
        ax.axvspan(lsb_opt.plateau_start, lsb_opt.plateau_end,
                   alpha=0.15, color="red", label=f"LSB (opt)")

    ax.invert_yaxis()  # Cp convention: negative up
    ax.set_xlabel("x/c")
    ax.set_ylabel(r"$C_p$")
    ax.set_title("Pressure Coefficient Distribution")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Panel 2: Cf distribution
    ax = axes[1]
    ax.plot(sf_base.x_upper, sf_base.cf_upper, "b-", linewidth=1.5, alpha=0.8, label="Baseline")
    ax.plot(sf_opt.x_upper, sf_opt.cf_upper, "r--", linewidth=1.5, alpha=0.8, label="Optimized")

    # Mark Cf=0 line
    ax.axhline(y=0, color="gray", linewidth=0.8, linestyle=":", alpha=0.7, label=r"$C_f = 0$")

    # Mark separation and reattachment points
    if lsb_base.separation_point:
        ax.axvline(x=lsb_base.separation_point, color="blue", linestyle=":",
                   alpha=0.6, linewidth=1.0,
                   label=f"Sep (base)={lsb_base.separation_point:.3f}")
    if lsb_base.reattachment_point:
        ax.axvline(x=lsb_base.reattachment_point, color="blue", linestyle="--",
                   alpha=0.6, linewidth=1.0,
                   label=f"Reatt (base)={lsb_base.reattachment_point:.3f}")
    if lsb_opt.separation_point:
        ax.axvline(x=lsb_opt.separation_point, color="red", linestyle=":",
                   alpha=0.6, linewidth=1.0,
                   label=f"Sep (opt)={lsb_opt.separation_point:.3f}")
    if lsb_opt.reattachment_point:
        ax.axvline(x=lsb_opt.reattachment_point, color="red", linestyle="--",
                   alpha=0.6, linewidth=1.0,
                   label=f"Reatt (opt)={lsb_opt.reattachment_point:.3f}")

    ax.set_xlabel("x/c")
    ax.set_ylabel(r"$C_f$")
    ax.set_title("Skin Friction Distribution (Upper Surface)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    # LSB summary text box
    if lsb_base.lsb_detected or lsb_opt.lsb_detected:
        summary_lines = ["LSB Characteristics:"]
        if lsb_base.lsb_detected:
            summary_lines.append(
                f"  Baseline: sep={lsb_base.separation_point:.3f}, "
                f"reatt={lsb_base.reattachment_point:.3f}, "
                f"L={lsb_base.bubble_length:.3f}"
            )
        else:
            summary_lines.append("  Baseline: No LSB")
        if lsb_opt.lsb_detected:
            summary_lines.append(
                f"  Optimized: sep={lsb_opt.separation_point:.3f}, "
                f"reatt={lsb_opt.reattachment_point:.3f}, "
                f"L={lsb_opt.bubble_length:.3f}"
            )
        else:
            summary_lines.append("  Optimized: No LSB")

        # Add text box to Cf plot
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        ax.text(0.98, 0.98, "\n".join(summary_lines),
                transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=props)

    fig.suptitle("Flow Profile Comparison: Baseline vs. Optimized",
                 fontsize=14, fontweight="bold")

    output_path = output_dir / f"flow_profiles.{fmt}"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved: {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "plot_generation.log", encoding="utf-8"),
        ],
    )
    logger = logging.getLogger("plot_diagnostics")
    logger.info("Generating optimization diagnostics plots...")

    # Panel 1: Convergence History
    if args.history:
        plot_convergence_history(
            Path(args.history), output_dir, args.dpi, args.format
        )

    # Panel 2: Geometry Evolution
    if args.coords_baseline or args.coords_optimized:
        plot_geometry_comparison(
            Path(args.coords_baseline) if args.coords_baseline else None,
            Path(args.coords_optimized) if args.coords_optimized else None,
            output_dir, args.dpi, args.format,
        )

    # Panel 3: Flow Profiles
    if args.surface_baseline or args.surface_optimized:
        plot_flow_profiles(
            Path(args.surface_baseline) if args.surface_baseline else None,
            Path(args.surface_optimized) if args.surface_optimized else None,
            output_dir, args.dpi, args.format,
        )

    logger.info(f"All figures saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()