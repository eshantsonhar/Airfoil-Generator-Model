"""
Adjoint Gradient Extraction and Surface Sensitivity Projection.

Reads SU2_CFD_ADJ surface sensitivity output files and projects them
onto the 12 CST design variables using the analytic chain rule through
the CST parameterization (∂y/∂A_i = C(x) * B_i(x)).

Supports:
  - Surface sensitivity CSV files from SU2_CFD_ADJ
  - Projection onto Bernstein coefficient design variables
  - Gradient scaling and validation checks
"""

from __future__ import annotations

import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

from .cst import (
    N_DESIGN_VARS,
    CST_ORDER,
    bernstein_basis,
    class_function,
    project_surface_sensitivity_to_cst,
)

logger = logging.getLogger(__name__)


def parse_surface_sensitivity_file(
    filepath: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse an SU2 surface_adjoint CSV file containing nodal sensitivities.

    SU2 v8 format (surface_adjoint.csv):
      x, y, z, dJ/dx, dJ/dy, dJ/dz

    Parameters
    ----------
    filepath : Path
        Path to surface_adjoint.csv file.

    Returns
    -------
    x_surf : np.ndarray, shape (N,)
        x-coordinates of surface nodes.
    dJ_dx : np.ndarray, shape (N,)
    dJ_dy : np.ndarray, shape (N,)
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Surface sensitivity file not found: {filepath}")

    try:
        data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    except Exception as e:
        # Try space-delimited format
        data = np.loadtxt(filepath, skiprows=1)

    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 5:
        raise ValueError(
            f"Surface file has {data.shape[1]} columns, expected >= 5 "
            f"(x, y, z, dJ/dx, dJ/dy, ...)"
        )

    x_surf = data[:, 0]
    dJ_dx = data[:, 3]
    dJ_dy = data[:, 4]

    return x_surf, dJ_dx, dJ_dy


def detect_upper_lower_split(
    x_surf: np.ndarray,
    y_surf: np.ndarray,
    le_x_tolerance: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect upper and lower surface node indices from the surface mesh.

    The SU2 surface file goes clockwise around the airfoil:
      trailing edge upper → leading edge → trailing edge lower.

    The leading edge is the point with minimum x (and y near 0).
    Nodes before the LE are upper surface, after LE are lower surface.

    Parameters
    ----------
    x_surf : np.ndarray
        x-coordinates of surface nodes.
    y_surf : np.ndarray
        y-coordinates of surface nodes (used for disambiguation).
    le_x_tolerance : float
        Tolerance for detecting the leading edge minimum.

    Returns
    -------
    upper_indices : np.ndarray, dtype bool
    lower_indices : np.ndarray, dtype bool
    """
    # Find leading edge index (minimum x)
    le_idx = int(np.argmin(x_surf))

    # Verify this is the LE (should have y near 0)
    if abs(y_surf[le_idx]) > 0.05:
        logger.warning(
            f"LE candidate at x={x_surf[le_idx]:.4f} has y={y_surf[le_idx]:.4f}, "
            f"may not be true LE"
        )

    n = len(x_surf)
    upper_indices = np.zeros(n, dtype=bool)
    lower_indices = np.zeros(n, dtype=bool)

    # Upper surface: from TE (index 0) to LE (index le_idx)
    upper_indices[: le_idx + 1] = True
    # Lower surface: from LE (index le_idx) to TE (last index)
    lower_indices[le_idx:] = True

    # Handle leading edge duplicate: the LE point belongs to both
    # surfaces in the SU2 surface file. Assign it to upper only.
    lower_indices[le_idx] = False

    # Verify orientation: upper nodes should have positive y
    y_upper = y_surf[upper_indices]
    if np.mean(y_upper) < 0:
        logger.warning("Upper surface appears to have negative y. Swapping split.")
        upper_indices, lower_indices = lower_indices, upper_indices

    return upper_indices, lower_indices


def extract_adjoint_gradient(
    case_dir: Path,
    objective: str = "DRAG",
    n_dv: int = N_DESIGN_VARS,
) -> np.ndarray:
    """
    Extract the gradient of the objective w.r.t. 12 CST design variables
    from the SU2 adjoint surface sensitivity file(s) in case_dir.

    Steps:
    1. Locate the surface_adjoint CSV file.
    2. Parse nodal sensitivities (dJ/dx, dJ/dy).
    3. Detect upper/lower surface split.
    4. Project onto CST basis using chain rule.

    Parameters
    ----------
    case_dir : Path
        Directory containing SU2 adjoint output files.
    objective : str
        Objective function name (used for file pattern, default DRAG).
    n_dv : int
        Number of design variables (default 12).

    Returns
    -------
    grad : np.ndarray, shape (12,)
        Gradient vector: [dJ/dAu_0 ... dJ/dAu_5, dJ/dAl_0 ... dJ/dAl_5].
    """
    # Locate surface sensitivity file
    patterns = [
        "surface_adjoint*.csv",
        "surface_adjoint*.dat",
        "*surface_adjoint*",
        f"*{objective.lower()}*surface*",
    ]
    sens_files = []
    for pat in patterns:
        sens_files = list(case_dir.glob(pat))
        if sens_files:
            break

    if not sens_files:
        # Try to find any adjoint output file
        all_csv = list(case_dir.glob("*.csv"))
        adj_csv = [f for f in all_csv if "adj" in f.name.lower() or "sensi" in f.name.lower()]
        if adj_csv:
            sens_files = adj_csv

    if not sens_files:
        raise FileNotFoundError(
            f"No adjoint surface sensitivity file found in {case_dir}. "
            f"Searched patterns: {patterns}"
        )

    sens_file = sens_files[0]
    logger.info(f"Reading adjoint sensitivities from: {sens_file.name}")

    # Parse sensitivity data
    x_surf, dJ_dx, dJ_dy = parse_surface_sensitivity_file(sens_file)

    # Detect upper/lower split from y-coordinates
    y_surf = np.zeros_like(x_surf)  # We don't have y from the adjoint file header
    # Try to get y from a surface_flow file nearby
    flow_surf_files = list(case_dir.glob("surface_flow*"))
    if flow_surf_files:
        try:
            flow_data = np.loadtxt(flow_surf_files[0], delimiter=",", skiprows=1)
            if flow_data.ndim >= 2 and flow_data.shape[1] >= 2:
                y_surf = flow_data[:, 1]
        except Exception:
            pass

    # If we still don't have y, estimate from x and the sign of dJ_dy
    if np.all(np.abs(y_surf) < 1e-12):
        # Use the derivative signs: dJ/dy tends to be positive on upper,
        # negative on lower for drag minimization
        y_surf = np.sign(dJ_dy) * 0.01  # small proxy values

    upper_idx, lower_idx = detect_upper_lower_split(x_surf, y_surf)

    logger.info(
        f"Surface nodes: total={len(x_surf)}, "
        f"upper={np.sum(upper_idx)}, lower={np.sum(lower_idx)}"
    )

    # Project surface sensitivities onto CST coefficients
    grad = project_surface_sensitivity_to_cst(
        dJ_dx=dJ_dx,
        dJ_dy=dJ_dy,
        x_surf=x_surf,
        upper_indices=upper_idx,
        lower_indices=lower_idx,
        n_coeff=CST_ORDER,
    )

    # Validate gradient (check for NaN, Inf)
    if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
        logger.error("Adjoint gradient contains NaN/Inf values")
        return np.zeros(n_dv)

    # Clip extreme gradients
    grad_norm = np.linalg.norm(grad)
    max_grad_norm = 10.0
    if grad_norm > max_grad_norm:
        logger.warning(
            f"Gradient norm {grad_norm:.4f} > {max_grad_norm}, scaling down"
        )
        grad *= max_grad_norm / grad_norm

    logger.info(
        f"Adjoint gradient extracted: norm={np.linalg.norm(grad):.6f}, "
        f"min={np.min(grad):.6f}, max={np.max(grad):.6f}"
    )

    return grad


def extract_adjoint_gradient_from_data(
    dJ_dx: np.ndarray,
    dJ_dy: np.ndarray,
    x_surf: np.ndarray,
    upper_indices: np.ndarray,
    lower_indices: np.ndarray,
    n_coeff: int = CST_ORDER,
) -> np.ndarray:
    """
    Extract CST gradient from pre-parsed surface sensitivity data.

    Direct interface for use when data is already in memory
    (e.g., from SU2 Python wrapper or in-memory results).

    Parameters
    ----------
    dJ_dx, dJ_dy : np.ndarray
        Surface sensitivity components.
    x_surf : np.ndarray
        x-coordinates of surface nodes.
    upper_indices, lower_indices : np.ndarray
        Boolean index arrays.
    n_coeff : int
        Number of CST coefficients (default 6).

    Returns
    -------
    grad : np.ndarray, shape (12,)
    """
    return project_surface_sensitivity_to_cst(
        dJ_dx=dJ_dx,
        dJ_dy=dJ_dy,
        x_surf=x_surf,
        upper_indices=upper_indices,
        lower_indices=lower_indices,
        n_coeff=n_coeff,
    )


def verify_adjoint_gradient(
    grad_adjoint: np.ndarray,
    grad_fd: Optional[np.ndarray] = None,
    cosine_threshold: float = 0.9,
) -> Dict[str, Any]:
    """
    Verify adjoint gradient against finite-difference reference (if provided)
    or check internal consistency.

    Parameters
    ----------
    grad_adjoint : np.ndarray, shape (12,)
    grad_fd : np.ndarray, shape (12,), optional
        Finite-difference gradient for verification.

    Returns
    -------
    report : dict
        Verification report with cosine similarity, norm ratio, etc.
    """
    report: Dict[str, Any] = {
        "adjoint_norm": float(np.linalg.norm(grad_adjoint)),
        "is_valid": True,
        "warnings": [],
    }

    if grad_fd is not None:
        fd_norm = np.linalg.norm(grad_fd)
        report["fd_norm"] = float(fd_norm)

        if fd_norm > 1e-15 and report["adjoint_norm"] > 1e-15:
            cos_sim = np.dot(grad_adjoint, grad_fd) / (report["adjoint_norm"] * fd_norm)
            report["cosine_similarity"] = float(cos_sim)
            report["norm_ratio"] = float(report["adjoint_norm"] / fd_norm)

            if cos_sim < cosine_threshold:
                report["is_valid"] = False
                report["warnings"].append(
                    f"Cosine similarity {cos_sim:.4f} < threshold {cosine_threshold}"
                )

            if report["norm_ratio"] > 5.0 or report["norm_ratio"] < 0.2:
                report["warnings"].append(
                    f"Norm ratio {report['norm_ratio']:.4f} outside [0.2, 5.0]"
                )
        else:
            report["warnings"].append("Zero gradient detected in adjoint or FD")

    else:
        # Self-consistency checks
        if report["adjoint_norm"] < 1e-12:
            report["is_valid"] = False
            report["warnings"].append("Zero adjoint gradient")
        elif report["adjoint_norm"] > 100.0:
            report["warnings"].append(f"Large adjoint gradient: {report['adjoint_norm']:.4f}")

    return report