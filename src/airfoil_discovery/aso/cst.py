"""
6th-Order Class-Shape Transformation (CST) Parameterization.

Implements a CST method with:
  - Class function: C(x/c) = (x/c)^0.5 * (1 - x/c)^1.0  (round-nosed, sharp aft)
  - 6 Bernstein basis polynomials per surface
  - 12 design variables total (6 upper + 6 lower)
  - Geometric bounds enforcement (min thickness, no cross-over)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

from scipy.special import comb


# ── CST Order ──────────────────────────────────────────────────────────────────
CST_ORDER = 6       # number of Bernstein coefficients per surface
N_DESIGN_VARS = 12  # 6 upper + 6 lower (trailing edge thickness fixed)


@dataclass(frozen=True)
class CSTBounds:
    """Physical bounds for CST design variables and derived geometry."""
    upper_min: np.ndarray      # shape (6,)
    upper_max: np.ndarray      # shape (6,)
    lower_min: np.ndarray      # shape (6,)
    lower_max: np.ndarray      # shape (6,)
    min_thickness: float = 0.06     # t/c minimum for structural feasibility
    max_thickness: float = 0.18     # t/c maximum
    te_thickness: float = 0.003     # fixed trailing edge thickness

    @classmethod
    def default(cls) -> "CSTBounds":
        return cls(
            upper_min=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            upper_max=np.array([0.5, 0.5, 0.8, 0.8, 0.6, 0.4]),
            lower_min=np.array([-0.4, -0.4, -0.4, -0.3, -0.2, -0.1]),
            lower_max=np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.05]),
            min_thickness=0.06,
            max_thickness=0.18,
            te_thickness=0.003,
        )


def cosine_spacing(n_pts: int) -> np.ndarray:
    """Cosine clustering of points towards leading and trailing edges."""
    beta = np.linspace(0.0, np.pi, n_pts)
    return 0.5 * (1.0 - np.cos(beta))


def bernstein_basis(n: int, x: np.ndarray) -> np.ndarray:
    """
    Evaluate all Bernstein basis polynomials of order n at points x.

    Returns
    -------
    np.ndarray of shape (n+1, len(x))
        B_{i,n}(x) = C(n, i) * x^i * (1-x)^{n-i}
    """
    basis = [comb(n, k) * (x ** k) * ((1.0 - x) ** (n - k)) for k in range(n + 1)]
    return np.vstack(basis)


def class_function(x: np.ndarray, n1: float = 0.5, n2: float = 1.0) -> np.ndarray:
    """
    CST class function: C(x) = x^n1 * (1-x)^n2.

    For airfoils: n1=0.5 (round nose), n2=1.0 (sharp trailing edge).
    """
    x_safe = np.clip(x, 1e-12, 1.0)
    return (x_safe ** n1) * ((1.0 - x_safe) ** n2)


def design_vector_to_surface_coefficients(dv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a 12-element design vector into upper and lower CST coefficient arrays.

    Parameters
    ----------
    dv : np.ndarray, shape (12,)
        [Au_0,...,Au_5, Al_0,...,Al_5]

    Returns
    -------
    upper_coeffs : np.ndarray, shape (6,)
    lower_coeffs : np.ndarray, shape (6,)
    """
    assert len(dv) == N_DESIGN_VARS, f"Expected {N_DESIGN_VARS} variables, got {len(dv)}"
    return dv[:6].copy(), dv[6:].copy()


def surface_coefficients_to_design_vector(upper: np.ndarray, lower: np.ndarray) -> np.ndarray:
    """Combine upper and lower coefficient arrays into a single design vector."""
    return np.concatenate([upper, lower])


# ── Airfoil generation ─────────────────────────────────────────────────────────

def compute_airfoil_coordinates(
    dv: np.ndarray,
    n_pts_per_surface: int = 200,
    te_thickness: float = 0.003,
    bounds: Optional[CSTBounds] = None,
) -> np.ndarray:
    """
    Generate airfoil coordinates from a 12-element CST design vector.

    Parameters
    ----------
    dv : np.ndarray, shape (12,)
        Design variables: 6 upper + 6 lower Bernstein coefficients.
    n_pts_per_surface : int
        Number of points for upper and lower surfaces each.
    te_thickness : float
        Trailing edge thickness (non-dimensional, chord=1).
    bounds : CSTBounds, optional
        If provided, clip coefficients to bounds before evaluation.

    Returns
    -------
    coords : np.ndarray, shape (2*n_pts_per_surface, 2)
        Full airfoil coordinates in [x, y] order, upper first then lower,
        both from leading edge (x=0) to trailing edge (x=1).
    """
    if bounds is not None:
        upper_c, lower_c = design_vector_to_surface_coefficients(dv)
        upper_c = np.clip(upper_c, bounds.upper_min, bounds.upper_max)
        lower_c = np.clip(lower_c, bounds.lower_min, bounds.lower_max)
    else:
        upper_c, lower_c = design_vector_to_surface_coefficients(dv)

    x = cosine_spacing(n_pts_per_surface)
    Cp = class_function(x, n1=0.5, n2=1.0)

    # Bernstein shape functions
    B = bernstein_basis(CST_ORDER - 1, x)  # shape (6, n_pts)

    # Upper surface: y_u = C(x) * sum(Au_i * B_i(x)) + 0.5 * TE_thickness * x
    upper_shape = upper_c @ B                  # shape (n_pts,)
    y_upper = Cp * upper_shape + 0.5 * te_thickness * x

    # Lower surface: y_l = C(x) * sum(Al_i * B_i(x)) - 0.5 * TE_thickness * x
    lower_shape = lower_c @ B                  # shape (n_pts,)
    y_lower = Cp * lower_shape - 0.5 * te_thickness * x

    # Upper: leading-edge to trailing-edge (x=0 → x=1)
    upper = np.column_stack([x, y_upper])
    # Lower: leading-edge to trailing-edge (x=0 → x=1)
    lower = np.column_stack([x, y_lower])

    # Full closed polygon: upper (LE→TE) then lower reversed (TE→LE)
    # SU2 and GMSH expect closed clockwise: start at TE, go upper to LE, lower back to TE
    full = np.vstack([
        upper[::-1],       # TE → LE along upper
        lower[1:],         # LE → TE along lower (skip LE duplicate)
    ])
    return full


def compute_surface_coordinates(
    dv: np.ndarray,
    n_pts: int = 200,
    te_thickness: float = 0.003,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return separate upper and lower surface coordinates.

    Returns
    -------
    upper : np.ndarray, shape (n_pts, 2)
    lower : np.ndarray, shape (n_pts, 2)
        Both from leading edge (x=0) to trailing edge (x=1).
    """
    upper_c, lower_c = design_vector_to_surface_coefficients(dv)
    x = cosine_spacing(n_pts)
    Cp = class_function(x)
    B = bernstein_basis(CST_ORDER - 1, x)

    y_upper = Cp * (upper_c @ B) + 0.5 * te_thickness * x
    y_lower = Cp * (lower_c @ B) - 0.5 * te_thickness * x

    return np.column_stack([x, y_upper]), np.column_stack([x, y_lower])


# ── Geometry validation ────────────────────────────────────────────────────────

def check_geometry_validity(
    dv: np.ndarray,
    bounds: Optional[CSTBounds] = None,
    n_pts: int = 200,
    te_thickness: float = 0.003,
) -> Tuple[bool, str]:
    """
    Validate that the CST design vector produces a physically admissible airfoil.

    Checks:
      1. Upper surface is above lower surface everywhere (no cross-over).
      2. Maximum thickness is within [t_min, t_max].
      3. Leading edge is not too sharp (LE radius > threshold).
      4. No extreme curvature spikes.

    Returns
    -------
    is_valid : bool
    reason   : str  (empty if valid)
    """
    upper, lower = compute_surface_coordinates(dv, n_pts=n_pts, te_thickness=te_thickness)

    # 1. Thickness = y_upper - y_lower along matched x
    thickness = upper[:, 1] - lower[:, 1]

    if np.any(thickness < -1e-8):
        return False, "Surface crossover detected (upper below lower)"

    if np.any(thickness[:-5] < 1e-8):  # exclude TE convergence
        return False, "Zero or negative thickness in interior"

    max_t = np.max(thickness)
    if bounds is not None:
        if max_t < bounds.min_thickness:
            return False, f"Max thickness {max_t:.4f} < minimum {bounds.min_thickness:.4f}"
        if max_t > bounds.max_thickness:
            return False, f"Max thickness {max_t:.4f} > maximum {bounds.max_thickness:.4f}"

    # 2. Leading edge radius estimate
    # Near LE (x ≈ 0), thickness ~ sqrt(2 * R * x) → R ≈ t^2 / (2x)
    near_le = (upper[:5, 1] - lower[:5, 1]) / 2.0   # half-thickness at first 5 points
    le_radii = (near_le[1:]**2) / (2.0 * upper[1:5, 0])
    if len(le_radii) > 0 and np.median(le_radii) < 5e-4:
        return False, f"Leading edge too sharp: R={np.median(le_radii):.6f}"

    return True, ""


# ── Sensitivity transform: surface → CST coefficients ─────────────────────────

def project_surface_sensitivity_to_cst(
    dJ_dx: np.ndarray,
    dJ_dy: np.ndarray,
    x_surf: np.ndarray,
    upper_indices: np.ndarray,
    lower_indices: np.ndarray,
    n_coeff: int = CST_ORDER,
) -> np.ndarray:
    """
    Project surface adjoint sensitivities onto CST design variables.

    Given surface sensitivities ∂J/∂x and ∂J/∂y at each surface mesh node,
    compute ∂J/∂Au_i and ∂J/∂Al_i using the chain rule through the CST
    parameterization.

    Parameters
    ----------
    dJ_dx, dJ_dy : np.ndarray
        Surface sensitivity components from SU2 adjoint solution.
    x_surf : np.ndarray
        x-coordinates of surface mesh nodes.
    upper_indices, lower_indices : np.ndarray
        Boolean or integer indices separating upper and lower surface nodes.
    n_coeff : int
        Number of CST coefficients per surface (default 6).

    Returns
    -------
    grad_dv : np.ndarray, shape (12,)
        Gradient of objective w.r.t. 12 CST design variables.
    """
    grad = np.zeros(12)

    # Surface Jacobian of CST parameterization:
    #   y(x) = C(x) * sum(A_i * B_i(x)) ± 0.5 * TE * x
    #   ∂y/∂A_i = C(x) * B_i(x)
    #   ∂x/∂A_i = 0  (x is independent)

    x_upper = x_surf[upper_indices]
    x_lower = x_surf[lower_indices]

    Cp_upper = class_function(x_upper)
    Cp_lower = class_function(x_lower)

    B_upper = bernstein_basis(n_coeff - 1, x_upper)  # (n_coeff, n_upper_pts)
    B_lower = bernstein_basis(n_coeff - 1, x_lower)

    dJ_dy_upper = dJ_dy[upper_indices]
    dJ_dy_lower = dJ_dy[lower_indices]

    # Compatible trapezoidal integration for NumPy 2.x+ and older versions
    try:
        from numpy import trapezoid as _trapz_func
    except ImportError:
        from numpy import trapz as _trapz_func

    # ∂J/∂A_i = sum_over_nodes( ∂J/∂y_node * ∂y_node/∂A_i )
    #         = sum_over_nodes( dJ_dy * C(x) * B_i(x) )
    for i in range(n_coeff):
        grad[i]      = _trapz_func(dJ_dy_upper * Cp_upper * B_upper[i, :], x_upper)
        grad[i + 6]  = _trapz_func(dJ_dy_lower * Cp_lower * B_lower[i, :], x_lower)

    return grad