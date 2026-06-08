"""
Low-Reynolds-number CFD physics utilities.

WHY THIS MODULE EXISTS
======================
At Re = 10,000–50,000 the boundary layer is almost entirely laminar.  A
fully-turbulent RANS model (k-ω SST without transition) treats the entire
boundary layer as turbulent from the leading edge.  This produces:

  • Skin friction 3–10× too high  →  Cd ~ 0.1 instead of 0.01–0.03
  • No laminar separation bubble  →  Cl/Cd ~ 3–7 instead of 15–40
  • Incorrect stall prediction

The Langtry–Menter γ–Reθ transition model (LM2009, also called "SST-LM" or
"4-equation SST") adds two transport equations:
  • γ  (intermittency): 0 = fully laminar, 1 = fully turbulent
  • Reθ (transition momentum-thickness Reynolds number)

These equations detect:
  1. Natural transition via Tollmien–Schlichting instabilities
  2. Separation-induced transition (laminar separation bubble)
  3. Bypass transition under high freestream turbulence

LAMINAR SEPARATION BUBBLE (LSB) PHYSICS
========================================
At low Re, the adverse pressure gradient on the suction surface causes the
laminar boundary layer to separate before it can transition.  The separated
shear layer then transitions to turbulence and reattaches, forming a closed
recirculation zone (the LSB).  The LSB:
  • Adds a small drag increment (displacement thickness effect)
  • Pins the transition location
  • Bursts (open separation) at high AoA → abrupt stall

Without transition modeling, SU2 with SST predicts turbulent reattachment
immediately at the leading edge, eliminating the LSB entirely and producing
massive drag overprediction.

BOUNDARY LAYER RESOLUTION
==========================
The viscous sublayer (y+ < 5) must be resolved with at least one cell inside
y+ < 1 for wall-resolved RANS.  The first cell height Δy₁ is computed from:

    u_τ = sqrt(τ_w / ρ)  where  τ_w = C_f * 0.5 * ρ * U²
    C_f ≈ 0.026 * Re^(-1/7)  (turbulent flat-plate estimate, conservative)
    Δy₁ = y⁺_target * ν / u_τ

At Re = 30,000, U = 1 m/s, ν = 3.33e-5 m²/s:
    C_f ≈ 0.026 * 30000^(-1/7) ≈ 0.0055
    u_τ = sqrt(0.0055 * 0.5 * 1.225 * 1²) ≈ 0.058 m/s
    Δy₁ (y+=1) ≈ 3.33e-5 / 0.058 ≈ 5.7e-4 m  (chord = 1 m)

Wait — that is for turbulent BL.  For a laminar BL (Blasius):
    C_f ≈ 0.664 / sqrt(Re_x)  at x = 0.1c, Re_x = 3000
    C_f ≈ 0.664 / sqrt(3000) ≈ 0.012
    u_τ = sqrt(0.012 * 0.5 * 1.225) ≈ 0.086 m/s
    Δy₁ (y+=1) ≈ 3.33e-5 / 0.086 ≈ 3.9e-4 m

We use the MORE CONSERVATIVE (smaller) turbulent estimate to guarantee y+<1
everywhere, including near the leading edge stagnation region.

NON-DIMENSIONALISATION
======================
All geometry is normalised to chord c = 1.  The freestream velocity is set to
match the target Reynolds number:

    U_∞ = Re * ν / c  where ν = μ / ρ

With ρ = 1.225 kg/m³ and μ chosen to give the target Re at U_∞ = 1 m/s:

    μ = ρ * U_∞ * c / Re

This keeps U_∞ = 1 m/s regardless of Re, which:
  • Avoids Mach number effects (Ma << 0.3 always)
  • Keeps the pressure reference consistent
  • Allows direct comparison of Cl, Cd across Re values

Scaling errors (e.g. wrong chord length, wrong μ) corrupt Re by a factor and
produce completely wrong transition locations and drag levels.
"""

from __future__ import annotations

import math


# ── Physical constants ────────────────────────────────────────────────────────

RHO_AIR = 1.225          # kg/m³  — ISA sea level
MU_AIR  = 1.7894e-5      # Pa·s   — ISA sea level dynamic viscosity
NU_AIR  = MU_AIR / RHO_AIR  # m²/s  kinematic viscosity

# Freestream turbulence intensity for a low-turbulence wind tunnel / free air.
# CRITICAL: SST-LM is sensitive to Tu_inf.
#   Tu > 1%  → bypass transition dominates, LSB suppressed, Cd too high
#   Tu < 0.05% → transition delayed unrealistically
#   Tu = 0.1–0.5% → appropriate for free-flight / low-turbulence tunnel
FREESTREAM_TURBULENCE_INTENSITY = 0.001   # 0.1 %  (fraction, not percent)

# Turbulent viscosity ratio at inlet.  Low value preserves laminar BL.
# μ_t / μ = 1–10 is standard for external aerodynamics.
FREESTREAM_TURB_VISCOSITY_RATIO = 5.0

# Specific dissipation rate ω at inlet derived from Tu and μ_t/μ:
#   k = 1.5 * (Tu * U)²
#   ω = ρ * k / (μ_t)  →  ω = (3/2) * Tu² * U² * ρ / (μ_t/μ * μ)
# We expose the ratio; SU2 accepts FREESTREAM_TURBULENCEINTENSITY directly.


def compute_first_cell_height(reynolds: float, y_plus_target: float = 0.8) -> float:
    """
    Compute the first boundary-layer cell height Δy₁ to achieve y⁺ ≤ y_plus_target.

    At Re = 10,000–50,000 the boundary layer is predominantly laminar.
    We use the Blasius laminar skin-friction at x = 0.1c as the reference
    location — this is where the suction peak and highest wall shear occur
    on a typical cambered airfoil.

    C_f(x) = 0.664 / sqrt(Re_x)   (Blasius laminar flat plate)

    This gives a LARGER u_τ than the turbulent estimate at these Re values,
    producing a SMALLER Δy₁ — the conservative (safe) direction.

    Parameters
    ----------
    reynolds : float
        Chord Reynolds number Re = U·c/ν  (chord = 1 m, U = 1 m/s)
    y_plus_target : float
        Target dimensionless wall distance.  Use 0.8 to guarantee y⁺ < 1.

    Returns
    -------
    float
        First cell height in metres (= chord fractions since c = 1).
    """
    # Local Re at x = 0.1c (near suction peak — highest wall shear)
    re_x = reynolds * 0.1

    # Blasius laminar skin friction at x = 0.1c
    cf = 0.664 / math.sqrt(max(re_x, 1.0))

    u_inf = 1.0  # non-dimensional, U_∞ = 1 m/s

    # Wall shear stress  τ_w = C_f * 0.5 * ρ * U²
    tau_w = cf * 0.5 * RHO_AIR * u_inf ** 2

    # Friction velocity  u_τ = sqrt(τ_w / ρ)
    u_tau = math.sqrt(max(tau_w / RHO_AIR, 1e-12))

    # Kinematic viscosity  ν = U * c / Re  (U=1, c=1)
    nu = u_inf / reynolds

    # First cell height  Δy₁ = y⁺ * ν / u_τ
    return y_plus_target * nu / u_tau


def compute_boundary_layer_thickness(reynolds: float, x_fraction: float = 1.0) -> float:
    """
    Estimate 99% boundary layer thickness at chord position x/c.

    Uses Blasius laminar BL formula (appropriate for low-Re laminar BL):
        δ ≈ 5.0 * x / sqrt(Re_x)

    This sets the total inflation layer thickness so all BL cells are inside
    the physical boundary layer.
    """
    re_x = reynolds * x_fraction
    if re_x <= 0:
        return 0.01
    return 5.0 * x_fraction / math.sqrt(re_x)


def compute_inflation_layers(
    first_height: float,
    growth_rate: float,
    total_thickness: float,
    min_layers: int = 30,
    max_layers: int = 50,
) -> int:
    """
    Compute the number of inflation layers needed to span total_thickness
    starting from first_height with geometric growth_rate.

    n = log(1 + total_thickness * (r-1) / h₁) / log(r)
    """
    if growth_rate <= 1.0 or first_height <= 0:
        return min_layers
    n = math.log(1.0 + total_thickness * (growth_rate - 1.0) / first_height) / math.log(growth_rate)
    return max(min_layers, min(max_layers, int(math.ceil(n))))


def su2_freestream_turbulence_params(reynolds: float) -> dict[str, float]:
    """
    Return SU2 inlet turbulence parameters for the γ–Reθ transition model.

    The Langtry–Menter model requires:
      FREESTREAM_TURBULENCEINTENSITY  — Tu (fraction, not %)
      FREESTREAM_TURB_VISCOSITY_RATIO — μ_t / μ

    These control:
      • The onset of bypass transition (high Tu → early transition)
      • The initial condition for the k and ω transport equations

    For free-flight / low-turbulence tunnel:  Tu = 0.001 (0.1%), μ_t/μ = 1–5
    For wind tunnel with screens:             Tu = 0.003 (0.3%), μ_t/μ = 5–10
    """
    _ = reynolds  # reserved for future Re-dependent correlations
    return {
        "turbulence_intensity": FREESTREAM_TURBULENCE_INTENSITY,
        "turb_viscosity_ratio": FREESTREAM_TURB_VISCOSITY_RATIO,
    }


def velocity_from_reynolds(reynolds: float, chord: float = 1.0) -> float:
    """
    Return freestream velocity U_∞ such that Re = ρ U c / μ.

    With ρ = RHO_AIR, μ = MU_AIR, c = chord:
        U = Re * μ / (ρ * c)
    """
    return reynolds * MU_AIR / (RHO_AIR * chord)


def dynamic_viscosity_for_unit_velocity(reynolds: float, chord: float = 1.0) -> float:
    """
    Return μ such that Re = ρ * 1.0 * chord / μ.

    Keeping U = 1 m/s and varying μ is numerically cleaner than varying U,
    because the pressure and velocity scales stay fixed.
    """
    return RHO_AIR * 1.0 * chord / reynolds
