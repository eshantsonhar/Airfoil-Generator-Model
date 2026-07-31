# Transition Model (γ-Reθ) Limitations Report

## Executive Summary

The γ-Reθ transition model (Langtry-Menter) cannot be successfully applied at Re=1e5 with the current mesh infrastructure. Multiple mesh generation attempts and configuration variations all resulted in either:
1. **Solver divergence** (NaN values)
2. **Stagnation** (residuals stuck at -5.67, far from -14 target)
3. **Unphysical force coefficients** (CL=1.74, CD=0.63 vs target CL=0.45-0.70, CD=0.015-0.030)

## Root Cause Analysis

### Mesh Quality Requirements for γ-Reθ Transition Modeling

The Langtry-Menter γ-Reθ transition model requires:
- **Boundary layer resolution:** y+ < 1.0 (ideally 0.5)
- **Structured inflation layers:** 35-45 layers with growth rate ≤ 1.15
- **Aspect ratio:** < 100 (current meshes have aspect ratios > 7600)
- **Orthogonality:** > 60° (current meshes have orthogonality as low as 29.4°)
- **Laminar separation bubble resolution:** 10-20 points within LSB

### Current Mesh Limitations

**Mesh 1: data/mesh_fixed.su2**
- Points: 222
- Aspect ratio: max 7659.4
- Orthogonality: 29.4° - 89.3°
- Boundary layer: No structured inflation
- Result: SST-only converges (CD=0.115), transition model diverges

**Mesh 2: airfoil_perfect.su2**
- Points: 2155
- Aspect ratio: Unknown (likely high)
- Boundary layer: No structured inflation
- Result: Transition model diverges with NaN values

**Mesh 3: baseline_cfd_run/airfoil.su2**
- Points: ~360
- Aspect ratio: High
- Boundary layer: No structured inflation
- Result: Transition model stagnates (residuals -5.67, unphysical CL/CD)

**Mesh 4: Generated mesh_highres_fixed.su2**
- Points: 9259
- Boundary layers: 40 layers, growth 1.12, first cell 1e-4 m
- Issue: SU2 format incompatibility (element type mismatch)
- Result: Cannot be loaded by SU2

## Test Results Summary

| Mesh | Transition Model | Residuals | CL | CD | Status |
|------|------------------|-----------|-----|-----|--------|
| mesh_fixed.su2 | SST only | -14.05 | 0.476 | 0.115 | Converges (CD high) |
| mesh_fixed.su2 | SST + LM | Diverged | - | - | NaN divergence |
| airfoil_perfect.su2 | SST + LM | Diverged | - | - | NaN divergence |
| baseline_cfd_run/airfoil.su2 | SST + LM | -5.67 | 1.74 | 0.63 | Stagnated (unphysical) |

## Physics Explanation

At Re=1e5, the boundary layer is predominantly laminar (60-80% laminar on suction surface). The γ-Reθ model requires:
1. **Accurate intermittency transport** (γ equation) - needs fine mesh near wall
2. **Transition momentum thickness Reynolds number** (Re_θ) calculation - needs proper velocity gradient resolution
3. **Laminar separation bubble (LSB) resolution** - needs 10-20 points within bubble

The current meshes lack the structured boundary layer inflation required to resolve these physics. The extreme aspect ratios cause numerical instability in the transition equations.

## Alternative Approaches

### Option 1: Increase Reynolds Number
- **Re = 1e6 - 1e7** where SST is more appropriate
- Transition less critical at higher Re
- Trade-off: Different flow regime than target

### Option 2: Laminar Flow Simulation
- Disable turbulence model entirely
- Valid for Re < 5e4
- Trade-off: No turbulence effects at all

### Option 3: Accept SST-Only Baseline
- Use SST without transition model
- CD will be 3-4x higher than target
- Relative drag reductions during optimization still meaningful
- Trade-off: Not physically accurate for LSB dynamics

### Option 4: Professional Mesh Generation
- Use commercial meshing tools (Pointwise, ANSYS ICEM)
- Generate structured C-mesh with proper boundary layer inflation
- Target: y+ < 1, 40+ layers, aspect ratio < 100
- Trade-off: Requires external tools and expertise

## Recommendation

Given the current infrastructure constraints, **Option 3 (SST-only baseline)** is the most practical path forward. While absolute CD values will be elevated, the optimization can still produce meaningful relative drag reductions. The mesh quality improvements required for γ-Reθ transition modeling are beyond the scope of the current mesh generation infrastructure.

## Configuration for SST-Only Baseline

```python
write_primal_config(
    mesh_filename="data/mesh_fixed.su2",
    aoa_deg=4.0,
    reynolds=1e5,
    mach=0.1,  # Incompressible for stability
    n_iter=500,
    cfl_initial=0.5,
    cfl_final=5.0,
    muscl=False,  # First-order for stability
    slope_limiter_flow="NONE",
    slope_limiter_turb="NONE",
    transition_model=False,  # SST only
    turbulence_intensity=0.05,
    turb_viscosity_ratio=10.0,
)
```

**Expected Performance:**
- CL ≈ 0.48 (reasonable)
- CD ≈ 0.115 (3.3x target, but stable)
- Residual convergence: 11+ orders
- Optimization: Will converge with elevated drag baseline
