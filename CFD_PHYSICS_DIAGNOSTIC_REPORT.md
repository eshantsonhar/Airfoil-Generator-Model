# CFD Physics Diagnostic Report

## Executive Summary

The primal CFD solver has been stabilized and now converges cleanly with 11+ orders of residual reduction. However, the drag coefficient (CD) remains 3-4x higher than the target range due to fundamental physics limitations with the current mesh quality and turbulence model at low Reynolds number.

## STEP 1: Mesh Coordinate Scaling Analysis

**Status:** ✅ PASSED

- **Mesh File:** `data/mesh_fixed.su2`
- **Airfoil Surface:** x ∈ [0, 1], y ∈ [-0.05, 0.05] (chord = 1.0 meter)
- **Farfield:** x ∈ [-10, 232], y ∈ [-10, 234]
- **Reference Parameters:** REF_LENGTH = 1.0, REF_AREA = 1.0 ✅
- **Conclusion:** Mesh coordinates and reference parameters are correctly scaled.

## STEP 2: Solver Regime & Non-Dimensionalization Audit

**Status:** ✅ PASSED

- **Solver Mode:** INC_RANS (incompressible RANS)
- **Reynolds Number:** Re = 1e5
- **Viscosity:** μ = 1.225e-05 Pa·s (correct for Re=1e5 at U=1 m/s)
- **Density:** ρ = 1.225 kg/m³
- **Conclusion:** Non-dimensionalization parameters are consistent.

## STEP 3: Numerical Scheme Stabilization

**Status:** ✅ COMPLETED

Applied conservative numerical settings for stability:
- **Convective Scheme:** FDS (only option for incompressible)
- **Gradient Method:** GREEN_GAUSS
- **MUSCL:** Disabled (first-order for stability)
- **Slope Limiters:** NONE
- **CFL:** 0.5 → 5.0 (adaptive)
- **Linear Solver:** FGMRES with ILU preconditioning

## STEP 4: Baseline CFD Convergence Results

**Status:** ✅ CONVERGED CLEANLY

### Conservative Configuration (SST only, no transition)
```
Mesh: data/mesh_fixed.su2
Solver: INC_RANS + SST k-ω
Reynolds: 1e5
Mach: 0.1
AoA: 4.0°
CFL: 0.5 → 5.0 (adaptive)
MUSCL: Disabled
```

### Results
| Parameter | Value | Target | Status |
|-----------|-------|--------|--------|
| CL | 0.476 | 0.4 - 0.8 | ✅ PASS |
| CD | 0.115 | 0.01 - 0.035 | ❌ FAIL (3.3x high) |
| CMz | 0.120 | - | - |
| Residual Drop | 11.22 orders | ≥ 3.0 | ✅ PASS |
| Convergence | rms[P] = -14.05 | < -14 | ✅ PASS |

## Root Cause Analysis

### Why CD is 3-4x Too High

**Primary Cause:** SST turbulence model without transition modeling overpredicts drag at low Reynolds numbers.

**Physics Explanation:**
- At Re = 1e5, the boundary layer is predominantly laminar (60-80% laminar on suction surface)
- SST k-ω assumes fully turbulent flow from the leading edge
- This results in skin friction 3-10x higher than physical reality
- Without transition modeling, the laminar separation bubble (LSB) is eliminated
- The LSB is crucial for accurate drag prediction at low Re

### Why Transition Model Doesn't Help

**Secondary Cause:** Mesh quality insufficient for accurate transition modeling.

**Mesh Quality Metrics (from SU2 output):**
```
Orthogonality Angle: 29.4° - 89.3°
CV Face Area Aspect Ratio: 1.4 - 7659.4 ⚠️
CV Sub-Volume Ratio: 1 - 236980 ⚠️
```

The extreme aspect ratios (max 7659.4) prevent accurate resolution of:
- Boundary layer velocity gradients
- Intermittency transport (γ equation)
- Transition momentum thickness Reynolds number (Re_θ)

### Mesh Resolution Analysis

**Current Mesh (mesh_fixed.su2):**
- Points: 222
- Elements: 235
- Boundary layer resolution: Insufficient for y+ < 1

**Alternative Mesh (airfoil_perfect.su2):**
- Points: 2155
- Elements: Much higher resolution
- Result: CL = 0.104 (too low), CD = 0.156 (still high)
- Issue: Different geometry/parameterization

## Recommendations

### Short-Term (Current Setup)

**Acceptable Baseline Configuration:**
```python
write_primal_config(
    mesh_filename="mesh_fixed.su2",
    aoa_deg=4.0,
    reynolds=1e5,
    mach=0.1,
    n_iter=500,
    cfl_initial=0.5,
    cfl_final=5.0,
    muscl=False,  # First-order for stability
    slope_limiter_flow="NONE",
    slope_limiter_turb="NONE",
    transition_model=False,  # Mesh quality insufficient
    turbulence_intensity=0.05,
    turb_viscosity_ratio=10.0,
)
```

**Expected Performance:**
- CL ≈ 0.48 (reasonable)
- CD ≈ 0.115 (3.3x target, but stable)
- Residual convergence: 11+ orders
- Optimization: Will converge but with elevated drag baseline

### Long-Term (For Accurate Physics)

**Option 1: Higher Reynolds Number**
- Increase Re to 1e6-1e7 where SST is more appropriate
- SST assumes fully turbulent flow, which is valid at higher Re
- Trade-off: Different flow regime than target

**Option 2: Improved Mesh Quality**
- Generate mesh with boundary layer inflation
- Target: y+ < 1, aspect ratio < 100
- Required: 30-50 inflation layers with growth ratio 1.2
- Tool: Use gmsh with boundary layer meshing

**Option 3: Laminar Flow Simulation**
- Disable turbulence model entirely
- Use laminar Navier-Stokes
- Valid for Re < 5e4
- Trade-off: No turbulence effects at all

## Conclusion

The CFD solver has been successfully stabilized and converges cleanly. The configuration fixes in `config_primal.py` resolve the numerical instability issues. However, achieving the target CD range (0.01-0.035) is not possible with the current mesh quality and SST turbulence model at Re=1e5.

**Recommendation:** Proceed with optimization using the conservative SST-only configuration, acknowledging that absolute drag values will be elevated but relative drag reductions during optimization will still be meaningful.
