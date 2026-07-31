# PDE-Constrained Aerodynamic Shape Optimization - Fixed Configuration
# This script runs the optimization pipeline with all identified fixes applied:
# 1. Fixed adjoint config: CFL_REDUCTION_ADJFLOW/TURB instead of invalid CFL_NUMBER_ADJFLOW/TURB
# 2. Reduced FD step size from 1e-3 to 1e-5 for gradient stability
# 3. Enabled MUSCL and proper slope limiters (VENKATAKRISHNAN_WANG/VENKATAKRISHNAN)
# 4. Added CFL adaptation parameters (CFL_ADAPT_PARAM, CFL_REDUCTION_TURB)
# 5. Added linear solver settings (FGMRES/ILU) and convergence criteria
# 6. Removed duplicate CONV_STARTITER config option

$ErrorActionPreference = "Stop"

# Set environment variables
$env:SU2_CFD_BIN = "bin\SU2_CFD.exe"
$env:SU2_DEF_BIN = "bin\SU2_DEF.exe"
$env:SU2_HOME = "bin"
$env:PATH = "bin;$env:PATH"
$env:PYTHONUNBUFFERED = "1"

# Configuration
$MESH_FILE = "baseline_cfd_run\airfoil.su2"
$OUTPUT_DIR = "aso_results_fixed"
$MAX_ITER = 5
$N_ITER_PRIMAL = 500
$N_ITER_ADJOINT = 100
$AOA = 4.0
$REYNOLDS = 1e5
$MACH = 0.1
$METHOD = "mma"
$TOL = 1e-4

Write-Host "=" * 70
Write-Host "  PDE-CONSTRAINED AERODYNAMIC SHAPE OPTIMIZATION"
Write-Host "  Configuration: All adjoint and primal config fixes applied"
Write-Host "=" * 70
Write-Host "  SU2_CFD: $env:SU2_CFD_BIN"
Write-Host "  SU2_DEF: $env:SU2_DEF_BIN"
Write-Host "  Mesh:    $MESH_FILE"
Write-Host "  Output:  $OUTPUT_DIR"
Write-Host "  Max Iter: $MAX_ITER"
Write-Host ""

# Clean previous results
if (Test-Path $OUTPUT_DIR) {
    Write-Host "Cleaning previous results..."
    Remove-Item -Recurse -Force $OUTPUT_DIR
}

# Run optimization
Write-Host "Starting optimization..."
$cmd = "python", "scripts\run_aso_pde_optimization.py",
       "--mesh", $MESH_FILE,
       "--output", $OUTPUT_DIR,
       "--max-iter", "$MAX_ITER",
       "--n-iter-primal", "$N_ITER_PRIMAL",
       "--n-iter-adjoint", "$N_ITER_ADJOINT",
       "--aoa", "$AOA",
       "--reynolds", "$REYNOLDS",
       "--mach", "$MACH",
       "--method", $METHOD,
       "--tol", "$TOL",
       "--no-preflight"

& $cmd
$exitCode = $LASTEXITCODE

Write-Host ""
Write-Host "=" * 70
Write-Host "  OPTIMIZATION COMPLETE"
Write-Host "=" * 70
Write-Host "  Exit code: $exitCode"
Write-Host "  Results: $OUTPUT_DIR"

# Check convergence history
$convergenceFile = "$OUTPUT_DIR\convergence_history.json"
if (Test-Path $convergenceFile) {
    Write-Host ""
    Write-Host "Convergence Summary:"
    $data = Get-Content $convergenceFile | ConvertFrom-Json
    Write-Host "  Total iterations: $($data.total_iterations)"
    Write-Host "  Converged: $($data.converged)"
    if ($data.iterations) {
        $first = $data.iterations[0]
        $last = $data.iterations[-1]
        Write-Host "  Initial Cd: $($first.cd), Final Cd: $($last.cd)"
        Write-Host "  Initial Cl: $($first.cl), Final Cl: $($last.cl)"
        Write-Host "  Initial |grad|: $($first.grad_norm), Final |grad|: $($last.grad_norm)"
    }
}

Write-Host ""
Write-Host "To view detailed logs, check: $OUTPUT_DIR\optimization.log"
Write-Host "=" * 70

exit $exitCode
