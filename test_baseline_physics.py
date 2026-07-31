"""
Baseline CFD Physics Verification Script

Runs a single SU2_CFD simulation on the baseline mesh and reports:
1. Max wall distance y+ reported by SU2
2. Final converged CL, CD, and Cm
3. Total residual drop (log10 reduction)

STOP GATE: Do NOT proceed to optimization until:
- CL ≈ 0.4 - 0.8
- CD ≈ 0.01 - 0.035
- Residual drop ≥ 3 orders of magnitude
"""

import subprocess
import re
import sys
from pathlib import Path

def parse_history_csv(history_path: Path) -> dict:
    """Parse SU2 history.csv to extract force coefficients and residuals."""
    if not history_path.exists():
        return {}
    
    data = {"CL": [], "CD": [], "CMz": [], "RMS_RES": []}
    
    with open(history_path, 'r') as f:
        lines = f.readlines()
    
    # Find header line (starts with "Inner_Iter")
    header_idx = None
    for i, line in enumerate(lines):
        if "Inner_Iter" in line:
            header_idx = i
            break
    
    if header_idx is None:
        return data
    
    headers = [h.strip().strip('"') for h in lines[header_idx].strip().split(',')]
    
    # Find column indices
    cl_idx = None
    cd_idx = None
    cm_idx = None
    res_idx = None
    
    for i, h in enumerate(headers):
        if h == "CL":
            cl_idx = i
        elif h == "CD":
            cd_idx = i
        elif h == "CMz":
            cm_idx = i
        elif h == "rms[P]":
            res_idx = i
    
    # Parse data lines
    for line in lines[header_idx+1:]:
        if not line.strip():
            continue
        values = line.strip().split(',')
        try:
            if cl_idx is not None and cl_idx < len(values):
                data["CL"].append(float(values[cl_idx]))
            if cd_idx is not None and cd_idx < len(values):
                data["CD"].append(float(values[cd_idx]))
            if cm_idx is not None and cm_idx < len(values):
                data["CMz"].append(float(values[cm_idx]))
            if res_idx is not None and res_idx < len(values):
                data["RMS_RES"].append(float(values[res_idx]))
        except (ValueError, IndexError):
            continue
    
    return data

def parse_surface_csv(surface_path: Path) -> dict:
    """Parse surface_flow.csv to extract y+ information."""
    if not surface_path.exists():
        return {"max_yplus": None}
    
    max_yplus = 0.0
    
    with open(surface_path, 'r') as f:
        lines = f.readlines()
    
    # Find header line
    header_idx = None
    for i, line in enumerate(lines):
        if "y_Plus" in line or "y+" in line:
            header_idx = i
            break
    
    if header_idx is None:
        return {"max_yplus": None}
    
    headers = lines[header_idx].strip().split(',')
    
    # Find y+ column index
    yplus_idx = None
    for i, h in enumerate(headers):
        if "y_Plus" in h or "y+" in h:
            yplus_idx = i
            break
    
    if yplus_idx is None:
        return {"max_yplus": None}
    
    # Parse data lines
    for line in lines[header_idx+1:]:
        if not line.strip():
            continue
        values = line.strip().split(',')
        try:
            yplus = float(values[yplus_idx])
            max_yplus = max(max_yplus, yplus)
        except (ValueError, IndexError):
            continue
    
    return {"max_yplus": max_yplus}

def run_baseline_cfd():
    """Run SU2_CFD on baseline mesh with fixed configuration."""
    workspace = Path(r"c:\Eshant_Sonhar\airfoil research paper\airfoil generator model")
    mesh_file = workspace / "baseline_cfd_run" / "airfoil.su2"  # Use known working mesh
    output_dir = workspace / "baseline_physics_test"
    output_dir.mkdir(exist_ok=True)
    
    # Generate config using the fixed config_primal module
    sys.path.insert(0, str(workspace / "src"))
    from airfoil_discovery.aso.config_primal import write_primal_config
    
    config_path = output_dir / "config_primal.cfg"
    write_primal_config(
        output_path=config_path,
        mesh_filename="airfoil.su2",
        aoa_deg=4.0,
        reynolds=1e5,
        mach=0.15,  # Compressible for transition model support
        n_iter=1500,  # More iterations for transition convergence
        cfl_initial=1.0,  # Standard CFL for transition
        cfl_final=30.0,
        muscl=True,  # MUSCL required for transition
        slope_limiter_flow="VENKATAKRISHNAN_WANG",
        slope_limiter_turb="NONE",
        transition_model=True,  # Enable γ-Reθ transition model
        turbulence_intensity=0.001,  # Low Tu for transition
        turb_viscosity_ratio=5.0,
    )
    
    # Copy mesh to output directory
    import shutil
    shutil.copy(mesh_file, output_dir / "airfoil.su2")
    
    # Run SU2_CFD
    print("Running SU2_CFD baseline test...")
    print(f"Config: {config_path}")
    print(f"Output dir: {output_dir}")
    
    su2_exe = workspace / "bin" / "SU2_CFD.exe"
    result = subprocess.run(
        [str(su2_exe), str(config_path)],
        cwd=str(output_dir),
        capture_output=True,
        text=True,
        timeout=1800  # 30 minutes
    )
    
    print("\n=== SU2_CFD STDOUT ===")
    print(result.stdout)
    if result.stderr:
        print("\n=== SU2_CFD STDERR ===")
        print(result.stderr)
    
    # Parse results
    history_path = output_dir / "history.csv"
    surface_path = output_dir / "surface_flow.csv"
    
    history_data = parse_history_csv(history_path)
    surface_data = parse_surface_csv(surface_path)
    
    # Extract final values
    final_cl = history_data["CL"][-1] if history_data["CL"] else None
    final_cd = history_data["CD"][-1] if history_data["CD"] else None
    final_cm = history_data["CMz"][-1] if history_data["CMz"] else None
    
    # Calculate residual drop (residuals are already in log10 scale)
    if len(history_data["RMS_RES"]) >= 2:
        initial_res = history_data["RMS_RES"][0]
        final_res = history_data["RMS_RES"][-1]
        residual_drop = final_res - initial_res  # More negative = more drop
    else:
        residual_drop = 0
    
    # Print diagnostic report
    print("\n" + "="*70)
    print("BASELINE CFD PHYSICS VERIFICATION REPORT")
    print("="*70)
    print(f"\nMesh: {mesh_file}")
    print(f"Max wall distance y+: {surface_data['max_yplus']:.4f}" if surface_data['max_yplus'] else "Max wall distance y+: N/A")
    print(f"\nFinal Force Coefficients:")
    print(f"  CL = {final_cl:.6f}" if final_cl else "  CL = N/A")
    print(f"  CD = {final_cd:.6f}" if final_cd else "  CD = N/A")
    print(f"  CMz = {final_cm:.6f}" if final_cm else "  CMz = N/A")
    print(f"\nResidual Convergence:")
    print(f"  Initial RMS Res = {history_data['RMS_RES'][0]:.6e}" if history_data['RMS_RES'] else "  Initial RMS Res = N/A")
    print(f"  Final RMS Res = {history_data['RMS_RES'][-1]:.6e}" if history_data['RMS_RES'] else "  Final RMS Res = N/A")
    print(f"  Log10 reduction = {residual_drop:.2f}")
    
    print("\n" + "="*70)
    print("STOP GATE CHECK")
    print("="*70)
    
    passed = True
    
    # Check CL range
    if final_cl is None:
        print("❌ CL: N/A - Cannot verify")
        passed = False
    elif 0.4 <= final_cl <= 0.8:
        print(f"✓ CL: {final_cl:.4f} [PASS - in range 0.4-0.8]")
    else:
        print(f"❌ CL: {final_cl:.4f} [FAIL - out of range 0.4-0.8]")
        passed = False
    
    # Check CD range
    if final_cd is None:
        print("❌ CD: N/A - Cannot verify")
        passed = False
    elif 0.01 <= final_cd <= 0.035:
        print(f"✓ CD: {final_cd:.4f} [PASS - in range 0.01-0.035]")
    else:
        print(f"❌ CD: {final_cd:.4f} [FAIL - out of range 0.01-0.035]")
        passed = False
    
    # Check residual drop (negative value = drop in log10 scale)
    if residual_drop <= -3.0:
        print(f"✓ Residual drop: {abs(residual_drop):.2f} orders [PASS - ≥ 3.0]")
    else:
        print(f"❌ Residual drop: {abs(residual_drop):.2f} orders [FAIL - < 3.0]")
        passed = False
    
    print("="*70)
    
    if passed:
        print("\n✓ ALL CHECKS PASSED - Proceed to optimization")
        return 0
    else:
        print("\n❌ CHECKS FAILED - Do NOT proceed to optimization")
        return 1

if __name__ == "__main__":
    import math
    sys.exit(run_baseline_cfd())
