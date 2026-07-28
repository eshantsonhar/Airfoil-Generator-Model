"""
Phase 3: Validate mesh and run baseline primal CFD simulation
"""
import sys
from pathlib import Path
import numpy as np
import subprocess
import os

PROJECT_ROOT = Path(__file__).resolve().parent
while not (PROJECT_ROOT / "bin").exists() and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent

MESH_PATH = PROJECT_ROOT / "data" / "cache" / "final_test" / "airfoil_perfect.su2"
SU2_CFD = str(PROJECT_ROOT / "bin" / "SU2_CFD.exe")
CASE_DIR = PROJECT_ROOT / "data" / "cache" / "final_test"


def parse_su2_mesh(filepath):
    lines = filepath.read_text().splitlines()
    
    # Header: first line has npoin, nelem, nmarker
    h = lines[0].strip().split()
    if len(h) >= 3 and h[0].isdigit():
        npoin, nelem, nmarker = int(h[0]), int(h[1]), int(h[2])
    else:
        npoin = int(next(l for l in lines if "NPOIN" in l).split("=")[1].strip())
        nelem = int(next(l for l in lines if "NELEM" in l).split("=")[1].strip())
        nmarker = int(next(l for l in lines if "NMARK" in l).split("=")[1].strip())
        
    pts = []
    mi = {}
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("MARKER_TAG"):
            name = line.split("=")[1].strip() if "=" in line else lines[i+1].strip()
            if "=" in line:
                count_line = lines[i+1].strip()
                count = int(count_line.split("=")[1].strip()) if "=" in count_line else int(count_line)
                i += 2 + count
            else:
                count = int(lines[i+1].strip())
                i += 2 + count
            mi[name] = count
            continue
        elif "MARKER_TAG" in line:
            parts = line.split("=")
            name = parts[1].strip()
            count_line = lines[i+1].strip()
            count = int(count_line.split("=")[1].strip())
            mi[name] = count
            i += 2 + count
            continue
        else:
            parts = line.split()
            if len(parts) in (2, 3):
                try:
                    x, y = float(parts[0]), float(parts[1])
                    pts.append([x, y])
                except ValueError:
                    pass
        i += 1
        
    pts = np.array(pts, dtype=float)
    return npoin, nelem, nmarker, pts, mi


# ── Step 1: Validate Mesh ──────────────────────────────
print("=" * 80)
print("PHASE 3: MESH VALIDATION & BASELINE CFD")
print("=" * 80)

print("\n1. Validating mesh file...")
npoin, nelem, nmarker, pts, mi = parse_su2_mesh(MESH_PATH)
actual_npoin = len(pts)

x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
chord = x_max - x_min

print(f"\n{'=' * 60}")
print("MESH VALIDATION")
print(f"{'=' * 60}")
print(f"  Nodes: {npoin} (parsed {actual_npoin})")
print(f"  Elements: {nelem}")
print(f"  Markers: {nmarker}")
for name, count in mi.items():
    print(f"    '{name}': {count} elements")
print(f"  Chord: {chord:.6f} m")
print(f"  X: [{x_min:.2f}, {x_max:.2f}]")
print(f"  Y: [{y_min:.2f}, {y_max:.2f}]")
print(f"  File: {MESH_PATH.stat().st_size / 1024:.1f} KB")
print(f"{'=' * 60}")

assert npoin > 10000, f"Too few nodes: {npoin}"
assert nelem > 11000, f"Too few elements: {nelem}"
assert mi.get('airfoil', 0) > 300, f"Too few airfoil elements: {mi.get('airfoil', 0)}"
assert mi.get('farfield', 0) > 100, f"Too few farfield elements: {mi.get('farfield', 0)}"
print("\n  [OK] Mesh validation PASSED")

# ── Step 2: Generate Config ────────────────────────────────────
print("\n2. Generating SU2 primal config...")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from airfoil_discovery.aso.config_primal import write_primal_config

cfg_path = CASE_DIR / "config_primal.cfg"
write_primal_config(
    output_path=cfg_path,
    mesh_filename="airfoil_perfect.su2",
    aoa_deg=4.0,
    reynolds=1e5,
    mach=0.1,
    n_iter=3000,
    cfl_initial=0.5,
    cfl_final=5.0,
    transition_model=True,
    turbulence_intensity=0.001,
    turb_viscosity_ratio=5.0,
)
print(f"  Config written: {cfg_path}")

# ── Step 3: Run Baseline CFD ───────────────────────────────────
print("\n3. Running baseline primal CFD...")
print(f"  SU2_CFD: {SU2_CFD}")
print(f"  Config: {cfg_path}")
print(f"  Mesh: {MESH_PATH}")

creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

try:
    result = subprocess.run(
        [SU2_CFD, cfg_path.name],
        cwd=str(CASE_DIR),
        capture_output=True,
        text=True,
        timeout=3600,
        creationflags=creation_flags,
    )
except subprocess.TimeoutExpired:
    print("  ERROR: SU2_CFD timed out after 3600s")
    sys.exit(1)
except FileNotFoundError:
    print(f"  ERROR: SU2_CFD not found at {SU2_CFD}")
    sys.exit(1)

(CASE_DIR / "su2_stdout.log").write_text(result.stdout, encoding="utf-8", errors="ignore")
(CASE_DIR / "su2_stderr.log").write_text(result.stderr, encoding="utf-8", errors="ignore")

print(f"  SU2_CFD exit code: {result.returncode}")

# ── Step 4: Extract Results ────────────────────────────────────
print("\n4. Extracting results from history.csv...")
history_path = CASE_DIR / "history.csv"
summary_path = CASE_DIR / "phase3_summary.json"

if history_path.exists():
    hist_lines = history_path.read_text().splitlines()
    print(f"  History file: {len(hist_lines)} lines")

    header = [h.strip().strip('"') for h in hist_lines[0].split(",")]
    print(f"  Headers: {header}")

    # Extract last 50 iterations for stability check
    data_lines = [line.strip() for line in hist_lines[1:] if line.strip() and line.strip() != ',']
    last_50_lines = data_lines[-50:] if len(data_lines) >= 50 else data_lines
    
    last_data = None
    for line in reversed(data_lines):
        if line:
            last_data = line
            break

    if last_data:
        values = [v.strip() for v in last_data.split(",")]
        mapping = dict(zip(header, values))

        cl = float(mapping.get("CL", mapping.get("LIFT", "0")))
        cd = float(mapping.get("CD", mapping.get("DRAG", "0")))
        total_iter = int(mapping.get("INNER_ITER", mapping.get("TIME_ITER", "0")))

        # Extract CL values from last 50 iterations for stability check
        cl_values = []
        for line in last_50_lines:
            vals = [v.strip() for v in line.split(",")]
            row_map = dict(zip(header, vals))
            if "CL" in row_map:
                try:
                    cl_values.append(float(row_map["CL"]))
                except:
                    pass
        
        # Check stability (standard deviation of last 50 CL values)
        stability_check = True
        if len(cl_values) >= 10:
            cl_std = np.std(cl_values)
            cl_mean = np.mean(cl_values)
            cl_cov = abs(cl_std / cl_mean) if cl_mean != 0 else float('inf')
            stability_check = cl_cov < 0.01  # Less than 1% coefficient of variation
            print(f"\n  Stability check (last {len(cl_values)} iterations):")
            print(f"    CL mean: {cl_mean:.6f}")
            print(f"    CL std:  {cl_std:.6f}")
            print(f"    CL CoV:  {cl_cov:.4f} ({'PASS' if stability_check else 'FAIL'})")

        rms_cols = [k for k in mapping if "RMS" in k.upper() or "rms" in k.lower()]
        residuals = {k: float(mapping[k]) for k in rms_cols if k in mapping}

        print(f"\n{'=' * 60}")
        print("BASELINE CFD RESULTS")
        print(f"{'=' * 60}")
        print(f"  Total iterations: {total_iter}")
        print(f"  CL: {cl:.6f}")
        print(f"  CD: {cd:.6f}")
        print(f"  L/D: {cl/cd:.2f}" if cd > 0 else "  L/D: INF")
        print(f"  Residuals:")
        for k, v in residuals.items():
            print(f"    {k}: {v:.6e}")
        print(f"{'=' * 60}")

        # Validation checks
        cl_ok = 0.3 <= cl <= 0.7
        cd_ok = 0.008 <= cd <= 0.025
        iter_ok = total_iter >= 1500
        
        if cl_ok:
            print(f"  [OK] CL={cl:.4f} within expected range [0.3, 0.7]")
        else:
            print(f"  [FAIL] CL={cl:.4f} outside expected range [0.3, 0.7]")
        
        if cd_ok:
            print(f"  [OK] CD={cd:.4f} within expected range [0.008, 0.025]")
        else:
            print(f"  [FAIL] CD={cd:.4f} outside expected range [0.008, 0.025]")
        
        if iter_ok:
            print(f"  [OK] Total iterations={total_iter} >= 1500")
        else:
            print(f"  [WARN] Total iterations={total_iter} < 1500")
        
        if stability_check:
            print(f"  [OK] Solution stable (CL variation < 1%)")
        else:
            print(f"  [WARN] Solution may not be fully stable")

        if residuals:
            first_rms = list(residuals.values())[0]
            last_rms = list(residuals.values())[-1]
            print(f"\n  Starting residual magnitude: {first_rms:.2e}")
            print(f"  Ending residual magnitude: {last_rms:.2e}")
            if last_rms > 0 and first_rms > 0:
                drop = abs(np.log10(first_rms / last_rms))
                print(f"  Orders of magnitude drop: {drop:.1f}")

        # Overall status
        overall_status = "PASSED" if (cl_ok and cd_ok and iter_ok and stability_check) else "FAILED"
        
        # Export summary JSON
        import json
        summary = {
            "status": overall_status,
            "total_iterations": total_iter,
            "cl": cl,
            "cd": cd,
            "ld": cl/cd if cd > 0 else None,
            "cl_in_range": bool(cl_ok),
            "cd_in_range": bool(cd_ok),
            "iterations_sufficient": bool(iter_ok),
            "solution_stable": bool(stability_check),
            "residuals": {k: float(v) for k, v in residuals.items()}
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"\n  Summary exported: {summary_path}")
        print(f"  Overall status: {overall_status}")
    else:
        print("  No data found in history file")
        summary_path.write_text(json.dumps({"status": "FAILED", "reason": "No history data"}, indent=2))
else:
    print("  No history.csv found")
    summary_path.write_text(json.dumps({"status": "FAILED", "reason": "No history file"}, indent=2))

print(f"\n{'=' * 80}")
print("PHASE 3 COMPLETE")
print("=" * 80)