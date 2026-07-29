#!/usr/bin/env python3
"""
Production Replica CFD Run — Full End-to-End Physics Extraction & Report Generation.

This script:
  1. Scans all phase5_output CFD evaluation directories
  2. Extracts convergence histories, aerodynamic coefficients, surface flows
  3. Computes physics audit (residual convergence, L/D, CL, CD)
  4. Generates production plots and markdown report
  5. Saves all artifacts to docs/reports/prod_run_latest/
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE5_DIR = PROJECT_ROOT / "phase5_output"
ASO_RESULTS_DIR = PROJECT_ROOT / "aso_results"
REPORT_DIR = PROJECT_ROOT / "docs" / "reports" / "prod_run_latest"
MESH_BASELINE = PHASE5_DIR / "mesh_baseline.su2"

N_DESIGN_VARS = 12
CST_ORDER = 6

# Colours
def ok(msg): return f"[OK]    {msg}"
def err(msg): return f"[ERROR] {msg}"
def info(msg): return f"[INFO]  {msg}"
def warn(msg): return f"[WARN]  {msg}"


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_history_csv(path: Path) -> list[dict]:
    """Parse SU2 history.csv into a list of dicts. Clean quoted/spaced column names."""
    if not path.exists():
        return []
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        # Clean column names: strip whitespace and quotes
        if hasattr(reader, 'fieldnames') and reader.fieldnames:
            reader.fieldnames = [fn.strip().strip('"') for fn in reader.fieldnames]
        for row in reader:
            # Also clean dict keys
            cleaned = {}
            for k, v in row.items():
                ck = k.strip().strip('"')
                cleaned[ck] = v
            rows.append(cleaned)
    return rows


def parse_surface_flow_csv(path: Path) -> list[dict]:
    """Parse SU2 surface_flow.csv. Clean quoted/spaced column names."""
    if not path.exists():
        return []
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        if hasattr(reader, 'fieldnames') and reader.fieldnames:
            reader.fieldnames = [fn.strip().strip('"') for fn in reader.fieldnames]
        for row in reader:
            cleaned = {}
            for k, v in row.items():
                ck = k.strip().strip('"')
                cleaned[ck] = v
            rows.append(cleaned)
    return rows


def extract_aerodynamics(rows: list[dict]) -> dict:
    """Extract final aerodynamic coefficients from history."""
    if not rows:
        return {"cl": None, "cd": None, "cm": None, "ld": None, "residual": None}

    last = rows[-1]
    cl = float(last.get("CL", last.get("CLIFT", 0)))
    cd = float(last.get("CD", last.get("CDRAG", 0)))
    cm = float(last.get("CM", last.get("CMOMENT", 0)))

    # Residual — look for rms[P] (pressure RMS residual)
    res_keys = ["rms[P]", "RMS_DENSITY", "RMS_RES", "RMS_PRESSURE", "RES", "ADJ_RES"]
    residual = None
    for k in res_keys:
        if k in last and str(last[k]).strip():
            try:
                residual = float(last[k])
            except ValueError:
                pass
            break

    # Also get initial residual for convergence check
    first = rows[0]
    for k in res_keys:
        if k in first and str(first[k]).strip():
            try:
                res_initial = float(first[k])
                res_final = residual or float(last.get(k, 0))
                residual = res_final
            except ValueError:
                pass
            break

    # SU2 outputs residuals in log10 format. Convert to linear.
    if residual is not None:
        residual_linear = 10.0 ** residual
    else:
        residual_linear = None

    ld = cl / cd if cd and cd > 0 else None

    return {"cl": cl, "cd": cd, "cm": cm, "ld": ld, "residual": residual_linear, "n_iter": len(rows)}


def parse_airfoil_mesh(path: Path) -> dict:
    """Parse SU2 mesh file to extract airfoil surface coordinates."""
    if not path.exists():
        return {"x": [], "y": [], "n_points": 0}

    text = path.read_text()
    x_upper, y_upper = [], []
    x_lower, y_lower = [], []
    in_marker = False
    marker_name = ""

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("MARKER_TAG="):
            marker_name = line.split("=")[1].strip()
            in_marker = True
        elif line.startswith("MARKER_ELEMS="):
            in_marker = True
        elif line == "NPOIN" or line.startswith("NELEM"):
            in_marker = False
        elif in_marker and line and not line.startswith(("MARKER_", "NMARK")):
            parts = line.split()
            if len(parts) == 3:
                try:
                    x, y = float(parts[0]), float(parts[1])
                    if "airfoil" in marker_name.lower() or "wing" in marker_name.lower():
                        x_upper.append(x)
                        y_upper.append(y)
                    elif "upper" in marker_name.lower():
                        x_upper.append(x)
                        y_upper.append(y)
                    elif "lower" in marker_name.lower():
                        x_lower.append(x)
                        y_lower.append(y)
                except ValueError:
                    pass

    # If no explicit markers, try surface markers
    if not x_upper and not x_lower:
        markers_started = False
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("MARKER_TAG="):
                markers_started = True
                continue
            if markers_started and line and not line.startswith(("MARKER_", "NMARK", "NELEM", "NPOIN")):
                parts = line.split()
                if len(parts) == 3:
                    try:
                        x, y = float(parts[0]), float(parts[1])
                        if y >= 0:
                            x_upper.append(x)
                            y_upper.append(y)
                        else:
                            x_lower.append(x)
                            y_lower.append(y)
                    except ValueError:
                        pass

    return {"x_upper": x_upper, "y_upper": y_upper,
            "x_lower": x_lower, "y_lower": y_lower}


# ── Phase 1: Pipeline Configuration & Sanity Check ──────────────────────────

def scan_evaluation_dirs():
    """Scan for all CFD evaluation directories and extract metadata."""
    eval_dirs = []
    opt_dir = PHASE5_DIR / "optimization"
    if opt_dir.exists():
        for d in sorted(opt_dir.iterdir()):
            if d.is_dir() and d.name.startswith("eval_"):
                eval_dirs.append(d)

    # Also check baseline
    baseline = PHASE5_DIR / "baseline_verification"
    if baseline.exists():
        eval_dirs.insert(0, baseline)  # put first

    return eval_dirs


def extract_baseline(eval_dirs):
    """Extract baseline data."""
    for d in eval_dirs:
        if "baseline" in str(d):
            hist = parse_history_csv(d / "history.csv")
            surface = parse_surface_flow_csv(d / "surface_flow.csv")
            return d, hist, surface
    return None, [], []


def sanity_check():
    """Run pre-checks for all prerequisites."""
    print("─" * 70)
    print("  PRODUCTION REPLICA RUN — SANITY CHECK")
    print("─" * 70)

    checks = []

    # Check phase5_output
    has_phase5 = PHASE5_DIR.exists()
    checks.append(("phase5_output/ directory", has_phase5))

    # Check baseline mesh
    has_mesh = MESH_BASELINE.exists()
    checks.append(("baseline mesh (mesh_baseline.su2)", has_mesh))

    # Check evaluation directories
    opt_dir = PHASE5_DIR / "optimization"
    n_evals = len(list(opt_dir.glob("eval_*"))) if opt_dir.exists() else 0
    checks.append((f"optimization evaluation dirs ({n_evals} found)", n_evals > 0))

    # Check baseline verification
    has_baseline = (PHASE5_DIR / "baseline_verification").exists()
    checks.append(("baseline verification dir", has_baseline))

    # Check Python packages
    for pkg in ["numpy", "scipy", "matplotlib"]:
        try:
            __import__(pkg)
            checks.append((f"Python: {pkg}", True))
        except ImportError:
            checks.append((f"Python: {pkg}", False))

    # Print results
    all_pass = True
    for name, passed in checks:
        sym = "PASS" if passed else "FAIL"
        print(f"  [{sym:4s}]  {name}")
        if not passed:
            all_pass = False

    print(f"\n  Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    return all_pass


# ── Phase 2: Extract All Data ────────────────────────────────────────────────

def extract_all_data(eval_dirs):
    """Extract aerodynamics, convergence, and geometry from all evaluations."""
    results = []
    baseline_result = None

    for d in eval_dirs:
        name = d.name
        hist = parse_history_csv(d / "history.csv")
        surface = parse_surface_flow_csv(d / "surface_flow.csv")
        aero = extract_aerodynamics(hist)

        # Try to get mesh (airfoil geometry)
        mesh_data = {}
        for mesh_name in ["airfoil.su2", "mesh_original.su2"]:
            m = d / mesh_name
            if m.exists():
                mesh_data = parse_airfoil_mesh(m)
                break

        result = {
            "name": name,
            "path": str(d),
            "aero": aero,
            "history": hist,
            "surface": surface,
            "mesh": mesh_data,
        }

        if "baseline" in name.lower():
            baseline_result = result
        else:
            results.append(result)

    return baseline_result, results


# ── Phase 3: Physics Verification ────────────────────────────────────────────

def verify_physics(baseline, evaluations):
    """Conduct rigorous physics and numerical audit."""
    print("─" * 70)
    print("  PHYSICS VERIFICATION AUDIT")
    print("─" * 70)

    audit = {"mesh": {}, "flow": {}, "aerodynamic": {}}

    # Mesh/Geometry audit
    if baseline and baseline["mesh"]:
        mesh = baseline["mesh"]
        n_upper = len(mesh.get("x_upper", []))
        n_lower = len(mesh.get("x_lower", []))
        audit["mesh"]["n_upper_points"] = n_upper
        audit["mesh"]["n_lower_points"] = n_lower

        # Check trailing edge closure
        if n_upper > 0 and n_lower > 0:
            xup, yup = np.array(mesh["x_upper"]), np.array(mesh["y_upper"])
            xlo, ylo = np.array(mesh["x_lower"]), np.array(mesh["y_lower"])
            te_gap = abs(yup[-1] - ylo[-1]) if len(yup) > 0 and len(ylo) > 0 else None
            audit["mesh"]["te_gap"] = te_gap
            print(f"  Trailing edge gap: {te_gap:.6f}" if te_gap else "  Trailing edge: N/A")

            # Check for NaN
            has_nan = np.any(np.isnan(xup)) or np.any(np.isnan(yup))
            audit["mesh"]["has_nan"] = bool(has_nan)
            print(f"  NaN in surface coords: {has_nan}")

            # Surface curvature (2nd derivative estimate)
            if len(xup) > 5 and len(yup) > 5:
                try:
                    dx = np.diff(xup)
                    dy = np.diff(yup)
                    curv = np.abs(np.diff(dy / np.maximum(dx, 1e-10))) / np.maximum(dx[:-1], 1e-10)
                    max_curv = float(np.max(curv))
                    audit["mesh"]["max_curvature"] = max_curv
                    print(f"  Max surface curvature: {max_curv:.4f}")
                except Exception:
                    pass

    # Flow/Residual audit
    print("\n  Flow Convergence:")
    for aero_hist in [("Baseline", baseline)] + [("Eval", e) for e in evaluations[:5]]:
        label, result = aero_hist
        if result and result["history"]:
            res = result["aero"].get("residual")
            n_iter = result["aero"].get("n_iter", 0)
            cl = result["aero"].get("cl")
            cd = result["aero"].get("cd")
            ld = result["aero"].get("ld")

            if res is not None:
                status = "CONVERGED" if res < 1e-4 else "PARTIAL" if res < 1e-3 else "MARGINAL"
                print(f"  {label}: residual={res:.6e}, {status}, CL={cl:.4f}, "
                      f"CD={cd:.6f}, L/D={ld:.2f} ({n_iter} iter)")
                audit["flow"][f"{label}_residual"] = res
                audit["flow"][f"{label}_converged"] = res < 1e-4

    # Aerodynamic plausibility audit
    print("\n  Aerodynamic Plausibility (Re=1e5, Ma=0.1, alpha=4 deg):")
    if baseline and baseline["aero"]["cl"]:
        cl = baseline["aero"]["cl"]
        cd = baseline["aero"]["cd"]
        ld = baseline["aero"]["ld"]

        # Physical bounds at Re=1e5
        plausible = (0.3 < cl < 1.5) and (cd is not None and cd < 0.1) and (ld is not None and ld > 5)
        audit["aerodynamic"]["baseline_plausible"] = bool(plausible)
        print(f"  Baseline: CL={cl:.4f}, CD={cd:.6f}, L/D={ld:.2f} -> "
              f"{'PLAUSIBLE' if plausible else 'SUSPICIOUS'}")

    # Best evaluation
    best_ld, best_name = 0, "N/A"
    for e in evaluations:
        ld = e["aero"].get("ld")
        if ld and ld > best_ld:
            best_ld = ld
            best_name = e["name"]
    audit["aerodynamic"]["best_ld"] = best_ld
    audit["aerodynamic"]["best_eval"] = best_name
    print(f"  Best L/D from optimization: {best_ld:.2f} ({best_name})")

    return audit


# ── Phase 4: Report Generation ───────────────────────────────────────────────

def generate_plots(baseline, evaluations, audit):
    """Generate all production plots."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("─" * 70)
    print("  GENERATING PRODUCTION PLOTS")
    print("─" * 70)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Plot 1: Airfoil geometry overlay (baseline + best)
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    if baseline and baseline["mesh"]:
        m = baseline["mesh"]
        if m.get("x_upper") and m.get("y_upper"):
            ax1.plot(m["x_upper"], m["y_upper"], "b-", label="Baseline Upper", lw=2)
        if m.get("x_lower") and m.get("y_lower"):
            ax1.plot(m["x_lower"], m["y_lower"], "b--", label="Baseline Lower", lw=2)

    # Best eval geometry
    best_name = audit["aerodynamic"]["best_eval"]
    for e in evaluations:
        if e["name"] == best_name and e["mesh"]:
            m = e["mesh"]
            if m.get("x_upper") and m.get("y_upper"):
                ax1.plot(m["x_upper"], m["y_upper"], "r-", label="Optimized Upper", lw=2)
            if m.get("x_lower") and m.get("y_lower"):
                ax1.plot(m["x_lower"], m["y_lower"], "r--", label="Optimized Lower", lw=2)
            break

    ax1.set_xlabel("x/c")
    ax1.set_ylabel("y/c")
    ax1.set_title("Baseline vs. Optimized Airfoil Geometry")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")
    fig1.savefig(REPORT_DIR / "airfoil_geometry_overlay.png", dpi=150, bbox_inches="tight")
    print(ok(f"airfoil_geometry_overlay.png saved"))
    plt.close(fig1)

    # Plot 2: L/D evolution across optimization
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))

    evals_ld = []
    evals_cl = []
    evals_cd = []
    eval_labels = []

    for e in evaluations:
        ld = e["aero"].get("ld")
        cl = e["aero"].get("cl")
        cd = e["aero"].get("cd")
        if ld and cl and cd:
            try:
                idx = int(e["name"].split("_")[1]) if "eval_" in e["name"] else 0
            except (ValueError, IndexError):
                idx = len(evals_ld)
            evals_ld.append((idx, ld))
            evals_cl.append((idx, cl))
            evals_cd.append((idx, cd))

    if evals_ld:
        evals_ld.sort()
        evals_cl.sort()
        evals_cd.sort()
        iters = [e[0] for e in evals_ld]
        ld_vals = [e[1] for e in evals_ld]
        cl_vals = [e[1] for e in evals_cl]
        cd_vals = [e[1] for e in evals_cd]

        ax2a.plot(iters, ld_vals, "g-o", label="L/D")
        ax2a.set_xlabel("Evaluation #")
        ax2a.set_ylabel("L/D")
        ax2a.set_title("Lift-to-Drag Ratio Evolution")
        ax2a.grid(True, alpha=0.3)
        ax2a.legend()

        ax2b.plot(iters, cl_vals, "b-s", label="C_L")
        ax2b.plot(iters, cd_vals, "r-^", label="C_D")
        ax2b.set_xlabel("Evaluation #")
        ax2b.set_ylabel("Coefficient")
        ax2b.set_title("C_L / C_D Evolution")
        ax2b.grid(True, alpha=0.3)
        ax2b.legend()

    fig2.tight_layout()
    fig2.savefig(REPORT_DIR / "ld_evolution.png", dpi=150, bbox_inches="tight")
    print(ok("ld_evolution.png saved"))
    plt.close(fig2)

    # Plot 3: Residual convergence history (baseline)
    if baseline and baseline["history"]:
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        rows = baseline["history"]
        res_key = None
        for k in ["rms[P]", "RMS_DENSITY", "RMS_RES", "RMS_PRESSURE"]:
            if k in rows[0]:
                res_key = k
                break

        if res_key:
            residuals = []
            for r in rows:
                try:
                    residuals.append(float(r.get(res_key, 0)))
                except (ValueError, TypeError):
                    residuals.append(0)

            if residuals:
                ax3.semilogy(range(len(residuals)), residuals, "k-", lw=1)
                ax3.axhline(y=1e-4, color="r", ls="--", label="Target (1e-4)")
                ax3.set_xlabel("Iteration")
                ax3.set_ylabel("Residual")
                ax3.set_title(f"Flow Solver Convergence ({res_key})")
                ax3.grid(True, alpha=0.3)
                ax3.legend()
                fig3.savefig(REPORT_DIR / "residual_convergence.png", dpi=150, bbox_inches="tight")
                print(ok("residual_convergence.png saved"))
        plt.close(fig3)

    # Plot 4: Surface pressure coefficient
    if baseline and baseline["surface"]:
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        surf = baseline["surface"]
        x_vals, cp_vals = [], []
        for row in surf:
            x_vals.append(float(row.get("x", row.get("Position_x", 0))))
            cp_vals.append(float(row.get("Cp", row.get("Pressure_Coefficient", 0))))

        if x_vals and cp_vals:
            ax4.plot(x_vals, cp_vals, "b-", lw=2)
            ax4.invert_yaxis()
            ax4.set_xlabel("x/c")
            ax4.set_ylabel("C_p")
            ax4.set_title("Surface Pressure Coefficient (Baseline)")
            ax4.grid(True, alpha=0.3)
            fig4.savefig(REPORT_DIR / "pressure_distribution.png", dpi=150, bbox_inches="tight")
            print(ok("pressure_distribution.png saved"))
        plt.close(fig4)

    print(ok("All plots generated successfully"))
    return True


def generate_report(baseline, evaluations, audit, plot_dir):
    """Generate comprehensive production run markdown report."""
    print("─" * 70)
    print("  GENERATING PRODUCTION RUN REPORT")
    print("─" * 70)

    # Count evaluations
    n_total = len(evaluations)
    n_converged = sum(1 for e in evaluations if e["aero"].get("residual") is not None
                      and e["aero"]["residual"] < 1e-4)

    # Aerodynamic statistics
    ld_vals = [e["aero"]["ld"] for e in evaluations if e["aero"].get("ld")]
    cl_vals = [e["aero"]["cl"] for e in evaluations if e["aero"].get("cl")]
    cd_vals = [e["aero"]["cd"] for e in evaluations if e["aero"].get("cd")]

    baseline_ld = baseline["aero"]["ld"] if baseline and baseline["aero"].get("ld") else 0
    baseline_cl = baseline["aero"]["cl"] if baseline and baseline["aero"].get("cl") else 0
    baseline_cd = baseline["aero"]["cd"] if baseline and baseline["aero"].get("cd") else 0

    best_ld_idx = np.argmax(ld_vals) if ld_vals else -1
    best_ld = ld_vals[best_ld_idx] if best_ld_idx >= 0 else 0
    best_cl = cl_vals[best_ld_idx] if best_ld_idx >= 0 and best_ld_idx < len(cl_vals) else 0
    best_cd = cd_vals[best_ld_idx] if best_ld_idx >= 0 and best_ld_idx < len(cd_vals) else 0

    ld_gain = ((best_ld - baseline_ld) / baseline_ld * 100) if baseline_ld > 0 else 0

    report = f"""# Production Replica Run Report

## Executive Summary

- **Run Type:** Aerodynamic Shape Optimization (CFD-based)
- **Total Evaluations:** {n_total}
- **Converged CFD Cases:** {n_converged}/{n_total}
- **Baseline L/D:** {baseline_ld:.2f}
- **Best L/D:** {best_ld:.2f}
- **L/D Improvement:** {ld_gain:+.1f}%
- **Baseline C_L:** {baseline_cl:.4f} -> **Best C_L:** {best_cl:.4f}
- **Baseline C_D:** {baseline_cd:.6f} -> **Best C_D:** {best_cd:.6f}

## Telemetry & Resource Usage

| Metric | Value |
|--------|-------|
| Optimization Eval Count | {n_total} |
| Max Iterations per CFD | {max([e['aero'].get('n_iter', 0) for e in evaluations]) if evaluations else 0} |
| Converged Cases | {n_converged}/{n_total} |
| Best Evaluation | {audit['aerodynamic'].get('best_eval', 'N/A')} |

## Physics & Design Audit

### Mesh/Geometry Quality
| Metric | Value |
|--------|-------|
| Upper surface points | {audit['mesh'].get('n_upper_points', 'N/A')} |
| Lower surface points | {audit['mesh'].get('n_lower_points', 'N/A')} |
| Trailing edge gap | {audit['mesh'].get('te_gap', 'N/A')} |
| NaN in coordinates | {audit['mesh'].get('has_nan', 'N/A')} |
| Max curvature | {audit['mesh'].get('max_curvature', 'N/A')} |

### Flow Convergence
| Case | Residual | Converged? | C_L | C_D | L/D |
|------|----------|------------|-----|-----|-----|
"""

    if baseline:
        r = baseline["aero"]
        conv = "Yes" if r.get("residual") and r["residual"] < 1e-4 else "No"
        res_val = r.get('residual')
        res_str = f"{res_val:.2e}" if res_val is not None else "N/A"
        cl_val = r.get('cl') or 0.0
        cd_val = r.get('cd') or 0.0
        ld_val = r.get('ld') or 0.0
        report += f"| Baseline | {res_str} | {conv} | {cl_val:.4f} | {cd_val:.6f} | {ld_val:.2f} |\n"

    for e in evaluations[:10]:
        r = e["aero"]
        conv = "Yes" if r.get("residual") and r["residual"] < 1e-4 else "No"
        res_str = f"{r['residual']:.2e}" if r.get("residual") else "N/A"
        cl_val = r.get('cl') or 0.0
        cd_val = r.get('cd') or 0.0
        ld_val = r.get('ld') or 0.0
        report += f"| {e['name'][:25]} | {res_str} | {conv} | {cl_val:.4f} | {cd_val:.6f} | {ld_val:.2f} |\n"

    report += f"""
### Aerodynamic Plausibility
| Check | Value |
|-------|-------|
| Baseline CL in range [0.3, 1.5] | {'Yes' if audit['aerodynamic'].get('baseline_plausible') else 'No'} |
| Physical CD (<0.1) | {'Yes' if baseline and baseline['aero'].get('cd', 1) < 0.1 else 'No'} |
| L/D > 5 (low-Re reasonable) | {'Yes' if baseline_ld > 5 else 'No'} |

## Artifact Manifest

| Artifact | Path |
|----------|------|
| Airfoil geometry overlay | `airfoil_geometry_overlay.png` |
| L/D & C_L/C_D evolution | `ld_evolution.png` |
| Residual convergence | `residual_convergence.png` |
| Pressure distribution | `pressure_distribution.png` |
| Production report | `production_run_report.md` |
| CFD surface data | `phase5_output/` eval directories |
| Optimization results | `aso_results/optimization_summary.txt` |
"""

    # Write report
    report_path = REPORT_DIR / "production_run_report.md"
    report_path.write_text(report)
    print(ok(f"Report written to {report_path}"))

    # Also save artifact manifest
    manifest = {
        "report_dir": str(REPORT_DIR),
        "plots": [
            "airfoil_geometry_overlay.png",
            "ld_evolution.png",
            "residual_convergence.png",
            "pressure_distribution.png",
        ],
        "report": "production_run_report.md",
        "n_evaluations": n_total,
        "n_converged": n_converged,
        "baseline_ld": baseline_ld,
        "best_ld": best_ld,
        "ld_gain_pct": ld_gain,
    }
    manifest_path = REPORT_DIR / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(ok(f"Manifest written to {manifest_path}"))

    return report


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    start_time = time.time()

    print("=" * 70)
    print("  PRODUCTION REPLICA CFD OPTIMIZATION RUN")
    print("=" * 70)

    # Phase 1: Sanity check
    print()
    all_ok = sanity_check()
    if not all_ok:
        print(warn("Some checks failed, but proceeding with available data"))

    # Phase 2: Scan & extract
    print(f"\n{'─'*70}")
    print("  SCANNING EVALUATION DIRECTORIES & EXTRACTING DATA")
    print(f"{'─'*70}")

    eval_dirs = scan_evaluation_dirs()
    print(info(f"Found {len(eval_dirs)} evaluation directories"))

    baseline, evaluations = extract_all_data(eval_dirs)
    print(ok(f"Baseline: {baseline['name'] if baseline else 'NOT FOUND'}"))
    print(ok(f"Optimization evals: {len(evaluations)}"))

    # Phase 3: Physics verification
    print()
    audit = verify_physics(baseline, evaluations)

    # Phase 4: Generate plots & report
    print()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    generate_plots(baseline, evaluations, audit)

    print()
    report = generate_report(baseline, evaluations, audit, REPORT_DIR)

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print("  PRODUCTION REPLICA RUN SUMMARY")
    print(f"{'='*70}")
    print(f"  Total time:       {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Evaluations:      {len(evaluations)}")
    print(f"  Reports/plots:    {REPORT_DIR}")
    if baseline:
        print(f"  Baseline L/D:     {baseline['aero'].get('ld', 0):.2f}")
    else:
        print("  Baseline: N/A")
    best_ld_val = max([e["aero"]["ld"] for e in evaluations if e["aero"].get("ld")], default=0)
    print(f"  Best L/D:         {best_ld_val:.2f}")
    if baseline and baseline["aero"].get("ld", 0) > 0:
        print(f"  L/D Gain:         +{(best_ld_val/baseline['aero']['ld'] - 1)*100:.1f}%")

    print(f"\n{ok('Production replica run COMPLETE')}")
    print("=" * 70)

    return report


if __name__ == "__main__":
    sys.exit(0 if main() else 1)