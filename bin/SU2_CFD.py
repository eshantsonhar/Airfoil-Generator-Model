#!/usr/bin/env python3
"""
Mock SU2_CFD solver — generates realistic CFD output files for pipeline testing.

This script is invoked by the optimizer in place of the real SU2_CFD binary.
It:
  1. Reads the config file to get flow conditions
  2. Generates a realistic history.csv with convergence behavior
  3. Generates surface_flow.csv with Cp distribution
  4. Generates surface sensitivity files for gradient extraction
  5. Runs for a configurable duration to simulate solver time
  6. Returns exit code 0

Usage: SU2_CFD config_primal.cfg
"""
import csv
import math
import os
import random
import sys
import time
from pathlib import Path

random.seed(42)

def parse_config(cfg_path: Path) -> dict:
    """Parse SU2 config file for key parameters."""
    params = {
        "aoa": 4.0,
        "reynolds": 1e5,
        "mach": 0.1,
        "n_iter": 200,
        "cfl": 3.0,
        "math_problem": "DIRECT",  # DIRECT, DISCRETE_ADJOINT, ELASTICITY
        "objective": "DRAG",
        "surface_filename": "surface_flow",
    }
    if not cfg_path.exists():
        return params

    text = cfg_path.read_text()
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("%") or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k == "AoA":
                params["aoa"] = float(v)
            elif k == "REYNOLDS_NUMBER":
                params["reynolds"] = float(v)
            elif k == "MACH_NUMBER":
                params["mach"] = float(v)
            elif k == "EXT_ITER":
                params["n_iter"] = int(v)
            elif k == "CFL_NUMBER":
                params["cfl"] = float(v)
            elif k == "MATH_PROBLEM":
                params["math_problem"] = v.upper()
            elif k == "OBJECTIVE_FUNCTION":
                params["objective"] = v.upper()
            elif k == "SURFACE_FILENAME":
                params["surface_filename"] = v.strip()
    return params


def generate_history(params: dict, output_dir: Path) -> None:
    """Generate a realistic SU2 history.csv with convergence."""
    n_iter = min(params["n_iter"], 200)
    aoa = params["aoa"]
    reynolds = params["reynolds"]

    # Realistic low-Re airfoil aerodynamics at alpha=4, Re=1e5
    # NACA 0012-like: CL ~ 0.4-0.5, CD ~ 0.02-0.03
    target_cl = 0.45 + 0.05 * (aoa / 4.0)
    target_cd = 0.022 + 0.003 * (aoa / 4.0)

    # Convergence: residual drops from ~1e-2 to ~1e-6
    res_initial = -2.0  # log10
    res_final = -6.0    # log10

    history_path = output_dir / "history.csv"
    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            '"Inner_Iter"', '"rms[P]"', '"rms[U]"', '"rms[V]"',
            '"rms[k]"', '"rms[w]"', '"CD"', '"CL"', '"CEff"', '"CMz"'
        ])

        for i in range(n_iter):
            # Exponential convergence
            frac = i / max(n_iter - 1, 1)
            res_log = res_initial + (res_final - res_initial) * (1 - math.exp(-5 * frac))
            # Add noise
            res_log += random.gauss(0, 0.05) * (1 - frac)

            # CL converges with some oscillation
            cl_conv = target_cl * (1 - 0.3 * math.exp(-3 * frac))
            cl_conv += random.gauss(0, 0.002) * (1 - frac)

            # CD converges
            cd_conv = target_cd * (1 + 0.5 * math.exp(-4 * frac))
            cd_conv += random.gauss(0, 0.0003) * (1 - frac)

            eff = cl_conv / cd_conv if cd_conv > 0 else 0
            cm = -0.05 + 0.01 * (aoa / 4.0) + random.gauss(0, 0.001) * (1 - frac)

            writer.writerow([
                f"{i:12d}",
                f"{res_log:20.9f}",
                f"{res_log - 0.5:20.9f}",
                f"{res_log - 0.3:20.9f}",
                f"{res_log - 1.0:20.9f}",
                f"{res_log - 0.8:20.9f}",
                f"{cd_conv:20.9f}",
                f"{cl_conv:20.9f}",
                f"{eff:20.9f}",
                f"{cm:20.9f}",
            ])

    print(f"  Generated history.csv: {n_iter} iterations, CL~{target_cl:.4f}, CD~{target_cd:.6f}")


def generate_surface_flow(params: dict, output_dir: Path) -> None:
    """Generate surface_flow.csv with Cp distribution."""
    aoa = params["aoa"]
    n_points = 200
    surface_path = output_dir / "surface_flow.csv"

    with open(surface_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['"x"', '"y"', '"Cp"', '"Cf"', '"Pressure"', '"Temperature"'])

        for i in range(n_points):
            frac = i / (n_points - 1)
            # Cosine-like spacing for airfoil
            theta = math.pi * frac
            x = 0.5 * (1 - math.cos(theta))
            # Upper surface (first half) and lower surface (second half)
            if i < n_points // 2:
                # Upper surface
                y = 0.06 * math.sin(theta) * (1 - x)  # simplified airfoil shape
                # Cp: suction peak near leading edge
                cp = -0.8 * (1 - 0.5 * math.exp(-10 * x)) - 0.2 * x
            else:
                # Lower surface
                y = -0.06 * math.sin(theta) * (1 - x)
                cp = 0.3 * (1 - 0.3 * math.exp(-5 * x)) - 0.1 * x

            # Add AoA effect
            cp += 0.02 * aoa * (1 - x)

            cf = 0.005 * (1 - 0.5 * math.exp(-3 * x))
            pressure = 101325 + 0.5 * 1.225 * 10**2 * cp
            temp = 288.15

            writer.writerow([f"{x:.10f}", f"{y:.10f}", f"{cp:.10f}",
                           f"{cf:.10f}", f"{pressure:.6f}", f"{temp:.6f}"])

    print(f"  Generated surface_flow.csv: {n_points} surface points")


def generate_sensitivity(params: dict, output_dir: Path) -> None:
    """Generate surface sensitivity file for adjoint gradient extraction."""
    n_points = 200
    # Use the configured surface filename from adjoint config
    base_name = params.get("surface_filename", "surface_adjoint")
    sens_path = output_dir / f"{base_name}.csv"

    with open(sens_path, "w", newline="") as f:
        writer = csv.writer(f)
        # SU2 adjoint format: x, y, z, dJ/dx, dJ/dy, dJ/dz
        writer.writerow(['"x"', '"y"', '"z"', '"dJ/dx"', '"dJ/dy"', '"dJ/dz"'])
        for i in range(n_points):
            frac = i / (n_points - 1)
            theta = math.pi * frac
            x = 0.5 * (1 - math.cos(theta))
            if i < n_points // 2:
                y = 0.06 * math.sin(theta) * (1 - x)
            else:
                y = -0.06 * math.sin(theta) * (1 - x)
            # Sensitivity: higher near leading edge
            sens_x = -0.01 * math.exp(-5 * x) * (1 + 0.2 * random.gauss(0, 1))
            sens_y = 0.005 * math.exp(-3 * x) * (1 + 0.2 * random.gauss(0, 1))
            writer.writerow([f"{x:.10f}", f"{y:.10f}", "0.0", f"{sens_x:.10e}", f"{sens_y:.10e}", "0.0"])

    print(f"  Generated {base_name}.csv: {n_points} points")


def generate_restart(params: dict, output_dir: Path) -> None:
    """Generate restart flow file."""
    restart_path = output_dir / "restart_flow.dat"
    n_cells = 5000
    with open(restart_path, "w") as f:
        f.write(f"VARIABLES=\"x\",\"y\"\n")
        f.write(f"ZONE T=Flow, N={n_cells}, E=0, F=POINT\n")
        for i in range(min(n_cells, 100)):
            f.write(f"{random.random():.10f} {random.random():.10f}\n")
    print(f"  Generated restart_flow.dat")


def generate_deformed_mesh(params: dict, output_dir: Path) -> None:
    """Generate deformed mesh file for mesh deformation."""
    mesh_path = output_dir / "mesh_deformed.su2"
    # Copy original mesh as deformed (in reality, SU2_DEF would deform it)
    original_mesh = output_dir / "mesh_original.su2"
    if original_mesh.exists():
        import shutil
        shutil.copy2(original_mesh, mesh_path)
    else:
        # Create a minimal mesh file
        with open(mesh_path, "w") as f:
            f.write("NDIME= 2\n")
            f.write("NELEM= 4\n")
            f.write("NPOIN= 5\n")
            f.write("NMARK= 2\n")
            f.write("MARKER_TAG= airfoil\n")
            f.write("MARKER_ELEMS= 2\n")
            f.write("MARKER_TAG= farfield\n")
            f.write("MARKER_ELEMS= 2\n")
    print(f"  Generated mesh_deformed.su2")


def main():
    if len(sys.argv) < 2:
        print("Usage: SU2_CFD config.cfg")
        sys.exit(1)

    cfg_path = Path(sys.argv[1])
    output_dir = cfg_path.parent

    print(f"Mock SU2_CFD: Reading config from {cfg_path}")
    params = parse_config(cfg_path)
    print(f"  AoA={params['aoa']} deg, Re={params['reynolds']:.1e}, "
          f"Ma={params['mach']}, iter={params['n_iter']}")
    print(f"  Mode: {params['math_problem']}, Objective: {params['objective']}")

    # Simulate solver runtime (0.1s per 100 iterations)
    sim_time = 0.1 * (params["n_iter"] / 100)
    print(f"  Simulating solver run ({sim_time:.2f}s)...")
    time.sleep(sim_time)

    # Generate output files based on mode
    if params["math_problem"] == "DISCRETE_ADJOINT":
        # Adjoint mode: generate history and surface sensitivities
        generate_history(params, output_dir)
        generate_sensitivity(params, output_dir)
        generate_restart(params, output_dir)
    elif params["math_problem"] == "ELASTICITY":
        # Mesh deformation mode
        generate_deformed_mesh(params, output_dir)
    else:
        # Direct/primal mode: full output
        generate_history(params, output_dir)
        generate_surface_flow(params, output_dir)
        generate_sensitivity(params, output_dir)
        generate_restart(params, output_dir)

    print("Mock SU2_CFD: Completed successfully")
    sys.exit(0)


if __name__ == "__main__":
    main()