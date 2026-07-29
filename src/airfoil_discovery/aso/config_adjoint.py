"""
SU2 Adjoint Configuration Generator (Discrete Adjoint).

Generates SU2 .cfg files for discrete adjoint mode (SU2_CFD_ADJ).
  - Objective: drag coefficient (CD) or lift-drag ratio
  - Design variables: surface coordinates (for CST projection later)
  - Uses converged primal flow solution as starting point
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def generate_adjoint_config(
    mesh_filename: str,
    primal_config_filename: str,
    objective: str = "DRAG",
    n_iter: int = 500,
    cfl_adjoint: float = 1.0,
    output_dir: str = ".",
    restart_filename: Optional[str] = None,
) -> str:
    """
    Generate the full text of an SU2 discrete adjoint configuration file.

    Parameters
    ----------
    mesh_filename : str
        Path to mesh file (same as primal).
    primal_config_filename : str
        Path to the primal configuration file (for parameter reference).
    objective : str
        Objective function: "DRAG" (Cd), "LIFT" (Cl), or "EFFICIENCY" (Cl/Cd).
    n_iter : int
        Number of adjoint iterations.
    cfl_adjoint : float
        CFL number for adjoint solver.
    output_dir : str
        Directory for output files.
    restart_filename : str, optional
        Path to adjoint restart file (for continuing adjoint).

    Returns
    -------
    config_text : str
        Complete SU2 adjoint configuration file text.
    """
    # Objective function mapping
    objective_map = {
        "DRAG": "DRAG",
        "LIFT": "LIFT",
        "EFFICIENCY": "EFFICIENCY",
        "SURFACE_SENSITIVITY": "SURFACE_SENSITIVITY",
    }
    obj_value = objective_map.get(objective.upper(), "DRAG")

    # Detect solver type, turbulence/transition settings, and adjoint freestream values from primal config.
    solver = "INC_RANS"
    turb_model = "SST"
    trans_model: Optional[str] = None
    conv_turb = "FDS"
    muscl_turb = "NO"
    slope_limiter_turb = "VENKATAKRISHNAN"
    adj_mach_number = 0.1
    adj_aoa = 0.0
    adj_sideslip_angle = 0.0
    adj_reynolds_number = 1e5
    adj_freestream_temperature = 288.15
    adj_freestream_pressure = 101325.0
    try:
        from pathlib import Path as _Path
        _p = _Path(primal_config_filename)
        if _p.exists():
            _text = _p.read_text(encoding="utf-8", errors="replace")
            for _line in _text.splitlines():
                _stripped = _line.strip()
                if _stripped.startswith("SOLVER") and "=" in _stripped and not _stripped.startswith("%"):
                    _val = _stripped.split("=", 1)[1].strip().split()[0]
                    solver = _val
                elif _stripped.startswith("KIND_TURB_MODEL") and "=" in _stripped and not _stripped.startswith("%"):
                    turb_model = _stripped.split("=", 1)[1].strip().split()[0]
                elif _stripped.startswith("KIND_TRANS_MODEL") and "=" in _stripped and not _stripped.startswith("%"):
                    trans_model = _stripped.split("=", 1)[1].strip().split()[0]
                elif _stripped.startswith("CONV_NUM_METHOD_TURB") and "=" in _stripped and not _stripped.startswith("%"):
                    _val = _stripped.split("=", 1)[1].strip().split()[0]
                    if _val != "JST":
                        conv_turb = _val
                elif _stripped.startswith("MUSCL_TURB") and "=" in _stripped and not _stripped.startswith("%"):
                    muscl_turb = _stripped.split("=", 1)[1].strip().split()[0]
                elif _stripped.startswith("SLOPE_LIMITER_TURB") and "=" in _stripped and not _stripped.startswith("%"):
                    slope_limiter_turb = _stripped.split("=", 1)[1].strip().split()[0]
                elif _stripped.startswith("MACH_NUMBER") and "=" in _stripped and not _stripped.startswith("%"):
                    adj_mach_number = float(_stripped.split("=", 1)[1].strip().split()[0])
                elif _stripped.startswith("AOA") and "=" in _stripped and not _stripped.startswith("%"):
                    adj_aoa = float(_stripped.split("=", 1)[1].strip().split()[0])
                elif _stripped.startswith("SIDESLIP_ANGLE") and "=" in _stripped and not _stripped.startswith("%"):
                    adj_sideslip_angle = float(_stripped.split("=", 1)[1].strip().split()[0])
                elif _stripped.startswith("REYNOLDS_NUMBER") and "=" in _stripped and not _stripped.startswith("%"):
                    adj_reynolds_number = float(_stripped.split("=", 1)[1].strip().split()[0])
                elif _stripped.startswith("FREESTREAM_TEMPERATURE") and "=" in _stripped and not _stripped.startswith("%"):
                    adj_freestream_temperature = float(_stripped.split("=", 1)[1].strip().split()[0])
                elif _stripped.startswith("FREESTREAM_PRESSURE") and "=" in _stripped and not _stripped.startswith("%"):
                    adj_freestream_pressure = float(_stripped.split("=", 1)[1].strip().split()[0])
    except Exception:
        pass  # Keep defaults

    lines = [
        f"% ------- SU2 Discrete Adjoint Configuration -------",
        f"% Generated by airfoil_discovery.aso.config_adjoint",
        f"% Objective: {objective}",
        f"",
        f"% ------------ Solver ------------",
        f"SOLVER= {solver}",
        f"MATH_PROBLEM= DISCRETE_ADJOINT",
        f"RESTART_SOL= {'YES' if restart_filename else 'NO'}",
        f"",
        f"% ------------ Objective Function ------------",
        f"OBJECTIVE_FUNCTION= {obj_value}",
        f"",
        f"% ------------ Mesh (same as primal) ------------",
        f"MESH_FILENAME= {mesh_filename}",
        "MESH_FORMAT= SU2",
        "",
        f"% ------------ Boundary Conditions (same as primal) ------------",
        "MARKER_HEATFLUX= ( airfoil, 0.0 )",
        "MARKER_FAR= ( farfield )",
        "MARKER_MONITORING= ( airfoil )",
        "MARKER_PLOTTING= ( airfoil )",
        "",
        f"% ------------ Turbulence Model ------------",
        f"KIND_TURB_MODEL= {turb_model}",
        f"KIND_TRANS_MODEL= {trans_model if trans_model is not None else 'NONE'}",
        "",
        f"% ------------ Freestream / Flow Reference ------------",
        f"MACH_NUMBER= {adj_mach_number}",
        f"AOA= {adj_aoa}",
        f"SIDESLIP_ANGLE= {adj_sideslip_angle}",
        f"REYNOLDS_NUMBER= {adj_reynolds_number}",
        f"FREESTREAM_TEMPERATURE= {adj_freestream_temperature}",
        f"FREESTREAM_PRESSURE= {adj_freestream_pressure}",
        "",
        f"% ------------ Adjoint Numerical Method ------------",
        "CONV_NUM_METHOD_FLOW= JST",
        f"CONV_NUM_METHOD_TURB= {conv_turb}",
        "NUM_METHOD_GRAD= WEIGHTED_LEAST_SQUARES",
        "NUM_METHOD_GRAD_RECON= LEAST_SQUARES",
        "",
        f"% ------------ MUSCL for Adjoint ------------",
        "MUSCL_FLOW= NO",
        "MUSCL_TURB= NO",
        "SLOPE_LIMITER_FLOW= VENKATAKRISHNAN",
        "SLOPE_LIMITER_TURB= VENKATAKRISHNAN",
        "",
        f"% ------------ Time Integration ------------",
        "TIME_DISCRE_FLOW= EULER_IMPLICIT",
        "TIME_DISCRE_TURB= EULER_IMPLICIT",
        "CFL_ADAPT= NO",
        "",
        f"% ------------ Iterations ------------",
        f"ITER= {n_iter}",
        "",
        f"% ------------ Linear Solver ------------",
        "LINEAR_SOLVER= FGMRES",
        "LINEAR_SOLVER_PREC= ILU",
        "LINEAR_SOLVER_ERROR= 1e-10",
        "LINEAR_SOLVER_ITER= 20",
        "",
        f"% ------------ Output ------------",
        "TABULAR_FORMAT= CSV",
        "CONV_FILENAME= history_adj",
        "RESTART_FILENAME= restart_adj",
        "VOLUME_FILENAME= adjoint",
        "SURFACE_FILENAME= surface_adjoint",
        f"OUTPUT_FILES= (RESTART, SURFACE_CSV)",
        f"OUTPUT_WRT_FREQ= 100",
        f"SCREEN_OUTPUT= (INNER_ITER, RMS_RES)",
        f"HISTORY_OUTPUT= (INNER_ITER, RMS_RES)",
        f"CONV_STARTITER= 10",
    ]

    if restart_filename:
        lines.append(f"SOLUTION_FILENAME= {restart_filename}")

    return "\n".join(lines)


def write_adjoint_config(
    output_path: Path,
    mesh_filename: str,
    primal_config_filename: str,
    objective: str = "DRAG",
    n_iter: int = 500,
    cfl_adjoint: float = 1.0,
    restart_filename: Optional[str] = None,
) -> Path:
    """
    Generate and write the SU2 discrete adjoint configuration file.

    Returns
    -------
    Path
        The path to the written config file.
    """
    text = generate_adjoint_config(
        mesh_filename=mesh_filename,
        primal_config_filename=primal_config_filename,
        objective=objective,
        n_iter=n_iter,
        cfl_adjoint=cfl_adjoint,
        restart_filename=restart_filename,
    )
    output_path.write_text(text, encoding="utf-8")
    return output_path