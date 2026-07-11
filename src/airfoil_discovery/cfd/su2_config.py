from __future__ import annotations

from pathlib import Path

import numpy as np

from airfoil_discovery.cfd.physics import RHO_AIR, dynamic_viscosity_for_unit_velocity
from airfoil_discovery.config import Settings
from airfoil_discovery.schemas import CandidateDesign


def build_stage_config(
    stage: int,
    candidate: CandidateDesign,
    mesh_path: Path,
    aoa: float,
    settings: Settings,
    restart_path: Path | None = None,
    *,
    time_discre_flow: str = "EULER_IMPLICIT",
    turbulence_intensity: float | None = None,
    turb_viscosity_ratio: float | None = None,
) -> str:
    solver = settings.solver
    if stage == 1:
        iter_count = solver.stage1_iter
        cfl = solver.stage1_cfl
        trans_model = "NONE"
        muscl = "NO"
        restart_sol = "NO"
        output_files = "(RESTART)"
    elif stage == 2:
        iter_count = solver.stage2_iter
        cfl = solver.stage2_cfl
        trans_model = "NONE"
        muscl = "YES"
        restart_sol = "YES"
        output_files = "(RESTART)"
    elif stage == 3:
        iter_count = solver.stage3_iter
        cfl = solver.stage3_cfl
        trans_model = "LM"
        muscl = "YES"
        restart_sol = "YES"
        output_files = "(RESTART, SURFACE_PARAVIEW)"
    else:
        raise ValueError(f"Unsupported stage: {stage}")

    aoa_rad = np.deg2rad(aoa)
    mu = dynamic_viscosity_for_unit_velocity(candidate.reynolds)
    mach_warning = ""
    if settings.flow.mach > 0.1:
        mach_warning = "% warning: freestream Mach number exceeds 0.1; incompressible assumptions may weaken"
    restart_rel = ""
    if restart_path is not None:
        restart_rel = (Path("..") / restart_path.parent.name / restart_path.name).as_posix()
    tu = settings.solver.stage3_turbulence_intensity if turbulence_intensity is None else turbulence_intensity
    tvr = settings.solver.stage3_turb_viscosity_ratio if turb_viscosity_ratio is None else turb_viscosity_ratio
    lines = [
        f"% stage: {stage}",
        f"% reynolds: {candidate.reynolds:.1f}",
        f"% mu: {mu:.8e}",
        f"% turbulence_intensity: {tu}",
        f"% turb_viscosity_ratio: {tvr}",
        mach_warning,
        "SOLVER= INC_RANS",
        "KIND_TURB_MODEL= SST",
        "MATH_PROBLEM= DIRECT",
        "VISCOSITY_MODEL= CONSTANT_VISCOSITY",
        f"MU_CONSTANT= {mu:.8e}",
        "INC_DENSITY_MODEL= CONSTANT",
        f"INC_DENSITY_INIT= {RHO_AIR}",
        f"INC_VELOCITY_INIT= ( {np.cos(aoa_rad):.8f}, {np.sin(aoa_rad):.8f}, 0.0 )",
        f"MACH_NUMBER= {settings.flow.mach}",
        f"AOA= {aoa}",
        f"REYNOLDS_NUMBER= {candidate.reynolds:.1f}",
        f"REF_LENGTH= {settings.flow.reference_length}",
        f"REF_AREA= {settings.flow.reference_area}",
        f"MESH_FILENAME= {mesh_path.name}",
        "MESH_FORMAT= SU2",
        "TABULAR_FORMAT= CSV",
        "MARKER_HEATFLUX= ( airfoil, 0.0 )",
        "MARKER_FAR= ( farfield )",
        "MARKER_MONITORING= ( airfoil )",
        "MARKER_PLOTTING= ( airfoil )",
        "NUM_METHOD_GRAD= WEIGHTED_LEAST_SQUARES",
        "NUM_METHOD_GRAD_RECON= LEAST_SQUARES",
        "CONV_NUM_METHOD_FLOW= FDS",
        f"TIME_DISCRE_FLOW= {time_discre_flow}",
        "TIME_DISCRE_TURB= EULER_IMPLICIT",
        f"KIND_TRANS_MODEL= {trans_model}",
        f"MUSCL_FLOW= {muscl}",
        f"MUSCL_TURB= {muscl}",
        "SLOPE_LIMITER_FLOW= VAN_ALBADA_EDGE" if muscl == "YES" else "SLOPE_LIMITER_FLOW= NONE",
        "SLOPE_LIMITER_TURB= VAN_ALBADA_EDGE" if muscl == "YES" else "SLOPE_LIMITER_TURB= NONE",
        f"ITER= {iter_count}",
        f"CFL_NUMBER= {cfl}",
        "CFL_ADAPT= YES",
        "CFL_ADAPT_PARAM= ( 0.5, 1.2, 0.5, 50.0 )",
        f"RESTART_SOL= {restart_sol}",
        f"OUTPUT_FILES= {output_files}",
        "CONV_FILENAME= history",
        "SCREEN_OUTPUT= (INNER_ITER, RMS_RES, AERO_COEFF)",
        "HISTORY_OUTPUT= (ITER, RMS_RES, AERO_COEFF)",
        "OUTPUT_WRT_FREQ= 100",
        "CONV_STARTITER= 100",
    ]
    if stage == 3:
        lines.extend(
            [
                f"FREESTREAM_TURBULENCEINTENSITY= {tu}",
                f"FREESTREAM_TURB2LAMVISCRATIO= {tvr}",
                f"% legacy_label: FREESTREAM_TURB_VISCOSITY_RATIO= {tvr}",
            ]
        )
    if restart_rel:
        lines.append(f"SOLUTION_FILENAME= {restart_rel}")
    return "\n".join(line for line in lines if line != "")


def write_stage_config(
    stage: int,
    candidate: CandidateDesign,
    mesh_path: Path,
    config_path: Path,
    aoa: float,
    settings: Settings,
    restart_path: Path | None = None,
    **kwargs: object,
) -> None:
    config_path.write_text(
        build_stage_config(stage, candidate, mesh_path, aoa, settings, restart_path, **kwargs),
        encoding="utf-8",
    )


def build_stage1_config(candidate: CandidateDesign, mesh_path: Path, aoa: float, settings: Settings) -> str:
    return build_stage_config(1, candidate, mesh_path, aoa, settings)


def build_stage2_config(
    candidate: CandidateDesign, mesh_path: Path, aoa: float, settings: Settings, restart_path: Path
) -> str:
    return build_stage_config(2, candidate, mesh_path, aoa, settings, restart_path)


def build_stage3_config(
    candidate: CandidateDesign, mesh_path: Path, aoa: float, settings: Settings, restart_path: Path
) -> str:
    return build_stage_config(3, candidate, mesh_path, aoa, settings, restart_path)
