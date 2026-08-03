"""
Mesh Deformation via SU2_DEF.

Wraps SU2's built-in grid deformation tool (SU2_DEF) to smoothly warp
the computational mesh around updated CST airfoil coordinates.

The deformation uses the linear elasticity analogy (FEA) built into SU2,
which preserves mesh quality (orthogonality, y+ distribution) for
moderate shape changes typical in gradient-based optimization.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .cst import compute_airfoil_coordinates, compute_surface_coordinates, N_DESIGN_VARS
from .subprocess_utils import run_solver_safe

logger = logging.getLogger(__name__)

STRUCTURAL_INTERIOR_X_MIN = 0.15
STRUCTURAL_INTERIOR_X_MAX = 0.80


def validate_geometric_integrity(
    dv: np.ndarray,
    te_thickness: float = 0.003,
    min_thickness_fraction: float = 0.02,  # 2% chord minimum thickness — HARD FLOOR, NEVER DISABLE
    n_pts: int = 200,
) -> tuple[bool, str]:
    """
    Validate geometric integrity of a design vector before CFD execution.

    *** v8 HARDENED VERSION — ALL 7 CHECKS ACTIVE. NEVER DISABLE OR BYPASS. ***

    This pre-CFD gate prevents non-physical geometries from reaching SU2_DEF or
    SU2_CFD. If any check fails, the proposed design vector MUST be rejected
    without calling any solver.

    Parameters
    ----------
    dv : np.ndarray, shape (12,)
        Design variable vector (CST coefficients)
    te_thickness : float
        Trailing edge thickness (non-dimensional, chord=1)
    min_thickness_fraction : float
        Minimum t/c as fraction of chord. Default 0.02 = 2% chord.
        DO NOT PASS 0.0 TO DISABLE THIS CHECK.
    n_pts : int
        Number of points for surface discretization

    Returns
    -------
    is_valid : bool
        True if geometry passes all checks, False otherwise
    reason : str
        Human-readable explanation of failure if is_valid=False
    """
    upper, lower = compute_surface_coordinates(dv, n_pts=n_pts, te_thickness=te_thickness)
    chord = 1.0  # Normalized chord length

    # ── Check 1: Self-intersection (upper must be strictly above lower) ────────
    thickness = upper[:, 1] - lower[:, 1]
    min_thickness = float(np.min(thickness))
    if min_thickness <= 0:
        return False, (
            f"[Check 1] Self-intersection: minimum thickness = {min_thickness:.6e} <= 0"
        )

    # ── Check 2: Structural interior minimum thickness (2% chord floor) ─────
    x_coords = upper[:, 0]
    interior_mask = (x_coords >= STRUCTURAL_INTERIOR_X_MIN) & (x_coords <= STRUCTURAL_INTERIOR_X_MAX)
    if np.any(interior_mask):
        interior_thickness = thickness[interior_mask]
        min_interior_thickness = float(np.min(interior_thickness))
        min_required = min_thickness_fraction * chord
        if min_interior_thickness < min_required:
            return False, (
                f"[Check 2] Minimum thickness violation: structural interior t/c={min_interior_thickness:.6f} "
                f"< required {min_required:.6f} (min_fraction={min_thickness_fraction:.4f})"
            )

    # ── Check 3: Must have positive maximum thickness ─────────────────────────
    max_thickness = float(np.max(thickness))
    min_max_required = 0.001 * chord  # 0.1% chord floor on max thickness
    if max_thickness < min_max_required:
        return False, (
            f"[Check 3] Max thickness too small: max t/c={max_thickness:.6f} "
            f"< {min_max_required:.6f}"
        )

    # ── Check 4: Trailing-edge thickness ──────────────────────────────────────
    te_mask = x_coords >= 0.98
    if np.any(te_mask):
        te_region_thickness = thickness[te_mask]
        avg_te_thickness = float(np.mean(te_region_thickness))
        min_te_required = 0.001 * chord  # 0.1% chord TE floor
        if avg_te_thickness < min_te_required:
            return False, (
                f"[Check 4] Trailing-edge thickness violation: avg TE t/c={avg_te_thickness:.6f} "
                f"< {min_te_required:.6f}"
            )

    # ── Check 5: Surface curvature / high-frequency spikes ────────────────────
    upper_y = upper[:, 1]
    lower_y = lower[:, 1]
    upper_curvature = np.abs(np.diff(upper_y, n=2))
    lower_curvature = np.abs(np.diff(lower_y, n=2))
    max_curvature = max(
        float(np.max(upper_curvature)),
        float(np.max(lower_curvature)),
    )
    curvature_threshold = 0.05
    if max_curvature > curvature_threshold:
        return False, (
            f"[Check 5] Excessive surface curvature/spike: max={max_curvature:.6f} "
            f"> threshold={curvature_threshold:.6f}"
        )

    # ── Check 6: Leading-edge curvature ───────────────────────────────────────
    le_mask = x_coords <= 0.10
    if np.any(le_mask):
        # curvature arrays are 2 shorter than x_coords due to np.diff(n=2)
        le_curv_mask = le_mask[:-2]
        if np.any(le_curv_mask):
            le_upper_curv = upper_curvature[le_curv_mask]
            le_lower_curv = lower_curvature[le_curv_mask]
            max_le_curv = max(
                float(np.max(le_upper_curv)) if len(le_upper_curv) > 0 else 0.0,
                float(np.max(le_lower_curv)) if len(le_lower_curv) > 0 else 0.0,
            )
            le_threshold = 0.04  # Stricter for leading edge
            if max_le_curv > le_threshold:
                return False, (
                    f"[Check 6] Excessive leading-edge curvature: {max_le_curv:.6f} "
                    f"> {le_threshold:.6f}"
                )

    # ── Check 7: Monotonic x-coordinates (no loops) ───────────────────────────
    if not np.all(np.diff(x_coords) > 0):
        return False, "[Check 7] Non-monotonic x-coordinates: surface loop detected"

    return True, "Geometry validation passed (all 7 checks active)"


def generate_su2_def_config(
    mesh_input: str,
    mesh_output: str,
    marker: str = "airfoil",
    n_iter: int = 500,
    young_modulus: float = 1e6,
    poisson_ratio: float = 0.3,
    surface_positions_filename: str = "surface_positions.dat",
    new_airfoil_dat: Optional[str] = None,
) -> str:
    """
    Generate an SU2_DEF configuration file for mesh deformation.

    SU2_DEF uses a linear elasticity (FEA) analogy to propagate
    boundary displacements into the volume mesh.

    Parameters
    ----------
    mesh_input : str
        Path to input mesh file (SU2 format).
    mesh_output : str
        Path to output deformed mesh file.
    marker : str
        Boundary marker to apply displacement to.
    n_iter : int
        Number of FEA iterations for deformation.
    young_modulus, poisson_ratio : float
        Material properties for elasticity analogy.
    surface_positions_filename : str
        Filename of the target SU2 surface positions file.
    new_airfoil_dat : str, optional
        Backward-compatible alias for ``surface_positions_filename``.

    Returns
    -------
    config_text : str
    """
    if new_airfoil_dat is not None:
        surface_positions_filename = new_airfoil_dat

    lines = [
        f"% ------- SU2_DEF Mesh Deformation Config -------",
        f"% Generated by airfoil_discovery.aso.mesh_deform",
        f"",
        f"% ------------ Solver ------------",
        "SOLVER= ELASTICITY",  # SU2_DEF uses ELASTICITY solver for mesh deformation
        "MATH_PROBLEM= DIRECT",
        "",
        f"% ------------ Mesh ------------",
        f"MESH_FILENAME= {mesh_input}",
        f"MESH_OUT_FILENAME= {mesh_output}",
        "MESH_FORMAT= SU2",
        "",
        f"% ------------ Boundary Conditions ------------",
        f"MARKER_HEATFLUX= ( {marker}, 0.0 )",
        "MARKER_FAR= ( farfield )",
        "",
        f"% ------------ Deformation Parameters ------------",
        f"DEFORM_STIFFNESS_TYPE= INVERSE_VOLUME",
        f"DEFORM_LINEAR_SOLVER= FGMRES",
        f"DEFORM_LINEAR_SOLVER_PREC= ILU",
        f"DEFORM_LINEAR_SOLVER_ITER= 100",
        f"DEFORM_LINEAR_SOLVER_ERROR= 1e-10",
        f"DEFORM_NONLINEAR_ITER= {n_iter}",
        f"DEFORM_CONSOLE_OUTPUT= YES",
        "",
        f"% ------------ Design Variables (Boundary Displacement) ------------",
        f"DV_KIND= SURFACE_FILE",
        f"DV_MARKER= ( {marker} )",
        f"DV_PARAM= ( {surface_positions_filename} )",
        "",
        f"% ------------ Elasticity Parameters ------------",
        f"DEFORM_ELASTICITY_MODULUS= {young_modulus}",
        f"DEFORM_POISSONS_RATIO= {poisson_ratio}",
        "",
        f"% ------------ Output ------------",
        "TABULAR_FORMAT= CSV",
        "CONV_FILENAME= history_def",
        "OUTPUT_FILES= (RESTART)",
        "OUTPUT_WRT_FREQ= 100",
    ]
    return "\n".join(lines)


def _parse_su2_nodes(mesh_path: Path) -> np.ndarray:
    """Return mesh point coordinates as an array with columns x, y."""
    nodes: List[List[float]] = []
    reading_nodes = False
    with mesh_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_str = line.strip()
            if line_str.startswith("NPOIN="):
                reading_nodes = True
                continue
            if reading_nodes:
                parts = line_str.split()
                if len(parts) < 2 or line_str.startswith("NELEM="):
                    break
                try:
                    nodes.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    break
    return np.asarray(nodes, dtype=float)


def _parse_marker_node_ids(mesh_path: Path, marker: str) -> List[int]:
    """Extract unique node IDs belonging to a boundary marker."""
    node_ids: List[int] = []
    seen = set()
    with mesh_path.open("r", encoding="utf-8", errors="replace") as f:
        lines = list(f)

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("MARKER_TAG="):
            marker_name = line.split("=", 1)[1].strip()
            if marker_name == marker:
                if i + 1 >= len(lines) or not lines[i + 1].strip().startswith("MARKER_ELEMS="):
                    raise ValueError(f"Malformed marker block for {marker!r} in {mesh_path}")
                n_elems = int(lines[i + 1].split("=", 1)[1].strip())
                for elem_line in lines[i + 2:i + 2 + n_elems]:
                    parts = elem_line.split()
                    if len(parts) < 3:
                        continue
                    # SU2 line boundary elements use VTK type 3 followed by two node IDs.
                    for token in parts[1:3]:
                        node_id = int(token)
                        if node_id not in seen:
                            node_ids.append(node_id)
                            seen.add(node_id)
                return node_ids
        i += 1

    markers = _parse_marker_names(mesh_path)
    raise ValueError(
        f"Marker {marker!r} not found in {mesh_path}. Available markers: {markers}"
    )


def _parse_marker_names(mesh_path: Path) -> List[str]:
    markers: List[str] = []
    with mesh_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_str = line.strip()
            if line_str.startswith("MARKER_TAG="):
                markers.append(line_str.split("=", 1)[1].strip())
    return markers


def _surface_y_lookup(dv: np.ndarray, te_thickness: float, n_pts: int = 1200) -> Dict[str, np.ndarray]:
    upper, lower = compute_surface_coordinates(dv, n_pts=n_pts, te_thickness=te_thickness)
    return {
        "x": upper[:, 0],
        "upper": upper[:, 1],
        "lower": lower[:, 1],
    }


def _surface_y_at(surface: Dict[str, np.ndarray], x: float, side: str) -> float:
    x_clipped = float(np.clip(x, 0.0, 1.0))
    return float(np.interp(x_clipped, surface["x"], surface[side]))


def write_surface_positions_file(
    mesh_path: Path,
    dv_old: np.ndarray,
    dv_new: np.ndarray,
    output_path: Path,
    marker: str = "airfoil",
    te_thickness: float = 0.003,
) -> float:
    """
    Write SU2's SURFACE_FILE target positions for the deformable marker.

    The file contains ``point_id x_new y_new`` rows. We keep x fixed and
    move y by the CST old-to-new delta for each boundary node.
    """
    nodes = _parse_su2_nodes(mesh_path)
    if nodes.size == 0:
        raise ValueError(f"No nodes parsed from mesh {mesh_path}")

    marker_node_ids = _parse_marker_node_ids(mesh_path, marker)
    old_surface = _surface_y_lookup(dv_old, te_thickness)
    new_surface = _surface_y_lookup(dv_new, te_thickness)

    lines = []
    max_target_displacement = 0.0
    for node_id in marker_node_ids:
        x_current, y_current = nodes[node_id]
        old_upper_y = _surface_y_at(old_surface, x_current, "upper")
        old_lower_y = _surface_y_at(old_surface, x_current, "lower")
        side = "upper" if abs(y_current - old_upper_y) <= abs(y_current - old_lower_y) else "lower"
        delta_y = _surface_y_at(new_surface, x_current, side) - _surface_y_at(old_surface, x_current, side)
        y_target = y_current + delta_y
        max_target_displacement = max(max_target_displacement, abs(delta_y))
        lines.append(f"{node_id}\t{x_current:.10f}\t{y_target:.10f}")

    if not lines:
        raise ValueError(f"No boundary nodes found for marker {marker!r} in {mesh_path}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return max_target_displacement


def write_airfoil_dat(
    dv: np.ndarray,
    output_path: Path,
    n_pts: int = 200,
    te_thickness: float = 0.003,
) -> None:
    """
    Write airfoil coordinates to a .dat file for SU2_DEF boundary displacement.

    The .dat file format is:
      airfoil_name
      x1  y1
      x2  y2
      ...

    Parameters
    ----------
    dv : np.ndarray, shape (12,)
        CST design variables.
    output_path : Path
        Path to write the .dat file.
    n_pts : int
        Number of points per surface.
    te_thickness : float
        Trailing edge thickness.
    """
    coords = compute_airfoil_coordinates(dv, n_pts_per_surface=n_pts, te_thickness=te_thickness)
    lines = ["airfoil"]
    for x, y in coords:
        lines.append(f"  {x:.10f}  {y:.10f}")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_su2_def(
    su2_def_bin: str,
    config_path: Path,
    work_dir: Path,
    timeout: float = 60.0,
) -> tuple[bool, int, str]:
    """
    Run SU2_DEF for mesh deformation with hardened execution safety.

    Parameters
    ----------
    su2_def_bin : str
        Path to SU2_DEF executable.
    config_path : Path
        Path to SU2_DEF configuration file.
    work_dir : Path
        Working directory (where mesh files are).
    timeout : float
        Timeout in seconds (default 60s).

    Returns
    -------
    success : bool
        True if SU2_DEF executed successfully
    return_code : int
        Process return code
    error_message : str
        Detailed error message if failed
    """
    cmd = [su2_def_bin, config_path.name]
    success, rc, stdout, stderr = run_solver_safe(
        cmd, work_dir, label="SU2_DEF", timeout=timeout,
    )

    # Save logs regardless of outcome
    (work_dir / "su2_def_stdout.log").write_text(stdout, encoding="utf-8", errors="ignore")
    (work_dir / "su2_def_stderr.log").write_text(stderr, encoding="utf-8", errors="ignore")

    if not success:
        error_msg = f"SU2_DEF failed (rc={rc}): {stderr[:500] if stderr else '(no output)'}"
        logger.error(error_msg)
        return False, rc, error_msg

    return True, rc, ""


def deform_mesh(
    su2_def_bin: str,
    original_mesh_path: Path,
    dv_old: np.ndarray,
    dv_new: np.ndarray,
    work_dir: Path,
    marker: str = "airfoil",
    n_pts: int = 200,
    te_thickness: float = 0.003,
    n_iter_def: int = 500,
) -> Optional[Path]:
    """
    Deform the mesh from the old airfoil shape to the new one.

    The process:
    1. Write old and new airfoil .dat files.
    2. Generate SU2_DEF config.
    3. Run SU2_DEF to deform the mesh.
    4. Return path to deformed mesh.

    Parameters
    ----------
    su2_def_bin : str
        Path to SU2_DEF executable.
    original_mesh_path : Path
        Path to the original (undeformed) mesh file.
    dv_old : np.ndarray, shape (12,)
        CST design variables of the old (baseline) airfoil.
    dv_new : np.ndarray, shape (12,)
        CST design variables of the new (target) airfoil.
    work_dir : Path
        Working directory for intermediate files.
    marker : str
        Boundary marker for displacement.
    n_pts : int
        Number of points for airfoil .dat files.
    te_thickness : float
        Trailing edge thickness.
    n_iter_def : int
        Number of SU2_DEF iterations.

    Returns
    -------
    deformed_mesh_path : Path or None
        Path to the deformed mesh if successful, None otherwise.
    """
    if dv_old.shape != dv_new.shape or dv_old.shape[0] != 12:
        logger.error(f"Invalid design vector shapes: old={dv_old.shape}, new={dv_new.shape}")
        return None
    if np.any(np.isnan(dv_old)) or np.any(np.isnan(dv_new)):
        logger.error("Design vectors contain NaN values")
        return None

    if not original_mesh_path.exists():
        logger.error(f"Original mesh not found: {original_mesh_path}")
        return None

    work_dir.mkdir(parents=True, exist_ok=True)

    # Write old/new airfoil coordinates for diagnostics, plus SU2's actual
    # surface-position file consumed by DV_KIND= SURFACE_FILE.
    old_dat = work_dir / "airfoil_old.dat"
    new_dat = work_dir / "airfoil_new.dat"
    write_airfoil_dat(dv_old, old_dat, n_pts=n_pts, te_thickness=te_thickness)
    write_airfoil_dat(dv_new, new_dat, n_pts=n_pts, te_thickness=te_thickness)

    # Copy original mesh to working directory
    mesh_input = work_dir / "mesh_original.su2"
    if original_mesh_path != mesh_input:
        shutil.copy2(original_mesh_path, mesh_input)

    # Output mesh path
    mesh_output = work_dir / "mesh_deformed.su2"
    surface_positions = work_dir / "surface_positions.dat"
    try:
        target_displacement = write_surface_positions_file(
            mesh_path=mesh_input,
            dv_old=dv_old,
            dv_new=dv_new,
            output_path=surface_positions,
            marker=marker,
            te_thickness=te_thickness,
        )
        logger.info(
            "Prepared SU2 surface positions for marker %s: max target displacement=%.6e",
            marker,
            target_displacement,
        )
    except Exception as e:
        logger.error(f"Could not prepare SU2 surface positions: {e}")
        return None

    # Generate SU2_DEF config
    def_config = work_dir / "config_deform.cfg"
    config_text = generate_su2_def_config(
        mesh_input=mesh_input.name,
        mesh_output=mesh_output.name,
        marker=marker,
        n_iter=n_iter_def,
        surface_positions_filename=surface_positions.name,
    )
    def_config.write_text(config_text, encoding="utf-8")

    # Run SU2_DEF with hardened failure detection
    success, rc, error_msg = run_su2_def(su2_def_bin, def_config, work_dir)
    
    # Additional hardened failure checks
    if not success:
        logger.error(f"Mesh deformation failed: {error_msg}")
        return None
    
    if rc != 0:
        logger.error(f"SU2_DEF returned non-zero exit code: {rc}")
        return None
    
    # Check if output file exists and validate mesh integrity
    if mesh_output.exists():
        try:
            # Parse mesh to detect corruption
            output_nodes = _parse_su2_nodes(mesh_output)
            input_nodes = _parse_su2_nodes(mesh_input)
            
            if len(output_nodes) == 0 or len(input_nodes) == 0:
                logger.error("Mesh parsing failed: empty node arrays detected")
                return None
            
            if len(output_nodes) != len(input_nodes):
                logger.warning(f"Mesh node count changed: {len(input_nodes)} -> {len(output_nodes)}")
            
            mesh_delta = float(np.max(np.abs(output_nodes - input_nodes)))
        except Exception as e:
            logger.error(f"Could not verify deformed mesh displacement or mesh is corrupted: {e}")
            return None
        
        if mesh_delta <= 1e-12:
            logger.error(
                "SU2_DEF produced an unchanged mesh for marker %s "
                "(max node displacement %.6e). Rejecting deformation.",
                marker,
                mesh_delta,
            )
            return None
        
        # Check for mesh corruption (NaN or Inf coordinates)
        if np.any(np.isnan(output_nodes)) or np.any(np.isinf(output_nodes)):
            logger.error("Deformed mesh contains NaN or Inf coordinates - mesh corruption detected")
            return None
        
        logger.info(f"Mesh deformed successfully: {mesh_output} (max node displacement={mesh_delta:.6e})")
        return mesh_output
    else:
        # Check for alternative output names that SU2_DEF might use
        possible_outputs = [
            mesh_output,
            work_dir / f"{mesh_input.stem}_deformed.su2",
            work_dir / f"{mesh_input.stem}_def.su2",
            work_dir / "mesh_out.su2"
        ]
        for alt_output in possible_outputs:
            if alt_output.exists():
                logger.info(f"Mesh deformed successfully (found as {alt_output.name}): {alt_output}")
                return alt_output
        
        # List files to help debug
        files_in_dir = list(work_dir.glob("*.su2"))
        logger.error(f"SU2_DEF succeeded but no output mesh found. Files in {work_dir}: {[f.name for f in files_in_dir]}")
        logger.error("Mesh deformation failed")
        return None


def compute_mesh_displacement(
    dv_old: np.ndarray,
    dv_new: np.ndarray,
    n_pts: int = 200,
    te_thickness: float = 0.003,
) -> np.ndarray:
    """
    Compute the maximum surface displacement between two airfoil shapes.

    Useful for checking if the shape change is small enough for
    linear elasticity deformation to be valid.

    Parameters
    ----------
    dv_old, dv_new : np.ndarray, shape (12,)
    n_pts : int
    te_thickness : float

    Returns
    -------
    max_displacement : float
        Maximum Euclidean distance between corresponding surface points.
    """
    coords_old = compute_airfoil_coordinates(dv_old, n_pts_per_surface=n_pts, te_thickness=te_thickness)
    coords_new = compute_airfoil_coordinates(dv_new, n_pts_per_surface=n_pts, te_thickness=te_thickness)

    displacements = np.sqrt(np.sum((coords_new - coords_old) ** 2, axis=1))
    return float(np.max(displacements))
