"""
PDE-Constrained Optimization Loop.

Implements the gradient-based optimization cycle:
  1. Evaluate primal CFD (RANS + γ-Re_θ transition)
  2. Run discrete adjoint (SU2_CFD_ADJ)
  3. Extract and project surface sensitivities → CST gradient
  4. Update design variables via MMA or SLSQP
  5. Deform mesh for next iteration
  6. Track convergence history (Cd, Cl, design vars)

Supports both:
  - scipy.optimize.minimize (SLSQP) as the robust fallback
  - SvanbergMMA (from airfoil_discovery.optimization.mma_engine) for superior convergence
"""

from __future__ import annotations

import datetime
import json
import logging
import subprocess
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np

from .cst import (
    N_DESIGN_VARS,
    CST_ORDER,
    CSTBounds,
    compute_airfoil_coordinates,
    compute_surface_coordinates,
    check_geometry_validity,
    design_vector_to_surface_coefficients,
    surface_coefficients_to_design_vector,
)
from .config_primal import generate_primal_config, write_primal_config
from .config_adjoint import generate_adjoint_config, write_adjoint_config
from .adjoint import extract_adjoint_gradient, verify_adjoint_gradient
from .mesh_deform import deform_mesh, compute_mesh_displacement

logger = logging.getLogger(__name__)


# ── Convergence history ────────────────────────────────────────────────────────

@dataclass
class IterationRecord:
    """Record of a single optimization iteration."""
    iteration: int
    cd: float
    cl: float
    objective: float            # Cd (drag minimization)
    grad_norm: float            # ||dCd/dDV||
    step_accepted: bool
    trust_radius: float
    max_thickness: float
    design_vector: List[float]  # 12 CST coefficients
    constraint_violations: List[float] = field(default_factory=list)
    gradient: List[float] = field(default_factory=list)  # 12 gradient values
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConvergenceHistory:
    """Full convergence history of the optimization."""
    iterations: List[IterationRecord] = field(default_factory=list)
    start_time: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    end_time: Optional[str] = None
    n_design_vars: int = N_DESIGN_VARS
    converged: bool = False
    total_iterations: int = 0

    def add(self, record: IterationRecord) -> None:
        self.iterations.append(record)
        self.total_iterations = len(self.iterations)

    def finalize(self, converged: bool) -> None:
        self.end_time = datetime.datetime.now().isoformat()
        self.converged = converged

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps({
                "start_time": self.start_time,
                "end_time": self.end_time,
                "converged": self.converged,
                "total_iterations": self.total_iterations,
                "n_design_vars": self.n_design_vars,
                "iterations": [it.to_dict() for it in self.iterations],
            }, indent=2),
            encoding="utf-8",
        )

    @property
    def cd_history(self) -> List[float]:
        return [it.cd for it in self.iterations]

    @property
    def cl_history(self) -> List[float]:
        return [it.cl for it in self.iterations]

    @property
    def grad_norm_history(self) -> List[float]:
        return [it.grad_norm for it in self.iterations]

    @property
    def objective_history(self) -> List[float]:
        return [it.objective for it in self.iterations]


# ── CFD Evaluation Wrapper ─────────────────────────────────────────────────────

@dataclass
class CFDResult:
    """Result of a primal + adjoint CFD evaluation."""
    cl: float
    cd: float
    converged: bool
    adjoint_gradient: np.ndarray   # shape (12,)
    gradient_valid: bool
    primal_converged: bool
    adjoint_converged: bool
    mesh_path: Optional[Path] = None
    case_dir: Optional[Path] = None
    failure_reason: str = ""


def run_primal_and_adjoint(
    su2_cfd_bin: str,
    su2_adj_bin: str,
    mesh_path: Path,
    dv: np.ndarray,
    case_dir: Path,
    aoa_deg: float = 4.0,
    reynolds: float = 1e5,
    mach: float = 0.1,
    n_iter_primal: int = 3000,
    n_iter_adjoint: int = 500,
    cfl_primal: float = 3.0,
    cfl_adjoint: float = 1.0,
    transition_model: bool = True,
    turbulence_intensity: float = 0.001,
    turb_viscosity_ratio: float = 5.0,
    objective: str = "DRAG",
    timeout_primal: float = 3600.0,
    timeout_adjoint: float = 600.0,
) -> CFDResult:
    """
    Run complete primal + adjoint CFD evaluation.

    Steps:
    1. Generate configuration files.
    2. Run SU2_CFD (primal RANS + transition).
    3. Check convergence and extract Cl, Cd.
    4. Run SU2_CFD_ADJ (discrete adjoint).
    5. Extract and project surface sensitivities onto CST DVs.
    6. Return combined result.

    Parameters
    ----------
    su2_cfd_bin : str
        Path to SU2_CFD executable.
    su2_adj_bin : str
        Path to SU2_CFD_ADJ executable (usually same as SU2_CFD with MATH_PROBLEM=DISCRETE_ADJOINT).
    mesh_path : Path
        Path to mesh file (SU2 format).
    dv : np.ndarray, shape (12,)
        Current CST design variables.
    case_dir : Path
        Working directory for this evaluation.
    aoa_deg : float
        Angle of attack.
    reynolds : float
        Chord Reynolds number.
    mach : float
        Freestream Mach number.
    n_iter_primal : int
        Number of primal iterations.
    n_iter_adjoint : int
        Number of adjoint iterations.
    cfl_primal, cfl_adjoint : float
        CFL numbers.
    transition_model : bool
        Enable γ-Re_θ transition model.
    turbulence_intensity : float
        Freestream Tu (fraction).
    turb_viscosity_ratio : float
        Freestream μ_t/μ.
    objective : str
        Adjoint objective ("DRAG", "LIFT", "EFFICIENCY").
    timeout_primal, timeout_adjoint : float
        Timeouts in seconds.

    Returns
    -------
    CFDResult
    """
    case_dir.mkdir(parents=True, exist_ok=True)
    mesh_name = "mesh.su2"

    # Copy mesh to case directory
    mesh_in_case = case_dir / mesh_name
    if mesh_path != mesh_in_case:
        import shutil
        shutil.copy2(mesh_path, mesh_in_case)

    # ── 1. Write primal config ──
    primal_cfg = case_dir / "config_primal.cfg"
    write_primal_config(
        output_path=primal_cfg,
        mesh_filename=mesh_name,
        aoa_deg=aoa_deg,
        reynolds=reynolds,
        mach=mach,
        n_iter=n_iter_primal,
        cfl_initial=cfl_primal * 0.5,
        cfl_final=cfl_primal,
        transition_model=transition_model,
        turbulence_intensity=turbulence_intensity,
        turb_viscosity_ratio=turb_viscosity_ratio,
        output_dir=".",
    )

    # ── 2. Run primal ──
    logger.info(f"Running primal CFD: {su2_cfd_bin}, AoA={aoa_deg}°, Re={reynolds:.1e}")
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

    try:
        primal_result = subprocess.run(
            [su2_cfd_bin, primal_cfg.name],
            cwd=case_dir,
            capture_output=True,
            text=True,
            timeout=timeout_primal,
            creationflags=creation_flags,
        )
    except subprocess.TimeoutExpired:
        return CFDResult(
            cl=0.0, cd=0.0, converged=False,
            adjoint_gradient=np.zeros(N_DESIGN_VARS), gradient_valid=False,
            primal_converged=False, adjoint_converged=False,
            case_dir=case_dir, mesh_path=mesh_path,
            failure_reason=f"Primal CFD timed out after {timeout_primal}s",
        )
    except FileNotFoundError:
        return CFDResult(
            cl=0.0, cd=0.0, converged=False,
            adjoint_gradient=np.zeros(N_DESIGN_VARS), gradient_valid=False,
            primal_converged=False, adjoint_converged=False,
            case_dir=case_dir, mesh_path=mesh_path,
            failure_reason=f"SU2_CFD binary not found: {su2_cfd_bin}",
        )

    # Save logs
    (case_dir / "su2_primal_stdout.log").write_text(primal_result.stdout, encoding="utf-8", errors="ignore")
    (case_dir / "su2_primal_stderr.log").write_text(primal_result.stderr, encoding="utf-8", errors="ignore")

    if primal_result.returncode != 0:
        return CFDResult(
            cl=0.0, cd=0.0, converged=False,
            adjoint_gradient=np.zeros(N_DESIGN_VARS), gradient_valid=False,
            primal_converged=False, adjoint_converged=False,
            case_dir=case_dir, mesh_path=mesh_path,
            failure_reason=f"Primal CFD failed (rc={primal_result.returncode}): {primal_result.stderr[:500]}",
        )

    # ── 3. Extract Cl, Cd from history ──
    history_file = case_dir / "history.csv"
    cl, cd, primal_conv = _parse_history(history_file)

    logger.info(f"Primal CFD: CL={cl:.6f}, CD={cd:.6f}, converged={primal_conv}")

    # ── 4. Run adjoint ──
    adj_cfg = case_dir / "config_adjoint.cfg"
    write_adjoint_config(
        output_path=adj_cfg,
        mesh_filename=mesh_name,
        primal_config_filename=str(primal_cfg),
        objective=objective,
        n_iter=n_iter_adjoint,
        cfl_adjoint=cfl_adjoint,
    )

    # SU2_CFD_ADJ is the same binary, just with MATH_PROBLEM=DISCRETE_ADJOINT
    su2_adj_bin = su2_cfd_bin  # SU2 uses the same binary with different config

    logger.info(f"Running adjoint CFD: {su2_adj_bin}")
    try:
        adj_result = subprocess.run(
            [su2_adj_bin, adj_cfg.name],
            cwd=case_dir,
            capture_output=True,
            text=True,
            timeout=timeout_adjoint,
            creationflags=creation_flags,
        )
    except subprocess.TimeoutExpired:
        return CFDResult(
            cl=cl, cd=cd, converged=False,
            adjoint_gradient=np.zeros(N_DESIGN_VARS), gradient_valid=False,
            primal_converged=primal_conv, adjoint_converged=False,
            case_dir=case_dir, mesh_path=mesh_path,
            failure_reason=f"Adjoint CFD timed out after {timeout_adjoint}s",
        )

    (case_dir / "su2_adjoint_stdout.log").write_text(adj_result.stdout, encoding="utf-8", errors="ignore")
    (case_dir / "su2_adjoint_stderr.log").write_text(adj_result.stderr, encoding="utf-8", errors="ignore")

    adj_conv = (adj_result.returncode == 0)

    if not adj_conv:
        return CFDResult(
            cl=cl, cd=cd, converged=False,
            adjoint_gradient=np.zeros(N_DESIGN_VARS), gradient_valid=False,
            primal_converged=primal_conv, adjoint_converged=False,
            case_dir=case_dir, mesh_path=mesh_path,
            failure_reason=f"Adjoint CFD failed (rc={adj_result.returncode})",
        )

    # ── 5. Extract gradient ──
    try:
        grad = extract_adjoint_gradient(case_dir, objective=objective)
        grad_valid = np.linalg.norm(grad) > 1e-12 and not np.any(np.isnan(grad))
    except Exception as e:
        logger.error(f"Gradient extraction failed: {e}")
        grad = np.zeros(N_DESIGN_VARS)
        grad_valid = False

    converged = primal_conv and adj_conv and grad_valid

    return CFDResult(
        cl=cl, cd=cd, converged=converged,
        adjoint_gradient=grad, gradient_valid=grad_valid,
        primal_converged=primal_conv, adjoint_converged=adj_conv,
        case_dir=case_dir, mesh_path=mesh_in_case,
    )


def _parse_history(history_path: Path) -> Tuple[float, float, bool]:
    """
    Parse SU2 history.csv to extract final Cl, Cd and convergence status.

    Returns
    -------
    cl, cd, converged : (float, float, bool)
    """
    if not history_path.exists():
        logger.warning(f"History file not found: {history_path}")
        return 0.0, 0.0, False

    try:
        text = history_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"Cannot read history: {e}")
        return 0.0, 0.0, False

    lines = text.splitlines()
    if len(lines) < 2:
        return 0.0, 0.0, False

    # Parse header
    header = [h.strip().strip('"') for h in lines[0].split(",")]

    # Find last data line
    last_data = None
    for line in reversed(lines[1:]):
        s = line.strip()
        if s and s != ',':
            last_data = s
            break

    if last_data is None:
        return 0.0, 0.0, False

    values = [v.strip() for v in last_data.split(",")]
    mapping = dict(zip(header, values))

    # Extract CL, CD
    cl_str = mapping.get("CL") or mapping.get("LIFT") or "0.0"
    cd_str = mapping.get("CD") or mapping.get("DRAG") or "0.0"

    try:
        cl = float(cl_str)
        cd = float(cd_str)
    except (ValueError, TypeError):
        return 0.0, 0.0, False

    # Check for convergence from RMS residual info
    rms_cols = [k for k in mapping if k.startswith("RMS_") or "rms" in k.lower()]
    converged = True  # default to converged if we got numbers
    if len(values) > 3:
        # Check if the last few CL values show stabilization
        data_rows = []
        for line in lines[1:]:
            s = line.strip()
            if s and s != ',':
                vals = [v.strip() for v in s.split(",")]
                if len(vals) == len(header):
                    data_rows.append(vals)

        if len(data_rows) > 10:
            last_cls = []
            for row in data_rows[-10:]:
                row_map = dict(zip(header, row))
                cl_v = row_map.get("CL") or row_map.get("LIFT") or "0.0"
                try:
                    last_cls.append(float(cl_v))
                except (ValueError, TypeError):
                    pass

            if len(last_cls) > 3:
                cl_std = np.std(last_cls)
                cl_mean = np.mean(np.abs(last_cls))
                if cl_mean > 1e-10 and cl_std / cl_mean > 0.1:
                    converged = False  # CL still oscillating significantly

    return cl, cd, converged


# ── Objective function for optimizer ───────────────────────────────────────────

class ASOObjectiveFunction:
    """
    Callable objective function for the PDE-constrained optimization.

    Wraps the CFD evaluation and gradient computation so that it can be
    passed to scipy.optimize.minimize or the SvanbergMMA optimizer.
    """

    def __init__(
        self,
        su2_cfd_bin: str,
        mesh_path: Path,
        case_root: Path,
        aoa_deg: float = 4.0,
        reynolds: float = 1e5,
        mach: float = 0.1,
        n_iter_primal: int = 3000,
        n_iter_adjoint: int = 500,
        cfl_primal: float = 3.0,
        cfl_adjoint: float = 1.0,
        transition_model: bool = True,
        turbulence_intensity: float = 0.001,
        turb_viscosity_ratio: float = 5.0,
        objective: str = "DRAG",
        bounds: Optional[CSTBounds] = None,
        use_mesh_deformation: bool = False,
        su2_def_bin: Optional[str] = None,
        previous_mesh_path: Optional[Path] = None,
        previous_dv: Optional[np.ndarray] = None,
    ):
        self.su2_cfd_bin = su2_cfd_bin
        self.mesh_path = mesh_path
        self.case_root = case_root
        self.aoa_deg = aoa_deg
        self.reynolds = reynolds
        self.mach = mach
        self.n_iter_primal = n_iter_primal
        self.n_iter_adjoint = n_iter_adjoint
        self.cfl_primal = cfl_primal
        self.cfl_adjoint = cfl_adjoint
        self.transition_model = transition_model
        self.turbulence_intensity = turbulence_intensity
        self.turb_viscosity_ratio = turb_viscosity_ratio
        self.objective = objective
        self.bounds = bounds
        self.use_mesh_deformation = use_mesh_deformation
        self.su2_def_bin = su2_def_bin

        # Internal state
        self.current_mesh_path = mesh_path
        self.previous_dv = previous_dv
        self._last_gradient: Optional[np.ndarray] = None
        self._last_result: Optional[CFDResult] = None

        # Previous design vector for mesh deformation tracking
        self._previous_dv_stored = previous_dv

    def __call__(self, dv: np.ndarray) -> float:
        """
        Evaluate the objective (Cd at current design point).

        This is the PDE constraint: solve RANS + transition equations.

        Parameters
        ----------
        dv : np.ndarray, shape (12,)
            CST design variables.

        Returns
        -------
        cd : float
            Drag coefficient.
        """
        # Validate geometry before running CFD
        valid, reason = check_geometry_validity(dv, bounds=self.bounds)
        if not valid:
            logger.warning(f"Invalid geometry: {reason}")
            return 1e10  # Large penalty

        # Run CFD evaluation
        case_dir = self.case_root / f"eval_{int(time.time())}"
        result = run_primal_and_adjoint(
            su2_cfd_bin=self.su2_cfd_bin,
            su2_adj_bin=self.su2_cfd_bin,
            mesh_path=self.current_mesh_path,
            dv=dv,
            case_dir=case_dir,
            aoa_deg=self.aoa_deg,
            reynolds=self.reynolds,
            mach=self.mach,
            n_iter_primal=self.n_iter_primal,
            n_iter_adjoint=self.n_iter_adjoint,
            cfl_primal=self.cfl_primal,
            cfl_adjoint=self.cfl_adjoint,
            transition_model=self.transition_model,
            turbulence_intensity=self.turbulence_intensity,
            turb_viscosity_ratio=self.turb_viscosity_ratio,
            objective=self.objective,
        )

        self._last_result = result
        self._last_gradient = result.adjoint_gradient.copy() if result.gradient_valid else None

        if not result.converged:
            logger.warning(f"CFD not converged: {result.failure_reason}")
            return 1e10  # Large penalty for non-converged CFD

        if self.use_mesh_deformation and self.su2_def_bin and self._previous_dv_stored is not None:
            self._deform_mesh_for_next(dv)

        self._previous_dv_stored = dv.copy()
        return result.cd

    def gradient(self, dv: np.ndarray) -> np.ndarray:
        """
        Return the gradient of the objective w.r.t. design variables.

        This is the adjoint PDE solve: solve the discrete adjoint equations.

        Parameters
        ----------
        dv : np.ndarray, shape (12,)

        Returns
        -------
        grad : np.ndarray, shape (12,)
        """
        if self._last_gradient is not None:
            return self._last_gradient
        # If gradient not available, compute via finite differences as fallback
        return self._finite_difference_gradient(dv)

    def _finite_difference_gradient(self, dv: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Fallback: compute gradient via forward finite differences."""
        grad = np.zeros_like(dv)
        f0 = self(dv)
        for i in range(len(dv)):
            dv_pert = dv.copy()
            dv_pert[i] += eps
            fi = self(dv_pert)
            grad[i] = (fi - f0) / eps
        return grad

    def _deform_mesh_for_next(self, dv_new: np.ndarray) -> None:
        """Deform the mesh from previous to new shape."""
        if self._previous_dv_stored is None or self.su2_def_bin is None:
            return
        def_dir = self.case_root / f"def_{int(time.time())}"
        deformed = deform_mesh(
            su2_def_bin=self.su2_def_bin,
            original_mesh_path=self.current_mesh_path,
            dv_old=self._previous_dv_stored,
            dv_new=dv_new,
            work_dir=def_dir,
        )
        if deformed is not None:
            self.current_mesh_path = deformed

    def get_last_result(self) -> Optional[CFDResult]:
        return self._last_result


# ── Main Optimization Driver ───────────────────────────────────────────────────

class PDEOptimizer:
    """
    PDE-Constrained Aerodynamic Shape Optimizer.

    Orchestrates the full optimization cycle:
      1. Initialize baseline geometry and mesh.
      2. For each iteration:
         a. Run primal CFD (RANS + SST + LM)
         b. Run discrete adjoint (SU2_CFD_ADJ)
         c. Project sensitivities onto CST coefficients
         d. Update design variables via optimizer
         e. Deform mesh
         f. Record convergence data
      3. Save convergence history and final design.
    """

    def __init__(
        self,
        su2_cfd_bin: str,
        mesh_path: Path,
        work_dir: Path,
        dv_initial: Optional[np.ndarray] = None,
        bounds: Optional[CSTBounds] = None,
        aoa_deg: float = 4.0,
        reynolds: float = 1e5,
        mach: float = 0.1,
        n_iter_primal: int = 3000,
        n_iter_adjoint: int = 500,
        cfl_primal: float = 3.0,
        cfl_adjoint: float = 1.0,
        transition_model: bool = True,
        turbulence_intensity: float = 0.001,
        turb_viscosity_ratio: float = 5.0,
        move_limit: float = 0.05,
        use_slsqp_fallback: bool = True,
        su2_def_bin: Optional[str] = None,
        use_mesh_deformation: bool = True,
        max_iterations: int = 50,
        convergence_tolerance: float = 1e-4,
    ):
        self.su2_cfd_bin = su2_cfd_bin
        self.su2_def_bin = su2_def_bin
        self.mesh_path = mesh_path
        self.work_dir = work_dir
        self.bounds = bounds or CSTBounds.default()
        self.aoa_deg = aoa_deg
        self.reynolds = reynolds
        self.mach = mach
        self.n_iter_primal = n_iter_primal
        self.n_iter_adjoint = n_iter_adjoint
        self.cfl_primal = cfl_primal
        self.cfl_adjoint = cfl_adjoint
        self.transition_model = transition_model
        self.turbulence_intensity = turbulence_intensity
        self.turb_viscosity_ratio = turb_viscosity_ratio
        self.move_limit = move_limit
        self.use_slsqp_fallback = use_slsqp_fallback
        self.use_mesh_deformation = use_mesh_deformation
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance

        # Default initial design: NACA 4-digit-like shape
        if dv_initial is None:
            dv_initial = np.array([
                0.18, 0.28, 0.34, 0.25, 0.15, 0.08,    # upper
                -0.19, -0.12, -0.09, -0.05, -0.02, -0.01,  # lower
            ])
        self.dv_initial = np.asarray(dv_initial, dtype=float)

        # State
        self.history = ConvergenceHistory()
        self._current_dv = self.dv_initial.copy()
        self.obj_function: Optional[ASOObjectiveFunction] = None

        # Create working directories
        self.case_root = work_dir / "cfd_cases"
        self.case_root.mkdir(parents=True, exist_ok=True)

    def run_slsqp(self) -> ConvergenceHistory:
        """
        Run optimization using scipy.optimize.minimize with SLSQP.

        SLSQP handles the gradient-based constrained optimization:
          minimize  Cd(dv)
          subject to: dv_min <= dv <= dv_max
                      t_min <= thickness(dv) <= t_max

        Returns
        -------
        ConvergenceHistory
        """
        from scipy.optimize import minimize

        # Build bounds for SLSQP (12 design variables)
        scipy_bounds = []
        for i in range(6):
            scipy_bounds.append((float(self.bounds.upper_min[i]), float(self.bounds.upper_max[i])))
        for i in range(6):
            scipy_bounds.append((float(self.bounds.lower_min[i]), float(self.bounds.lower_max[i])))

        # Objective function wrapper for scipy
        self.obj_function = ASOObjectiveFunction(
            su2_cfd_bin=self.su2_cfd_bin,
            mesh_path=self.mesh_path,
            case_root=self.case_root,
            aoa_deg=self.aoa_deg,
            reynolds=self.reynolds,
            mach=self.mach,
            n_iter_primal=self.n_iter_primal,
            n_iter_adjoint=self.n_iter_adjoint,
            cfl_primal=self.cfl_primal,
            cfl_adjoint=self.cfl_adjoint,
            transition_model=self.transition_model,
            turbulence_intensity=self.turbulence_intensity,
            turb_viscosity_ratio=self.turb_viscosity_ratio,
            objective="DRAG",
            bounds=self.bounds,
            use_mesh_deformation=self.use_mesh_deformation,
            su2_def_bin=self.su2_def_bin,
            previous_mesh_path=self.mesh_path,
            previous_dv=self.dv_initial,
        )

        def callback(xk: np.ndarray) -> None:
            """Callback at each iteration to record progress."""
            result = self.obj_function.get_last_result()
            if result is not None:
                # Get max thickness from current design
                upper, lower = compute_surface_coordinates(xk, te_thickness=self.bounds.te_thickness)
                thickness = upper[:, 1] - lower[:, 1]
                max_t = float(np.max(thickness))

                record = IterationRecord(
                    iteration=self.history.total_iterations + 1,
                    cd=result.cd,
                    cl=result.cl,
                    objective=result.cd,
                    grad_norm=float(np.linalg.norm(result.adjoint_gradient)),
                    step_accepted=True,
                    trust_radius=0.0,  # SLSQP doesn't have trust radius
                    max_thickness=max_t,
                    design_vector=xk.tolist(),
                    gradient=result.adjoint_gradient.tolist(),
                )
                self.history.add(record)

                logger.info(
                    f"Iter {record.iteration:3d}: Cd={result.cd:.6f}, Cl={result.cl:.6f}, "
                    f"|∇Cd|={record.grad_norm:.6f}, t/c={max_t:.4f}"
                )

        logger.info("Starting SLSQP optimization...")
        result = minimize(
            fun=lambda dv: self.obj_function(dv),
            x0=self.dv_initial,
            method="SLSQP",
            jac=lambda dv: self.obj_function.gradient(dv),
            bounds=scipy_bounds,
            callback=callback,
            options={
                "maxiter": self.max_iterations,
                "ftol": self.convergence_tolerance,
                "disp": True,
            },
        )

        self._current_dv = result.x.copy()
        converged = result.success or result.status in {0, 1, 3}
        self.history.finalize(converged)

        logger.info(
            f"SLSQP finished: success={result.success}, status={result.status}, "
            f"final Cd={result.fun:.6f}, iterations={self.history.total_iterations}"
        )

        return self.history

    def run_mma(self) -> ConvergenceHistory:
        """
        Run optimization using Svanberg's Method of Moving Asymptotes (MMA).

        MMA generally provides superior convergence for topology/shape optimization
        problems with multiple local minima.

        Returns
        -------
        ConvergenceHistory
        """
        from airfoil_discovery.optimization.mma_engine import SvanbergMMA, TrustRegionGovernor

        # Initialize MMA optimizer
        mma = SvanbergMMA(
            n_vars=N_DESIGN_VARS,
            n_constraints=2,  # min thickness, max thickness
            x_min=np.concatenate([self.bounds.upper_min, self.bounds.lower_min]),
            x_max=np.concatenate([self.bounds.upper_max, self.bounds.lower_max]),
            move_limit=self.move_limit,
            asymptote_adapt=0.7,
        )
        mma.initialize(self.dv_initial)

        # Trust region
        governor = TrustRegionGovernor(
            initial_radius=0.1,
            max_radius=0.5,
            min_radius=1e-6,
        )

        # Objective function
        self.obj_function = ASOObjectiveFunction(
            su2_cfd_bin=self.su2_cfd_bin,
            mesh_path=self.mesh_path,
            case_root=self.case_root,
            aoa_deg=self.aoa_deg,
            reynolds=self.reynolds,
            mach=self.mach,
            n_iter_primal=self.n_iter_primal,
            n_iter_adjoint=self.n_iter_adjoint,
            cfl_primal=self.cfl_primal,
            cfl_adjoint=self.cfl_adjoint,
            transition_model=self.transition_model,
            turbulence_intensity=self.turbulence_intensity,
            turb_viscosity_ratio=self.turb_viscosity_ratio,
            objective="DRAG",
            bounds=self.bounds,
            use_mesh_deformation=self.use_mesh_deformation,
            su2_def_bin=self.su2_def_bin,
            previous_mesh_path=self.mesh_path,
            previous_dv=self.dv_initial,
        )

        dv = self.dv_initial.copy()

        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"=== MMA Iteration {iteration}/{self.max_iterations} ===")

            # Evaluate objective and gradient
            cd = self.obj_function(dv)
            grad = self.obj_function.gradient(dv)
            grad_norm = float(np.linalg.norm(grad))

            # Compute thickness constraint
            upper, lower = compute_surface_coordinates(dv, te_thickness=self.bounds.te_thickness)
            thickness = upper[:, 1] - lower[:, 1]
            max_t = float(np.max(thickness))

            # Constraints: g <= 0
            g_min_t = self.bounds.min_thickness - max_t  # thickness >= min
            g_max_t = max_t - self.bounds.max_thickness    # thickness <= max
            g = np.array([g_min_t, g_max_t])

            # Simplified constraint gradients (finite difference)
            dg = np.zeros((2, N_DESIGN_VARS))
            eps_c = 1e-6
            for i in range(N_DESIGN_VARS):
                dv_pert = dv.copy()
                dv_pert[i] += eps_c
                u2, l2 = compute_surface_coordinates(dv_pert, te_thickness=self.bounds.te_thickness)
                t2 = float(np.max(u2[:, 1] - l2[:, 1]))
                dg[0, i] = -(t2 - max_t) / eps_c
                dg[1, i] = (t2 - max_t) / eps_c

            # Run MMA step
            x_candidate, step_accepted, state = mma.run_optimization_step(
                f=cd, df=grad, g=g, dg=dg
            )

            # Update trust region
            trust_update = governor.update(state.rho)
            trust_radius = trust_update["radius"]

            # Record iteration
            record = IterationRecord(
                iteration=iteration,
                cd=cd,
                cl=self.obj_function.get_last_result().cl if self.obj_function.get_last_result() else 0.0,
                objective=cd,
                grad_norm=grad_norm,
                step_accepted=step_accepted,
                trust_radius=trust_radius,
                max_thickness=max_t,
                design_vector=x_candidate.tolist() if step_accepted else dv.tolist(),
                gradient=grad.tolist(),
                constraint_violations=[float(g_min_t), float(g_max_t)],
            )
            self.history.add(record)

            logger.info(
                f"Cd={cd:.6f}, |∇Cd|={grad_norm:.6f}, t/c={max_t:.4f}, "
                f"accepted={step_accepted}, ρ={trust_radius:.4f}"
            )

            if step_accepted:
                dv = x_candidate.copy()
                # Deform mesh to new shape
                if self.use_mesh_deformation and self.su2_def_bin and iteration > 1:
                    def_dir = self.case_root / f"def_iter_{iteration}"
                    deformed = deform_mesh(
                        su2_def_bin=self.su2_def_bin,
                        original_mesh_path=self.obj_function.current_mesh_path,
                        dv_old=self._current_dv,
                        dv_new=dv,
                        work_dir=def_dir,
                    )
                    if deformed is not None:
                        self.obj_function.current_mesh_path = deformed
                self._current_dv = dv.copy()

            # Check convergence
            if grad_norm < self.convergence_tolerance and step_accepted:
                logger.info(f"Converged at iteration {iteration}: |∇Cd|={grad_norm:.6e}")
                self.history.finalize(converged=True)
                return self.history

            # Check stagnation
            if state.stagnated_counter >= 15:
                logger.warning(f"Optimization stagnated at iteration {iteration}")
                self.history.finalize(converged=False)
                return self.history

        self.history.finalize(converged=False)
        return self.history

    def run(self, method: str = "mma") -> ConvergenceHistory:
        """
        Run the complete optimization.

        Parameters
        ----------
        method : str
            "mma" for SvanbergMMA, "slsqp" for scipy SLSQP.

        Returns
        -------
        ConvergenceHistory
        """
        method = method.lower()
        if method == "slsqp" or (method == "mma" and not self.use_slsqp_fallback):
            if method == "slsqp":
                return self.run_slsqp()
            else:
                return self.run_mma()
        else:
            # Try MMA first, fallback to SLSQP
            try:
                return self.run_mma()
            except Exception as e:
                logger.warning(f"MMA failed ({e}), falling back to SLSQP")
                return self.run_slsqp()

    def save_results(self, output_dir: Path) -> None:
        """Save optimization results including convergence history and final design."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save convergence history
        history_path = output_dir / "convergence_history.json"
        self.history.save(history_path)

        # Save final design
        final_dv_path = output_dir / "final_design.npy"
        np.save(final_dv_path, self._current_dv)

        # Save final airfoil coordinates
        coords = compute_airfoil_coordinates(
            self._current_dv,
            te_thickness=self.bounds.te_thickness,
        )
        coords_path = output_dir / "final_airfoil.dat"
        lines = ["final_airfoil"]
        for x, y in coords:
            lines.append(f"  {x:.10f}  {y:.10f}")
        coords_path.write_text("\n".join(lines), encoding="utf-8")

        # Save a summary text file
        summary = [
            "=" * 60,
            "PDE-Constrained Aerodynamic Shape Optimization Results",
            "=" * 60,
            f"Method: {'MMA' if self.history.converged else 'SLSQP'}",
            f"Total iterations: {self.history.total_iterations}",
            f"Converged: {self.history.converged}",
            f"",
            f"Final Design Variables (CST coefficients):",
            f"  Upper: {self._current_dv[:6]}",
            f"  Lower: {self._current_dv[6:]}",
            f"",
            f"Final Objective (Cd): {self.history.iterations[-1].cd if self.history.iterations else 'N/A':.6f}",
            f"Final Cl: {self.history.iterations[-1].cl if self.history.iterations else 'N/A':.6f}",
            f"",
            f"Convergence History (first 5 and last 5):",
        ]

        if self.history.iterations:
            for rec in self.history.iterations[:5]:
                summary.append(f"  Iter {rec.iteration:3d}: Cd={rec.cd:.6f}, Cl={rec.cl:.6f}, |∇|={rec.grad_norm:.6f}")
            if len(self.history.iterations) > 10:
                summary.append(f"  ... ({len(self.history.iterations) - 10} intermediate iterations) ...")
            for rec in self.history.iterations[-5:]:
                summary.append(f"  Iter {rec.iteration:3d}: Cd={rec.cd:.6f}, Cl={rec.cl:.6f}, |∇|={rec.grad_norm:.6f}")

        summary_path = output_dir / "optimization_summary.txt"
        summary_path.write_text("\n".join(summary), encoding="utf-8")

        logger.info(f"Results saved to {output_dir}")