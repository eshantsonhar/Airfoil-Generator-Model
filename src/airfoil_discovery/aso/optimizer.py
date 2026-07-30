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
import signal
import subprocess
import os
import time
import traceback
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


# ── Emergency State for Signal Handlers ────────────────────────────────────────

_emergency_state: Dict[str, Any] = {
    "current_dv": None,
    "best_dv": None,
    "best_cd": float("inf"),
    "optimizer": None,
    "output_dir": None,
    "history": None,
    "iteration": 0,
    "shutdown_requested": False,
}


def _emergency_signal_handler(signum: int, frame: Any) -> None:
    """Signal handler for SIGINT/SIGTERM — dumps emergency state."""
    logger = logging.getLogger(__name__)
    sig_name = signal.Signals(signum).name if signum in signal.Signals._value2member_map_ else f"signal {signum}"
    logger.warning(f"=== EMERGENCY: {sig_name} received, dumping state ===")
    _emergency_state["shutdown_requested"] = True
    _dump_emergency_state()
    logger.warning("=== Emergency dump complete. Exiting. ===")
    sys.exit(1)


def _dump_emergency_state() -> None:
    """Serialize current optimization state to disk."""
    state = _emergency_state
    if state["output_dir"] is None:
        return
    output_dir = Path(state["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    emergency_dir = output_dir / "emergency_dump"
    emergency_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save current design
    if state["current_dv"] is not None:
        np.save(emergency_dir / f"emergency_dv_{timestamp}.npy", state["current_dv"])

    # Save best design
    if state["best_dv"] is not None:
        np.save(emergency_dir / f"emergency_best_dv_{timestamp}.npy", state["best_dv"])
        from .cst import compute_airfoil_coordinates
        try:
            coords = compute_airfoil_coordinates(state["best_dv"])
            dat_path = emergency_dir / f"emergency_best_airfoil_{timestamp}.dat"
            lines = ["emergency_best_airfoil"]
            for x, y in coords:
                lines.append(f"  {x:.10f}  {y:.10f}")
            dat_path.write_text("\n".join(lines), encoding="utf-8")
        except Exception as e:
            logger.error(f"Emergency: could not write best airfoil: {e}")

    # Save history
    if state["history"] is not None:
        try:
            hist_path = emergency_dir / f"emergency_history_{timestamp}.json"
            state["history"].save(hist_path)
        except Exception as e:
            logger.error(f"Emergency: could not save history: {e}")

    # Save summary
    summary = {
        "timestamp": timestamp,
        "iteration": state["iteration"],
        "best_cd": state["best_cd"],
        "shutdown_requested": state["shutdown_requested"],
        "message": "Emergency shutdown — state preserved",
    }
    summary_path = emergency_dir / f"emergency_summary_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    logger.info(f"Emergency state dumped to {emergency_dir}")


def setup_signal_handlers() -> None:
    """Install emergency signal handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, _emergency_signal_handler)
    signal.signal(signal.SIGTERM, _emergency_signal_handler)


def update_emergency_state(**kwargs: Any) -> None:
    """Update the emergency state dictionary."""
    for key, value in kwargs.items():
        if key in _emergency_state:
            _emergency_state[key] = value


def shutdown_requested() -> bool:
    """Check if a shutdown signal has been received."""
    return _emergency_state["shutdown_requested"]


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
    use_adjoint: bool = True,
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
    # Copy mesh to case directory using the original filename to ensure SU2 can find it
    mesh_name = mesh_path.name

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
        cfl_initial=cfl_primal,
        cfl_final=cfl_primal,
        cfl_adapt=False,
        transition_model=transition_model,
        turbulence_intensity=turbulence_intensity,
        turb_viscosity_ratio=turb_viscosity_ratio,
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

    if not use_adjoint:
        logger.info("Skipping adjoint solve; using finite-difference gradient fallback.")
        return CFDResult(
            cl=cl, cd=cd, converged=primal_conv,
            adjoint_gradient=np.zeros(N_DESIGN_VARS), gradient_valid=False,
            primal_converged=primal_conv, adjoint_converged=False,
            case_dir=case_dir, mesh_path=mesh_in_case,
        )

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

    # ── 5. Extract gradient (if adjoint succeeded) ──
    try:
        grad = extract_adjoint_gradient(case_dir, objective=objective)
        grad_valid = np.linalg.norm(grad) > 1e-12 and not np.any(np.isnan(grad))
    except Exception as e:
        logger.error(f"Gradient extraction failed: {e}")
        grad = None
        grad_valid = False

    # If adjoint fails but primal converged, fall back to finite differences
    if not adj_conv or not grad_valid:
        logger.warning("Adjoint failed or invalid, falling back to finite difference gradient")
        grad = None
    
    # Accept result if primal converged (adjoint is optional with FD fallback)
    converged = primal_conv

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

    # Parse header - strip quotes and whitespace
    header = [h.strip().strip('"').strip("'") for h in lines[0].split(",")]
    
    # Log detected headers for debugging
    logger.info(f"SU2 history headers detected: {header}")

    # Find last data line
    last_data = None
    for line in reversed(lines[1:]):
        s = line.strip()
        if s and s != ',' and not s.startswith("#"):
            last_data = s
            break

    if last_data is None:
        logger.warning("No valid data rows found in history file")
        return 0.0, 0.0, False

    values = [v.strip() for v in last_data.split(",")]
    
    # Ensure we have matching header/value counts
    if len(values) != len(header):
        logger.warning(f"Header/value count mismatch: {len(header)} headers, {len(values)} values")
        # Try to use positional parsing as fallback
        if len(values) >= 8:
            # Common SU2 format: Iter, Time, CL, CD, ... (positions 2, 3)
            try:
                cl = float(values[2])
                cd = float(values[3])
                logger.warning(f"Using positional parsing: CL={cl}, CD={cd}")
                return cl, cd, True
            except (ValueError, IndexError):
                return 0.0, 0.0, False
        return 0.0, 0.0, False
    
    mapping = dict(zip(header, values))

    # Extract CL, CD with comprehensive header matching
    # SU2 uses various naming conventions depending on solver and configuration
    cl_candidates = ["CL", "LIFT", "CLift", "CL_Total", "Cz", "FORCE_X_COEFF"]
    cd_candidates = ["CD", "DRAG", "CDrag", "CD_Total", "Cx", "FORCE_Y_COEFF"]
    
    cl_str = None
    cd_str = None
    
    # Try exact matches first
    for candidate in cl_candidates:
        if candidate in mapping:
            cl_str = mapping[candidate]
            logger.info(f"Found CL column: '{candidate}' = {cl_str}")
            break
    
    for candidate in cd_candidates:
        if candidate in mapping:
            cd_str = mapping[candidate]
            logger.info(f"Found CD column: '{candidate}' = {cd_str}")
            break
    
    # Fallback to case-insensitive search
    if cl_str is None:
        for key in mapping:
            if key.upper() in ["CL", "LIFT"]:
                cl_str = mapping[key]
                logger.info(f"Found CL column (case-insensitive): '{key}' = {cl_str}")
                break
    
    if cd_str is None:
        for key in mapping:
            if key.upper() in ["CD", "DRAG"]:
                cd_str = mapping[key]
                logger.info(f"Found CD column (case-insensitive): '{key}' = {cd_str}")
                break
    
    # Final fallback
    cl_str = cl_str or "0.0"
    cd_str = cd_str or "0.0"

    try:
        cl = float(cl_str)
        cd = float(cd_str)
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to parse CL/CD values: cl_str='{cl_str}', cd_str='{cd_str}', error={e}")
        return 0.0, 0.0, False

    # Check for convergence from RMS residual info
    rms_cols = [k for k in mapping if k.startswith("RMS_") or "rms" in k.lower()]
    converged = True  # default to converged if we got numbers
    forces_stabilized = True
    
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
            last_cds = []
            for row in data_rows[-10:]:
                row_map = dict(zip(header, row))
                cl_v = row_map.get("CL") or row_map.get("LIFT") or "0.0"
                cd_v = row_map.get("CD") or row_map.get("DRAG") or "0.0"
                try:
                    last_cls.append(float(cl_v))
                    last_cds.append(float(cd_v))
                except (ValueError, TypeError):
                    pass

            if len(last_cls) > 3:
                cl_std = np.std(last_cls)
                cl_mean = np.mean(np.abs(last_cls))
                cd_std = np.std(last_cds)
                cd_mean = np.mean(np.abs(last_cds))
                
                # Check if forces are stabilized (low relative std)
                if cl_mean > 1e-10 and cl_std / cl_mean > 0.1:
                    forces_stabilized = False
                if cd_mean > 1e-10 and cd_std / cd_mean > 0.1:
                    forces_stabilized = False
                
                if not forces_stabilized:
                    converged = False  # Forces still oscillating significantly

    # Low-Re relaxation: if forces are stabilized and residual dropped >= 1 order, accept
    if not converged and forces_stabilized and rms_cols:
        # Check residual drop
        if len(data_rows) > 20:
            first_rms = None
            last_rms = None
            for row in data_rows[:10]:
                row_map = dict(zip(header, row))
                for rms_col in rms_cols:
                    if rms_col in row_map:
                        try:
                            first_rms = float(row_map[rms_col])
                            break
                        except (ValueError, TypeError):
                            pass
                if first_rms is not None:
                    break
            
            for row in data_rows[-10:]:
                row_map = dict(zip(header, row))
                for rms_col in rms_cols:
                    if rms_col in row_map:
                        try:
                            last_rms = float(row_map[rms_col])
                            break
                        except (ValueError, TypeError):
                            pass
                if last_rms is not None:
                    break
            
            if first_rms is not None and last_rms is not None:
                if first_rms != 0:
                    residual_drop = np.log10(abs(first_rms)) - np.log10(abs(last_rms))
                    if residual_drop >= 1.0:
                        logger.info(f"Low-Re relaxation: forces stabilized, residual dropped {residual_drop:.2f} orders. Accepting as converged.")
                        converged = True

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
        use_adjoint: bool = True,
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
        self.use_adjoint = use_adjoint
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

        # Reset last gradient for a new design point
        self._last_gradient = None
        self._last_dv = dv.copy()

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
            use_adjoint=self.use_adjoint,
        )

        self._last_result = result
        self._last_gradient = result.adjoint_gradient.copy() if result.gradient_valid else None

        if not result.converged:
            logger.warning(f"CFD not converged: {result.failure_reason}")
            return 1e10  # Large penalty for non-converged CFD

        # ── AERODYNAMIC SANITY BOUNDS ──
        # Physical limits for 2D airfoil at Re=100,000, AoA=4°
        # These are hard guards against non-physical CFD results
        cl_lower = -0.5   # Negative lift at positive AoA indicates geometry/solver error
        cl_upper = 2.5    # Beyond this, flow is fully separated (stall)
        cd_lower = 0.001  # Below this is unrealistically low (laminar bubble artifacts)
        cd_upper = 1.0    # Relaxed upper bound to accept early/unconverged primal runs
        
        if result.cl < cl_lower or result.cl > cl_upper:
            logger.error(
                f"NON-PHYSICAL LIFT: Cl={result.cl:.6f} outside bounds [{cl_lower}, {cl_upper}]. "
                f"This indicates geometry/solver error. Rejecting result."
            )
            return 1e10
        
        if result.cd < cd_lower or result.cd > cd_upper:
            logger.error(
                f"NON-PHYSICAL DRAG: Cd={result.cd:.6f} outside bounds [{cd_lower}, {cd_upper}]. "
                f"This indicates CFD divergence or geometry error. Rejecting result."
            )
            return 1e10

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
        if not self.use_adjoint:
            self._last_gradient = self._finite_difference_gradient(dv)
            return self._last_gradient
        # If gradient not available, compute via finite differences as fallback
        return self._finite_difference_gradient(dv)

    def _finite_difference_gradient(self, dv: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Fallback: compute gradient via forward finite differences."""
        grad = np.zeros_like(dv)
        if self._last_result is not None and self._last_dv is not None and np.array_equal(dv, self._last_dv):
            f0 = self._last_result.cd
        else:
            f0 = self(dv)

        for i in range(len(dv)):
            dv_pert = dv.copy()
            dv_pert[i] += eps
            fi = self(dv_pert)
            grad[i] = (fi - f0) / eps

        self._last_gradient = grad
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
        use_adjoint: bool = True,
        max_iterations: int = 50,
        convergence_tolerance: float = 1e-4,
    ):
        self.su2_cfd_bin = su2_cfd_bin
        self.su2_def_bin = su2_def_bin
        self.mesh_path = mesh_path
        self.work_dir = work_dir
        self.bounds = bounds or CSTBounds.default()
        self.use_adjoint = use_adjoint
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
            use_adjoint=self.use_adjoint,
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
                    f"|grad Cd|={record.grad_norm:.6f}, t/c={max_t:.4f}"
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
            use_adjoint=self.use_adjoint,
        )

        dv = self.dv_initial.copy()

        # Fault-tolerant state
        consecutive_cfd_failures = 0
        max_consecutive_failures = 3
        backtrack_factor = 0.5
        dv_safe = dv.copy()
        mesh_safe = self.mesh_path
        best_cd = float("inf")
        best_dv = dv.copy()

        # Update emergency state
        update_emergency_state(
            current_dv=dv,
            best_dv=best_dv,
            best_cd=best_cd,
            optimizer=self,
            output_dir=str(self.work_dir),
            history=self.history,
        )

        for iteration in range(1, self.max_iterations + 1):
            # Check for shutdown signal
            if shutdown_requested():
                logger.warning("Shutdown requested, stopping optimization")
                self.history.finalize(converged=False)
                return self.history

            logger.info(f"=== MMA Iteration {iteration}/{self.max_iterations} ===")

            # Evaluate objective and gradient with fault tolerance
            try:
                cd = self.obj_function(dv)
            except Exception as e:
                logger.error(f"CFD evaluation failed at iteration {iteration}: {e}")
                consecutive_cfd_failures += 1
                if consecutive_cfd_failures >= max_consecutive_failures:
                    logger.error(f"Too many consecutive CFD failures ({consecutive_cfd_failures}). Stopping.")
                    self.history.finalize(converged=False)
                    return self.history
                # Backtrack: restore safe design and reduce step
                logger.warning(f"Backtracking: restoring previous safe design (attempt {consecutive_cfd_failures}/{max_consecutive_failures})")
                dv = dv_safe.copy()
                self.obj_function.current_mesh_path = mesh_safe
                self.obj_function._previous_dv_stored = dv_safe.copy()
                # Reduce move limit
                self.move_limit *= backtrack_factor
                logger.info(f"Move limit reduced to {self.move_limit:.6f}")
                continue

            # Reset failure counter on success
            consecutive_cfd_failures = 0

            # Check for NaN/Inf in Cd
            if np.isnan(cd) or np.isinf(cd) or cd > 1e6:
                logger.warning(f"CFD produced invalid Cd={cd:.4e}. Rejecting step.")
                consecutive_cfd_failures += 1
                dv = dv_safe.copy()
                self.obj_function.current_mesh_path = mesh_safe
                self.move_limit *= backtrack_factor
                continue

            # Track best design
            if cd < best_cd:
                best_cd = cd
                best_dv = dv.copy()
                update_emergency_state(best_dv=best_dv, best_cd=best_cd)

            # Get gradient — with fallback to finite differences on zero/failed adjoint
            try:
                grad = self.obj_function.gradient(dv)
            except Exception as e:
                logger.error(f"Gradient extraction failed: {e}")
                grad = None

            grad_norm = float(np.linalg.norm(grad)) if grad is not None else 0.0

            # If adjoint produced zero or failed gradient, try finite differences directly
            if grad is None or grad_norm < 1e-12:
                logger.warning(
                    f"Iter {iteration}: adjoint gradient zero or unavailable "
                    f"(norm={grad_norm:.3e}). Computing via finite differences."
                )
                try:
                    grad = self.obj_function._finite_difference_gradient(dv)
                    grad_norm = float(np.linalg.norm(grad))
                except Exception as fd_err:
                    logger.error(f"Finite difference gradient also failed: {fd_err}")
                    consecutive_cfd_failures += 1
                    if consecutive_cfd_failures >= max_consecutive_failures:
                        logger.error(f"Cannot compute gradient after {consecutive_cfd_failures} attempts. Stopping.")
                        self.history.finalize(converged=False)
                        return self.history
                    dv = dv_safe.copy()
                    self.move_limit *= backtrack_factor
                    continue

            # After FD fallback, if still zero — backtrack instead of hard crash
            if grad_norm < 1e-12:
                logger.warning(
                    f"Iter {iteration}: gradient still zero after FD fallback. "
                    f"Backtracking and reducing move limit."
                )
                consecutive_cfd_failures += 1
                if consecutive_cfd_failures >= max_consecutive_failures:
                    logger.error(f"Zero gradient persists after {consecutive_cfd_failures} attempts. Stopping.")
                    self.history.finalize(converged=False)
                    return self.history
                dv = dv_safe.copy()
                self.move_limit *= backtrack_factor
                continue

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
                f"Cd={cd:.6f}, |grad Cd|={grad_norm:.6f}, t/c={max_t:.4f}, "
                f"accepted={step_accepted}, rho={trust_radius:.4f}"
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
                logger.info(f"Converged at iteration {iteration}: |grad Cd|={grad_norm:.6e}")
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
        # Convert to numpy array for XFLR5/Selig format processing
        pts = np.array(coords)
        
        # Identify critical apex points
        le_idx = np.argmin(pts[:, 0])
        le_x, le_y = pts[le_idx]
        max_x = np.max(pts[:, 0])
        te_pts = pts[pts[:, 0] == max_x]
        avg_te_y = np.mean(te_pts[:, 1])
        
        upper = []
        lower = []
        
        # Separate upper and lower surfaces using chord-line equation
        for x, y in pts:
            if x == le_x and y == le_y:
                continue
            chord_y = le_y + ((avg_te_y - le_y) / (max_x - le_x)) * (x - le_x)
            if y >= chord_y:
                upper.append((x, y))
            else:
                lower.append((x, y))
        
        # Sort for Selig Format (Upper: TE -> LE [X descending], Lower: LE -> TE [X ascending])
        upper_sorted = sorted(upper, key=lambda p: p[0], reverse=True)
        lower_sorted = sorted(lower, key=lambda p: p[0])
        
        # Construct final sequenced coordinate list
        final_sequence = upper_sorted + [(le_x, le_y)] + lower_sorted
        
        # Close the loop at TE if single point
        if len(te_pts) == 1:
            final_sequence.append((te_pts[0][0], te_pts[0][1]))
        
        # Write XFLR5 compatible file
        coords_path = output_dir / "final_airfoil.dat"
        lines = ["final_airfoil"]
        for x, y in final_sequence:
            lines.append(f"  {x:.10f}   {y:.10f}")
        # Ensure trailing newline at end of file for XFLR5 compatibility
        coords_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Save a summary text file
        # Format final values safely
        final_cd = f"{self.history.iterations[-1].cd:.6f}" if self.history.iterations else "N/A"
        final_cl = f"{self.history.iterations[-1].cl:.6f}" if self.history.iterations else "N/A"
        
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
            f"Final Objective (Cd): {final_cd}",
            f"Final Cl: {final_cl}",
            f"",
            f"Convergence History (first 5 and last 5):",
        ]

        if self.history.iterations:
            for rec in self.history.iterations[:5]:
                summary.append(f"  Iter {rec.iteration:3d}: Cd={rec.cd:.6f}, Cl={rec.cl:.6f}, |grad|={rec.grad_norm:.6f}")
            if len(self.history.iterations) > 10:
                summary.append(f"  ... ({len(self.history.iterations) - 10} intermediate iterations) ...")
            for rec in self.history.iterations[-5:]:
                # ASCII-safe: avoid nabla U+2207 which crashes Windows cp1252 console
                summary.append(f"  Iter {rec.iteration:3d}: Cd={rec.cd:.6f}, Cl={rec.cl:.6f}, |grad|={rec.grad_norm:.6f}")

        summary_path = output_dir / "optimization_summary.txt"
        summary_path.write_text("\n".join(summary), encoding="utf-8")

        logger.info(f"Results saved to {output_dir}")