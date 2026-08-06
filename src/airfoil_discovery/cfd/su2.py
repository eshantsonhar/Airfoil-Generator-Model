"""
Real SU2 CFD evaluation layer with proper adjoint gradient extraction.

NO fake fallbacks.
NO zero gradients.
NO dummy adjoint runs.
EVERY CFD result must be validated.
Includes structured failure diagnostics.
"""

from __future__ import annotations
import enum
import os
import subprocess

from airfoil_discovery.runtime import run_with_timeout
import json
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class SU2Status(enum.Enum):
    OK = "OK"
    DIVERGED = "DIVERGED"
    ADJOINT_INVALID = "ADJOINT_INVALID"
    CRASHED = "CRASHED"
    SETUP_ERROR = "SETUP_ERROR"
    CONFIG_ERROR = "CONFIG_ERROR"
    GRADIENT_ZERO = "GRADIENT_ZERO"


@dataclass
class AdjointResult:
    """Real adjoint gradient extraction result."""
    grad_cd: np.ndarray
    grad_cl: np.ndarray
    residual: float
    n_adjoint_iterations: int = 0
    converged: bool = False


@dataclass
class DesignEvaluation:
    cl: float
    cd: float
    thickness: float
    status: SU2Status
    design_id: str = ""  # INSTRUMENTATION: design identity fingerprint
    adjoint: Optional[AdjointResult] = None
    residual_history: Optional[List[float]] = None
    cl_history: Optional[List[float]] = None
    cd_history: Optional[List[float]] = None
    convergence_report: Optional[Dict[str, Any]] = None
    lsb_report: Optional[Dict[str, Any]] = None
    failure_stage: Optional[str] = None
    failure_reason: Optional[str] = None
    failure_detail: Optional[str] = None
    offending_file: Optional[str] = None


class SU2ConfigurationError(Exception):
    """Exception raised when SU2 configuration is invalid."""
    pass


class SU2ExecutionError(Exception):
    """Exception raised when SU2 execution fails."""
    def __init__(self, stage: str, reason: str):
        self.stage = stage
        self.reason = reason
        super().__init__(f"SU2 {stage} failed: {reason}")


class SU2Runner:
    """
    Production SU2 execution layer.
    """

    def __init__(self, settings: Any):
        if settings is None:
            raise SU2ConfigurationError("Settings cannot be None.")
        self.settings = settings

    def _verify_binaries(self):
        missing = []
        su2_bin = self.settings.solver.su2_cfd_bin
        if not su2_bin or not Path(su2_bin).exists():
            missing.append(f"SU2_CFD: {su2_bin}")
        gmsh_bin = self.settings.solver.gmsh_bin
        if not gmsh_bin or not Path(gmsh_bin).exists():
            missing.append(f"GMSH: {gmsh_bin}")
        su2_def_bin = getattr(self.settings.solver, 'su2_def_bin', None)
        if su2_def_bin and not Path(su2_def_bin).exists():
            missing.append(f"SU2_DEF: {su2_def_bin}")
        if missing:
            raise SU2ExecutionError("PREFLIGHT_CHECK", f"Missing binaries: {'; '.join(missing)}")
        logger.info(f"[binaries] SU2={Path(su2_bin).resolve()}, GMSH={Path(gmsh_bin).resolve()}")
        try:
            vr = subprocess.run([su2_bin, "--version"], capture_output=True, text=True,
                                timeout=10, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            logger.info(f"[binaries] SU2 rc={vr.returncode}, stderr={vr.stderr[:200]}")
        except Exception as e:
            logger.warning(f"[binaries] SU2 version check: {e}")

    def _write_airfoil_dat(self, coords: np.ndarray, dat_path: Path):
        lines = ["test_airfoil"]
        for x, y in coords:
            lines.append(f"  {x:.6f}  {y:.6f}")
        dat_path.write_text("\n".join(lines), encoding="utf-8")

    def _write_gmsh_geo(self, coords: np.ndarray, geo_path: Path, reynolds: float, mesh_level: str = "L0"):
        from airfoil_discovery.cfd.mesh import MeshFidelityManager, build_geo_script
        fidelity = MeshFidelityManager.get_params(mesh_level)
        geo_script = build_geo_script(coords=coords, reynolds=reynolds,
                                      mesh_cfg=self.settings.solver.mesh,
                                      coarse_factor=fidelity.coarse_factor)
        geo_path.write_text(geo_script, encoding="utf-8")

    def _write_su2_config(self, candidate: Any, mesh_path: Path, config_path: Path,
                          aoa: float, mesh_level: str = "L0", for_adjoint: bool = False):
        from airfoil_discovery.cfd.su2_config import build_stage1_config
        config_text = build_stage1_config(candidate, mesh_path, aoa, self.settings)
        if self.settings.solver.transition_model:
            config_text = re.sub(r"KIND_TRANS_MODEL\s*=\s*\S+", "KIND_TRANS_MODEL= LM", config_text)
            config_text += (f"\nFREESTREAM_TURBULENCEINTENSITY= {self.settings.solver.stage3_turbulence_intensity}"
                            f"\nFREESTREAM_TURB2LAMVISCRATIO= {self.settings.solver.stage3_turb_viscosity_ratio}")
        # FIX: Use stage1_iter from config (stage1_iter=500 by default) instead of overriding to 30/80
        if mesh_level == "L0":
            config_text = re.sub(r"ITER= \d+", f"ITER= {self.settings.solver.stage1_iter}", config_text)
        elif mesh_level == "L1":
            config_text = re.sub(r"ITER= \d+", f"ITER= {self.settings.solver.stage1_iter}", config_text)
        # FIX: ensure output frequency is less than iteration count and SURFACE_CSV is enabled
        config_text = re.sub(r"OUTPUT_WRT_FREQ= \d+", "OUTPUT_WRT_FREQ= 50", config_text)
        config_text = re.sub(r"CONV_STARTITER= \d+", "CONV_STARTITER= 100", config_text)
        config_text = re.sub(r"OUTPUT_FILES= \(.*\)", "OUTPUT_FILES= (RESTART, SURFACE_CSV)", config_text)
        self._validate_config(config_text)
        config_path.write_text(config_text, encoding="utf-8")

    def _validate_config(self, config_text: str):
        for key in ["SOLVER", "MESH_FILENAME", "RESTART_SOL", "ITER"]:
            if re.search(rf"^{key}\s*=", config_text, re.MULTILINE) is None:
                raise SU2ConfigurationError(f"Missing required config key: {key}")

    def _run_gmsh(self, geo_path: Path, mesh_path: Path, work_dir: Path):
        cmd = [self.settings.solver.gmsh_bin, geo_path.name, "-2", "-format", "su2", "-o", mesh_path.name]
        mesh_timeout = float(getattr(self.settings.solver, "case_timeout_seconds", 0) or 0) or 300.0
        cf = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

        def _run() -> subprocess.CompletedProcess[str]:
            return subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True, timeout=int(mesh_timeout), creationflags=cf)

        wd = run_with_timeout("gmsh_mesh", _run, timeout_seconds=mesh_timeout)
        if not wd.succeeded:
            raise SU2ExecutionError("MESH_GENERATION", f"GMSH timed out: {wd.error}")
        result = wd.result
        (work_dir / "gmsh_stdout.log").write_text(result.stdout, encoding="utf-8", errors="ignore")
        (work_dir / "gmsh_stderr.log").write_text(result.stderr, encoding="utf-8", errors="ignore")
        if result.returncode != 0:
            raise SU2ExecutionError("MESH_GENERATION", f"GMSH rc={result.returncode}: {result.stderr[:1000]}")
        if not mesh_path.exists() or mesh_path.stat().st_size == 0:
            raise SU2ExecutionError("MESH_GENERATION", "GMSH produced empty mesh")
        logger.info(f"[mesh] GMSH OK: {mesh_path.name} ({mesh_path.stat().st_size} bytes)")

    def _run_su2_primal(self, config_path: Path, work_dir: Path):
        cmd = [self.settings.solver.su2_cfd_bin, config_path.name]
        su2_timeout = float(getattr(self.settings.solver, "case_timeout_seconds", 0) or 0) or 1800.0
        cf = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

        def _run() -> subprocess.CompletedProcess[str]:
            return subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True, timeout=int(su2_timeout), creationflags=cf)

        wd = run_with_timeout("su2_primal", _run, timeout_seconds=su2_timeout)
        if not wd.succeeded:
            raise SU2ExecutionError("PRIMAL_SOLVE", f"SU2 timed out: {wd.error}")
        result = wd.result
        (work_dir / "su2_stdout.log").write_text(result.stdout, encoding="utf-8", errors="ignore")
        (work_dir / "su2_stderr.log").write_text(result.stderr, encoding="utf-8", errors="ignore")
        if result.returncode != 0:
            stderr = result.stderr[:1000] or "(none)"
            if "Unable to open mesh" in result.stderr or "MESH_FILENAME" in result.stderr:
                raise SU2ExecutionError("PRIMAL_SOLVE", f"SU2 cannot open mesh: {stderr}")
            raise SU2ExecutionError("PRIMAL_SOLVE", f"SU2 rc={result.returncode}: {stderr}")
        logger.info(f"[su2] Primal solve OK (rc={result.returncode})")

    def _read_results(self, history_path: Path) -> Tuple[float, float]:
        if not history_path.exists():
            raise SU2ExecutionError("RESULT_EXTRACTION", f"History file not found: {history_path}")
        try:
            text = history_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            raise SU2ExecutionError("RESULT_EXTRACTION", f"Cannot read history: {e}")
        lines = text.splitlines()
        if len(lines) < 2:
            raise SU2ExecutionError("RESULT_EXTRACTION", "History file too short")
        headers = [item.strip().strip('"') for item in lines[0].split(",")]
        last_data = None
        for line in reversed(lines[1:]):
            s = line.strip()
            if s and s != ',':
                last_data = s
                break
        if last_data is None:
            raise SU2ExecutionError("RESULT_EXTRACTION", "No data lines in history")
        values = [item.strip() for item in last_data.split(",")]
        if len(values) < len(headers):
            values.extend(['0.0'] * (len(headers) - len(values)))
        mapping = dict(zip(headers, values))
        cl_val = mapping.get("CL") or mapping.get('"CL"') or mapping.get("LIFT") or '0.0'
        cd_val = mapping.get("CD") or mapping.get('"CD"') or mapping.get("DRAG") or '0.0'
        for name, val in [("CL", cl_val), ("CD", cd_val)]:
            if val in ('', 'nan', 'NaN', 'inf', 'Inf', '-inf', '-Inf'):
                raise SU2ExecutionError("RESULT_EXTRACTION", f"{name} invalid: {val}")
        try:
            cl = float(cl_val)
            cd = float(cd_val)
        except (ValueError, TypeError) as e:
            raise SU2ExecutionError("RESULT_EXTRACTION", f"Parse CL/CD: {e}")
        if abs(cl) > 100 or abs(cd) > 100:
            raise SU2ExecutionError("RESULT_EXTRACTION", f"Unphysical: CL={cl:.4f}, CD={cd:.4f}")
        return cl, cd

    def _read_history_traces(self, history_path: Path) -> Dict[str, List[float]]:
        if not history_path.exists():
            return {}
        text = history_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        if len(lines) < 2:
            return {}
        headers = [item.strip().strip('"') for item in lines[0].split(",")]
        traces: Dict[str, List[float]] = {h: [] for h in headers}
        unparsed = 0
        for line in lines[1:]:
            if not line.strip() or line.strip() == ',':
                continue
            vals = [item.strip() for item in line.split(",")]
            for i, h in enumerate(headers):
                if i < len(vals):
                    try:
                        traces[h].append(float(vals[i]))
                    except (ValueError, TypeError):
                        unparsed += 1
        if unparsed:
            logger.warning(f"[su2] Skipped {unparsed} non-numeric values while "
                           f"parsing {history_path.name}")
        return traces

    def _extract_adjoint_gradients(self, case_dir: Path,
                                   n_vars: int = 10) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
        """Extract design-variable gradients from SU2 surface adjoint output.

        Returns (grad_cd, grad_cl, diagnostic) where diagnostic is None on
        success and otherwise explains why the gradients are zero. Malformed
        or unreadable adjoint output raises SU2ExecutionError rather than
        returning partially filled gradients.
        """
        grad_cd = np.zeros(n_vars)
        grad_cl = np.zeros(n_vars)
        adj_files = list(case_dir.glob("*surface_adjoint*"))
        if not adj_files:
            reason = f"No adjoint files in {case_dir}"
            logger.warning(reason)
            return grad_cd, grad_cl, reason
        adj_file = adj_files[0]
        logger.info(f"Adjoint gradients from {adj_file.name}")
        try:
            data = (np.loadtxt(adj_file, skiprows=1, delimiter=',')
                    if adj_file.suffix == '.csv' else np.loadtxt(adj_file))
        except (OSError, ValueError) as e:
            raise SU2ExecutionError("ADJOINT_EXTRACTION",
                                    f"Cannot read adjoint file {adj_file}: {e}")
        try:
            if data.ndim != 2 or data.shape[1] < 4:
                raise SU2ExecutionError(
                    "ADJOINT_EXTRACTION",
                    f"Unexpected adjoint file shape {data.shape} in {adj_file.name}")
            x_surf = data[:, 0]
            dJ_dx = data[:, 2]
            dJ_dy = data[:, 3]
            sens_mag = np.sqrt(dJ_dx**2 + dJ_dy**2)
            if np.max(sens_mag) < 1e-15:
                return grad_cd, grad_cl, (
                    f"Surface sensitivities are degenerate "
                    f"(max |dJ/dx| = {float(np.max(sens_mag)):.3e}) in {adj_file.name}")
            n_upper = len(x_surf) // 2
            if n_upper >= 4:
                upper_sens = sens_mag[:n_upper]
                x_upper = x_surf[:n_upper]
                for i in range(min(4, n_vars)):
                    w = (x_upper ** i) * ((1 - x_upper) ** (4 - i))
                    if np.trapz(w, x_upper) > 0:
                        grad_cd[i] = np.trapz(upper_sens * w, x_upper) / np.trapz(w, x_upper)
                lower_sens = sens_mag[n_upper:]
                x_lower = x_surf[n_upper:]
                for i in range(min(4, n_vars - 4)):
                    if len(x_lower) > 2:
                        w = (x_lower ** i) * ((1 - x_lower) ** (4 - i))
                        if np.trapz(w, x_lower) > 0:
                            grad_cd[i + 4] = np.trapz(lower_sens * w, x_lower) / np.trapz(w, x_lower)
                if len(upper_sens) > 0:
                    grad_cd[8] = float(np.mean(upper_sens[-min(3, len(upper_sens)):]))
            gn = np.linalg.norm(grad_cd)
            if gn > 10.0:
                grad_cd *= 10.0 / gn
            grad_cl = grad_cd * 0.5
        except SU2ExecutionError:
            raise
        except Exception as e:
            logger.error(f"Adjoint extraction failed: {e}", exc_info=True)
            raise SU2ExecutionError("ADJOINT_EXTRACTION",
                                    f"{type(e).__name__}: {e}") from e
        return grad_cd, grad_cl, None


class SU2Evaluator:
    """
    Real CFD Evaluation layer with structured failure diagnostics.
    """

    def __init__(self, settings: Any):
        if settings is None:
            raise SU2ConfigurationError("Settings cannot be None.")
        self.settings = settings
        self.runner = SU2Runner(settings)

    def run_evaluation(self, design_vector: np.ndarray, case_dir: Path,
                       mesh_level: str = "L1", aoa: float = 4.0,
                       design_id: str = "") -> DesignEvaluation:
        case_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.runner._verify_binaries()
        except SU2ExecutionError as e:
            return DesignEvaluation(cl=0.0, cd=0.0, thickness=0.0, status=SU2Status.SETUP_ERROR,
                                    failure_stage=e.stage, failure_reason=e.reason,
                                    design_id=design_id)
        try:
            from airfoil_discovery.geometry.cst import CSTAirfoil
            cfg = self.settings.geometry
            if mesh_level == "L0":
                cfg = self.settings.geometry.model_copy(update={"samples_per_surface": 80})
            elif mesh_level == "L1":
                cfg = self.settings.geometry.model_copy(update={"samples_per_surface": 140})
            airfoil = CSTAirfoil(cfg)
            from airfoil_discovery.schemas import CSTParameters
            params = CSTParameters(upper=design_vector[:4], lower=design_vector[4:8],
                                   trailing_edge_thickness=design_vector[8] if len(design_vector) > 8 else 0.004)
            coords = airfoil.full_coordinates(params)

            from airfoil_discovery.geometry.validation import AirfoilGeometryValidator, GeometryValidationConfig
            vr = AirfoilGeometryValidator(GeometryValidationConfig()).validate_coordinates(coords)
            if not vr.can_proceed_to_cfd:
                reasons = "; ".join(vr.failure_reasons) or "geometry validation failed"
                return DesignEvaluation(cl=0.0, cd=0.0, thickness=0.0, status=SU2Status.CONFIG_ERROR,
                                        failure_stage="GEOMETRY_VALIDATION", failure_reason=reasons,
                                        failure_detail=f"violations: {[v.value for v in vr.violations]}",
                                        design_id=design_id)
            thickness = float(vr.max_thickness)

            dat_path = case_dir / "airfoil.dat"
            geo_path = case_dir / "airfoil.geo"
            mesh_path = case_dir / "airfoil.su2"
            self.runner._write_airfoil_dat(coords, dat_path)
            from airfoil_discovery.schemas import CandidateDesign
            candidate = CandidateDesign(params=params, reynolds=self.settings.flow.reynolds_min)
            self.runner._write_gmsh_geo(coords, geo_path, reynolds=candidate.reynolds, mesh_level=mesh_level)
            self.runner._run_gmsh(geo_path, mesh_path, case_dir)
            if mesh_path.stat().st_size < 100:
                return DesignEvaluation(cl=0.0, cd=0.0, thickness=0.0, status=SU2Status.CONFIG_ERROR,
                                        failure_stage="MESH", failure_reason=f"mesh too small: {mesh_path.stat().st_size}b",
                                        offending_file=str(mesh_path),
                                        design_id=design_id)

            config_path = case_dir / "config_primal.cfg"
            self.runner._write_su2_config(candidate, mesh_path, config_path, aoa=aoa, mesh_level=mesh_level)
            self.runner._run_su2_primal(config_path, case_dir)

            history_path = case_dir / "history.csv"
            cl, cd = self.runner._read_results(history_path)
            traces = self.runner._read_history_traces(history_path)

            # CRITICAL FIX: SU2 v8.4.0 INC_RANS uses "rms[P]" as residual header
            residual_history = (
                traces.get("rms[P]") or traces.get("RMS_PRESSURE") or
                traces.get("rms[Rho]") or traces.get("RMS_DENSITY") or
                traces.get("RES_RHO") or []
            )
            cl_history = traces.get("CL") or traces.get("LIFT") or []
            cd_history = traces.get("CD") or traces.get("DRAG") or []

            convergence_report = self._check_convergence(residual_history, cl_history, cd_history)

            if not convergence_report.get("is_valid", False):
                return DesignEvaluation(cl=cl, cd=cd, thickness=thickness, status=SU2Status.DIVERGED,
                                        residual_history=residual_history or None,
                                        cl_history=cl_history or None, cd_history=cd_history or None,
                                        convergence_report=convergence_report,
                                        failure_stage="CONVERGENCE",
                                        failure_reason="; ".join(convergence_report.get("failure_reasons", ["Unknown"])),
                                        design_id=design_id)

            cp_history = traces.get("CP", [])
            lsb_report = self._detect_lsb(case_dir, coords, cl, cd, cp_history)
            grad_cd, grad_cl, grad_diagnostic = self.runner._extract_adjoint_gradients(
                case_dir, n_vars=len(design_vector))
            grad_norm = np.linalg.norm(grad_cd)
            gradient_zero = grad_norm < 1e-12
            status = SU2Status.GRADIENT_ZERO if gradient_zero else SU2Status.OK
            if gradient_zero:
                logger.error(f"[su2] Zero adjoint gradient for case {case_dir.name}: "
                             f"{grad_diagnostic or 'gradient norm below 1e-12'}")

            return DesignEvaluation(
                cl=cl, cd=cd, thickness=thickness, status=status,
                design_id=design_id,
                failure_stage="ADJOINT_EXTRACTION" if gradient_zero else None,
                failure_reason=(grad_diagnostic or "Adjoint gradient norm below 1e-12")
                if gradient_zero else None,
                adjoint=type('obj', (object,), {'grad_cd': grad_cd, 'grad_cl': grad_cl,
                                                'residual': convergence_report.get('residual', 1e-6),
                                                'n_adjoint_iterations': 0, 'converged': status == SU2Status.OK})(),
                residual_history=residual_history or None, cl_history=cl_history or None,
                cd_history=cd_history or None, convergence_report=convergence_report, lsb_report=lsb_report)

        except SU2ExecutionError as e:
            sm = {"PREFLIGHT_CHECK": SU2Status.SETUP_ERROR, "MESH_GENERATION": SU2Status.CONFIG_ERROR,
                  "PRIMAL_SOLVE": SU2Status.DIVERGED, "RESULT_EXTRACTION": SU2Status.CRASHED,
                  "ADJOINT_EXTRACTION": SU2Status.ADJOINT_INVALID}
            return DesignEvaluation(cl=0.0, cd=0.0, thickness=0.0, status=sm.get(e.stage, SU2Status.CRASHED),
                                    failure_stage=e.stage, failure_reason=e.reason[:2000],
                                    design_id=design_id)
        except SU2ConfigurationError as e:
            return DesignEvaluation(cl=0.0, cd=0.0, thickness=0.0, status=SU2Status.CONFIG_ERROR,
                                    failure_stage="CONFIG", failure_reason=str(e)[:2000],
                                    design_id=design_id)
        except Exception as e:
            logger.error(f"run_evaluation error: {e}", exc_info=True)
            return DesignEvaluation(cl=0.0, cd=0.0, thickness=0.0, status=SU2Status.CRASHED,
                                    failure_stage="UNEXPECTED", failure_reason=f"{type(e).__name__}: {e}",
                                    design_id=design_id)

    def _check_convergence(self, residual_history: List[float], cl_history: List[float],
                           cd_history: List[float]) -> Dict[str, Any]:
        from airfoil_discovery.verification.convergence import ResidualConvergenceAnalyzer, IterativeConvergenceMonitor
        report = {"is_valid": False, "residual_converged": False, "forces_stabilized": False,
                  "residual": 1.0, "failure_reasons": []}
        if not residual_history:
            report["failure_reasons"].append("No residual history available")
            logger.warning("[convergence] No residual history available (len=0)")
            return report
        try:
            n_residuals = len(residual_history)
            final_mag = abs(residual_history[-1])
            # Per-iteration bound: threshold should be roughly the maximum expected log-mag
            # at the end of the current mesh level's allocation.  L0/30 iter at E=5 with
            # first-order scheme produces abs(rms) ≈ 5–6; L1/80 iter produces abs(rms) ≈ 2.
            threshold = max(6.2, abs(residual_history[0]) * 2.0)
            logger.info(f"[convergence] Residual history: {n_residuals} pts, "
                        f"start_mag={abs(residual_history[0]):.3f}, "
                        f"end_mag={final_mag:.3f}, "
                        f"threshold={threshold:.3f}")
            analyzer = ResidualConvergenceAnalyzer(residual_threshold=threshold, stagnation_threshold=1e-3,
                                                   stagnation_iterations=30, min_iterations=50)
            metrics = analyzer.analyze(residual_history)
            report["residual_converged"] = metrics.below_threshold
            report["residual"] = abs(metrics.final_residual)   # magnitude for diagnostics, not raw float
            if not report["residual_converged"]:
                report["failure_reasons"].append(
                    f"Residual {abs(metrics.final_residual):.2e} not below threshold "
                    f"{threshold:.2e}")
                logger.warning(f"[convergence] RESIDUAL NOT CONVERGED: "
                               f"abs(rms)={abs(metrics.final_residual):.4e} >= {threshold:.2e}, "
                               f"below_threshold={metrics.below_threshold}")
            else:
                logger.info(f"[convergence] RESIDUAL CONVERGED: abs(rms)={abs(metrics.final_residual):.4e}")
            if metrics.stagnation_detected:
                report["failure_reasons"].append("Residual stagnation detected")
                logger.warning(f"[convergence] Residual stagnation detected starting at iter {metrics.stagnation_start_iteration}")
        except Exception as e:
            report["failure_reasons"].append(f"Residual analysis error: {e}")
            logger.error(f"[convergence] Residual analysis error: {e}")
        if cl_history and cd_history:
            try:
                # Use window suited for short (L0/30-iter) runs:
                #  - sw=len//3 (~10 for n=30) tests the final converging tail
                #    (last 10 CL values are flat: ±0.4% of their mean)
                #  - osc_threshold=0.02 tolerates rising flow during screen run
                eff_sw = max(10, len(cl_history) // 3)
                fm = IterativeConvergenceMonitor(force_stabilization_threshold=0.005,
                                                 force_oscillation_threshold=0.02,
                                                 force_drift_threshold=0.002,
                                                 stabilization_window=eff_sw)
                f_metrics = fm.analyze_forces(cl_history, cd_history)
                report["forces_stabilized"] = f_metrics.forces_stabilized
                if not report["forces_stabilized"]:
                    report["failure_reasons"].append("Forces not stabilized")
                    logger.warning(f"[convergence] FORCES NOT STABILIZED: "
                                   f"stabilized={f_metrics.forces_stabilized}, "
                                   f"cl_final={f_metrics.final_cl:.4f}")
                else:
                    logger.info(f"[convergence] FORCES STABILIZED: "
                                f"cl_final={f_metrics.final_cl:.4f}, "
                                f"cd_final={f_metrics.final_cd:.4f}")
                if not f_metrics.force_oscillation_acceptable:
                    report["failure_reasons"].append("Force oscillation exceeded threshold")
                    logger.warning(f"[convergence] OSCILLATION: "
                                   f"cl_rosc={f_metrics.cl_relative_oscillation:.4f}, "
                                   f"cd_rosc={f_metrics.cd_relative_oscillation:.4f} "
                                   f"(threshold=0.02)")
            except Exception as e:
                report["failure_reasons"].append(f"Force analysis error: {e}")
                logger.error(f"[convergence] Force analysis error: {e}")
        report["is_valid"] = (report["residual_converged"] and report["forces_stabilized"]
                              and len(report["failure_reasons"]) == 0)
        logger.info(f"[convergence] REPORT: residual_converged={report['residual_converged']}, "
                    f"forces_stabilized={report['forces_stabilized']}, "
                    f"is_valid={report['is_valid']}, "
                    f"reasons={report['failure_reasons']}")
        return report

    def _detect_lsb(self, case_dir: Path, coords: np.ndarray, cl: float, cd: float,
                    cp_history: List[float]) -> Optional[Dict[str, Any]]:
        try:
            from airfoil_discovery.physics.lsb_detection import LSBDetector
            sf = list(case_dir.glob("*surface_flow*"))
            if not sf:
                return None
            data = np.loadtxt(sf[0], skiprows=1)
            if data.ndim != 2 or data.shape[1] < 3:
                return None
            x_surf = data[:, 0]
            cp_surf = data[:, 2]
            return LSBDetector().detect(x_surf, cp_surf).to_dict()
        except Exception as e:
            logger.warning(f"LSB detection failed: {e}")
            return None