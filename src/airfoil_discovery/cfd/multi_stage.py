from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from airfoil_discovery.cfd.su2_config import write_stage_config
from airfoil_discovery.schemas import CandidateDesign


@dataclass(slots=True)
class StageResult:
    stage: int
    mesh_path: Path
    config_path: Path
    restart_path: Path
    cl: float
    cd: float
    converged: bool
    wall_clock_s: float
    iter_count: int
    restart_source_path: str | None = None
    residual_final: float | None = None
    coefficient_variation: float | None = None


@dataclass(slots=True)
class MultiStageResult:
    stages: list[StageResult]
    final_cl: float
    final_cd: float
    converged: bool
    manifest_path: Path
    surface_path: Path | None = None


@dataclass(slots=True)
class MISResult:
    mesh_levels: list[str]
    node_counts: list[int]
    cl_values: list[float]
    cd_values: list[float]
    gci_cl: float
    gci_cd: float
    mesh_dependent: bool
    mesh_independence_penalty: float


@dataclass(slots=True)
class UQResult:
    parent_case_key: str
    scenarios: list[dict[str, float | str | None]]
    cv_cl: float
    cv_cd: float
    numerically_sensitive: bool


class StageFailureError(RuntimeError):
    def __init__(self, stage: int, aoa: float, candidate_signature: str, run_dir: Path):
        super().__init__(f"Stage {stage} failed for AoA={aoa} candidate={candidate_signature}")
        self.stage = stage
        self.aoa = aoa
        self.candidate_signature = candidate_signature
        self.run_dir = run_dir


def _read_history(history_path: Path) -> list[dict[str, float]]:
    if not history_path.exists():
        return []
    lines = [line.strip().replace('"', "") for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    headers = [col.strip() for col in lines[0].split(",")]
    rows: list[dict[str, float]] = []
    for line in lines[1:]:
        values = [value.strip() for value in line.split(",")]
        rows.append({header: float(value) for header, value in zip(headers, values) if value})
    return rows


def _check_convergence_coefficients(history: list[dict[str, float]], window: int = 100, tol: float = 0.005) -> tuple[bool, float]:
    if len(history) < window:
        return False, math.inf
    tail = history[-window:]
    cls = np.array([row["CL"] for row in tail], dtype=float)
    cds = np.array([row["CD"] for row in tail], dtype=float)
    cl_var = np.ptp(cls) / (abs(np.mean(cls)) + 1e-12)
    cd_var = np.ptp(cds) / (abs(np.mean(cds)) + 1e-12)
    metric = float(max(cl_var, cd_var))
    return metric < tol, metric


def _check_convergence_residual(history: list[dict[str, float]], required_drop: float = 6.0) -> tuple[bool, float]:
    if len(history) < 2:
        return False, 0.0
    residual_key = "RMS_RES" if "RMS_RES" in history[0] else next((key for key in history[0] if "RMS" in key), None)
    if residual_key is None:
        return False, 0.0
    initial = max(history[0][residual_key], 1e-30)
    final = max(history[-1][residual_key], 1e-30)
    drop = math.log10(initial / final)
    return drop >= required_drop, float(final)


def compute_gci(f1: float, f2: float, f3: float, r: float, safety_factor: float = 1.25) -> float:
    if abs(f2 - f1) < 1e-12 or abs(f3 - f2) < 1e-12:
        return 0.0
    p = math.log(abs((f3 - f2) / (f2 - f1))) / math.log(r)
    if abs(r**p - 1.0) < 1e-12:
        return 0.0
    f_exact = f1 + (f1 - f2) / (r**p - 1.0)
    if abs(f_exact) < 1e-12:
        return 0.0
    return safety_factor * abs(f1 - f_exact) / abs(f_exact)


class MultiStageRunner:
    def __init__(
        self,
        settings: Any,
        *,
        airfoil: Any,
        mesh_writer: Callable[[CandidateDesign, Path, float], Path],
        su2_runner: Callable[[Path, Path, str], None],
        coefficient_reader: Callable[[Path], tuple[float, float]],
        history_resolver: Callable[[Path], Path],
    ):
        self.settings = settings
        self.airfoil = airfoil
        self.mesh_writer = mesh_writer
        self.su2_runner = su2_runner
        self.coefficient_reader = coefficient_reader
        self.history_resolver = history_resolver

    def run_aoa(self, candidate: CandidateDesign, base_dir: Path, aoa: float) -> MultiStageResult:
        signature = candidate.params.rounded_signature(self.settings.optimization.duplicate_rounding)
        stages: list[StageResult] = []
        restart_source: Path | None = None
        for stage, coarse_factor in [
            (1, self.settings.solver.stage1_coarse_factor),
            (2, self.settings.solver.stage2_coarse_factor),
            (3, self.settings.solver.stage3_coarse_factor),
        ]:
            stage_dir = base_dir / f"stage_{stage}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            mesh_path = self.mesh_writer(candidate, stage_dir, coarse_factor)
            config_path = stage_dir / "config.cfg"
            write_stage_config(stage, candidate, mesh_path, config_path, aoa, self.settings, restart_source)
            started = time.time()
            self.su2_runner(stage_dir, config_path, f"stage{stage}")
            wall_clock = max(0.0, time.time() - started)
            restart_path = self._resolve_restart(stage_dir)
            if not restart_path.exists() or restart_path.stat().st_size == 0:
                raise StageFailureError(stage, aoa, signature, stage_dir)
            history_path = self.history_resolver(stage_dir)
            cl, cd = self.coefficient_reader(history_path)
            history = _read_history(history_path)
            coeff_ok, coeff_metric = _check_convergence_coefficients(
                history, self.settings.solver.convergence_window, self.settings.solver.convergence_cl_cd_tol
            )
            resid_ok, residual_final = _check_convergence_residual(
                history, self.settings.solver.convergence_residual_drop
            )
            stages.append(
                StageResult(
                    stage=stage,
                    mesh_path=mesh_path,
                    config_path=config_path,
                    restart_path=restart_path,
                    cl=cl,
                    cd=cd,
                    converged=bool(coeff_ok and resid_ok),
                    wall_clock_s=wall_clock,
                    iter_count=getattr(self.settings.solver, f"stage{stage}_iter"),
                    restart_source_path=str(restart_source) if restart_source else None,
                    residual_final=residual_final,
                    coefficient_variation=coeff_metric,
                )
            )
            restart_source = restart_path
        manifest_path = base_dir / "stage_manifest.json"
        manifest = {
            "candidate_signature": signature,
            "aoa": aoa,
            "stages": [
                {
                    **asdict(stage_result),
                    "mesh_path": str(stage_result.mesh_path),
                    "config_path": str(stage_result.config_path),
                    "restart_path": str(stage_result.restart_path),
                }
                for stage_result in stages
            ],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        final = stages[-1]
        return MultiStageResult(
            stages=stages,
            final_cl=final.cl,
            final_cd=final.cd,
            converged=final.converged,
            manifest_path=manifest_path,
            surface_path=self._resolve_surface(base_dir / "stage_3"),
        )

    @staticmethod
    def _resolve_restart(stage_dir: Path) -> Path:
        for name in ("restart_flow.dat", "solution_flow.dat"):
            candidate = stage_dir / name
            if candidate.exists():
                return candidate
        return stage_dir / "restart_flow.dat"

    @staticmethod
    def _resolve_surface(stage_dir: Path) -> Path | None:
        patterns = [
            "surface_flow.vtu",
            "surface_flow_*.vtu",
            "*.vtu",
            "surface_flow.vtk",
        ]
        for pattern in patterns:
            matches = sorted(stage_dir.glob(pattern))
            if matches:
                return matches[-1]
        return None


class MeshIndependenceStudy:
    def __init__(self, runner: Callable[[CandidateDesign, Path, float, float], tuple[int, float, float]]):
        self.runner = runner

    def run(self, candidate: CandidateDesign, base_dir: Path, aoa: float) -> MISResult:
        levels = [("baseline", 1.0), ("1.5x", 1.0 / 1.5), ("2.0x", 0.5)]
        node_counts: list[int] = []
        cls: list[float] = []
        cds: list[float] = []
        for name, coarse_factor in levels:
            node_count, cl, cd = self.runner(candidate, base_dir / name, aoa, coarse_factor)
            node_counts.append(node_count)
            cls.append(cl)
            cds.append(cd)
        gci_cl = compute_gci(cls[0], cls[1], cls[2], 1.5)
        gci_cd = compute_gci(cds[0], cds[1], cds[2], 1.5)
        mesh_dependent = gci_cl > 0.02 or gci_cd > 0.05
        return MISResult(
            mesh_levels=[name for name, _ in levels],
            node_counts=node_counts,
            cl_values=cls,
            cd_values=cds,
            gci_cl=gci_cl,
            gci_cd=gci_cd,
            mesh_dependent=mesh_dependent,
            mesh_independence_penalty=2.0 if mesh_dependent else 0.0,
        )


class UQEngine:
    def __init__(self, rerun_case: Callable[[Any, str], dict[str, float | None]]):
        self.rerun_case = rerun_case

    def run_top_candidates(self, candidates: list[Any], database: Any) -> list[UQResult]:
        top = sorted(candidates, key=lambda item: float(item["score"]), reverse=True)[:5]
        scenario_names = ["baseline", "finer_mesh", "first_order_time", "tu_minus_50", "tu_plus_50"]
        results: list[UQResult] = []
        for candidate in top:
            rows = [self.rerun_case(candidate, scenario) for scenario in scenario_names]
            cls = np.array([float(row["cl"]) for row in rows], dtype=float)
            cds = np.array([float(row["cd"]) for row in rows], dtype=float)
            cv_cl = float(np.std(cls) / max(abs(np.mean(cls)), 1e-12))
            cv_cd = float(np.std(cds) / max(abs(np.mean(cds)), 1e-12))
            sensitive = cv_cl > 0.05 or cv_cd > 0.10
            if hasattr(database, "insert_uq_result"):
                database.insert_uq_result(candidate["case_key"], rows, cv_cl, cv_cd, sensitive)
            results.append(
                UQResult(
                    parent_case_key=str(candidate["case_key"]),
                    scenarios=rows,
                    cv_cl=cv_cl,
                    cv_cd=cv_cd,
                    numerically_sensitive=sensitive,
                )
            )
        return results
