from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class CSTParameters:
    upper: np.ndarray
    lower: np.ndarray
    trailing_edge_thickness: float

    def as_vector(self) -> np.ndarray:
        return np.concatenate(
            [self.upper.astype(float), self.lower.astype(float), np.array([self.trailing_edge_thickness])]
        )

    def rounded_signature(self, decimals: int = 5) -> str:
        values = np.round(self.as_vector(), decimals=decimals)
        return ",".join(f"{value:.{decimals}f}" for value in values)


@dataclass(slots=True)
class GeometryMetrics:
    max_thickness: float
    max_camber: float
    leading_edge_radius: float
    smoothness_score: float
    curvature_spike: float
    prior_score: float
    is_valid: bool
    rejection_reason: str | None = None


@dataclass(slots=True)
class CandidateDesign:
    params: CSTParameters
    reynolds: float
    geometry_metrics: GeometryMetrics | None = None
    surrogate_prediction: dict[str, float] | None = None
    acquisition_score: float | None = None


@dataclass(slots=True)
class PolarPoint:
    aoa_deg: float
    cl: float
    cd: float
    converged: bool = True
    design_id: str = ""  # INSTRUMENTATION: which design was evaluated

    @property
    def efficiency(self) -> float:
        return self.cl / max(self.cd, 1.0e-8)


@dataclass(slots=True)
class TransitionPhysics:
    aoa_deg: float
    x_tr: float | None
    x_sep: float | None
    x_reat: float | None
    bubble_length: float
    cp_min: float
    x_cp_min: float
    lsb_detected: bool
    transition_inconsistent: bool
    unrealistic_early_transition: bool
    fully_laminar: bool
    physics_violation_penalty: float


@dataclass(slots=True)
class SimulationResult:
    candidate: CandidateDesign
    polar: list[PolarPoint]
    score: float
    stall_angle_deg: float
    cd_at_cruise: float
    separation_penalty: float
    instability_penalty: float
    archive_url: str | None = None
    local_run_dir: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    # INSTRUMENTATION: identity tracking
    evaluated_design_id: str = ""
    stored_geometry_id: str = ""
    flags: list[str] = field(default_factory=list)
