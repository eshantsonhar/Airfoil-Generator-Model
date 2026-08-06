"""
Flow Diagnostics and LSB Characterization Module.

Provides utilities to parse surface flow solutions from SU2 (surface_flow.csv)
and perform automated extraction of:
  - Global aerodynamic metrics (CL, CD, CM, L/D)
  - Laminar Separation Bubble (LSB) coordinates via Cf-based detection
  - Pressure coefficient (Cp) and skin friction (Cf) distributions
  - Baseline vs. optimized comparisons

LSB extraction follows the standard approach:
  - Separation: Cf crosses from positive to negative
  - Transition zone: negative Cf region
  - Reattachment: Cf crosses from negative back to positive
  - Bubble length: L_sep = x_reattach - x_separate
"""

from __future__ import annotations

import numpy as np
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

from airfoil_discovery.cfd.su2_csv import last_row_mapping, lookup_float, read_csv_table

logger = logging.getLogger(__name__)


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class SurfaceFlowData:
    """Parsed surface flow data from SU2 surface_flow.csv."""
    x: np.ndarray
    y: np.ndarray
    cp: np.ndarray
    cf: np.ndarray
    pressure: np.ndarray
    density: np.ndarray
    velocity: np.ndarray
    mach: np.ndarray
    temperature: np.ndarray
    n_nodes: int = 0
    has_upper_lower_split: bool = False
    upper_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    lower_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))

    @classmethod
    def from_csv(cls, filepath: Path) -> "SurfaceFlowData":
        """Parse SU2 surface_flow.csv file."""
        if not filepath.exists():
            raise FileNotFoundError(f"Surface flow file not found: {filepath}")
        try:
            data = np.loadtxt(filepath, delimiter=",", skiprows=1)
        except Exception:
            data = np.loadtxt(filepath, skiprows=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        x = data[:, 0].copy()
        y = data[:, 1].copy()
        n_cols = data.shape[1]
        cp = data[:, 8].copy() if n_cols > 8 else np.zeros_like(x)
        cf = data[:, 9].copy() if n_cols > 9 else np.zeros_like(x)
        pressure = data[:, 3].copy() if n_cols > 3 else np.zeros_like(x)
        temperature = data[:, 4].copy() if n_cols > 4 else np.zeros_like(x)
        density = data[:, 5].copy() if n_cols > 5 else np.zeros_like(x)
        velocity = data[:, 6].copy() if n_cols > 6 else np.zeros_like(x)
        mach = data[:, 7].copy() if n_cols > 7 else np.zeros_like(x)
        obj = cls(
            x=x, y=y, cp=cp, cf=cf,
            pressure=pressure, density=density,
            velocity=velocity, mach=mach, temperature=temperature,
            n_nodes=len(x),
        )
        obj._detect_upper_lower_split()
        return obj

    def _detect_upper_lower_split(self) -> None:
        """Detect upper/lower surface split from surface coordinates."""
        if self.n_nodes < 4:
            return
        le_idx = int(np.argmin(self.x))
        n = self.n_nodes
        upper = np.zeros(n, dtype=bool)
        lower = np.zeros(n, dtype=bool)
        upper[:le_idx + 1] = True
        lower[le_idx:] = True
        lower[le_idx] = False
        if np.mean(self.y[upper]) < 0:
            upper, lower = lower, upper
            lower[le_idx] = False
        self.upper_indices = upper
        self.lower_indices = lower
        self.has_upper_lower_split = True

    @property
    def x_upper(self) -> np.ndarray:
        return self.x[self.upper_indices]

    @property
    def y_upper(self) -> np.ndarray:
        return self.y[self.upper_indices]

    @property
    def cp_upper(self) -> np.ndarray:
        return self.cp[self.upper_indices]

    @property
    def cf_upper(self) -> np.ndarray:
        return self.cf[self.upper_indices]

    @property
    def x_lower(self) -> np.ndarray:
        return self.x[self.lower_indices]

    @property
    def y_lower(self) -> np.ndarray:
        return self.y[self.lower_indices]

    @property
    def cp_lower(self) -> np.ndarray:
        return self.cp[self.lower_indices]

    @property
    def cf_lower(self) -> np.ndarray:
        return self.cf[self.lower_indices]


@dataclass
class LSBResult:
    """LSB extraction result."""
    lsb_detected: bool
    separation_point: Optional[float]
    reattachment_point: Optional[float]
    bubble_length: Optional[float]
    transition_onset: Optional[float]
    bubble_center: Optional[float]
    cf_min: Optional[float]
    cf_min_location: Optional[float]
    plateau_start: Optional[float]
    plateau_end: Optional[float]
    plateau_length: Optional[float]
    warnings: List[str] = field(default_factory=list)


@dataclass
class AerodynamicMetrics:
    """Global aerodynamic performance metrics."""
    cl: float
    cd: float
    cm: float
    efficiency: float = 0.0
    cd_pressure: float = 0.0
    cd_friction: float = 0.0
    lsd_detected: bool = False
    lsb: Optional[LSBResult] = None
    flow_separated: bool = False
    max_cp: float = 0.0
    min_cp: float = 0.0

    def __post_init__(self) -> None:
        if self.cd > 0:
            self.efficiency = self.cl / self.cd


# ── Parsing ────────────────────────────────────────────────────────────────────

def parse_surface_flow(filepath: Path) -> SurfaceFlowData:
    """Parse SU2 surface flow CSV into structured data."""
    return SurfaceFlowData.from_csv(filepath)


# ── LSB Detection ──────────────────────────────────────────────────────────────

def extract_lsb_from_cf(
    x: np.ndarray,
    cf: np.ndarray,
    min_bubble_length: float = 0.01,
    cf_threshold: float = 0.0,
) -> LSBResult:
    """
    Detect LSB from skin friction (Cf) distribution on upper surface.

    Physical model:
      - Laminar region: Cf > 0 (attached flow)
      - Separation: Cf crosses below 0
      - Separated region: Cf < 0 (reverse flow, LSB)
      - Reattachment: Cf crosses above 0 (turbulent reattachment)
    """
    warnings = []
    n = len(x)
    signs = np.sign(cf - cf_threshold)

    crossings = []
    for i in range(1, n):
        if signs[i] != signs[i-1] and signs[i-1] != 0:
            x0, x1 = x[i-1], x[i]
            cf0, cf1 = cf[i-1], cf[i]
            if abs(cf1 - cf0) > 1e-15:
                frac = (cf_threshold - cf0) / (cf1 - cf0)
                x_cross = x0 + frac * (x1 - x0)
                crossings.append({
                    "x": float(x_cross),
                    "direction": "pos_to_neg" if signs[i-1] > 0 else "neg_to_pos",
                })

    if len(crossings) < 2:
        return LSBResult(
            lsb_detected=False,
            separation_point=None,
            reattachment_point=None,
            bubble_length=None,
            transition_onset=None,
            bubble_center=None,
            cf_min=float(np.min(cf)),
            cf_min_location=float(x[np.argmin(cf)]),
            plateau_start=None, plateau_end=None, plateau_length=None,
            warnings=["No Cf zero-crossings detected"],
        )

    separation = None
    reattachment = None
    for c in crossings:
        if c["direction"] == "pos_to_neg" and separation is None:
            separation = c["x"]
        elif c["direction"] == "neg_to_pos" and separation is not None and reattachment is None:
            reattachment = c["x"]
            break

    if separation is not None and reattachment is None:
        warnings.append("Open separation - Cf remains negative toward TE")
        reattachment = 1.0

    bubble_length = None
    bubble_center = None
    if separation is not None and reattachment is not None:
        bubble_length = reattachment - separation
        bubble_center = 0.5 * (separation + reattachment)

    lsb_detected = False
    if bubble_length is not None and bubble_length >= min_bubble_length:
        lsb_detected = True
    elif bubble_length is not None and bubble_length < min_bubble_length:
        warnings.append(f"Bubble length {bubble_length:.4f} < min {min_bubble_length}")

    transition_onset = None
    if separation is not None and reattachment is not None:
        in_bubble = (x >= separation) & (x <= reattachment)
        if np.any(in_bubble):
            cf_min_idx = np.argmin(cf[in_bubble])
            transition_onset = float(x[in_bubble][cf_min_idx])

    cf_min = float(np.min(cf))
    cf_min_loc = float(x[np.argmin(cf)])

    return LSBResult(
        lsb_detected=lsb_detected,
        separation_point=separation,
        reattachment_point=reattachment,
        bubble_length=bubble_length,
        transition_onset=transition_onset,
        bubble_center=bubble_center,
        cf_min=cf_min,
        cf_min_location=cf_min_loc,
        plateau_start=None, plateau_end=None, plateau_length=None,
        warnings=warnings,
    )


def extract_lsb_from_cp(
    x: np.ndarray,
    cp: np.ndarray,
    plateau_threshold: float = 0.3,
    min_plateau_length: float = 0.02,
) -> LSBResult:
    """
    Detect LSB from pressure coefficient (Cp) plateau on upper surface.

    An LSB manifests as a region of nearly constant Cp (pressure plateau)
    followed by rapid pressure recovery.
    """
    warnings = []
    dcp_dx = np.gradient(cp, x)
    plateau_mask = np.abs(dcp_dx) < plateau_threshold

    regions = []
    in_plateau = False
    start_idx = 0
    for i in range(len(x)):
        if plateau_mask[i] and not in_plateau:
            in_plateau = True
            start_idx = i
        elif not plateau_mask[i] and in_plateau:
            in_plateau = False
            length = x[i] - x[start_idx]
            if length >= min_plateau_length:
                regions.append((start_idx, i, length))
    if in_plateau:
        length = x[-1] - x[start_idx]
        if length >= min_plateau_length:
            regions.append((start_idx, len(x) - 1, length))

    if not regions:
        return LSBResult(
            lsb_detected=False, separation_point=None, reattachment_point=None,
            bubble_length=None, transition_onset=None, bubble_center=None,
            cf_min=None, cf_min_location=None,
            plateau_start=None, plateau_end=None, plateau_length=None,
            warnings=["No Cp plateau detected"],
        )

    valid_regions = [(s, e, l) for s, e, l in regions if x[s] < 0.6]
    if not valid_regions:
        valid_regions = regions
    best = max(valid_regions, key=lambda r: r[2])
    plateau_start = float(x[best[0]])
    plateau_end = float(x[best[1]])
    plateau_length = float(best[2])

    separation = plateau_start
    reattachment = plateau_end
    bubble_length = plateau_length
    lsb_detected = plateau_length >= min_plateau_length
    bubble_center = 0.5 * (plateau_start + plateau_end)

    transition_onset = None
    dcp_dx_after = dcp_dx[best[1]:] if best[1] < len(dcp_dx) else np.array([])
    if len(dcp_dx_after) > 0:
        for i in range(len(dcp_dx_after)):
            if dcp_dx_after[i] < -1.0:
                idx = best[1] + i
                if idx < len(x):
                    transition_onset = float(x[idx])
                    break

    return LSBResult(
        lsb_detected=lsb_detected,
        separation_point=separation,
        reattachment_point=reattachment,
        bubble_length=bubble_length,
        transition_onset=transition_onset,
        bubble_center=bubble_center,
        cf_min=None, cf_min_location=None,
        plateau_start=plateau_start, plateau_end=plateau_end,
        plateau_length=plateau_length,
        warnings=warnings,
    )


# ── Aerodynamic Metrics ────────────────────────────────────────────────────────

def compute_aerodynamic_metrics(
    history_file: Optional[Path] = None,
    surface_file: Optional[Path] = None,
    cl: Optional[float] = None,
    cd: Optional[float] = None,
) -> AerodynamicMetrics:
    """Compute aerodynamic metrics from SU2 history and/or surface flow files."""
    if history_file is not None and history_file.exists():
        cl_val, cd_val, cm_val = _parse_aero_from_history(history_file)
    else:
        cl_val, cd_val, cm_val = cl or 0.0, cd or 0.0, 0.0

    metrics = AerodynamicMetrics(cl=cl_val, cd=cd_val, cm=cm_val)

    if surface_file is not None and surface_file.exists():
        try:
            sf = SurfaceFlowData.from_csv(surface_file)
            if sf.has_upper_lower_split:
                lsb_cf = extract_lsb_from_cf(sf.x_upper, sf.cf_upper)
                metrics.lsb = lsb_cf
                metrics.lsd_detected = lsb_cf.lsb_detected
                metrics.flow_separated = lsb_cf.separation_point is not None
                metrics.min_cp = float(np.min(sf.cp_upper))
                metrics.max_cp = float(np.max(sf.cp))
        except Exception as e:
            logger.warning(f"Surface flow analysis failed: {e}")

    return metrics


def _parse_aero_from_history(history_path: Path) -> Tuple[float, float, float]:
    """Parse final CL, CD, CMz from SU2 history.csv."""
    try:
        headers, rows = read_csv_table(history_path)
    except Exception:
        return 0.0, 0.0, 0.0
    mapping = last_row_mapping(headers, rows)
    if mapping is None:
        return 0.0, 0.0, 0.0
    cl = lookup_float(mapping, ["CL", "LIFT"], 0.0)
    cd = lookup_float(mapping, ["CD", "DRAG"], 0.0)
    cm = lookup_float(mapping, ["CMz", "CM", "MOMENT"], 0.0)
    return cl, cd, cm


def compare_baseline_optimized(
    baseline_metrics: AerodynamicMetrics,
    optimized_metrics: AerodynamicMetrics,
) -> Dict[str, Any]:
    """Compare aerodynamic performance between baseline and optimized designs."""
    delta_cl = optimized_metrics.cl - baseline_metrics.cl
    delta_cd = optimized_metrics.cd - baseline_metrics.cd
    delta_eff = optimized_metrics.efficiency - baseline_metrics.efficiency

    lsb_improved = None
    lsb_reduction = None
    if baseline_metrics.lsb is not None and optimized_metrics.lsb is not None:
        bl = baseline_metrics.lsb.bubble_length or 0
        ol = optimized_metrics.lsb.bubble_length or 0
        if bl > 0:
            lsb_reduction = (bl - ol) / bl * 100
            lsb_improved = lsb_reduction > 0

    return {
        "delta_cl": delta_cl,
        "delta_cd": delta_cd,
        "delta_efficiency": delta_eff,
        "cd_reduction_percent": -delta_cd / baseline_metrics.cd * 100 if baseline_metrics.cd > 0 else 0,
        "cl_improvement_percent": delta_cl / baseline_metrics.cl * 100 if baseline_metrics.cl > 0 else 0,
        "efficiency_improvement_percent": delta_eff / baseline_metrics.efficiency * 100
        if baseline_metrics.efficiency > 0 else 0,
        "lsb_reduction_percent": lsb_reduction,
        "lsb_improved": lsb_improved,
        "baseline": {
            "cl": baseline_metrics.cl, "cd": baseline_metrics.cd,
            "efficiency": baseline_metrics.efficiency,
            "lsb_detected": baseline_metrics.lsd_detected,
            "lsb_length": baseline_metrics.lsb.bubble_length if baseline_metrics.lsb else None,
        },
        "optimized": {
            "cl": optimized_metrics.cl, "cd": optimized_metrics.cd,
            "efficiency": optimized_metrics.efficiency,
            "lsb_detected": optimized_metrics.lsd_detected,
            "lsb_length": optimized_metrics.lsb.bubble_length if optimized_metrics.lsb else None,
        },
    }