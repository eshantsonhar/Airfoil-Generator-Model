from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import meshio
import numpy as np


class ExtractionError(RuntimeError):
    pass


@dataclass(slots=True)
class SurfaceDistributions:
    aoa: float
    upper_x: np.ndarray
    upper_cp: np.ndarray
    upper_cf: np.ndarray
    upper_gamma: np.ndarray
    lower_x: np.ndarray
    lower_cp: np.ndarray
    lower_cf: np.ndarray
    lower_gamma: np.ndarray
    x_tr: float | None
    x_sep: float | None
    x_reat: float | None
    bubble_length: float
    cp_min: float
    x_cp_min: float


class PhysicsExtractor:
    CP_NAMES = ("Pressure_Coefficient", "Cp", "PRESSURE_COEFFICIENT")
    GAMMA_NAMES = ("Intermittency", "Gamma", "Turbulent_Intermittency")
    CF_VECTOR_NAMES = ("Skin_Friction_Coefficient", "Skin_Friction_Coefficient_Vector")
    CF_X_NAMES = ("Skin_Friction_Coefficient_x", "Cf_x")
    CF_Y_NAMES = ("Skin_Friction_Coefficient_y", "Cf_y")

    def extract(self, vtu_path: Path, aoa: float) -> SurfaceDistributions:
        if not vtu_path.exists():
            raise ExtractionError(f"Surface file not found: {vtu_path}")
        if vtu_path.suffix.lower() == ".json":
            data = json.loads(vtu_path.read_text(encoding="utf-8"))
            x = np.asarray(data["x"], dtype=float)
            y = np.asarray(data["y"], dtype=float)
            cp = np.asarray(data["cp"], dtype=float)
            cf = np.asarray(data["cf"], dtype=float)
            gamma = np.asarray(data.get("gamma", np.zeros_like(x)), dtype=float)
        else:
            x, y, cp, cf, gamma = self._read_vtu(vtu_path)
        upper_mask = y >= 0.0
        lower_mask = ~upper_mask
        upper = self._prepare_side(x[upper_mask], cp[upper_mask], cf[upper_mask], gamma[upper_mask])
        lower = self._prepare_side(x[lower_mask], cp[lower_mask], cf[lower_mask], gamma[lower_mask])
        x_tr = self._first_location(upper["x"], upper["gamma"] > 0.1)
        x_sep = self._first_location(upper["x"], upper["cf"] < 0.0)
        x_reat = None
        if x_sep is not None:
            downstream = upper["x"] > x_sep
            x_reat = self._first_location(upper["x"][downstream], upper["cf"][downstream] > 0.0)
        bubble_length = float(x_reat - x_sep) if x_sep is not None and x_reat is not None else 0.0
        cp_min_idx = int(np.argmin(upper["cp"]))
        return SurfaceDistributions(
            aoa=aoa,
            upper_x=upper["x"],
            upper_cp=upper["cp"],
            upper_cf=upper["cf"],
            upper_gamma=upper["gamma"],
            lower_x=lower["x"],
            lower_cp=lower["cp"],
            lower_cf=lower["cf"],
            lower_gamma=lower["gamma"],
            x_tr=x_tr,
            x_sep=x_sep,
            x_reat=x_reat,
            bubble_length=bubble_length,
            cp_min=float(upper["cp"][cp_min_idx]),
            x_cp_min=float(upper["x"][cp_min_idx]),
        )

    def write_distributions(self, distributions: list[SurfaceDistributions], output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() not in {".h5", ".hdf5"}:
            output_path = output_path.with_suffix(".h5")
        with h5py.File(output_path, "w") as handle:
            for dist in distributions:
                group = handle.create_group(f"aoa_{dist.aoa:+06.2f}".replace(".", "p"))
                group.attrs["aoa"] = dist.aoa
                for key, value in dist.__dict__.items():
                    if isinstance(value, np.ndarray):
                        group.create_dataset(key, data=value)
                    elif value is None:
                        group.attrs[key] = np.nan
                    else:
                        group.attrs[key] = value
        return output_path

    def read_distributions(self, path: Path) -> dict[str, Any]:
        if path.suffix.lower() in {".h5", ".hdf5"}:
            payload: dict[str, Any] = {}
            with h5py.File(path, "r") as handle:
                for key, group in handle.items():
                    payload[key] = {
                        name: group[name][:].tolist() for name in group.keys()
                    }
                    for attr_name, attr_val in group.attrs.items():
                        payload[key][attr_name] = None if isinstance(attr_val, float) and np.isnan(attr_val) else attr_val
            return payload
        return json.loads(path.read_text(encoding="utf-8"))

    def _read_vtu(self, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mesh = meshio.read(path)
        if mesh.points.size == 0:
            raise ExtractionError(f"No points found in VTU: {path}")
        x = np.asarray(mesh.points[:, 0], dtype=float)
        y = np.asarray(mesh.points[:, 1], dtype=float)
        point_data = mesh.point_data
        cp = self._get_array(point_data, self.CP_NAMES, path)
        gamma = self._get_array(point_data, self.GAMMA_NAMES, path, default=np.zeros_like(cp))
        cf = self._extract_cf(point_data, cp, x, y)
        return x, y, cp, cf, gamma

    def _extract_cf(
        self,
        point_data: dict[str, Any],
        cp: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        vector = self._get_optional_array(point_data, self.CF_VECTOR_NAMES)
        if vector is not None:
            vector = np.asarray(vector, dtype=float)
            tangent = self._surface_tangent(x, y)
            return np.sum(vector[:, :2] * tangent, axis=1)
        cf_x = self._get_optional_array(point_data, self.CF_X_NAMES)
        cf_y = self._get_optional_array(point_data, self.CF_Y_NAMES)
        if cf_x is not None and cf_y is not None:
            tangent = self._surface_tangent(x, y)
            return np.asarray(cf_x, dtype=float) * tangent[:, 0] + np.asarray(cf_y, dtype=float) * tangent[:, 1]
        if cf_x is not None:
            return np.asarray(cf_x, dtype=float)
        raise ExtractionError("Required skin-friction fields were not found in VTU output")

    def _surface_tangent(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        dx = np.gradient(x)
        dy = np.gradient(y)
        mag = np.maximum(np.sqrt(dx**2 + dy**2), 1e-12)
        return np.column_stack([dx / mag, dy / mag])

    def _get_array(
        self,
        point_data: dict[str, Any],
        names: tuple[str, ...],
        path: Path,
        default: np.ndarray | None = None,
    ) -> np.ndarray:
        arr = self._get_optional_array(point_data, names)
        if arr is None:
            if default is not None:
                return default
            raise ExtractionError(f"Missing required fields {names} in {path}")
        return np.asarray(arr, dtype=float)

    @staticmethod
    def _get_optional_array(point_data: dict[str, Any], names: tuple[str, ...]) -> Any | None:
        lowered = {key.lower(): value for key, value in point_data.items()}
        for name in names:
            if name in point_data:
                return point_data[name]
            if name.lower() in lowered:
                return lowered[name.lower()]
        return None

    def _prepare_side(self, x: np.ndarray, cp: np.ndarray, cf: np.ndarray, gamma: np.ndarray) -> dict[str, np.ndarray]:
        if x.size == 0:
            grid = np.linspace(0.0, 1.0, 200)
            zeros = np.zeros_like(grid)
            return {"x": grid, "cp": zeros, "cf": zeros, "gamma": zeros}
        order = np.argsort(x)
        x_sorted = np.asarray(x[order], dtype=float)
        cp_sorted = np.asarray(cp[order], dtype=float)
        cf_sorted = np.asarray(cf[order], dtype=float)
        gamma_sorted = np.asarray(gamma[order], dtype=float)
        x_unique, unique_idx = np.unique(x_sorted, return_index=True)
        cp_unique = cp_sorted[unique_idx]
        cf_unique = cf_sorted[unique_idx]
        gamma_unique = gamma_sorted[unique_idx]
        if x_unique.size < 200:
            grid = np.linspace(float(x_unique.min()), float(x_unique.max()), 200)
            cp_unique = np.interp(grid, x_unique, cp_unique)
            cf_unique = np.interp(grid, x_unique, cf_unique)
            gamma_unique = np.interp(grid, x_unique, gamma_unique)
            x_unique = grid
        return {"x": x_unique, "cp": cp_unique, "cf": cf_unique, "gamma": gamma_unique}

    @staticmethod
    def _first_location(x: np.ndarray, mask: np.ndarray) -> float | None:
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return None
        return float(x[int(idx[0])])
