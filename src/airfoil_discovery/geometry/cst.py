from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import comb

from airfoil_discovery.config import GeometryConfig
from airfoil_discovery.schemas import CSTParameters, GeometryMetrics


def cosine_spacing(n: int) -> np.ndarray:
    beta = np.linspace(0.0, np.pi, n)
    return 0.5 * (1.0 - np.cos(beta))


@dataclass(slots=True)
class CSTAirfoil:
    config: GeometryConfig

    def bernstein(self, n: int, x: np.ndarray) -> np.ndarray:
        basis = [comb(n, k) * (x**k) * ((1.0 - x) ** (n - k)) for k in range(n + 1)]
        return np.vstack(basis)

    def class_function(self, x: np.ndarray, n1: float = 0.5, n2: float = 1.0) -> np.ndarray:
        x_safe = np.clip(x, 1.0e-8, 1.0)
        return (x_safe**n1) * ((1.0 - x_safe) ** n2)

    def shape_function(self, coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
        n = len(coeffs) - 1
        return coeffs @ self.bernstein(n, x)

    def surface_y(self, coeffs: np.ndarray, x: np.ndarray, te_thickness: float, sign: float) -> np.ndarray:
        return self.class_function(x) * self.shape_function(coeffs, x) + sign * 0.5 * te_thickness * x

    def coordinates(self, params: CSTParameters) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = cosine_spacing(self.config.samples_per_surface)
        yu = self.surface_y(params.upper, x, params.trailing_edge_thickness, +1.0)
        yl = self.surface_y(params.lower, x, params.trailing_edge_thickness, -1.0)
        return x, yu, yl

    def full_coordinates(self, params: CSTParameters) -> np.ndarray:
        x, yu, yl = self.coordinates(params)
        upper = np.column_stack([x[::-1], yu[::-1]])
        lower = np.column_stack([x[1:], yl[1:]])
        return np.vstack([upper, lower])

    def geometry_metrics(self, params: CSTParameters) -> GeometryMetrics:
        x, yu, yl = self.coordinates(params)
        thickness = yu - yl
        camber = 0.5 * (yu + yl)

        dy_dx = np.gradient(yu, x, edge_order=2)
        d2y_dx2 = np.gradient(dy_dx, x, edge_order=2)
        curvature = np.abs(d2y_dx2) / np.power(1.0 + dy_dx**2, 1.5)
        curvature_spike = float(np.max(curvature))
        smoothness = float(np.exp(-self.config.smoothness_penalty_scale * np.std(np.diff(curvature))))

        le_radius = self._estimate_le_radius(x, yu, yl)
        max_thickness = float(np.max(thickness))
        max_camber = float(np.max(camber))

        valid, reason = self._validate_metrics(max_thickness, max_camber, le_radius, curvature_spike, thickness, camber)
        prior = self._prior_score(max_thickness, max_camber, le_radius, curvature_spike, smoothness, valid)
        return GeometryMetrics(
            max_thickness=max_thickness,
            max_camber=max_camber,
            leading_edge_radius=le_radius,
            smoothness_score=smoothness,
            curvature_spike=curvature_spike,
            prior_score=prior,
            is_valid=valid and prior >= self.config.prior_threshold,
            rejection_reason=None if valid and prior >= self.config.prior_threshold else reason or "low_prior_score",
        )

    def _estimate_le_radius(self, x: np.ndarray, yu: np.ndarray, yl: np.ndarray) -> float:
        # Near the leading edge, CST airfoils follow y_t ~ sqrt(2 R x).
        # Estimating radius from y^2 / (2 x) is more stable than a quadratic
        # fit because the local behavior is not parabolic in x.
        half_thickness = 0.5 * (yu - yl)
        sample_end = min(6, len(x))
        x_local = np.clip(x[1:sample_end], 1.0e-8, None)
        y_local = np.clip(half_thickness[1:sample_end], 0.0, None)
        if len(x_local) == 0:
            return 0.0
        radii = (y_local**2) / (2.0 * x_local)
        return float(np.mean(radii))

    def _validate_metrics(
        self,
        max_thickness: float,
        max_camber: float,
        le_radius: float,
        curvature_spike: float,
        thickness: np.ndarray,
        camber: np.ndarray,
    ) -> tuple[bool, str | None]:
        if not self.config.thickness_bounds[0] <= max_thickness <= self.config.thickness_bounds[1]:
            return False, "thickness_out_of_bounds"
        if not self.config.camber_bounds[0] <= max_camber <= self.config.camber_bounds[1]:
            return False, "camber_out_of_bounds"
        if le_radius < self.config.min_le_radius:
            return False, "leading_edge_too_sharp"
        if curvature_spike > self.config.max_curvature_spike:
            return False, "curvature_spike"
        if np.min(thickness) <= 0.0:
            return False, "self_intersection"
        if np.max(np.abs(np.diff(camber, n=2))) > 0.015:
            return False, "camber_oscillation"
        return True, None

    def _prior_score(
        self,
        max_thickness: float,
        max_camber: float,
        le_radius: float,
        curvature_spike: float,
        smoothness: float,
        valid: bool,
    ) -> float:
        if not valid:
            return 0.0
        thickness_center = np.mean(self.config.thickness_bounds)
        camber_center = np.mean(self.config.camber_bounds)
        thickness_reward = np.exp(-((max_thickness - thickness_center) / 0.02) ** 2)
        camber_reward = np.exp(-((max_camber - camber_center) / 0.015) ** 2)
        le_reward = np.clip(le_radius / (3.0 * self.config.min_le_radius), 0.0, 1.0)
        curvature_penalty = np.exp(-curvature_spike / self.config.max_curvature_spike)
        return float(0.30 * smoothness + 0.20 * le_reward + 0.25 * thickness_reward + 0.25 * camber_reward) * curvature_penalty
