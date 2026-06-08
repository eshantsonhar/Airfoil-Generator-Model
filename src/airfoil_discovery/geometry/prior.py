from __future__ import annotations

import numpy as np

from airfoil_discovery.config import GeometryConfig
from airfoil_discovery.geometry.cst import CSTAirfoil
from airfoil_discovery.schemas import CSTParameters, GeometryMetrics


class GeometryPriorFilter:
    def __init__(self, config: GeometryConfig):
        self.config = config
        self.airfoil = CSTAirfoil(config)

    def sample_parameters(self, rng: np.random.Generator) -> CSTParameters:
        upper = rng.uniform(*self.config.upper_bounds, size=4)
        lower = rng.uniform(*self.config.lower_bounds, size=4)
        te = float(rng.uniform(*self.config.te_thickness_bounds))
        return CSTParameters(upper=upper, lower=lower, trailing_edge_thickness=te)

    def evaluate(self, params: CSTParameters) -> GeometryMetrics:
        return self.airfoil.geometry_metrics(params)

    def is_valid(self, params: CSTParameters) -> tuple[bool, GeometryMetrics]:
        metrics = self.evaluate(params)
        return metrics.is_valid, metrics
