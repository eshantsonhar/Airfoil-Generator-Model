from __future__ import annotations

import numpy as np

from airfoil_discovery.config import load_settings
from airfoil_discovery.geometry import GeometryPriorFilter
from airfoil_discovery.schemas import CSTParameters


def test_geometry_filter_accepts_reasonable_shape() -> None:
    settings = load_settings("config/default.yaml")
    geometry_filter = GeometryPriorFilter(settings.geometry)
    params = CSTParameters(
        upper=np.array([0.17933770871077237, 0.05130698471822659, 0.34051670543061097, 0.10406137837150395]),
        lower=np.array([-0.1929602781744858, 0.05213286159233149, -0.09608367605263071, 0.038860098546347754]),
        trailing_edge_thickness=0.006757454502482736,
    )
    valid, metrics = geometry_filter.is_valid(params)
    assert valid
    assert metrics.prior_score >= settings.geometry.prior_threshold


def test_geometry_filter_rejects_sharp_invalid_shape() -> None:
    settings = load_settings("config/default.yaml")
    geometry_filter = GeometryPriorFilter(settings.geometry)
    params = CSTParameters(
        upper=np.array([0.39, 0.39, 0.39, 0.39]),
        lower=np.array([-0.29, -0.29, -0.29, -0.29]),
        trailing_edge_thickness=0.001,
    )
    valid, _ = geometry_filter.is_valid(params)
    assert not valid


def test_geometry_filter_finds_valid_sample_from_prior_space() -> None:
    settings = load_settings("config/default.yaml")
    geometry_filter = GeometryPriorFilter(settings.geometry)
    rng = np.random.default_rng(7)

    for _ in range(2000):
        params = geometry_filter.sample_parameters(rng)
        valid, metrics = geometry_filter.is_valid(params)
        if valid:
            assert metrics.prior_score >= settings.geometry.prior_threshold
            return

    raise AssertionError("Expected to find at least one valid CST sample within 2000 draws.")
