"""
Geometry module for airfoil parameterization and governance.

This module provides:
- CST (Class-Shape Transformation) airfoil parameterization
- Geometric governance and validation
- Airfoil shape manifold for realistic geometry validation
"""

from __future__ import annotations

from airfoil_discovery.geometry.cst import CSTAirfoil, cosine_spacing
from airfoil_discovery.geometry.governance import (
    GeometryGovernor,
    GeometryGovernanceConfig,
    GeometryGovernanceReport,
    GeometryValidityStatus,
    GeometryViolationType,
    ThicknessMetrics,
    LeadingEdgeMetrics,
    CurvatureMetrics,
    SurfaceAngleMetrics,
    CSTCoefficientMetrics,
)
from airfoil_discovery.geometry.manifold import (
    AirfoilManifold,
    ManifoldConfig,
    ManifoldQueryResult,
    ManifoldStatus,
    create_default_manifold,
)
from airfoil_discovery.geometry.prior import GeometryPriorFilter

__all__ = [
    # CST parameterization
    "CSTAirfoil",
    "cosine_spacing",
    # Governance
    "GeometryGovernor",
    "GeometryGovernanceConfig",
    "GeometryGovernanceReport",
    "GeometryValidityStatus",
    "GeometryViolationType",
    "ThicknessMetrics",
    "LeadingEdgeMetrics",
    "CurvatureMetrics",
    "SurfaceAngleMetrics",
    "CSTCoefficientMetrics",
    # Manifold
    "AirfoilManifold",
    "ManifoldConfig",
    "ManifoldQueryResult",
    "ManifoldStatus",
    "create_default_manifold",
    # Prior
    "GeometryPriorFilter",
]