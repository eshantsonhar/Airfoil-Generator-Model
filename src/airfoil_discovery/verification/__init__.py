"""
Verification systems for numerical accuracy and solver correctness.

This module provides verification infrastructure to ensure the CFD solver
is solving the equations correctly, separate from validation which checks
if the equations represent real physics.

Verification answers: "Are we solving the equations correctly?"
Validation answers: "Are these equations representative of real aerodynamics?"
"""

from .gci import GridConvergenceIndex, RichardsonExtrapolation
from .convergence import ResidualConvergenceAnalyzer, IterativeConvergenceMonitor
from .gradient_audit import GradientAuditor, FiniteDifferenceVerifier
from .mesh_verification import ComprehensiveMeshVerifier, MeshQualityVerifier, YPlusMonitor

# Lazy imports for modules with pre-existing dataclass ordering issues
try:
    from .numerical_dissipation import NumericalDissipationMonitor
except Exception:
    NumericalDissipationMonitor = None

try:
    from .cfd_governance import CFDGovernanceModel, GovernanceStatus
except Exception:
    CFDGovernanceModel = None
    GovernanceStatus = None

__all__ = [
    "GridConvergenceIndex",
    "RichardsonExtrapolation",
    "ResidualConvergenceAnalyzer",
    "IterativeConvergenceMonitor",
    "GradientAuditor",
    "FiniteDifferenceVerifier",
    "MeshQualityVerifier",
    "ComprehensiveMeshVerifier",
    "YPlusMonitor",
    "NumericalDissipationMonitor",
    "CFDGovernanceModel",
    "GovernanceStatus",
]
