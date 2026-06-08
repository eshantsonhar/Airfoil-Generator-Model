"""
Physics analysis modules for low-Re transitional flow.

This module provides physics-based analysis for laminar separation
bubbles (LSB), transition modeling, and flow diagnostics.
"""

from .lsb_detection import LSBDetector, LSBMetrics, LSBClassification
from .transition_governance import TransitionModelGovernor, TransitionDiagnostics

__all__ = [
    "LSBDetector",
    "LSBMetrics",
    "LSBClassification",
    "TransitionModelGovernor",
    "TransitionDiagnostics",
]
