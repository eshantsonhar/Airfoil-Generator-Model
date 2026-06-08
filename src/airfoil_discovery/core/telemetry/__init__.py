"""
Research telemetry dashboard for real-time metrics.

Tracks residuals, force histories, transition movement, bubble metrics,
FD gradient error, KKT metrics, gain ratio, mesh quality, intermittency
evolution, numerical dissipation, CFL evolution, solver stiffness,
oscillation spectra, trust-region behavior. All metrics are timestamped,
exportable, queryable, archived, and plotted.
"""

from .telemetry import ResearchTelemetry, TelemetryMetric, TelemetrySnapshot
from .metrics_store import MetricsStore

__all__ = [
    "ResearchTelemetry",
    "TelemetryMetric",
    "TelemetrySnapshot",
    "MetricsStore",
]
