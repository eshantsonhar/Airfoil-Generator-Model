"""
Pipeline-side telemetry bridge (safe for subprocess execution).
"""

from __future__ import annotations

import time
from typing import Any

from airfoil_discovery.core.telemetry import ResearchTelemetry
from airfoil_discovery.ui.telemetry_hub import TelemetryEventWriter, writer_from_env


class PipelineTelemetryBridge:
    """Unified telemetry for optimization runs."""

    def __init__(
        self,
        run_id: str,
        db_path: str | None = None,
        event_writer: TelemetryEventWriter | None = None,
    ):
        from pathlib import Path

        self.run_id = run_id
        self.events = event_writer or writer_from_env()
        telemetry_db = Path("data/telemetry/metrics.db")
        if db_path:
            telemetry_db = Path(db_path)
        self.research = ResearchTelemetry(db_path=telemetry_db, run_id=run_id)
        self._started = time.time()

    def emit(self, event_type: str, **payload: Any) -> None:
        payload.setdefault("run_id", self.run_id)
        self.events.emit(event_type, **payload)

    def snapshot(self, iteration: int | None = None, **metrics: Any) -> None:
        snap = self.research.create_snapshot(iteration=iteration, **metrics)
        self.research.record_snapshot(snap)
        self.emit(
            "telemetry_snapshot",
            iteration=iteration,
            **{k: v for k, v in snap.to_dict().items() if v is not None},
        )

    def heartbeat(self, component: str, state: str, **details: Any) -> None:
        self.emit(
            "heartbeat",
            component=component,
            state=state,
            elapsed_s=time.time() - self._started,
            **details,
        )

    def watchdog_event(self, operation: str, status: str, **details: Any) -> None:
        self.emit("watchdog", operation=operation, status=status, **details)

    def failure(self, category: str, message: str, **details: Any) -> None:
        self.emit(
            "failure",
            category=category,
            message=message,
            severity="fatal",
            **details,
        )
