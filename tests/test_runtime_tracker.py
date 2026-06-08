from __future__ import annotations

import json
from pathlib import Path

from airfoil_discovery.pipeline import RuntimeTracker


def test_runtime_tracker_records_running_cases(tmp_path: Path) -> None:
    runtime_path = tmp_path / "runtime.json"
    tracker = RuntimeTracker(runtime_path)
    tracker.initialize(total_iterations=2, batch_size=3, max_parallel_workers=1)

    tracker.on_case_event(
        {
            "event": "case_started",
            "case_id": "run_000_test",
            "reynolds": 30000.0,
        }
    )

    data = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert data["running_cases_count"] == 1
    assert data["running_cases"][0]["case_id"] == "run_000_test"
    assert data["running_cases"][0]["elapsed_s"] >= 0.0

    tracker.on_case_event({"event": "case_completed", "case_id": "run_000_test"})

    data = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert data["running_cases_count"] == 0
    assert data["completed_cases"] == 1
    assert data["avg_case_runtime_s"] is not None
