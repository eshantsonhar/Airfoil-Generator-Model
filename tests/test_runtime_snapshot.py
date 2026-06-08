from __future__ import annotations

import json
from pathlib import Path

from airfoil_discovery.runtime import refresh_runtime_snapshot


def test_runtime_snapshot_updates_elapsed_time(tmp_path: Path, monkeypatch) -> None:
    runtime_path = tmp_path / "latest_runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "status": "running",
                "completed_cases": 0,
                "total_planned_cases": 1,
                "running_cases": [
                    {
                        "case_id": "run_000_test",
                        "reynolds": 25000.0,
                        "start_ts": 100.0,
                        "elapsed_s": 0.0,
                        "eta_s": None,
                    }
                ],
                "debug_events": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("airfoil_discovery.runtime.snapshot.time.time", lambda: 160.0)

    payload = refresh_runtime_snapshot(runtime_path)
    assert payload is not None
    assert payload["running_cases"][0]["elapsed_s"] == 60.0
    assert payload["running_cases_count"] == 1
