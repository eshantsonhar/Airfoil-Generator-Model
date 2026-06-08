from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any

def refresh_runtime_snapshot(runtime_path: Path) -> dict[str, Any] | None:
    if not runtime_path.exists():
        return None
    try:
        payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    
    # Pass-through for Production ASO schema
    if "stationarity" in payload:
        return payload

    now = time.time()
    running_cases = payload.get("running_cases") or []
    completed_cases = int(payload.get("completed_cases") or 0)
    avg_case_runtime_s = payload.get("avg_case_runtime_s")

    for case in running_cases:
        start_ts = float(case.get("start_ts") or now)
        case["elapsed_s"] = max(0.0, now - start_ts)
        case["eta_s"] = None if avg_case_runtime_s is None else max(0.0, float(avg_case_runtime_s) - case["elapsed_s"])

    payload["running_cases_count"] = len(running_cases)
    return payload
