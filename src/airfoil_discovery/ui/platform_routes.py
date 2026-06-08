"""
Extended platform API: WebSocket telemetry, failures, run configuration.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from airfoil_discovery.ui.telemetry_hub import DEFAULT_EVENT_PATH, get_telemetry_hub

router = APIRouter(tags=["platform"])

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
SAVED_CONFIG_DIR = PROJECT_ROOT / "data" / "configs"
DIAGNOSTICS_ROOT = PROJECT_ROOT / "data" / "cache" / "diagnostics"
FAILURES_ROOT = PROJECT_ROOT / "data" / "failures"
TELEMETRY_DB = PROJECT_ROOT / "data" / "telemetry" / "metrics.db"


class RunConfigPatch(BaseModel):
    reynolds_min: float | None = None
    reynolds_max: float | None = None
    mach: float | None = None
    iterations: int | None = Field(default=None, ge=1)
    batch_size: int | None = Field(default=None, ge=1)
    move_limit: float | None = Field(default=None, gt=0, le=1)
    trust_region_initial: float | None = Field(default=None, gt=0)
    case_timeout_seconds: float | None = Field(default=None, ge=0)
    mesh_surface_points: int | None = Field(default=None, ge=50)
    convergence_residual_drop: float | None = None
    watchdog_su2_timeout: float | None = Field(default=None, ge=60)


@router.websocket("/ws/telemetry")
async def telemetry_ws(websocket: WebSocket) -> None:
    hub = get_telemetry_hub()
    await hub.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await hub.disconnect(websocket)


@router.get("/api/telemetry/replay")
def telemetry_replay(limit: int = 500) -> dict[str, Any]:
    hub = get_telemetry_hub()
    events = list(hub.buffer)[-max(1, min(limit, 5000)) :]
    return {"events": events, "count": len(events)}


@router.get("/api/failures")
def list_failures() -> dict[str, Any]:
    roots = [FAILURES_ROOT, DIAGNOSTICS_ROOT]
    entries: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*"), key=lambda p: p.stat().st_mtime if p.is_file() else 0, reverse=True):
            if path.is_file() and path.suffix in {".log", ".json", ".txt", ".md"}:
                entries.append({
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "name": path.name,
                    "size": path.stat().st_size,
                    "modified": path.stat().st_mtime,
                })
            if len(entries) >= 200:
                break
    return {"failures": entries[:200]}


@router.get("/api/failures/content")
def failure_content(path: str, tail: int = 200) -> dict[str, Any]:
    target = (PROJECT_ROOT / path).resolve()
    if not str(target).startswith(str(PROJECT_ROOT.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    text = target.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    tail = max(1, min(tail, 2000))
    return {"path": path, "lines": lines[-tail:], "truncated": len(lines) > tail}


@router.get("/api/config/current")
def current_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise HTTPException(status_code=404, detail="Config not found")
    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    digest = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()[:16]
    return {"config": data, "hash": digest, "path": str(CONFIG_PATH)}


@router.post("/api/config/save")
def save_config(patch: RunConfigPatch, name: str = "custom") -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise HTTPException(status_code=404, detail="Config not found")
    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if patch.reynolds_min is not None:
        data.setdefault("flow", {})["reynolds_min"] = patch.reynolds_min
    if patch.reynolds_max is not None:
        data.setdefault("flow", {})["reynolds_max"] = patch.reynolds_max
    if patch.mach is not None:
        data.setdefault("flow", {})["mach"] = patch.mach
    if patch.iterations is not None:
        data.setdefault("optimization", {})["iterations"] = patch.iterations
    if patch.batch_size is not None:
        data.setdefault("optimization", {})["batch_size"] = patch.batch_size
    if patch.case_timeout_seconds is not None:
        data.setdefault("solver", {})["case_timeout_seconds"] = patch.case_timeout_seconds
    if patch.mesh_surface_points is not None:
        data.setdefault("solver", {}).setdefault("mesh", {})["surface_points"] = patch.mesh_surface_points
    if patch.convergence_residual_drop is not None:
        data.setdefault("solver", {})["convergence_residual_drop"] = patch.convergence_residual_drop

    SAVED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAVED_CONFIG_DIR / f"{name}.yaml"
    out_path.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()[:16]
    return {"saved": str(out_path.relative_to(PROJECT_ROOT)), "hash": digest}


@router.get("/api/config/saved")
def list_saved_configs() -> dict[str, Any]:
    SAVED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for path in sorted(SAVED_CONFIG_DIR.glob("*.yaml")):
        items.append({
            "name": path.stem,
            "path": str(path.relative_to(PROJECT_ROOT)),
            "hash": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
        })
    return {"configs": items}


@router.get("/api/watchdog/status")
def watchdog_status() -> dict[str, Any]:
    runtime_path = PROJECT_ROOT / "data" / "logs" / "latest_runtime.json"
    runtime: dict[str, Any] = {}
    if runtime_path.exists():
        try:
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        except Exception:
            runtime = {}
    return {
        "watchdog_status": runtime.get("watchdog_status", "UNKNOWN"),
        "last_heartbeat_ts": runtime.get("last_heartbeat_ts", 0),
        "job_age_s": runtime.get("job_age_s", 0),
        "running_cases": runtime.get("running_cases", []),
    }
