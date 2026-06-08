"""
Live telemetry hub: append-only event log + WebSocket fan-out.

The optimization pipeline runs in a separate process; it writes JSONL events.
The FastAPI server tails the log and streams to connected WebSocket clients.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Set

from fastapi import WebSocket


DEFAULT_EVENT_PATH = Path("data/logs/telemetry_events.jsonl")
DEFAULT_BUFFER_SIZE = 5000


@dataclass
class TelemetryEvent:
    timestamp: float
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            **self.payload,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class TelemetryEventWriter:
    """Append-only writer for subprocess / pipeline use."""

    def __init__(self, path: Path | None = None):
        self.path = path or DEFAULT_EVENT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, event_type: str, **payload: Any) -> None:
        row = TelemetryEvent(
            timestamp=time.time(),
            event_type=event_type,
            payload=payload,
        )
        line = row.to_json() + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)

    def clear(self) -> None:
        with self._lock:
            self.path.write_text("", encoding="utf-8")


def writer_from_env() -> TelemetryEventWriter:
    import os

    path = os.getenv("AIRFOIL_TELEMETRY_PATH")
    return TelemetryEventWriter(Path(path) if path else None)


class TelemetryHub:
    """In-process hub: buffers events and broadcasts to WebSocket subscribers."""

    def __init__(
        self,
        event_path: Path | None = None,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
    ):
        self.event_path = event_path or DEFAULT_EVENT_PATH
        self.buffer: Deque[dict[str, Any]] = deque(maxlen=buffer_size)
        self._clients: Set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._tail_pos = 0
        self._watcher_task: asyncio.Task | None = None

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._clients.add(websocket)
        # Replay buffered history
        for event in list(self.buffer):
            try:
                await websocket.send_json(event)
            except Exception:
                break

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(websocket)

    async def broadcast(self, event: dict[str, Any]) -> None:
        self.buffer.append(event)
        async with self._lock:
            dead: list[WebSocket] = []
            for client in self._clients:
                try:
                    await client.send_json(event)
                except Exception:
                    dead.append(client)
            for client in dead:
                self._clients.discard(client)

    async def start_watcher(self, poll_interval: float = 0.25) -> None:
        if self._watcher_task is not None:
            return
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.event_path.exists():
            self.event_path.write_text("", encoding="utf-8")
        self._tail_pos = self.event_path.stat().st_size

        async def _loop() -> None:
            while True:
                await self._tail_file()
                await asyncio.sleep(poll_interval)

        self._watcher_task = asyncio.create_task(_loop())

    async def stop_watcher(self) -> None:
        if self._watcher_task:
            self._watcher_task.cancel()
            try:
                await self._watcher_task
            except asyncio.CancelledError:
                pass
            self._watcher_task = None

    async def _tail_file(self) -> None:
        if not self.event_path.exists():
            return
        size = self.event_path.stat().st_size
        if size < self._tail_pos:
            self._tail_pos = 0
        if size == self._tail_pos:
            return
        with self.event_path.open("r", encoding="utf-8") as handle:
            handle.seek(self._tail_pos)
            chunk = handle.read()
            self._tail_pos = handle.tell()
        for line in chunk.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            await self.broadcast(event)


_hub: TelemetryHub | None = None


def get_telemetry_hub() -> TelemetryHub:
    global _hub
    if _hub is None:
        _hub = TelemetryHub()
    return _hub
