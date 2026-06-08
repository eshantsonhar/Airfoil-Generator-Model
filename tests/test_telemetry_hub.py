import json
from pathlib import Path

from airfoil_discovery.ui.telemetry_hub import TelemetryEventWriter, TelemetryHub


def test_event_writer_appends_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    writer = TelemetryEventWriter(path)
    writer.emit("test_event", value=1.0)
    writer.emit("test_event", value=2.0)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event_type"] == "test_event"


def test_hub_buffer_replay() -> None:
    hub = TelemetryHub(buffer_size=10)
    import asyncio

    async def run() -> None:
        await hub.broadcast({"event_type": "a", "x": 1})
        await hub.broadcast({"event_type": "b", "x": 2})

    asyncio.run(run())
    assert len(hub.buffer) == 2
