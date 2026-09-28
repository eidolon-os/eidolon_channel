"""The runtime's panel payloads against the vectors the korvo-1 firmware parses.

Fed the golden inputs (the sample apartment, the panel placed in the living
room, the device states the generator used, a camera nobody reports, a Host at
UTC+8), the runtime must produce the golden snapshot and, after the voice
command, the golden delta. Envelope ids and timestamps are per send and are not
compared. The upward vectors are the bodies a panel publishes, fed in as the
transport would hand them over.
"""

from __future__ import annotations

import json
from pathlib import Path

import eidolon_sdk
import pytest
from eidolon_sdk.biz.smarthome import OP_DELTA, OP_SNAPSHOT, initial_state

from eidolon.capability_runtime.smarthome import SmartHomeRuntime, apply_command, panel_command

from .helpers import OWNER, Clock, FakeRegistry, RecordingPanels, cmd, home, request

GOLDEN = Path(eidolon_sdk.__file__).resolve().parents[1] / "contracts/smarthome/v1/golden"
PANEL = "korvo1-golden"
# The generator's device states: readings a virtual device cannot produce itself.
UNREPORTED = "entry.camera"
STATES = {
    "living.main_light": {"on": True, "level": 60},
    "master.bedside": {"on": True, "level": 30},
    "living.curtain": {"position": 70},
    "living.purifier": {"on": True, "speed": 30},
    "living.speaker": {"on": True, "volume": 35},
    "living.ac": {"current_c": 28},
    "master.ac": {"current_c": 27},
    "bath.water_heater": {"on": True},
    "living.thermo": {"temp_c": 24.5, "humidity": 48},
}


class GoldenProvider:
    """The virtual Provider's semantics over the generator's states."""

    def __init__(self) -> None:
        self.state: dict[str, dict] = {}

    async def reconcile(self, owner_id, devices):
        for device in devices:
            self.state.setdefault(
                device.device_id, initial_state(device.type) | STATES.get(device.device_id, {})
            )

    async def states(self, owner_id, devices):
        return {d.device_id: self.state[d.device_id] for d in devices if d.device_id != UNREPORTED}

    async def execute(self, owner_id, device, command):
        self.state[device.device_id] = apply_command(
            device.type, self.state[device.device_id], command
        )
        return self.state[device.device_id]


def golden(name: str) -> dict:
    path = GOLDEN / name
    assert path.exists(), f"the smart home panel vector is not at {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def comparable(envelope: dict) -> dict:
    return {key: value for key, value in envelope.items() if key not in {"id", "ts"}}


def golden_runtime(panels: RecordingPanels) -> SmartHomeRuntime:
    return SmartHomeRuntime(
        registry=FakeRegistry({OWNER: home(placements={PANEL: "living"})}),
        panels=panels,
        providers={"virtual": GoldenProvider()},
        now_ms=Clock(),
        utc_offset_minutes=lambda _now_ms: 480,
    )


async def test_snapshot_and_delta_match_the_golden_vectors():
    panels = RecordingPanels()
    runtime = golden_runtime(panels)
    await runtime.attach_panel(OWNER, PANEL)
    await runtime.execute(
        OWNER, request("turn-golden-1", cmd("living.ac", "on_off", "on"), label="面板语音")
    )
    for op, name in ((OP_SNAPSHOT, "panel-snapshot.json"), (OP_DELTA, "panel-delta.json")):
        envelope = panel_command(device_ref=PANEL, op=op, payload=panels.last(PANEL, op))
        assert comparable(envelope) == comparable(golden(name)), name
        assert envelope["id"].startswith(f"{op}:")


async def test_the_golden_panel_requests_are_carried_out():
    panels = RecordingPanels()
    runtime = golden_runtime(panels)
    await runtime.attach_panel(OWNER, PANEL)
    await runtime.handle_panel_request(OWNER, PANEL, golden("panel-execute-request.json"))
    touch = panels.last(PANEL, OP_DELTA)
    assert (touch["seq"], touch["source"]) == (1, {"kind": "touch", "label": None})
    assert touch["changes"] == [
        {"device_id": "living.main_light", "online": True, "state": {"on": True, "level": 40}}
    ]
    await runtime.handle_panel_request(OWNER, PANEL, golden("panel-scene-request.json"))
    movie = panels.last(PANEL, OP_DELTA)
    assert movie["seq"] == 2
    assert {c["device_id"]: c["state"] for c in movie["changes"]} == {
        "living.main_light": {"on": True, "level": 15},
        "living.curtain": {"position": 0},
        "living.tv": {"on": True, "volume": 20, "muted": False},
    }
    await runtime.handle_panel_request(OWNER, PANEL, golden("panel-sync-request.json"))
    assert panels.ops(PANEL)[-1] == (OP_SNAPSHOT, 2)


def test_only_panel_ops_are_wrapped():
    with pytest.raises(ValueError):
        panel_command(device_ref=PANEL, op="room.join", payload={})
