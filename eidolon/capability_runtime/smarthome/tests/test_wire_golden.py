"""Channel presentation remains compatible with the Korvo firmware wire vectors."""

from __future__ import annotations

import json
from pathlib import Path

import eidolon_sdk
import pytest
from eidolon_sdk.biz.smarthome import OP_DELTA, OP_SNAPSHOT, initial_state

from eidolon.capability_runtime.smarthome import SmartHomeRuntime, panel_command

from .helpers import OWNER, Clock, RecordingPanels, home

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


def golden(name: str) -> dict:
    path = GOLDEN / name
    assert path.exists(), f"the smart home panel vector is not at {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def comparable(envelope: dict) -> dict:
    return {key: value for key, value in envelope.items() if key not in {"id", "ts"}}


class GoldenBackend:
    async def snapshot(self, owner_id):
        registry = home(placements={PANEL: "living"})
        return {
            "registry": registry.model_dump(mode="json"),
            "status": {
                d.device_id: {
                    "online": d.device_id != UNREPORTED,
                    "state": initial_state(d.type) | STATES.get(d.device_id, {}),
                }
                for d in registry.devices
            },
        }


async def test_snapshot_matches_the_firmware_golden_vector():
    panels = RecordingPanels()
    runtime = SmartHomeRuntime(
        backend=GoldenBackend(), panels=panels, now_ms=Clock(), utc_offset_minutes=lambda _: 480
    )
    await runtime.attach_panel(OWNER, PANEL)
    envelope = panel_command(
        device_ref=PANEL, op=OP_SNAPSHOT, payload=panels.last(PANEL, OP_SNAPSHOT)
    )
    assert comparable(envelope) == comparable(golden("panel-snapshot.json"))


def test_delta_wire_still_matches_the_firmware_golden_vector():
    vector = golden("panel-delta.json")
    # The envelope contract remains supported even though polling uses full snapshots.
    from eidolon_sdk.biz.smarthome import PanelDelta

    payload = PanelDelta.model_validate(vector["payload"])
    envelope = panel_command(device_ref=PANEL, op=OP_DELTA, payload=payload.model_dump(mode="json"))
    assert comparable(envelope) == comparable(vector)


def test_only_panel_ops_are_wrapped():
    with pytest.raises(ValueError):
        panel_command(device_ref=PANEL, op="room.join", payload={})
