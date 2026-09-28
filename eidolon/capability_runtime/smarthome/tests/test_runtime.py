"""Channel only projects authoritative state and translates trusted panel input."""

import asyncio

import pytest
from eidolon_sdk.biz.smarthome import (
    OP_SNAPSHOT,
    ExecuteResult,
    PanelExecute,
    PanelSync,
    initial_state,
)

from eidolon.capability_runtime.smarthome import SmartHomeRuntime

from .helpers import OWNER, PANEL, Clock, RecordingPanels, cmd, home, request


class Backend:
    def __init__(self):
        self.registry = home()
        self.states = {
            d.device_id: {"online": True, "state": initial_state(d.type)}
            for d in self.registry.devices
        }
        self.calls = []

    async def snapshot(self, owner_id):
        import copy

        return {
            "registry": self.registry.model_dump(mode="json"),
            "status": copy.deepcopy(self.states),
        }

    async def execute(self, owner_id, value):
        self.calls.append((owner_id, value))
        return ExecuteResult(request_id=value.request_id, error="DEVICE_OFFLINE")


@pytest.fixture
def setup():
    backend, panels = Backend(), RecordingPanels()
    runtime = SmartHomeRuntime(backend=backend, panels=panels, now_ms=Clock())
    return runtime, backend, panels


async def test_attach_and_fresh_state_same_registry(setup):
    runtime, backend, panels = setup
    await runtime.attach_panel(OWNER, PANEL)
    backend.states["living.main_light"]["state"]["on"] = True
    await runtime.refresh_active_panels()
    value = panels.last(PANEL, OP_SNAPSHOT)
    assert value["seq"] == 1
    assert (
        next(x for x in value["devices"] if x["device_id"] == "living.main_light")["state"]["on"]
        is True
    )
    await runtime.refresh_active_panels()
    assert len(panels.sent) == 2


async def test_touch_identity_and_deadline_come_from_binding(setup):
    runtime, backend, _ = setup
    await runtime.handle_panel_execute(
        OWNER, PANEL, PanelExecute(request_id="touch", commands=(cmd("living.ac", "on_off", "on"),))
    )
    owner, value = backend.calls[0]
    assert owner == OWNER and value.origin.device_ref == PANEL and value.origin.kind == "touch"
    assert value.deadline_ms == Clock()() + 3000


async def test_changed_registry_sends_full_snapshot(setup):
    runtime, backend, panels = setup
    await runtime.attach_panel(OWNER, PANEL)
    backend.registry = home(revision=2)
    await runtime.refresh_active_panels()
    assert panels.last(PANEL, OP_SNAPSHOT)["revision"] == 2


async def test_sync_and_detach(setup):
    runtime, _, panels = setup
    await runtime.attach_panel(OWNER, PANEL)
    await runtime.attach_panel(OWNER, "other")
    await runtime.handle_panel_sync(OWNER, PANEL, PanelSync(known_revision=0, known_seq=0))
    assert len(panels.to(PANEL)) == 2 and len(panels.to("other")) == 1
    runtime.detach_panel(OWNER, PANEL)
    assert PANEL not in runtime._owners[OWNER].panels


async def test_unreachable_panel_does_not_prevent_other_projection(setup):
    runtime, backend, panels = setup
    await runtime.attach_panel(OWNER, PANEL)
    await runtime.attach_panel(OWNER, "other")
    panels.unreachable.add(PANEL)
    backend.states["living.main_light"]["state"]["on"] = True
    await runtime.refresh_active_panels()
    assert len(panels.to("other")) == 2


async def test_malformed_input_does_not_execute(setup):
    runtime, backend, _ = setup
    await runtime.handle_panel_request(OWNER, PANEL, {"owner_id": "forged", "request_id": "x"})
    assert backend.calls == []


async def test_panel_send_can_block_without_blocking_backend(setup):
    runtime, backend, panels = setup
    await runtime.attach_panel(OWNER, PANEL)
    blocked = asyncio.Event()

    async def slow(*args):
        await blocked.wait()

    panels.send = slow
    backend.states["living.main_light"]["state"]["on"] = True
    projection = asyncio.create_task(runtime.refresh_active_panels())
    await asyncio.sleep(0)
    assert not projection.done()
    await asyncio.wait_for(
        backend.execute(OWNER, request("voice", cmd("living.ac", "on_off", "on"))), 0.1
    )
    blocked.set()
    await projection
