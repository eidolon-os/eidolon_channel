"""Channel only projects authoritative state and translates trusted panel input."""

import asyncio

import pytest
from eidolon_sdk.biz.smarthome import (
    OP_DELTA,
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


class PushingBackend(Backend):
    """A Hub that also streams observed changes."""

    def __init__(self):
        super().__init__()
        self.queue: asyncio.Queue = asyncio.Queue()
        self.latest = 0

    async def changes(self, owner_id, *, since, timeout_ms):
        if timeout_ms == 0:
            return {"changes": [], "seq": self.latest}
        change = await self.queue.get()
        self.latest = change["seq"]
        return {"changes": [change], "seq": change["seq"]}

    def push(self, device_id, *, reachable=True, state=None):
        self.latest += 1
        self.queue.put_nowait(
            {
                "device_id": device_id,
                "reachable": reachable,
                "state": state,
                "observed_at_ms": 1,
                "seq": self.latest,
            }
        )


async def test_observed_changes_reach_panels_as_deltas():
    backend, panels = PushingBackend(), RecordingPanels()
    runtime = SmartHomeRuntime(backend=backend, panels=panels, now_ms=Clock())
    await runtime.attach_panel(OWNER, PANEL)
    backend.push("living.main_light", state={"on": True, "level": 90})
    for _ in range(50):
        await asyncio.sleep(0.01)
        if panels.to(PANEL, OP_DELTA):
            break
    delta = panels.last(PANEL, OP_DELTA)
    assert delta["seq"] == 1 and delta["changes"][0] == {
        "device_id": "living.main_light",
        "online": True,
        "state": {"on": True, "level": 90},
    }
    assert delta["source"] == {"kind": "automation", "label": "observed"}
    # The poll afterwards sees the same state and sends nothing new.
    backend.states["living.main_light"]["state"] = {"on": True, "level": 90}
    await runtime.refresh_active_panels()
    assert panels.ops(PANEL) == [(OP_SNAPSHOT, 0), (OP_DELTA, 1)]
    # Unreachable: online false, last state kept for the tile.
    backend.push("living.main_light", reachable=False)
    for _ in range(50):
        await asyncio.sleep(0.01)
        if len(panels.to(PANEL, OP_DELTA)) == 2:
            break
    assert panels.last(PANEL, OP_DELTA)["changes"][0]["online"] is False
    # A device the registry does not know is not a delta.
    backend.push("ghost.device", state={"on": True})
    await asyncio.sleep(0.05)
    assert len(panels.to(PANEL, OP_DELTA)) == 2
    runtime.detach_panel(OWNER, PANEL)
    await asyncio.sleep(0)
    await runtime.close()


async def test_backend_without_changes_is_polled_only(setup):
    runtime, backend, panels = setup
    await runtime.attach_panel(OWNER, PANEL)
    assert runtime._owners[OWNER].watcher is None
