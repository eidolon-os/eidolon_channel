"""Execution invariants and panel sync, against the real virtual Provider.

The invariants are the plan's §6.3: only a Provider's success is shown as done,
a request id never runs twice, a late request runs nothing and a command a
Provider is still busy with is unknown and never resent, a scene is its
commands reported one by one, a panel is a cache kept in order by (revision,
seq), and one Owner can neither see nor move another's devices.
"""

from __future__ import annotations

import asyncio

import pytest
from eidolon_sdk.biz.smarthome import (
    OP_DELTA,
    OP_SNAPSHOT,
    Device,
    PanelExecute,
    PanelSync,
    Registry,
    initial_state,
    panel_request,
    validate_execute_result,
)
from eidolon_sdk.biz.smarthome.samples import apartment
from pydantic import ValidationError

from eidolon.capability_runtime.smarthome import (
    IdempotencyConflict,
    SmartHomeRuntime,
    VirtualProvider,
)
from eidolon.capability_runtime.smarthome.runtime import local_utc_offset_minutes

from .helpers import (
    NOW_MS,
    OTHER_OWNER,
    OWNER,
    PANEL,
    Clock,
    FakeRegistry,
    RecordingPanels,
    cmd,
    device_state,
    home,
    request,
    scene,
)

AC_ON = cmd("living.ac", "on_off", "on")


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def registry():
    other = Registry(
        revision=4,
        devices=(
            Device(device_id="living.ac", name="他家空调", type="climate"),
            Device(device_id="b.lamp", name="他家的灯", type="light"),
        ),
    )
    return FakeRegistry({OWNER: home(), OTHER_OWNER: other})


@pytest.fixture
def panels():
    return RecordingPanels()


@pytest.fixture
def make_runtime(tmp_path, registry, panels, clock):
    def make(**kwargs) -> SmartHomeRuntime:
        provider = VirtualProvider(tmp_path / "virtual.sqlite3")
        provider.initialize()
        kwargs.setdefault("providers", {"virtual": provider})
        return SmartHomeRuntime(registry=registry, panels=panels, now_ms=clock, **kwargs)

    return make


@pytest.fixture
def runtime(make_runtime):
    return make_runtime()


async def test_a_success_reports_the_state_and_reaches_every_panel_as_one_delta(runtime, panels):
    await runtime.attach_panel(OWNER, PANEL)
    await runtime.attach_panel(OWNER, "panel-bedroom")
    result = await runtime.execute(OWNER, request("turn-1", AC_ON, label="面板语音"))
    (only,) = result.results
    assert only.status == "succeeded"
    assert only.state == initial_state("climate") | {"on": True}
    for panel in (PANEL, "panel-bedroom"):
        delta = panels.last(panel, OP_DELTA)
        assert delta["seq"] == 1 and delta["revision"] == 1
        assert delta["source"] == {"kind": "voice", "label": "面板语音"}
        assert delta["changes"] == [{"device_id": "living.ac", "online": True, "state": only.state}]


async def test_a_repeated_request_id_returns_the_first_result_and_runs_nothing(runtime, panels):
    await runtime.attach_panel(OWNER, PANEL)
    toggle = request("turn-1", cmd("living.main_light", "on_off", "toggle"))
    first = await runtime.execute(OWNER, toggle)
    again = await runtime.execute(OWNER, toggle)
    assert again == first and first.results[0].state["on"] is True
    assert panels.ops(PANEL) == [(OP_SNAPSHOT, 0), (OP_DELTA, 1)]
    with pytest.raises(IdempotencyConflict):
        await runtime.execute(OWNER, request("turn-1", cmd("living.main_light", "on_off", "off")))


async def test_concurrent_repeats_run_once(runtime):
    toggle = request("turn-1", cmd("living.main_light", "on_off", "toggle"))
    results = await asyncio.gather(*(runtime.execute(OWNER, toggle) for _ in range(5)))
    assert {r.results[0].state["on"] for r in results} == {True}


async def test_the_idempotency_window_is_bounded_in_time_and_size(make_runtime, clock):
    runtime = make_runtime(idempotency_ttl_ms=1_000, idempotency_capacity=2)
    toggle = request("turn-1", cmd("living.main_light", "on_off", "toggle"), deadline_ms=NOW_MS * 2)
    await runtime.execute(OWNER, toggle)
    clock.now_ms += 1_000
    assert (await runtime.execute(OWNER, toggle)).results[0].state["on"] is False
    for other in ("turn-2", "turn-3"):
        await runtime.execute(OWNER, request(other, AC_ON, deadline_ms=NOW_MS * 2))
    assert (await runtime.execute(OWNER, toggle)).results[0].state["on"] is True


async def test_a_request_already_past_its_deadline_is_refused_whole(runtime, panels):
    await runtime.attach_panel(OWNER, PANEL)
    late = request("turn-1", AC_ON, cmd("nowhere", "on_off", "on"), deadline_ms=NOW_MS)
    result = await runtime.execute(OWNER, late)
    assert (result.error, result.results) == ("DEADLINE_EXCEEDED", ())
    assert await runtime.execute(OWNER, late) == result
    assert panels.ops(PANEL) == [(OP_SNAPSHOT, 0)]
    await runtime.handle_panel_sync(OWNER, PANEL, PanelSync())
    assert device_state(panels.last(PANEL, OP_SNAPSHOT), "living.ac")["on"] is False


async def test_commands_reached_after_the_deadline_are_not_attempted(make_runtime, clock, tmp_path):
    inner = VirtualProvider(tmp_path / "late.sqlite3")
    inner.initialize()

    class Slow:
        reconcile, states = inner.reconcile, inner.states

        async def execute(self, owner_id, device, command):
            clock.now_ms += 5_000
            return await inner.execute(owner_id, device, command)

    runtime = make_runtime(providers={"virtual": Slow()})
    commands = (AC_ON, cmd("living.tv", "on_off", "on"))
    result = await runtime.execute(OWNER, request("turn-1", *commands))
    validate_execute_result(commands, result)
    assert [(r.status, r.code) for r in result.results] == [
        ("succeeded", None),
        ("failed", "DEADLINE_EXCEEDED"),
    ]


async def test_a_provider_that_outlives_the_deadline_is_unknown_not_failed(make_runtime, registry):
    class Stuck:
        async def reconcile(self, owner_id, devices):
            pass

        async def states(self, owner_id, devices):
            return {d.device_id: initial_state(d.type) for d in devices}

        async def execute(self, owner_id, device, command):
            await asyncio.sleep(5)

    runtime = make_runtime(providers={"virtual": Stuck()})
    result = await runtime.execute(OWNER, request("turn-1", AC_ON, deadline_ms=NOW_MS + 20))
    assert (result.results[0].status, result.results[0].code) == ("unknown", "DEADLINE_EXCEEDED")


async def test_each_command_stands_alone_and_is_reported_honestly(runtime, panels):
    await runtime.attach_panel(OWNER, PANEL)
    commands = (
        AC_ON,
        cmd("projector", "on_off", "on"),
        cmd("living.main_light", "level", "set", value=150),
        cmd("living.thermo", "measure", "read"),
        cmd("living.main_light", "thermostat", "set_target", celsius=22),
        cmd("living.curtain", "position", "open"),
    )
    result = await runtime.execute(OWNER, request("multi-1", *commands, kind="text", label="回家"))
    validate_execute_result(commands, result)
    assert [(r.device_id, r.status, r.code) for r in result.results] == [
        ("living.ac", "succeeded", None),
        ("projector", "failed", "UNKNOWN_DEVICE"),
        ("living.main_light", "failed", "OUT_OF_RANGE"),
        ("living.thermo", "failed", "UNSUPPORTED_COMMAND"),
        ("living.main_light", "failed", "UNSUPPORTED_COMMAND"),
        ("living.curtain", "succeeded", None),
    ]
    delta = panels.last(PANEL, OP_DELTA)
    assert delta["source"] == {"kind": "text", "label": "回家"}
    assert [c["device_id"] for c in delta["changes"]] == ["living.ac", "living.curtain"]


async def test_one_device_commanded_twice_is_one_change_with_its_final_state(runtime, panels):
    await runtime.attach_panel(OWNER, PANEL)
    await runtime.execute(
        OWNER,
        request(
            "scene-1",
            cmd("living.main_light", "on_off", "on"),
            cmd("living.main_light", "level", "set", value=15),
        ),
    )
    assert panels.last(PANEL, OP_DELTA)["changes"] == [
        {"device_id": "living.main_light", "online": True, "state": {"on": True, "level": 15}},
    ]


async def test_nothing_changed_means_no_delta_and_no_seq(runtime, panels):
    await runtime.attach_panel(OWNER, PANEL)
    result = await runtime.execute(OWNER, request("turn-1", cmd("living.ac", "on_off", "off")))
    assert result.results[0].status == "succeeded"
    toggled_back = request(
        "turn-2",
        cmd("living.tv", "on_off", "toggle"),
        cmd("living.tv", "on_off", "toggle"),
    )
    await runtime.execute(OWNER, toggled_back)
    assert panels.ops(PANEL) == [(OP_SNAPSHOT, 0)]


async def test_an_owner_can_neither_see_nor_move_another_owners_devices(runtime, panels):
    await runtime.attach_panel(OWNER, PANEL)
    await runtime.attach_panel(OTHER_OWNER, "their-panel")
    result = await runtime.execute(OWNER, request("turn-1", cmd("b.lamp", "on_off", "on")))
    assert (result.results[0].status, result.results[0].code) == ("failed", "UNKNOWN_DEVICE")
    await runtime.execute(OWNER, request("turn-2", AC_ON))
    assert panels.ops("their-panel") == [(OP_SNAPSHOT, 0)]
    their = await runtime.execute(
        OTHER_OWNER, request("turn-2", cmd("living.ac", "on_off", "toggle"))
    )
    # Same device id, same request id, different Owner: its own device, its own run.
    assert their.results[0].state["on"] is True
    assert panels.last("their-panel", OP_DELTA)["seq"] == 1
    assert panels.last(PANEL, OP_DELTA)["seq"] == 1
    snapshot = panels.last("their-panel", OP_SNAPSHOT)
    assert {d["device_id"] for d in snapshot["devices"]} == {"living.ac", "b.lamp"}


async def test_state_survives_a_new_runtime(make_runtime, panels):
    first = make_runtime()
    await first.execute(
        OWNER, request("turn-1", cmd("living.curtain", "position", "set", value=35))
    )
    second = make_runtime()
    await second.attach_panel(OWNER, PANEL)
    assert device_state(panels.last(PANEL, OP_SNAPSHOT), "living.curtain") == {"position": 35}


async def test_seq_is_the_owners_and_moves_by_one_per_delta_across_attach_and_detach(
    runtime, panels
):
    await runtime.attach_panel(OWNER, PANEL)
    await runtime.execute(OWNER, request("t1", AC_ON))
    runtime.detach_panel(OWNER, PANEL)
    await runtime.execute(OWNER, request("t2", cmd("living.tv", "on_off", "on")))
    await runtime.attach_panel(OWNER, "panel-bedroom")
    await runtime.attach_panel(OWNER, PANEL)
    await runtime.execute(OWNER, request("t3", cmd("living.speaker", "on_off", "on")))
    assert panels.ops(PANEL) == [(OP_SNAPSHOT, 0), (OP_DELTA, 1), (OP_SNAPSHOT, 2), (OP_DELTA, 3)]
    assert panels.ops("panel-bedroom") == [(OP_SNAPSHOT, 2), (OP_DELTA, 3)]
    runtime.detach_panel(OWNER, "never-attached")


async def test_a_registry_change_is_a_fresh_snapshot_for_every_panel(runtime, registry, panels):
    await runtime.attach_panel(OWNER, PANEL)
    await runtime.attach_panel(OWNER, "panel-bedroom")
    await runtime.execute(OWNER, request("t1", cmd("living.speaker", "on_off", "on")))
    current = registry.registries[OWNER].model_dump(mode="json")
    added = {"device_id": "living.lamp", "name": "落地灯", "type": "light", "area_id": "living"}
    registry.registries[OWNER] = home(
        revision=2,
        devices=[d for d in current["devices"] if d["device_id"] != "living.speaker"] + [added],
        placements={PANEL: "living", "panel-bedroom": "master"},
    )
    await runtime.on_registry_changed(OWNER)
    for panel, area in ((PANEL, "living"), ("panel-bedroom", "master")):
        assert panels.ops(panel)[-1] == (OP_SNAPSHOT, 1)
        snapshot = panels.last(panel, OP_SNAPSHOT)
        assert snapshot["revision"] == 2 and snapshot["panel_area_id"] == area
        ids = [d["device_id"] for d in snapshot["devices"]]
        assert "living.speaker" not in ids and ids[-1] == "living.lamp"
        assert device_state(snapshot, "living.lamp") == initial_state("light")
    await runtime.execute(OWNER, request("t2", cmd("living.lamp", "on_off", "on")))
    assert panels.last(PANEL, OP_DELTA)["revision"] == 2
    assert panels.last(PANEL, OP_DELTA)["seq"] == 2
    # A removed device's state is forgotten, not parked for its return.
    registry.registries[OWNER] = home(revision=3)
    await runtime.on_registry_changed(OWNER)
    assert device_state(panels.last(PANEL, OP_SNAPSHOT), "living.speaker") == initial_state("media")


async def test_sync_answers_only_the_panel_that_asked(runtime, panels):
    await runtime.attach_panel(OWNER, PANEL)
    await runtime.attach_panel(OWNER, "panel-bedroom")
    await runtime.execute(OWNER, request("t1", AC_ON))
    await runtime.handle_panel_sync(OWNER, PANEL, PanelSync(known_revision=1, known_seq=0))
    assert panels.ops(PANEL)[-1] == (OP_SNAPSHOT, 1)
    assert panels.ops("panel-bedroom")[-1] == (OP_DELTA, 1)
    assert device_state(panels.last(PANEL, OP_SNAPSHOT), "living.ac")["on"] is True


async def test_touch_is_stamped_from_the_binding_and_scoped_to_its_panel(runtime, panels):
    await runtime.attach_panel(OWNER, PANEL)
    touch = PanelExecute(
        request_id="touch-1",
        commands=(cmd("living.main_light", "on_off", "toggle"),),
    )
    first = await runtime.handle_panel_execute(OWNER, PANEL, touch)
    assert first.request_id == "touch-1" and first.results[0].state["on"] is True
    assert panels.last(PANEL, OP_DELTA)["source"] == {"kind": "touch", "label": None}
    assert (await runtime.handle_panel_execute(OWNER, PANEL, touch)) == first
    # Another panel's counter may land on the same id; it is a different touch.
    other = await runtime.handle_panel_execute(OWNER, "panel-bedroom", touch)
    assert other.results[0].state["on"] is False
    # And a voice turn with that id is not either panel's.
    voice = await runtime.execute(
        OWNER, request("touch-1", cmd("living.main_light", "on_off", "toggle"))
    )
    assert voice.results[0].state["on"] is True
    with pytest.raises(ValidationError):
        PanelExecute.model_validate(touch.model_dump() | {"origin": {"kind": "voice"}})


async def test_the_panels_area_comes_from_its_placement(runtime, panels):
    await runtime.attach_panel(OWNER, PANEL)
    await runtime.attach_panel(OWNER, "unplaced")
    assert panels.last(PANEL, OP_SNAPSHOT)["panel_area_id"] == "living"
    assert panels.last("unplaced", OP_SNAPSHOT)["panel_area_id"] is None


async def test_a_device_no_provider_serves_is_offline_and_refuses_commands(
    runtime, registry, panels
):
    current = registry.registries[OWNER].model_dump(mode="json")
    bridged = {"device_id": "hall.light", "name": "走廊灯", "type": "light", "provider": "hass"}
    registry.registries[OWNER] = home(devices=current["devices"] + [bridged])
    await runtime.attach_panel(OWNER, PANEL)
    snapshot = panels.last(PANEL, OP_SNAPSHOT)
    hall = next(d for d in snapshot["devices"] if d["device_id"] == "hall.light")
    assert (hall["online"], hall["state"]) == (False, None)
    result = await runtime.execute(OWNER, request("t1", cmd("hall.light", "on_off", "on")))
    assert (result.results[0].status, result.results[0].code) == ("failed", "DEVICE_OFFLINE")


async def test_an_unreachable_panel_neither_fails_the_command_nor_starves_the_others(
    runtime, panels
):
    await runtime.attach_panel(OWNER, PANEL)
    await runtime.attach_panel(OWNER, "panel-bedroom")
    panels.unreachable.add("panel-bedroom")
    result = await runtime.execute(OWNER, request("t1", AC_ON))
    assert result.results[0].status == "succeeded"
    assert panels.ops(PANEL)[-1] == (OP_DELTA, 1)
    # It hears seq 2 next, sees the gap and asks; the command was not re-run meanwhile.
    panels.unreachable.clear()
    assert (await runtime.execute(OWNER, request("t1", AC_ON))) == result
    await runtime.execute(OWNER, request("t2", cmd("living.tv", "on_off", "on")))
    assert panels.ops("panel-bedroom") == [(OP_SNAPSHOT, 0), (OP_DELTA, 2)]


async def test_a_registry_that_cannot_be_read_executes_nothing_and_is_retried(runtime, registry):
    saved = registry.registries.pop(OWNER)
    with pytest.raises(KeyError):
        await runtime.execute(OWNER, request("t1", AC_ON))
    registry.registries[OWNER] = saved
    assert (await runtime.execute(OWNER, request("t1", AC_ON))).results[0].status == "succeeded"


# -- scenes ------------------------------------------------------------------


async def test_a_scene_runs_its_stored_actions_in_order_as_one_delta(runtime, panels):
    await runtime.attach_panel(OWNER, PANEL)
    result = await runtime.execute(OWNER, scene("turn-1", "scene.away", label="离家"))
    actions = apartment().scene("scene.away").actions
    validate_execute_result(actions, result)
    assert {r.status for r in result.results} == {"succeeded"}
    delta = panels.last(PANEL, OP_DELTA)
    assert (delta["seq"], delta["source"]) == (1, {"kind": "voice", "label": "离家"})
    # Only what moved: the lights, AC and TV were already off and the door locked.
    assert [c["device_id"] for c in delta["changes"]] == ["whole.vacuum"]


async def test_a_scene_is_expanded_from_the_registry_as_it_is_now(runtime, registry, panels):
    current = registry.registries[OWNER].model_dump(mode="json")
    movie = next(s for s in current["scenes"] if s["scene_id"] == "scene.movie")
    movie["actions"] = [{"device_id": "living.speaker", "trait": "on_off", "command": "on"}]
    registry.registries[OWNER] = home(revision=2, scenes=current["scenes"])
    await runtime.on_registry_changed(OWNER)
    result = await runtime.execute(OWNER, scene("turn-1", "scene.movie"))
    assert [r.device_id for r in result.results] == ["living.speaker"]


async def test_an_unknown_scene_is_refused_whole(runtime, panels):
    await runtime.attach_panel(OWNER, PANEL)
    result = await runtime.execute(OWNER, scene("turn-1", "scene.party"))
    assert (result.error, result.results) == ("UNKNOWN_SCENE", ())
    assert panels.ops(PANEL) == [(OP_SNAPSHOT, 0)]


async def test_the_scene_is_part_of_what_a_request_id_stands_for(runtime):
    await runtime.execute(OWNER, scene("turn-1", "scene.movie"))
    with pytest.raises(IdempotencyConflict):
        await runtime.execute(OWNER, scene("turn-1", "scene.sleep"))
    with pytest.raises(IdempotencyConflict):
        await runtime.execute(OWNER, request("turn-1", AC_ON))


async def test_a_scene_button_on_a_panel_is_a_touch(runtime, panels):
    await runtime.attach_panel(OWNER, PANEL)
    touch = PanelExecute(request_id="touch-1", scene_id="scene.movie")
    result = await runtime.handle_panel_execute(OWNER, PANEL, touch)
    validate_execute_result(apartment().scene("scene.movie").actions, result)
    assert panels.last(PANEL, OP_DELTA)["source"] == {"kind": "touch", "label": None}


# -- the panel's own requests --------------------------------------------------


async def test_a_panel_request_body_is_dispatched_by_type(runtime, panels):
    await runtime.attach_panel(OWNER, PANEL)
    touch = PanelExecute(request_id="touch-1", commands=(AC_ON,))
    await runtime.handle_panel_request(OWNER, PANEL, panel_request(touch))
    assert panels.last(PANEL, OP_DELTA)["source"]["kind"] == "touch"
    await runtime.handle_panel_request(OWNER, PANEL, panel_request(PanelSync(known_seq=0)))
    assert panels.ops(PANEL)[-1] == (OP_SNAPSHOT, 1)


@pytest.mark.parametrize(
    "body",
    [
        None,
        "smarthome.sync",
        [],
        {"schema_v": 2, "type": "smarthome.sync", "payload": {}},
        {"schema_v": 1, "type": "smarthome.forget", "payload": {}},
        {"schema_v": 1, "type": "smarthome.sync", "payload": {}, "owner_id": "owner-b"},
        {"schema_v": 1, "type": "smarthome.execute", "payload": {"request_id": "t1"}},
        {
            "schema_v": 1,
            "type": "smarthome.execute",
            "payload": {
                "request_id": "t1",
                "scene_id": "scene.movie",
                "commands": [{"device_id": "living.ac", "trait": "on_off", "command": "on"}],
            },
        },
        {
            "schema_v": 1,
            "type": "smarthome.execute",
            "payload": {"request_id": "t1", "scene_id": "scene.movie", "origin": {"kind": "voice"}},
        },
    ],
)
async def test_a_malformed_panel_request_is_dropped_without_raising(runtime, panels, body):
    await runtime.attach_panel(OWNER, PANEL)
    await runtime.handle_panel_request(OWNER, PANEL, body)
    assert panels.ops(PANEL) == [(OP_SNAPSHOT, 0)]


async def test_a_panel_request_that_cannot_be_served_does_not_raise(runtime, registry, panels):
    first = PanelExecute(request_id="touch-1", commands=(AC_ON,))
    await runtime.handle_panel_request(OWNER, PANEL, panel_request(first))
    reused = PanelExecute(request_id="touch-1", scene_id="scene.movie")
    await runtime.handle_panel_request(OWNER, PANEL, panel_request(reused))
    panels.unreachable.add(PANEL)
    await runtime.handle_panel_request(OWNER, PANEL, panel_request(PanelSync()))
    del registry.registries[OTHER_OWNER]
    await runtime.handle_panel_request(OTHER_OWNER, PANEL, panel_request(PanelSync()))
    assert panels.sent == []


# -- local time ----------------------------------------------------------------


async def test_the_snapshot_carries_the_hosts_utc_offset_at_that_moment(make_runtime, panels):
    asked = []

    def offset(now_ms: int) -> int:
        asked.append(now_ms)
        return 480

    runtime = make_runtime(utc_offset_minutes=offset)
    await runtime.attach_panel(OWNER, PANEL)
    assert panels.last(PANEL, OP_SNAPSHOT)["utc_offset_minutes"] == 480
    assert asked == [NOW_MS]


def test_the_default_offset_is_this_hosts_local_one():
    offset = local_utc_offset_minutes(NOW_MS)
    assert -720 <= offset <= 840
