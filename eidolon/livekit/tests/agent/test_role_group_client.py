"""Channel transport tests; media adapters are fakes, not hardware evidence."""

import asyncio
import json

import aiohttp
from aiohttp import web
import pytest

from eidolon_sdk.biz.control.coordination_stream import OpenScene, ROLE_GROUP_STREAM_PATH
from eidolon_sdk.device_foundation.v1.testing import named_device_instance_id
from eidolon.livekit.agent.coordination.client import RoleGroupClient


def opening():
    def ref(name):
        return dict(
            device_instance_id=named_device_instance_id(name),
            owner_domain_id="owner-domain",
            owner_domain_generation=1,
            claim_generation=1,
            trust_epoch=1,
        )

    return OpenScene.model_validate(
        dict(
            type="open",
            owner_id="owner",
            selection=dict(
                scenario="ip_role_group",
                session_id="scene",
                input_device=ref("input"),
                members=[dict(companion_id=k, output_device=ref(k)) for k in ("a", "b")],
            ),
        )
    )


def frame(kind, **data):
    return json.dumps(dict(type=kind, stream_id="stream", session_id="scene", **data))


def output(client, kind, *, turn="turn", epoch=1, companion="a", **data):
    device = client.opened.selection.members[0].output_device.device_instance_id
    if kind == "reply_start":
        data["companion_id"] = companion
    client.accept(frame(kind, request_id=turn, turn_id=turn, device_id=device, epoch=epoch, **data))


async def stop_ok(device):
    return True


async def unused(*args):
    raise AssertionError("no reply expected")


def prepared(**kwargs):
    client = RoleGroupClient(opening(), present=unused, stop=stop_ok, **kwargs)
    client._running = True
    client.accept(frame("prepared", policy="semantic-step-v2", physical_devices_ready=False))
    return client


def capture(client, *, capture="one", epoch=1):
    client.press(capture)
    client.release(capture)
    client.accept(frame("capturing", capture_id=capture, epoch=epoch))


async def settle(client):
    tasks = list(client._tasks)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_round_ui_completion_is_revoked_by_new_ptt_before_delivery():
    delivered = []
    async def state(frame):
        delivered.append(frame.epoch)
    client = prepared(on_state=state)
    capture(client)
    client.accept(frame('state', capture_id='one', state='waiting', members={}, epoch=1))
    client.press('next')
    await settle(client)
    assert not delivered
    client.release('next')
    client.accept(frame('capturing', capture_id='next', epoch=2))
    client.accept(frame('state', capture_id='one', state='waiting', members={}, epoch=1))
    await settle(client)
    assert not delivered
    client.accept(frame('state', capture_id='next', state='waiting', members={}, epoch=2))
    await settle(client)
    assert delivered == [2]


async def test_local_press_stops_all_members_before_agent_ack_and_drops_late_output():
    client = prepared()
    stopped = []
    gate = asyncio.Event()

    async def stop(device):
        stopped.append(device)
        await gate.wait()
        return True

    client.stop = stop
    capture(client)
    await asyncio.sleep(0)
    assert set(stopped) == set(client.members)  # parallel, no ACK required
    client.press("two")
    output(client, "reply_start", turn="obsolete", epoch=1)
    assert client._playback is None
    gate.set()
    await settle(client)


async def test_no_played_receipt_until_correlated_physical_completion():
    client = prepared()
    played = asyncio.Event()
    consumed = asyncio.Event()

    async def present(start, text, speaking):
        assert [chunk async for chunk in text] == ["hello"]
        speaking()
        consumed.set()
        await played.wait()
        return True

    client.present = present
    capture(client)
    output(client, "reply_start")
    output(client, "reply_delta", text="hello")
    output(client, "reply_end")
    await consumed.wait()
    assert not any(x["type"] == "receipt" for x in client._outbound._queue)
    played.set()
    await settle(client)
    receipts = [x for x in client._outbound._queue if x["type"] == "receipt"]
    assert len(receipts) == 1 and receipts[0]["result"] == "completed"
    with pytest.raises(ValueError, match="replayed reply"):
        output(client, "reply_start")


async def test_press_cancels_playback_and_suppresses_cancel_ignoring_receipt():
    client = prepared()
    started = asyncio.Event()

    async def present(start, text, speaking):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return True  # a late adapter result must not confirm the old turn

    client.present = present
    capture(client)
    output(client, "reply_start")
    await started.wait()
    client.press("two")
    await settle(client)
    assert not any(x["type"] == "receipt" for x in client._outbound._queue)


@pytest.mark.parametrize("fault", ["device", "companion", "stream", "overlap", "future_epoch"])
async def test_wrong_routing_fails_closed(fault):
    client = prepared()
    capture(client)
    data = dict(
        request_id="turn",
        turn_id="turn",
        epoch=1,
        companion_id="a",
        device_id=next(iter(client.members)),
    )
    if fault == "device":
        data["device_id"] = named_device_instance_id("outsider")
    if fault == "companion":
        data["companion_id"] = "b"
    if fault == "future_epoch":
        data["epoch"] = 2
    if fault == "overlap":
        output(client, "reply_start")
    raw = frame("reply_start", **data)
    if fault == "stream":
        raw = raw.replace('"stream"', '"other"')
    with pytest.raises(ValueError):
        client.accept(raw)
    client._revoke()
    await settle(client)


async def test_media_queue_is_bounded_without_blocking_control_reader():
    client = prepared()
    capture(client)
    output(client, "reply_start")
    for _ in range(64):
        output(client, "reply_delta", text="x")
    with pytest.raises(asyncio.QueueFull):
        output(client, "reply_delta", text="x")
    client.press("two")  # still synchronous and can revoke full media queue
    await settle(client)


@pytest.mark.parametrize("stop_success", [True, False])
async def test_real_tcp_disconnect_independently_stops_every_endpoint(stop_success):
    stopped = []

    async def stop(device):
        stopped.append(device)
        return stop_success

    async def handler(request):
        assert request.headers["Authorization"] == "Bearer test-only"
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        assert (await socket.receive_json())["type"] == "open"
        await socket.send_str(
            frame("prepared", policy="semantic-step-v2", physical_devices_ready=False)
        )
        await socket.close()
        return socket

    app = web.Application()
    app.router.add_get(ROLE_GROUP_STREAM_PATH, handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    client = RoleGroupClient(opening(), present=unused, stop=stop)
    try:
        async with aiohttp.ClientSession() as http:
            with pytest.raises(ConnectionError):
                await client.run(http, base_url=f"http://127.0.0.1:{port}", token="test-only")
        assert set(stopped) == set(client.members)
        assert client.closed.is_set()
        assert client.cleanup_ok is stop_success
    finally:
        await runner.cleanup()


async def test_local_stop_failure_blocks_speech_without_destroying_scene():
    client = prepared()

    async def failed(device):
        return False

    client.stop = failed
    capture(client)
    await settle(client)
    assert not client._failure.is_set()
    output(client, "reply_start")
    await settle(client)
    assert client._playback is None
    assert not client._failure.is_set()
    client.press("two")
    await settle(client)


async def test_close_cannot_be_reopened_by_a_late_capture_ack():
    client = prepared()
    client.press("one")
    client.release("one")
    client.close()
    client.accept(frame("capturing", capture_id="one", epoch=1))
    output(client, "reply_start")
    assert client._playback is None
    with pytest.raises(ConnectionError):
        client.press("two")
    await settle(client)


async def test_speaking_and_scene_state_do_not_advance_physical_completion():
    client = prepared()
    capture(client)
    client.accept(
        frame("state", capture_id="one", epoch=1, state="waiting", members={"a": "waiting", "b": "waiting"})
    )
    assert client.state.state == "waiting"
    assert not any(x["type"] == "receipt" for x in client._outbound._queue)
    with pytest.raises(ValueError, match="unknown scene state member"):
        client.accept(frame("state", capture_id="one", epoch=1, state="waiting", members={"outsider": "waiting"}))
    await settle(client)

async def test_failed_round_state_survives_stop_but_cannot_clear_new_capture():
    client = prepared()
    capture(client)
    client.accept(frame('stop', request_id='failure-stop', device_id=named_device_instance_id('a'), epoch=2))
    # The failure stop revoked the old playback epoch, but this terminal outcome
    # belongs to the current capture and must reach the device UI.
    client.accept(frame('state', capture_id='one', epoch=2, state='failed',
                        members={'a': 'waiting', 'b': 'waiting'}, outcome='error',
                        error_code='DECISION_TIMEOUT'))
    assert client.state.error_code == 'DECISION_TIMEOUT'
    client.press('two')
    client.release('two')
    client.accept(frame('state', capture_id='one', epoch=2, state='waiting',
                        members={}, outcome='finished'))
    assert client.state is None
    await settle(client)


async def test_agent_confirmation_reuses_local_stop_even_after_completion():
    client = prepared()
    calls = []
    async def stop(device):
        calls.append(device)
        return True
    client.stop = stop
    capture(client)
    await settle(client)
    for device in client.members:
        client.accept(frame("stop", request_id=device, device_id=device,
                            capture_id="one", epoch=1))
    await settle(client)
    assert len(calls) == len(client.members)
    receipts = [x for x in client._outbound._queue if x["type"] == "receipt"]
    assert len(receipts) == len(client.members)
    assert all(x["result"] == "completed" for x in receipts)


async def test_stop_execution_budget_excludes_lock_wait_and_stale_controls_are_fenced():
    client = prepared(stop_timeout=.1)
    calls = []
    async def stop(device):
        calls.append(device)
        await asyncio.sleep(.06)
        return True
    client.stop = stop
    device = next(iter(client.members))
    assert all(await asyncio.gather(client._stop_device(device), client._stop_device(device)))
    assert len(calls) == 2
    client._last_epoch = 3
    result = await client._stop_device(device, epoch=2)
    assert result.error_code == "TEAM_STOP_SUPERSEDED"
    assert len(calls) == 2


async def test_failed_stop_receipt_keeps_connection_available_for_fresh_capture():
    client = prepared()
    async def failed(device):
        return False
    client.stop = failed
    capture(client)
    for device in client.members:
        client.accept(frame("stop", request_id=device, device_id=device,
                            capture_id="one", epoch=1))
    await settle(client)
    receipts = [x for x in client._outbound._queue if x["type"] == "receipt"]
    assert all(x["error_code"] == "TEAM_STOP_UNCONFIRMED" for x in receipts)
    assert not client._failure.is_set()
    client.stop = stop_ok
    capture(client, capture="two", epoch=3)
    await settle(client)
    assert all(t.result() for t in client._capture_stops["two"].values())


async def test_failed_presentation_discards_inflight_text_without_losing_scene():
    client = prepared()
    async def failed(*args):
        return False
    client.present = failed
    capture(client)
    output(client, "reply_start")
    await settle(client)
    output(client, "reply_delta", text="already queued before failure")
    output(client, "reply_end")
    assert not client._failure.is_set()
    receipts = [x for x in client._outbound._queue if x["type"] == "receipt"]
    assert receipts[-1]["error_code"] == "TEAM_PLAYBACK_UNCONFIRMED"
    client.press("retry")
    await settle(client)
