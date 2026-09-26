import asyncio
import json
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
from eidolon_sdk.biz.control.shared_session import SharedSessionInvitation
from eidolon_sdk.biz.control.channel_binding import ChannelBinding
from eidolon_sdk.device_foundation.v1 import DeviceRef
from eidolon_sdk.biz.contracts import CONTROL_TOPIC

from eidolon.channel_provider.contracts import BackendUnavailable, InvalidTransition
from .test_livekit_adapter import _adapter, _spec, _listening


def invitation(device, *, now=None):
    now = now or int(time.time() * 1000)
    return SharedSessionInvitation(
        session_id="team-1",
        deadline_ms=now + 10000,
        device_ref=DeviceRef(
            device_instance_id=device,
            owner_domain_id="owner-domain-1",
            owner_domain_generation=1,
            claim_generation=1,
            trust_epoch=1,
        ),
        channel=ChannelBinding(
            channel_id="temporary",
            purpose="shared-session",
            kinds=("reliable-data",),
            binding_format="test/v1",
            issued_at_ms=now,
            expires_at_ms=now + 20000,
            opaque_binding="dGVzdA==",
        ),
    )


def receipt(connection, device, *, ref="invite-1", status="accepted", sender=None, op="shared-session.invite"):
    connection.handlers["data_received"](
        SimpleNamespace(
            topic=CONTROL_TOPIC,
            participant=SimpleNamespace(identity=sender or device),
            data=json.dumps(
                dict(
                    v=1,
                    kind="ack",
                    ref=ref,
                    device_id=device,
                    op=op,
                    status=status,
                )
            ).encode(),
        )
    )


async def setup(monkeypatch):
    adapter, _ = _adapter()
    grant = await adapter.open(_spec(), issued_at_ms=1000)
    connection = await _listening(monkeypatch, adapter, grant)
    sent = []
    published = asyncio.Event()

    async def publish(data, **kwargs):
        sent.append((json.loads(data), kwargs))
        published.set()

    connection.local_participant = SimpleNamespace(publish_data=publish)
    return adapter, grant, connection, sent, published


async def test_delivery_uses_existing_listener_and_targeted_reliable_packet(monkeypatch):
    adapter, grant, conn, sent, published = await setup(monkeypatch)
    device = grant.handle["device"]
    task = asyncio.create_task(
        adapter.deliver_shared_invitation(grant.handle, invitation(device), command_id="invite-1")
    )
    await asyncio.wait_for(published.wait(), 1)
    assert sent[0][1] == dict(reliable=True, topic=CONTROL_TOPIC, destination_identities=[device])
    assert not task.done()  # Publishing is not an acceptance receipt.
    receipt(conn, device, ref="old")
    receipt(conn, device, sender="other-device")
    assert not task.done()
    receipt(conn, device)
    assert await task == "accepted"
    await adapter.shutdown()


@pytest.mark.parametrize(
    "status,expected",
    [
        ("unsupported", "failed"),
        ("rejected", "rejected"),
        ("completed", "succeeded"),
        ("error", "failed"),
    ],
)
async def test_refusal_and_success_are_not_confused(monkeypatch, status, expected):
    adapter, grant, conn, _, published = await setup(monkeypatch)
    device = grant.handle["device"]
    task = asyncio.create_task(
        adapter.deliver_shared_invitation(grant.handle, invitation(device), command_id="invite-1")
    )
    await asyncio.wait_for(published.wait(), 1)
    receipt(conn, device, status=status)
    assert await task == expected
    await adapter.shutdown()


async def test_disconnection_fails_pending_delivery_and_releases_slot(monkeypatch):
    adapter, grant, conn, _, published = await setup(monkeypatch)
    task = asyncio.create_task(
        adapter.deliver_shared_invitation(
            grant.handle, invitation(grant.handle["device"]), command_id="invite-1"
        )
    )
    await asyncio.wait_for(published.wait(), 1)
    conn.drop()
    with pytest.raises(BackendUnavailable):
        await task
    assert not adapter._control_receipts
    await adapter.shutdown()


async def test_playback_stop_bypasses_pending_control_without_losing_correlation(monkeypatch):
    from eidolon_sdk.biz.control.protocol import build_command_envelope

    adapter, grant, connection, sent, published = await setup(monkeypatch)
    device = grant.handle["device"]
    commands = [build_command_envelope(command_id=ref, device_id=device,
                payload={}, op=op) for ref, op in
                (("join", "room.join"), ("stop", "playback.stop"))]
    tasks = []
    try:
        tasks.append(asyncio.create_task(adapter.deliver_control(
            grant.handle, commands[0], wait_for_terminal=True)))
        await asyncio.wait_for(published.wait(), 1)
        published.clear()
        tasks.append(asyncio.create_task(adapter.deliver_control(
            grant.handle, commands[1], wait_for_terminal=True)))
        await asyncio.wait_for(published.wait(), 1)
        assert [item[0]["id"] for item in sent] == ["join", "stop"]
        # A duplicate urgent command must not replace the original receipt future.
        with pytest.raises(InvalidTransition):
            await adapter.deliver_control(grant.handle, commands[1], wait_for_terminal=True)
        ordinary = build_command_envelope(command_id="join-2", device_id=device,
                                          payload={}, op="room.join")
        with pytest.raises(InvalidTransition):
            await adapter.deliver_control(grant.handle, ordinary, wait_for_terminal=True)
        receipt(connection, device, ref="stop", op="playback.stop", status="accepted")
        receipt(connection, device, ref="stop", op="room.join", status="completed")
        await asyncio.sleep(0)
        assert not any(task.done() for task in tasks)
        receipt(connection, device, ref="stop", op="playback.stop", status="completed")
        assert await tasks[1] == "succeeded"
        assert not tasks[0].done()
        receipt(connection, device, ref="join", op="room.join", status="completed")
        assert await tasks[0] == "succeeded"
        assert not adapter._control_receipts
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await adapter.shutdown()


async def test_slow_stop_receipt_does_not_block_another_device(monkeypatch):
    from eidolon_sdk.biz.control.protocol import build_command_envelope

    adapter, first, connection, sent, published = await setup(monkeypatch)
    second = await adapter.open(replace(_spec(), device_id="second-device"), issued_at_ms=1000)
    other_connection = await _listening(monkeypatch, adapter, second)
    other_published = asyncio.Event()

    async def publish(data, **kwargs):
        assert kwargs["destination_identities"] == [second.handle["device"]]
        other_published.set()

    other_connection.local_participant = SimpleNamespace(publish_data=publish)
    tasks = []
    try:
        # Identical command IDs across distinct device channels remain independent.
        for grant in (first, second):
            command = build_command_envelope(command_id="stop", device_id=grant.handle["device"],
                                             payload={}, op="playback.stop")
            tasks.append(asyncio.create_task(adapter.deliver_control(
                grant.handle, command, wait_for_terminal=True)))
        await asyncio.wait_for(asyncio.gather(published.wait(), other_published.wait()), 1)
        receipt(other_connection, second.handle["device"], ref="stop",
                op="playback.stop", status="completed")
        assert await asyncio.wait_for(tasks[1], 1) == "succeeded"
        assert not tasks[0].done()
        connection.drop()
        with pytest.raises(BackendUnavailable):
            await tasks[0]
        assert not adapter._control_receipts
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await adapter.shutdown()


async def test_expired_invitation_never_sends(monkeypatch):
    adapter, grant, _, sent, _ = await setup(monkeypatch)
    with pytest.raises(InvalidTransition):
        await adapter.deliver_shared_invitation(
            grant.handle, invitation(grant.handle["device"], now=1000), command_id="invite-1"
        )
    assert not sent
    await adapter.shutdown()


async def test_cancellation_releases_slot_and_late_ack_cannot_complete_new_command(monkeypatch):
    adapter, grant, conn, _, published = await setup(monkeypatch)
    value = invitation(grant.handle["device"])
    first = asyncio.create_task(
        adapter.deliver_shared_invitation(grant.handle, value, command_id="invite-1")
    )
    await asyncio.wait_for(published.wait(), 1)
    with pytest.raises(InvalidTransition):
        await adapter.deliver_shared_invitation(grant.handle, value, command_id="concurrent")
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert not adapter._control_receipts
    published.clear()
    second = asyncio.create_task(
        adapter.deliver_shared_invitation(grant.handle, value, command_id="invite-2")
    )
    await asyncio.wait_for(published.wait(), 1)
    receipt(conn, grant.handle["device"])
    assert not second.done()
    receipt(conn, grant.handle["device"], ref="invite-2")
    assert await second == "accepted"
    await adapter.shutdown()


async def test_receipt_timeout_covers_publish_and_cleans_up(monkeypatch):
    adapter, grant, conn, _, _ = await setup(monkeypatch)
    value = invitation(grant.handle["device"])
    # Bound publish itself, not just the wait following it.
    value = value.model_copy(update={"deadline_ms": int(time.time() * 1000) + 40})

    async def hanging_publish(*args, **kwargs):
        await asyncio.Event().wait()

    conn.local_participant.publish_data = hanging_publish
    with pytest.raises(BackendUnavailable, match="timed out"):
        await adapter.deliver_shared_invitation(grant.handle, value, command_id="invite-1")
    assert not adapter._control_receipts
    await adapter.shutdown()


async def test_stop_listening_terminates_receipt_wait(monkeypatch):
    adapter, grant, _, _, published = await setup(monkeypatch)
    task = asyncio.create_task(
        adapter.deliver_shared_invitation(
            grant.handle, invitation(grant.handle["device"]), command_id="invite-1"
        )
    )
    await asyncio.wait_for(published.wait(), 1)
    await adapter.stop_accepting(grant.handle)
    with pytest.raises(BackendUnavailable):
        await task
    assert not adapter._control_receipts
    await adapter.shutdown()


async def test_malformed_or_unresolved_sender_is_not_an_ack(monkeypatch):
    adapter, grant, conn, _, published = await setup(monkeypatch)
    device = grant.handle["device"]
    task = asyncio.create_task(
        adapter.deliver_shared_invitation(grant.handle, invitation(device), command_id="invite-1")
    )
    await asyncio.wait_for(published.wait(), 1)
    for data in (b"null", b"{bad", b"[]", b'{"v":true}'):
        conn.handlers["data_received"](
            SimpleNamespace(
                topic=CONTROL_TOPIC, data=data, participant=SimpleNamespace(identity=device)
            )
        )
    conn.handlers["data_received"](
        SimpleNamespace(
            topic=CONTROL_TOPIC,
            participant=None,
            data=json.dumps(
                dict(
                    v=1,
                    kind="ack",
                    ref="invite-1",
                    device_id=device,
                    op="shared-session.invite",
                    status="accepted",
                )
            ).encode(),
        )
    )
    assert not task.done()
    receipt(conn, device)
    assert await task == "accepted"
    await adapter.shutdown()


async def test_room_join_waits_for_completion_and_ignores_other_operation(monkeypatch):
    from eidolon_sdk.biz.control.protocol import build_command_envelope
    adapter, grant, connection, sent, published = await setup(monkeypatch)
    device = grant.handle["device"]
    command = build_command_envelope(command_id="join", device_id=device,
        payload={}, op="room.join")
    task = asyncio.create_task(adapter.deliver_control(grant.handle, command, wait_for_terminal=True))
    await asyncio.wait_for(published.wait(), 1)
    receipt(connection, device, ref="join", op="room.join", status="accepted")
    receipt(connection, device, ref="join", op="shared-session.invite", status="completed")
    await asyncio.sleep(0)
    assert not task.done()
    receipt(connection, device, ref="join", op="room.join", status="completed")
    assert await task == "succeeded"
    assert not adapter._control_receipts
    await adapter.shutdown()
