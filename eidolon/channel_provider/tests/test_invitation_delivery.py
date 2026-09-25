import asyncio
import json
import time
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


def receipt(connection, device, *, ref="invite-1", status="accepted", sender=None):
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
                    op="shared-session.invite",
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
    assert not adapter._invitation_receipts
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
    assert not adapter._invitation_receipts
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
    assert not adapter._invitation_receipts
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
    assert not adapter._invitation_receipts
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
