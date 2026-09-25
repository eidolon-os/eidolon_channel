import asyncio
import json
from dataclasses import replace

import pytest
from eidolon_sdk.biz.control.shared_session import SharedSessionSelection
from eidolon_sdk.device_foundation.v1.testing import named_device_instance_id
from eidolon.channel_provider.contracts import (
    ProvisionRequest,
    Forbidden,
    IdempotencyConflict,
    InvalidTransition,
    RevokeRequest,
)
from eidolon.channel_provider.ports import ChannelGrant
from .helpers import encoded, provision_payload, revoke_payload
from .test_service import _service


@pytest.mark.parametrize("presence", [False, None])
async def test_unreachable_member_refuses_before_creating_room(tmp_path, presence):
    service, _, adapter, clock, selected, requests = await setup(tmp_path)
    async def unavailable(handle):
        clock[0] += 21000
        return presence
    adapter.device_is_on_channel = unavailable
    with pytest.raises(InvalidTransition, match="reachable"):
        async with service.shared_transport(selected, requests, authenticated_owner_id="owner_1"):
            pytest.fail("must not invite unreachable members")
    assert not adapter.shared_created
    assert not service._shared_scopes


async def test_provision_replay_preserves_shared_scope_but_refresh_retires_it(tmp_path):
    service, _, adapter, _, selected, requests = await setup(tmp_path)
    entered = asyncio.Event()

    async def run():
        async with service.shared_transport(selected, requests, authenticated_owner_id="owner_1"):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(run())
    await asyncio.wait_for(entered.wait(), 1)
    await service.provision(requests[0])
    assert not task.done()
    assert not adapter.closed
    raw = provision_payload(device_id=requests[0].device_id)
    raw["device_ref"] = requests[0].device_ref.model_dump(mode="json")
    raw["operation"] = "channel.refresh-device"
    raw["operation_id"] = "refresh-after-shared"
    raw["device"]["manifest_revision"] = "sha256:manifest-2"
    response = await asyncio.wait_for(service.provision(ProvisionRequest.parse(encoded(raw))), 1)
    assert json.loads(response)["operation"] == "channel.provisioned-device"
    with pytest.raises(asyncio.CancelledError):
        await task
    assert adapter.closed[0]["resource"] == "temporary"
    assert not service._shared_scopes


async def setup(tmp_path):
    clock = [1700000000000]
    service, store, adapter = _service(tmp_path, clock)
    requests = []
    for i in range(2):
        raw = provision_payload(device_id=named_device_instance_id(f"endpoint-{i}"))
        raw["operation_id"] = f"enrollment-{i}"
        req = ProvisionRequest.parse(encoded(raw))
        await service.provision(req)
        requests.append(req)
    selected = SharedSessionSelection(
        session_id="group-1",
        devices=tuple(r.device_ref for r in requests),
        input_device_id=requests[0].device_id,
    )
    adapter.on_channel = True
    adapter.invites = []
    adapter.shared_created = []
    adapter.ack_status = "accepted"

    async def open_shared(specs, *, input_device_id, issued_at_ms):
        adapter.shared_created.append(tuple(specs))
        return {
            s.device_id: ChannelGrant(
                binding_format="test/v1",
                payload=b"{}",
                expires_at_ms=issued_at_ms + 30000,
                handle={"resource": "temporary", "device": s.device_id},
            )
            for s in specs
        }

    async def deliver(handle, invitation, *, command_id):
        adapter.invites.append((handle, invitation, command_id))
        return adapter.ack_status

    adapter.open_shared = open_shared
    adapter.deliver_shared_invitation = deliver
    return service, store, adapter, clock, selected, tuple(requests)


async def test_scope_joins_all_members_then_closes_only_temporary_transport(tmp_path):
    service, store, adapter, _, selected, requests = await setup(tmp_path)
    before = tuple(store.active_provisions())
    async with service.shared_transport(
        selected, requests, authenticated_owner_id="owner_1"
    ) as ready:
        assert ready["session_id"] == selected.session_id
        assert ready["state"] == "transport_ready"
        assert len(adapter.invites) == 2
        assert not adapter.closed
        assert adapter.sessions_opened == []
    assert len(adapter.closed) == 1
    assert adapter.closed[0]["resource"] == "temporary"
    assert tuple(store.active_provisions()) == before
    assert not service._shared_scopes


@pytest.mark.parametrize("case", ["owner", "fingerprint", "missing"])
async def test_invalid_authority_cannot_create_a_shared_room(tmp_path, case):
    service, _, adapter, _, selected, requests = await setup(tmp_path)
    owner = "owner_1"
    if case == "owner":
        owner = "another"
    if case == "fingerprint":
        requests = (replace(requests[0], fingerprint="forged"), requests[1])
    if case == "missing":
        requests = requests[:1]
    with pytest.raises((Forbidden, IdempotencyConflict, InvalidTransition)):
        async with service.shared_transport(selected, requests, authenticated_owner_id=owner):
            pytest.fail("must not admit")
    assert not adapter.shared_created


async def test_refused_invitation_closes_room_and_releases_members(tmp_path):
    service, _, adapter, _, selected, requests = await setup(tmp_path)
    adapter.ack_status = "rejected"
    with pytest.raises(InvalidTransition):
        async with service.shared_transport(selected, requests, authenticated_owner_id="owner_1"):
            pytest.fail("must not admit")
    assert len(adapter.closed) == 1
    assert not service._shared_scopes


async def test_cancellation_in_scope_closes_room(tmp_path):
    service, _, adapter, _, selected, requests = await setup(tmp_path)
    entered = asyncio.Event()

    async def run():
        async with service.shared_transport(selected, requests, authenticated_owner_id="owner_1"):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(run())
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(adapter.closed) == 1
    assert not service._shared_scopes


async def test_revocation_cancels_scope_before_retiring_original_channel(tmp_path):
    service, _, adapter, _, selected, requests = await setup(tmp_path)
    entered = asyncio.Event()

    async def run():
        async with service.shared_transport(selected, requests, authenticated_owner_id="owner_1"):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(run())
    await asyncio.wait_for(entered.wait(), 1)
    raw = revoke_payload(device_id=requests[0].device_id)
    raw["device_ref"] = requests[0].device_ref.model_dump(mode="json")
    await service.revoke(RevokeRequest.parse(encoded(raw)))
    with pytest.raises(asyncio.CancelledError):
        await task
    assert adapter.closed[0]["resource"] == "temporary"
    assert len(adapter.closed) == 2


async def test_changed_manifest_with_copied_fingerprint_is_rejected(tmp_path):
    service, _, adapter, _, selected, requests = await setup(tmp_path)
    changed = replace(requests[0].device, display_name="forged-new-spec")
    requests = (replace(requests[0], device=changed), requests[1])
    with pytest.raises(IdempotencyConflict):
        async with service.shared_transport(selected, requests, authenticated_owner_id="owner_1"):
            pytest.fail("must not admit")
    assert not adapter.shared_created


async def test_shared_reservation_blocks_second_scope_and_single_chat(tmp_path):
    service, _, _, _, selected, requests = await setup(tmp_path)
    async with service.shared_transport(selected, requests, authenticated_owner_id="owner_1"):
        with pytest.raises(InvalidTransition):
            async with service.shared_transport(
                selected, requests, authenticated_owner_id="owner_1"
            ):
                pytest.fail("duplicate reservation")
        with pytest.raises(InvalidTransition):
            await service._converge_serving(
                requests[0].device_ref,
                serving=True,
                conversation_id="single-1",
                session_intent="user_initiated",
            )


async def test_accepted_ack_without_actual_membership_cannot_admit(tmp_path):
    service, _, adapter, clock, selected, requests = await setup(tmp_path)

    async def present(handle):
        if handle.get("resource") == "temporary":
            clock[0] += 21000
            return False
        return True

    adapter.device_is_on_channel = present
    with pytest.raises(InvalidTransition, match="deadline"):
        async with service.shared_transport(selected, requests, authenticated_owner_id="owner_1"):
            pytest.fail("ACK alone must not admit")
    assert len(adapter.invites) == 2
    assert len(adapter.closed) == 1


async def test_close_failure_keeps_reservation_until_retry_succeeds(tmp_path):
    from eidolon.channel_provider.contracts import BackendUnavailable

    service, _, adapter, _, selected, requests = await setup(tmp_path)
    original_close = adapter.close

    async def failing_close(handle):
        raise BackendUnavailable("room deletion failed")

    adapter.close = failing_close

    async def run():
        async with service.shared_transport(selected, requests, authenticated_owner_id="owner_1"):
            pass

    with pytest.raises(BackendUnavailable):
        await asyncio.create_task(run())
    assert service._shared_scopes
    adapter.close = original_close
    await service._retire_shared_scope(requests[0].device_id)
    assert not service._shared_scopes
    assert len(adapter.closed) == 1


async def test_failed_send_cancels_other_send_before_room_cleanup(tmp_path):
    from eidolon.channel_provider.contracts import BackendUnavailable

    service, _, adapter, _, selected, requests = await setup(tmp_path)
    waiting = asyncio.Event()
    cancelled = asyncio.Event()

    async def deliver(handle, invitation, *, command_id):
        if invitation.device_ref == requests[0].device_ref:
            await waiting.wait()
            raise BackendUnavailable("send failed")
        waiting.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    adapter.deliver_shared_invitation = deliver
    original_close = adapter.close

    async def close(handle):
        assert cancelled.is_set()
        await original_close(handle)

    adapter.close = close
    with pytest.raises(BackendUnavailable):
        async with service.shared_transport(selected, requests, authenticated_owner_id="owner_1"):
            pytest.fail("must not admit")
    assert len(adapter.closed) == 1
    assert not service._shared_scopes


async def test_revocation_does_not_hold_lifecycle_lock_while_caller_unwinds(tmp_path):
    service, _, adapter, _, selected, requests = await setup(tmp_path)
    entered = asyncio.Event()
    released = asyncio.Event()

    async def run():
        async with service.shared_transport(selected, requests, authenticated_owner_id="owner_1"):
            try:
                entered.set()
                await asyncio.Event().wait()
            finally:
                # Business cleanup can use another existing Provider operation.
                await service._converge_serving(
                    requests[0].device_ref,
                    serving=False,
                    conversation_id="old-single",
                    session_intent="user_initiated",
                )
                released.set()

    task = asyncio.create_task(run())
    await asyncio.wait_for(entered.wait(), 1)
    raw = revoke_payload(device_id=requests[0].device_id)
    raw["device_ref"] = requests[0].device_ref.model_dump(mode="json")
    await asyncio.wait_for(service.revoke(RevokeRequest.parse(encoded(raw))), 1)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert released.is_set()
    assert adapter.closed[0]["resource"] == "temporary"


@pytest.mark.parametrize("mutation", ["none", "operation", "manifest", "duplicate", "bad_ref"])
async def test_current_specifications_use_provider_ledger_and_reject_tampering(tmp_path, mutation):
    service, _, adapter, _, selected, requests = await setup(tmp_path)
    specifications = []
    for request in requests:
        raw = provision_payload(device_id=request.device_id)
        specifications.append(
            {"device_ref": request.device_ref.model_dump(mode="json"), "device": raw["device"]}
        )
    if mutation == "operation":
        specifications[0]["operation_id"] = "forged"
    elif mutation == "manifest":
        specifications[0]["device"]["display_name"] = "forged"
    elif mutation == "duplicate":
        specifications[1] = specifications[0]
    elif mutation == "bad_ref":
        specifications[0]["device_ref"]["device_instance_id"] = []
    if mutation == "none":
        async with service.shared_transport_from_specifications(
            selected,
            tuple(specifications),
            authenticated_owner_id="owner_1",
        ) as ready:
            assert ready["state"] == "transport_ready"
        assert len(adapter.closed) == 1
    else:
        with pytest.raises((InvalidTransition, IdempotencyConflict)):
            async with service.shared_transport_from_specifications(
                selected,
                tuple(specifications),
                authenticated_owner_id="owner_1",
            ):
                pytest.fail("tampered specification admitted")
        assert not adapter.shared_created


async def test_recovering_control_room_is_observed_before_new_invitation(tmp_path):
    service, _, adapter, _, selection, requests = await setup(tmp_path)
    restored = asyncio.Event()
    observed = asyncio.Event()
    async def presence(handle):
        if handle.get("resource") == "temporary":
            return True
        observed.set()
        return restored.is_set()
    adapter.device_is_on_channel = presence
    async def visit():
        async with service.shared_transport(selection, requests, authenticated_owner_id="owner_1"):
            return True
    task = asyncio.create_task(visit())
    await observed.wait()
    await asyncio.sleep(0)
    try:
        assert not task.done()
        assert not adapter.shared_created and not adapter.invites
        restored.set()
        assert await asyncio.wait_for(task, 1)
        assert len(adapter.shared_created) == 1 and len(adapter.invites) == 2
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
