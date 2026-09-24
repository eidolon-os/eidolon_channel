import asyncio

import pytest
from eidolon_sdk.biz.control.shared_session import SharedSessionSelection
from eidolon_sdk.device_foundation.v1.testing import named_device_instance_id
from eidolon.channel_provider.contracts import (
    Forbidden,
    InvalidTransition,
    UnknownChannel,
    ProvisionRequest,
    RevokeRequest,
)
from .helpers import encoded, provision_payload, revoke_payload
from .test_service import _service


async def setup_selection(tmp_path, second_owner="owner_1"):
    clock = [1_700_000_000_000]
    service, store, adapter = _service(tmp_path, clock)
    refs = []
    for i, owner in enumerate(("owner_1", second_owner)):
        payload = provision_payload(
            device_id=named_device_instance_id(f"endpoint-{i}"), owner_id=owner
        )
        payload["operation_id"] = f"enrollment-{i}"
        req = ProvisionRequest.parse(encoded(payload))
        await service.provision(req)
        refs.append(req.device_ref)
    selection = SharedSessionSelection(
        session_id="shared-1", devices=tuple(refs), input_device_id=refs[1].device_instance_id
    )
    return service, store, adapter, clock, selection


async def test_selection_reads_existing_channels_without_resources_or_credentials(tmp_path):
    service, store, adapter, _, request = await setup_selection(tmp_path)
    before = (len(adapter.opened), len(adapter.closed))
    snapshots = await service.inspect_shared_selection(request, authenticated_owner_id="owner_1")
    assert tuple(s.device_ref for s in snapshots) == request.devices
    assert (len(adapter.opened), len(adapter.closed)) == before
    assert adapter.sessions_opened == []
    assert adapter.sessions_closed == []
    assert all(
        set(s.model_dump())
        == {
            "device_ref",
            "channel_id",
            "manifest_revision",
            "expires_at_ms",
            "on_channel",
            "observed_at_ms",
        }
        for s in snapshots
    )
    assert snapshots[0].channel_id == store.active_device(request.devices[0]).channel_id


@pytest.mark.parametrize("owner", ["", "owner_2"])
async def test_selection_checks_authenticated_business_owner(tmp_path, owner):
    service, _, adapter, _, request = await setup_selection(tmp_path)
    with pytest.raises(Forbidden):
        await service.inspect_shared_selection(request, authenticated_owner_id=owner)
    assert adapter.sessions_opened == []
    assert adapter.presence_reads == []


async def test_same_domain_does_not_make_another_owners_device_selectable(tmp_path):
    service, _, adapter, _, request = await setup_selection(tmp_path, second_owner="owner_2")
    with pytest.raises(Forbidden):
        await service.inspect_shared_selection(request, authenticated_owner_id="owner_1")
    assert adapter.sessions_opened == []
    assert adapter.presence_reads == []


@pytest.mark.parametrize("present", [True, False, None])
async def test_selection_observes_transport_presence_not_enrollment(tmp_path, present):
    service, _, adapter, clock, request = await setup_selection(tmp_path)
    adapter.on_channel = present
    snapshots = await service.inspect_shared_selection(request, authenticated_owner_id="owner_1")
    assert [s.on_channel for s in snapshots] == [present, present]
    assert all(s.observed_at_ms == clock[0] for s in snapshots)
    assert len(adapter.presence_reads) == 2
    assert adapter.sessions_opened == []


@pytest.mark.parametrize("failure", ["error", "timeout", "invalid"])
async def test_unobservable_transport_is_unknown_not_offline(tmp_path, monkeypatch, failure):
    service, _, adapter, _, request = await setup_selection(tmp_path)
    monkeypatch.setattr("eidolon.channel_provider.service.PRESENCE_READ_TIMEOUT_SECONDS", 0.01)

    async def read(handle):
        if failure == "error":
            raise OSError("unreachable")
        if failure == "timeout":
            await asyncio.Event().wait()
        return "true"

    adapter.device_is_on_channel = read
    snapshots = await service.inspect_shared_selection(request, authenticated_owner_id="owner_1")
    assert all(s.on_channel is None for s in snapshots)
    assert adapter.sessions_opened == []


@pytest.mark.parametrize("change", ["revoke", "expire", "refresh"])
async def test_presence_wait_does_not_lock_lifecycle_or_return_stale_selection(tmp_path, change):
    service, _, adapter, clock, request = await setup_selection(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()

    async def read(handle):
        entered.set()
        await release.wait()
        return True

    adapter.device_is_on_channel = read
    task = asyncio.create_task(
        service.inspect_shared_selection(request, authenticated_owner_id="owner_1")
    )
    try:
        await asyncio.wait_for(entered.wait(), 1)
        if change == "revoke":
            payload = revoke_payload(device_ref=request.devices[1].model_dump(mode="json"))
            await asyncio.wait_for(service.revoke(RevokeRequest.parse(encoded(payload))), 1)
        elif change == "expire":
            clock[0] += 1_800_001
        else:
            payload = provision_payload(device_id=request.devices[1].device_instance_id)
            payload["device"]["manifest"]["title"] = "Updated device"
            payload["device"]["manifest_revision"] = "sha256:manifest-2"
            payload.update(operation="channel.refresh-device", operation_id="refresh-during-probe")
            await asyncio.wait_for(service.provision(ProvisionRequest.parse(encoded(payload))), 1)
        release.set()
        with pytest.raises(InvalidTransition if change == "refresh" else UnknownChannel):
            await task
    finally:
        release.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_cancelling_inspection_cancels_probes_without_starting_anything(tmp_path):
    service, _, adapter, _, request = await setup_selection(tmp_path)
    entered = asyncio.Event()
    stopped = []

    async def read(handle):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.append(handle)

    adapter.device_is_on_channel = read
    task = asyncio.create_task(
        service.inspect_shared_selection(request, authenticated_owner_id="owner_1")
    )
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(stopped) == 2
    assert adapter.sessions_opened == []


@pytest.mark.parametrize("change", ["expired", "revoked", "unknown", "epoch"])
async def test_every_selected_device_must_still_have_its_exact_live_lifecycle(tmp_path, change):
    service, store, adapter, clock, request = await setup_selection(tmp_path)
    if change == "expired":
        clock[0] += 1_800_001
    elif change == "revoked":
        payload = revoke_payload(device_ref=request.devices[1].model_dump(mode="json"))
        await service.revoke(RevokeRequest.parse(encoded(payload)))
    else:
        payload = request.model_dump(mode="json")
        if change == "epoch":
            payload["devices"][1]["trust_epoch"] += 1
        else:
            payload["devices"][1]["device_instance_id"] = named_device_instance_id("unknown")
            payload["input_device_id"] = payload["devices"][1]["device_instance_id"]
        request = SharedSessionSelection.model_validate(payload)
    before = (len(adapter.opened), len(adapter.closed))
    with pytest.raises(UnknownChannel):
        await service.inspect_shared_selection(request, authenticated_owner_id="owner_1")
    assert (len(adapter.opened), len(adapter.closed)) == before
    assert adapter.sessions_opened == []
    if change == "expired":
        assert (
            store.active_device(request.devices[0]) is not None
        )  # inspection made no persistence change
