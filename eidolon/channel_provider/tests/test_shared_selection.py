import pytest
from eidolon_sdk.biz.control.shared_session import SharedSessionSelection
from eidolon_sdk.device_foundation.v1.testing import named_device_instance_id
from eidolon.channel_provider.contracts import (
    Forbidden,
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
        set(s.model_dump()) == {"device_ref", "channel_id", "manifest_revision", "expires_at_ms"}
        for s in snapshots
    )
    assert snapshots[0].channel_id == store.active_device(request.devices[0]).channel_id


@pytest.mark.parametrize("owner", ["", "owner_2"])
async def test_selection_checks_authenticated_business_owner(tmp_path, owner):
    service, _, adapter, _, request = await setup_selection(tmp_path)
    with pytest.raises(Forbidden):
        await service.inspect_shared_selection(request, authenticated_owner_id=owner)
    assert adapter.sessions_opened == []


async def test_same_domain_does_not_make_another_owners_device_selectable(tmp_path):
    service, _, adapter, _, request = await setup_selection(tmp_path, second_owner="owner_2")
    with pytest.raises(Forbidden):
        await service.inspect_shared_selection(request, authenticated_owner_id="owner_1")
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
