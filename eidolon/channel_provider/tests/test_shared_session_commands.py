import asyncio

import pytest
from eidolon.channel_provider.contracts import (
    IdempotencyConflict,
    InvalidTransition,
    BackendUnavailable,
)
from .test_shared_transport_scope import setup
from .helpers import provision_payload


async def fixture(tmp_path):
    service, _, adapter, _, selection, requests = await setup(tmp_path)
    specs = tuple(
        {
            "device_ref": r.device_ref.model_dump(mode="json"),
            "device": provision_payload(device_id=r.device_id)["device"],
        }
        for r in requests
    )
    return service, adapter, selection, specs


async def test_open_survives_request_return_and_replay_then_explicit_close(tmp_path):
    service, adapter, selection, specs = await fixture(tmp_path)
    ready = await service.open_shared_session(selection, specs, authenticated_owner_id="owner_1")
    replay = await service.open_shared_session(selection, specs, authenticated_owner_id="owner_1")
    assert ready == replay and ready["state"] == "transport_ready"
    assert len(adapter.shared_created) == 1 and not adapter.closed
    await service.close_shared_session(selection.session_id, authenticated_owner_id="another")
    assert not adapter.closed
    await service.close_shared_session(selection.session_id, authenticated_owner_id="owner_1")
    await service.close_shared_session(selection.session_id, authenticated_owner_id="owner_1")
    assert len(adapter.closed) == 1 and not service._shared_visits


async def test_cancelled_request_does_not_end_admitted_transport(tmp_path):
    service, adapter, selection, specs = await fixture(tmp_path)
    blocked, release = asyncio.Event(), asyncio.Event()
    deliver = adapter.deliver_shared_invitation

    async def wait(*a, **kw):
        blocked.set()
        await release.wait()
        return await deliver(*a, **kw)

    adapter.deliver_shared_invitation = wait
    request = asyncio.create_task(
        service.open_shared_session(selection, specs, authenticated_owner_id="owner_1")
    )
    await blocked.wait()
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    release.set()
    assert (await service.open_shared_session(selection, specs, authenticated_owner_id="owner_1"))[
        "state"
    ] == "transport_ready"
    assert not adapter.closed
    await service.shutdown()
    assert not service._shared_visits


async def test_conflicting_repeat_is_rejected(tmp_path):
    service, _, selection, specs = await fixture(tmp_path)
    await service.open_shared_session(selection, specs, authenticated_owner_id="owner_1")
    try:
        with pytest.raises(IdempotencyConflict):
            await service.open_shared_session(
                selection, specs[::-1], authenticated_owner_id="owner_1"
            )
    finally:
        await service.shutdown()


async def test_close_during_admission_unblocks_start_and_cleans(tmp_path):
    service, adapter, selection, specs = await fixture(tmp_path)
    blocked = asyncio.Event()

    async def wait(*a, **kw):
        blocked.set()
        await asyncio.Event().wait()

    adapter.deliver_shared_invitation = wait
    request = asyncio.create_task(
        service.open_shared_session(selection, specs, authenticated_owner_id="owner_1")
    )
    await blocked.wait()
    await service.close_shared_session(selection.session_id, authenticated_owner_id="owner_1")
    with pytest.raises(InvalidTransition):
        await request
    assert len(adapter.closed) == 1


async def test_failed_cleanup_is_not_success_and_next_close_retries(tmp_path):
    service, adapter, selection, specs = await fixture(tmp_path)
    await service.open_shared_session(selection, specs, authenticated_owner_id="owner_1")
    original = adapter.close

    async def fail(handle):
        raise BackendUnavailable("cleanup failed")

    adapter.close = fail
    with pytest.raises(BackendUnavailable):
        await service.close_shared_session(selection.session_id, authenticated_owner_id="owner_1")
    assert service._shared_visits
    adapter.close = original
    await service.close_shared_session(selection.session_id, authenticated_owner_id="owner_1")
    assert not service._shared_visits and len(adapter.closed) == 1


async def test_close_before_background_visit_starts_unblocks_open(tmp_path):
    service, adapter, selection, specs = await fixture(tmp_path)
    request = asyncio.create_task(
        service.open_shared_session(selection, specs, authenticated_owner_id="owner_1")
    )
    await asyncio.sleep(0)
    await service.close_shared_session(selection.session_id, authenticated_owner_id="owner_1")
    with pytest.raises(InvalidTransition):
        await asyncio.wait_for(request, 1)
    assert not service._shared_visits and not adapter.shared_created
