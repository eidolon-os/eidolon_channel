from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from eidolon_sdk.device_foundation.v1 import DeviceRef

from eidolon_sdk.biz.persona import ResolvedRuntimeIdentity as ResolvedContext
from eidolon.livekit.agent.runtime.resolver import (
    CompanionInteractionContext,
    DeviceConnectionContext,
    DeviceTokenResolverError,
    make_device_token_resolver,
    resolve_channel_context,
)

from eidolon_sdk.device_foundation.v1.testing import named_device_instance_id

# Tests name the device they mean; the name becomes a real device
# instance id, which is a digest of a key and never a chosen string.
_DEVICE_1 = named_device_instance_id("device-1")


def runtime_context(*, device_id=None, companion_id="companion-1"):
    return ResolvedContext(
        owner_id="owner-1",
        companion_id=companion_id,
        memory_realm_id="realm-1",
        genome_id="genome-1",
        schema_version="eidolon.persona_genome",
        genome_hash="genome-hash",
        realizer_version="eidolon.persona_realizer",
        device_id=device_id,
    )


def room(metadata: str):
    participant = SimpleNamespace(identity=_DEVICE_1, metadata=metadata)
    return SimpleNamespace(remote_participants={_DEVICE_1: participant})


pytestmark = pytest.mark.asyncio


def device_ref(device_id: str = _DEVICE_1) -> DeviceRef:
    return DeviceRef(
        device_instance_id=device_id,
        owner_domain_id="owner-domain-1",
        owner_domain_generation=1,
        claim_generation=1,
        trust_epoch=1,
    )


async def test_a_device_nobody_answers_through_resolves_without_a_runtime_lookup():
    mounts = AsyncMock()
    mounts.resolve.return_value = DeviceConnectionContext(
        owner_id="owner-1",
        device_id=_DEVICE_1,
        device_ref=device_ref(),
        mount_revision=7,
        answering_companion_id=None,
    )
    runtime = AsyncMock()

    context = await resolve_channel_context(
        runtime=runtime,
        mounts=mounts,
        identity=_DEVICE_1,
        metadata={"kind": "device", "owner_id": "owner-1"},
    )

    assert isinstance(context, DeviceConnectionContext)
    assert context.answering_companion_id is None
    runtime.resolve_companion.assert_not_called()


async def test_an_assigned_device_resolves_a_complete_companion_interaction_context():
    mounts = AsyncMock()
    mounts.resolve.return_value = DeviceConnectionContext(
        owner_id="owner-1",
        device_id=_DEVICE_1,
        device_ref=device_ref(),
        mount_revision=7,
        answering_companion_id="companion-1",
    )
    runtime = AsyncMock()
    runtime.resolve_companion.return_value = runtime_context(device_id=_DEVICE_1)

    context = await resolve_channel_context(
        runtime=runtime,
        mounts=mounts,
        identity=_DEVICE_1,
        metadata={"kind": "device", "owner_id": "owner-1"},
    )

    assert isinstance(context, CompanionInteractionContext)
    assert context.runtime.device_id == _DEVICE_1
    assert context.mount_revision == 7
    runtime.resolve_companion.assert_awaited_once_with("companion-1", device_id=_DEVICE_1)


async def test_companion_participant_needs_no_device():
    runtime = AsyncMock()
    runtime.resolve_companion.return_value = runtime_context(device_id=None)

    context = await resolve_channel_context(
        runtime=runtime,
        mounts=None,
        identity="companion-1",
        metadata={
            "kind": "companion",
            "owner_id": "owner-1",
            "companion_id": "companion-1",
        },
    )

    assert isinstance(context, CompanionInteractionContext)
    assert context.runtime.device_id is None
    runtime.resolve_companion.assert_awaited_once_with("companion-1", device_id=None)


async def test_device_context_fails_closed_on_owner_mismatch():
    mounts = AsyncMock()
    mounts.resolve.return_value = DeviceConnectionContext(
        owner_id="owner-2",
        device_id=_DEVICE_1,
        device_ref=device_ref(),
        mount_revision=1,
        answering_companion_id=None,
    )

    with pytest.raises(DeviceTokenResolverError, match="owner"):
        await resolve_channel_context(
            runtime=AsyncMock(),
            mounts=mounts,
            identity=_DEVICE_1,
            metadata={"kind": "device", "owner_id": "owner-1"},
        )


async def test_audio_token_resolver_rejects_a_device_nobody_answers_through_before_signing():
    mounts = AsyncMock()
    mounts.resolve.return_value = DeviceConnectionContext(
        owner_id="owner-1",
        device_id=_DEVICE_1,
        device_ref=device_ref(),
        mount_revision=1,
        answering_companion_id=None,
    )
    runtime = AsyncMock()
    resolve = make_device_token_resolver(
        room=room('{"kind":"device","owner_id":"owner-1"}'),
        runtime=runtime,
        mounts=mounts,
        session_id="room-1",
        jwt_secret="secret-with-enough-entropy",
    )

    with pytest.raises(DeviceTokenResolverError, match="no companion"):
        await resolve()
    runtime.resolve_companion.assert_not_called()


async def test_the_owner_entrance_gets_the_owners_default():
    """Tier three of the resolution order, and the only place it applies.

    "Which Companion answers when nothing named one" is one field on the Owner,
    and this is the ingress that asks for it: a person talking through the app
    named no body and no Companion.
    """
    runtime = AsyncMock()
    runtime.resolve_owner.return_value = runtime_context()

    context = await resolve_channel_context(
        runtime=runtime,
        mounts=AsyncMock(),
        identity="owner-1",
        metadata={"kind": "owner", "owner_id": "owner-1"},
    )

    assert isinstance(context, CompanionInteractionContext)
    runtime.resolve_owner.assert_awaited_once_with("owner-1")
    runtime.resolve_companion.assert_not_called()


async def test_an_explicit_companion_outranks_the_bodys_assignment():
    """Tier one over tier two, on the same request.

    The metadata is trusted only as far as the checks below: the resolved
    Companion has to belong to the same Owner and the same device, so naming
    one cannot reach across Owners.
    """
    mounts = AsyncMock()
    mounts.resolve.return_value = DeviceConnectionContext(
        owner_id="owner-1",
        device_id=_DEVICE_1,
        device_ref=device_ref(),
        mount_revision=7,
        answering_companion_id="companion-assigned",
    )
    runtime = AsyncMock()
    runtime.resolve_companion.return_value = runtime_context(
        device_id=_DEVICE_1, companion_id="companion-asked-for"
    )

    context = await resolve_channel_context(
        runtime=runtime,
        mounts=mounts,
        identity=_DEVICE_1,
        metadata={
            "kind": "device",
            "owner_id": "owner-1",
            "companion_id": "companion-asked-for",
        },
    )

    assert isinstance(context, CompanionInteractionContext)
    runtime.resolve_companion.assert_awaited_once_with("companion-asked-for", device_id=_DEVICE_1)


async def test_an_unassigned_device_is_not_given_the_owners_default():
    """The deliberate *absence* of a fallback, pinned so it stays deliberate.

    A physical device with no assignment is a mounted Body carrying nobody. It
    would be easy to read "the default answers when nothing named a Companion"
    as covering this too — and that reading is what must not happen: if an
    unassigned speaker already spoke with the Owner's default, then assigning a
    Companion to it would change nothing observable, and "which Eidolon is in
    this device" would have two answers, the assignment and the fallback.

    The Owner's default answers the *app* entrance, where there is no body to
    assign. A body answers for whoever is assigned to it, or for nobody.
    """
    mounts = AsyncMock()
    mounts.resolve.return_value = DeviceConnectionContext(
        owner_id="owner-1",
        device_id=_DEVICE_1,
        device_ref=device_ref(),
        mount_revision=7,
        answering_companion_id=None,
    )
    runtime = AsyncMock()

    context = await resolve_channel_context(
        runtime=runtime,
        mounts=mounts,
        identity=_DEVICE_1,
        metadata={"kind": "device", "owner_id": "owner-1"},
    )

    assert isinstance(context, DeviceConnectionContext)
    runtime.resolve_owner.assert_not_called()
    runtime.resolve_companion.assert_not_called()


@pytest.mark.parametrize("kind", ["owner", "user"])
async def test_virtual_ingress_explicit_target_uses_shared_resolution(kind):
    runtime = AsyncMock()
    runtime.resolve_companion.return_value = runtime_context(companion_id="visitor")
    context = await resolve_channel_context(
        runtime=runtime,
        mounts=None,
        identity="owner-1",
        metadata={"kind": kind, "owner_id": "owner-1", "companion_id": "visitor"},
    )
    assert context.runtime.companion_id == "visitor"
    assert context.runtime.device_id is None
    runtime.resolve_owner.assert_not_called()


@pytest.mark.parametrize("kind", ["owner", "user"])
async def test_virtual_default_authority_mismatch_is_rejected_before_signing(kind):
    runtime = AsyncMock()
    runtime.resolve_owner.return_value = runtime_context().model_copy(
        update={"owner_id": "other-owner"}
    )
    resolve = make_device_token_resolver(
        room=room(f'{{"kind":"{kind}","owner_id":"owner-1"}}'),
        runtime=runtime,
        session_id="session-1",
        jwt_secret="secret-with-enough-entropy",
    )
    with pytest.raises(DeviceTokenResolverError, match="mismatch"):
        await resolve()


async def test_temporary_target_is_session_scoped_and_next_session_returns_to_assignment():
    import json

    from eidolon_sdk.biz.runtime import RuntimeTokenVerifier
    from eidolon.livekit.agent.runtime.services import ChannelRuntimeServices

    mounts = AsyncMock()
    mounts.resolve.return_value = DeviceConnectionContext(
        owner_id="owner-1",
        device_id=_DEVICE_1,
        device_ref=device_ref(),
        mount_revision=7,
        answering_companion_id="resident",
    )
    runtime = AsyncMock()
    runtime.resolve_companion.side_effect = lambda target, *, device_id: runtime_context(
        device_id=device_id, companion_id=target
    )
    metadata = {"kind": "device", "owner_id": "owner-1", "companion_id": "visitor"}
    active_room = room(json.dumps(metadata))
    services = ChannelRuntimeServices(runtime=runtime, mounts=mounts)
    secret = "secret-with-enough-entropy-for-tests"
    resolve = make_device_token_resolver(
        room=active_room,
        runtime=runtime,
        mounts=mounts,
        context_resolver=services.resolve_room,
        session_id="temporary-session",
        jwt_secret=secret,
    )
    token = await resolve()
    scope = await RuntimeTokenVerifier(secret=secret).verify(token)
    assert (scope.companion_id, scope.device_id, scope.session_id) == (
        "visitor",
        _DEVICE_1,
        "temporary-session",
    )
    # Even an upstream metadata change must not change an active session's role.
    metadata.pop("companion_id")
    active_room.remote_participants[_DEVICE_1].metadata = json.dumps(metadata)
    assert await resolve() == token
    assert (await services.resolve_room(active_room)).companion_id == "visitor"
    runtime.resolve_companion.assert_awaited_once_with("visitor", device_id=_DEVICE_1)

    next_services = ChannelRuntimeServices(runtime=runtime, mounts=mounts)
    next_resolve = make_device_token_resolver(
        room=active_room,
        runtime=runtime,
        mounts=mounts,
        context_resolver=next_services.resolve_room,
        session_id="next-session",
        jwt_secret=secret,
    )
    next_scope = await RuntimeTokenVerifier(secret=secret).verify(await next_resolve())
    assert next_scope.companion_id == "resident"
    assert next_scope.session_id == "next-session"
    assert mounts.resolve.return_value.answering_companion_id == "resident"
    assert [call[0] for call in mounts.mock_calls] == ["resolve", "resolve"]
