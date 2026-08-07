from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from eidolon_sdk.biz.persona import ResolvedRuntimeIdentity as ResolvedContext
from eidolon.livekit.agent.runtime.resolver import (
    CompanionInteractionContext,
    DeviceConnectionContext,
    DeviceTokenResolverError,
    make_device_token_resolver,
    resolve_channel_context,
)


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
    participant = SimpleNamespace(identity="device-1", metadata=metadata)
    return SimpleNamespace(remote_participants={"device-1": participant})


pytestmark = pytest.mark.asyncio


async def test_unattached_device_resolves_to_device_connection_without_runtime_lookup():
    mounts = AsyncMock()
    mounts.resolve.return_value = DeviceConnectionContext(
        owner_id="owner-1",
        device_id="device-1",
        mount_revision=7,
        attached_companion_id=None,
    )
    runtime = AsyncMock()

    context = await resolve_channel_context(
        runtime=runtime,
        mounts=mounts,
        identity="device-1",
        metadata={"kind": "device", "owner_id": "owner-1"},
    )

    assert isinstance(context, DeviceConnectionContext)
    assert context.attached_companion_id is None
    runtime.resolve_companion.assert_not_called()


async def test_attached_device_resolves_complete_companion_interaction_context():
    mounts = AsyncMock()
    mounts.resolve.return_value = DeviceConnectionContext(
        owner_id="owner-1",
        device_id="device-1",
        mount_revision=7,
        attached_companion_id="companion-1",
    )
    runtime = AsyncMock()
    runtime.resolve_companion.return_value = runtime_context(device_id="device-1")

    context = await resolve_channel_context(
        runtime=runtime,
        mounts=mounts,
        identity="device-1",
        metadata={"kind": "device", "owner_id": "owner-1"},
    )

    assert isinstance(context, CompanionInteractionContext)
    assert context.runtime.device_id == "device-1"
    assert context.mount_revision == 7
    runtime.resolve_companion.assert_awaited_once_with("companion-1", device_id="device-1")


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
        device_id="device-1",
        mount_revision=1,
        attached_companion_id=None,
    )

    with pytest.raises(DeviceTokenResolverError, match="owner"):
        await resolve_channel_context(
            runtime=AsyncMock(),
            mounts=mounts,
            identity="device-1",
            metadata={"kind": "device", "owner_id": "owner-1"},
        )


async def test_audio_token_resolver_rejects_unattached_device_before_signing():
    mounts = AsyncMock()
    mounts.resolve.return_value = DeviceConnectionContext(
        owner_id="owner-1",
        device_id="device-1",
        mount_revision=1,
        attached_companion_id=None,
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
