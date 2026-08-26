"""Avatar face resolution stays outside the audio pipeline and degrades safely."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from eidolon_sdk.device_foundation.v1 import DeviceRef
from eidolon_sdk.biz.persona import ResolvedRuntimeIdentity

from eidolon.livekit.agent.runtime.resolver import DeviceConnectionContext
from eidolon.livekit.agent.runtime.services import ChannelRuntimeServices
from eidolon.livekit.avatar.face_source import resolve_session_face_image

from eidolon_sdk.device_foundation.v1.testing import named_device_instance_id

# Tests name the device they mean; the name becomes a real device
# instance id, which is a digest of a key and never a chosen string.
_ESP32_A = named_device_instance_id("esp32-a")

pytestmark = pytest.mark.asyncio
JPEG = b"\xff\xd8\xffconfigured-face-bytes\xff\xd9"


def _participant(identity: str, metadata: str) -> SimpleNamespace:
    return SimpleNamespace(identity=identity, metadata=metadata)


def _room_with(*participants) -> SimpleNamespace:
    return SimpleNamespace(remote_participants={p.identity: p for p in participants})


def _context(*, device_id: str | None) -> ResolvedRuntimeIdentity:
    return ResolvedRuntimeIdentity(
        owner_id="owner-a",
        companion_id="companion-a",
        memory_realm_id="realm-a",
        genome_id="genome-a",
        schema_version="eidolon.persona_genome",
        genome_hash="pg_hash",
        realizer_version="eidolon.persona_realizer",
        device_id=device_id,
    )


def _runtime(*, face: bytes | None = JPEG) -> AsyncMock:
    runtime = AsyncMock()
    runtime.resolve_companion.return_value = _context(device_id=_ESP32_A)
    runtime.resolve_owner.return_value = _context(device_id=None)
    runtime.get_companion_face.return_value = face
    return runtime


def _mounts() -> AsyncMock:
    mounts = AsyncMock()
    mounts.resolve.return_value = DeviceConnectionContext(
        owner_id="owner-a",
        device_id=_ESP32_A,
        device_ref=DeviceRef(
            device_instance_id=_ESP32_A,
            owner_domain_id="owner-domain-a",
            owner_domain_generation=1,
            claim_generation=1,
            trust_epoch=1,
        ),
        mount_revision=2,
        attached_companion_id="companion-a",
    )
    return mounts


async def test_resolves_configured_face_for_mounted_device() -> None:
    runtime = _runtime()
    services = ChannelRuntimeServices(runtime=runtime, mounts=_mounts())
    room = _room_with(
        _participant(
            _ESP32_A,
            f'{{"kind":"device","device_id":"{_ESP32_A}","owner_id":"owner-a"}}',
        )
    )

    assert (
        await resolve_session_face_image(
            room,
            context_resolver=services.resolve_room,
            runtime_client=runtime,
        )
        == JPEG
    )
    runtime.resolve_companion.assert_awaited_once_with("companion-a", device_id=_ESP32_A)
    runtime.get_companion_face.assert_awaited_once_with("companion-a")


async def test_resolves_configured_face_for_owner() -> None:
    runtime = _runtime()
    services = ChannelRuntimeServices(runtime=runtime, mounts=_mounts())
    room = _room_with(_participant("owner-a", '{"kind":"owner","owner_id":"owner-a"}'))

    assert (
        await resolve_session_face_image(
            room,
            context_resolver=services.resolve_room,
            runtime_client=runtime,
        )
        == JPEG
    )
    runtime.resolve_owner.assert_awaited_once_with("owner-a")


async def test_none_when_no_face_configured() -> None:
    runtime = _runtime(face=None)
    services = ChannelRuntimeServices(runtime=runtime, mounts=_mounts())
    room = _room_with(_participant("owner-a", '{"kind":"owner","owner_id":"owner-a"}'))

    assert (
        await resolve_session_face_image(
            room,
            context_resolver=services.resolve_room,
            runtime_client=runtime,
        )
        is None
    )


async def test_none_when_no_participant() -> None:
    runtime = _runtime()
    services = ChannelRuntimeServices(runtime=runtime, mounts=_mounts())
    assert (
        await resolve_session_face_image(
            _room_with(),
            context_resolver=services.resolve_room,
            runtime_client=runtime,
        )
        is None
    )


async def test_authority_failure_never_breaks_avatar_or_audio() -> None:
    runtime = _runtime()
    services = ChannelRuntimeServices(runtime=runtime, mounts=_mounts())
    runtime.resolve_owner.side_effect = RuntimeError("authority unavailable")
    room = _room_with(_participant("owner-a", '{"kind":"owner","owner_id":"owner-a"}'))

    assert (
        await resolve_session_face_image(
            room,
            context_resolver=services.resolve_room,
            runtime_client=runtime,
        )
        is None
    )
