"""Device token resolver for LiveKit sessions.

Verifies the closure correctly:
  - dispatches participant.metadata.kind=device/owner
  - caches the resolved token (one HTTP + one sign per session)
  - propagates System Data errors as DeviceTokenResolverError
  - fails clearly when no participant is connected yet
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import jwt
import pytest

from eidolon_sdk.biz.persona import ResolvedRuntimeIdentity as ResolvedContext
from eidolon_sdk.biz.system_data import SystemDataNotFound
from eidolon.livekit.agent.runtime.resolver import (
    DeviceConnectionContext,
    DeviceTokenResolverError,
    RoomNotConnectedError,
    make_device_token_resolver,
    wait_for_runtime_participant_metadata,
)


def _fake_admin() -> AsyncMock:
    return AsyncMock()


def _mounts(*, device_id: str = "esp32-007") -> AsyncMock:
    mounts = AsyncMock()
    mounts.resolve.return_value = DeviceConnectionContext(
        owner_id="owner-1",
        device_id=device_id,
        mount_revision=3,
        attached_companion_id="companion-1",
    )
    return mounts


def _participant(
    identity: str,
    metadata: str = "",
    *,
    can_publish: bool | None = None,
) -> SimpleNamespace:
    participant = SimpleNamespace(identity=identity, metadata=metadata)
    if can_publish is not None:
        participant.permissions = SimpleNamespace(can_publish=can_publish)
    return participant


def _room_with(*participants, connected: bool = True) -> SimpleNamespace:
    # LiveKit Room exposes ``remote_participants`` as a dict keyed by
    # identity. Match that shape so the resolver's defensive code works.
    return SimpleNamespace(
        remote_participants={p.identity: p for p in participants},
        isconnected=lambda: connected,
    )


pytestmark = pytest.mark.asyncio
SECRET = "test-secret-with-enough-entropy-32b"


def _ctx(*, device_id: str | None = "dev-1") -> ResolvedContext:
    return ResolvedContext(
        owner_id="owner-1",
        companion_id="companion-1",
        memory_realm_id="realm-1",
        genome_id="genome-1",
        schema_version="eidolon.persona_genome",
        genome_hash="pg_resolver",
        realizer_version="eidolon.persona_realizer",
        device_id=device_id,
    )


async def test_resolver_dispatches_to_device_for_kind_device():
    admin = _fake_admin()
    admin.resolve_companion.return_value = _ctx(device_id="esp32-007")
    mounts = _mounts()
    room = _room_with(
        _participant(
            "esp32-007",
            '{"kind":"device","device_id":"esp32-007","owner_id":"owner-1"}',
        )
    )
    resolve = make_device_token_resolver(
        room=room,
        runtime=admin,
        mounts=mounts,
        jwt_secret=SECRET,
    )
    token = await resolve()
    payload = jwt.decode(token, SECRET, algorithms=["HS256"])
    assert payload["owner_id"] == "owner-1"
    assert payload["companion_id"] == "companion-1"
    assert "memory_realm_id" not in payload
    assert "genome_id" not in payload
    assert payload["device_id"] == "esp32-007"
    assert payload["runtime_token_version"] == 5
    assert "actor_kind" not in payload
    assert "actor_id" not in payload
    mounts.resolve.assert_awaited_once_with(owner_id="owner-1", device_id="esp32-007")
    admin.resolve_companion.assert_awaited_once_with("companion-1", device_id="esp32-007")


async def test_resolver_ignores_non_publishing_system_participant_before_device():
    admin = _fake_admin()
    admin.resolve_companion.return_value = _ctx(device_id="esp32-007")
    mounts = _mounts()
    room = _room_with(
        _participant("eidolon-hub-control-test", can_publish=False),
        _participant(
            "esp32-007",
            '{"kind":"device","device_id":"esp32-007","owner_id":"owner-1"}',
            can_publish=True,
        ),
    )
    resolve = make_device_token_resolver(
        room=room,
        runtime=admin,
        mounts=mounts,
        jwt_secret=SECRET,
    )

    token = await resolve()

    payload = jwt.decode(token, SECRET, algorithms=["HS256"])
    assert payload["device_id"] == "esp32-007"
    admin.resolve_companion.assert_awaited_once_with("companion-1", device_id="esp32-007")


async def test_wait_for_runtime_participant_returns_device_metadata_not_hub():
    room = _room_with(
        _participant("eidolon-hub-control-test", can_publish=False),
        _participant(
            "esp32-007",
            (
                '{"kind":"device","device_id":"esp32-007",'
                '"interaction_mode":"full_duplex",'
                '"session_intent":"presence_initiated"}'
            ),
            can_publish=True,
        ),
    )

    identity, metadata = await wait_for_runtime_participant_metadata(
        room,
        timeout_sec=0,
    )

    assert identity == "esp32-007"
    assert metadata["interaction_mode"] == "full_duplex"
    assert metadata["session_intent"] == "presence_initiated"


async def test_wait_for_runtime_participant_rejects_unconnected_room():
    room = _room_with(connected=False)

    with pytest.raises(RoomNotConnectedError, match="unconnected room"):
        await wait_for_runtime_participant_metadata(room, timeout_sec=60)


async def test_resolver_dispatches_to_owner_for_kind_owner():
    admin = _fake_admin()
    admin.resolve_owner.return_value = _ctx(device_id=None)
    room = _room_with(_participant("owner-1", '{"kind": "owner", "owner_id": "owner-1"}'))
    resolve = make_device_token_resolver(
        room=room,
        runtime=admin,
        jwt_secret=SECRET,
    )
    token = await resolve()
    payload = jwt.decode(token, SECRET, algorithms=["HS256"])
    assert payload["owner_id"] == "owner-1"
    assert payload["companion_id"] == "companion-1"
    assert "memory_realm_id" not in payload
    assert "genome_id" not in payload
    assert "actor_kind" not in payload
    assert "actor_id" not in payload
    assert "device_id" not in payload
    admin.resolve_owner.assert_awaited_once_with("owner-1")


async def test_resolver_raises_when_metadata_missing_kind():
    """Phase 33.A5 tightening: no silent device-fallback for missing
    or unknown ``kind``. Supported clients tag kind explicitly. A participant with no
    ``kind`` is a misconfigured / pre-32-era client — surface it
    loudly rather than guess."""
    from eidolon.livekit.agent.runtime.resolver import DeviceTokenResolverError

    admin = _fake_admin()
    room = _room_with(_participant("old-esp", metadata=""))
    resolve = make_device_token_resolver(
        room=room,
        runtime=admin,
        jwt_secret=SECRET,
    )
    with pytest.raises(DeviceTokenResolverError) as exc_info:
        await resolve()
    # message mentions the missing-kind problem so ops can grep it
    assert "kind" in str(exc_info.value)


async def test_resolver_raises_when_kind_unknown():
    """Same strictness for ``kind=anonymous`` or other bogus values."""
    from eidolon.livekit.agent.runtime.resolver import DeviceTokenResolverError

    admin = _fake_admin()
    room = _room_with(_participant("x", '{"kind": "anonymous"}'))
    resolve = make_device_token_resolver(
        room=room,
        runtime=admin,
        jwt_secret=SECRET,
    )
    with pytest.raises(DeviceTokenResolverError):
        await resolve()


async def test_resolver_caches_token_across_calls():
    """Once resolved, subsequent invocations return the same cached
    token without hitting admin again. One LK session = one HTTP +
    one sign — the whole point of the cache."""
    admin = _fake_admin()
    admin.resolve_companion.return_value = _ctx(device_id="dev-1")
    mounts = _mounts(device_id="dev-1")
    room = _room_with(_participant("dev-1", '{"kind":"device","owner_id":"owner-1"}'))
    resolve = make_device_token_resolver(
        room=room,
        runtime=admin,
        mounts=mounts,
        jwt_secret=SECRET,
    )
    t1 = await resolve()
    t2 = await resolve()
    assert t1 == t2
    admin.resolve_companion.assert_awaited_once_with("companion-1", device_id="dev-1")


async def test_resolver_coalesces_concurrent_first_calls():
    admin = _fake_admin()

    async def resolve_companion(_companion_id, *, device_id):
        await asyncio.sleep(0.01)
        return _ctx(device_id=device_id)

    admin.resolve_companion.side_effect = resolve_companion
    mounts = _mounts(device_id="dev-1")
    room = _room_with(_participant("dev-1", '{"kind":"device","owner_id":"owner-1"}'))
    resolve = make_device_token_resolver(
        room=room,
        runtime=admin,
        mounts=mounts,
        jwt_secret=SECRET,
    )

    tokens = await asyncio.gather(resolve(), resolve(), resolve())

    assert len(set(tokens)) == 1
    mounts.resolve.assert_awaited_once_with(owner_id="owner-1", device_id="dev-1")
    admin.resolve_companion.assert_awaited_once_with("companion-1", device_id="dev-1")


async def test_resolver_propagates_system_data_404_as_resolver_error():
    """Chat-time, not session-construction. Closure wraps admin errors
    in DeviceTokenResolverError so the gRPC LLM raises a clean
    APIConnectionError up to the LK pipeline."""
    admin = _fake_admin()
    admin.resolve_companion.side_effect = SystemDataNotFound("companion 'ghost' not found")
    mounts = _mounts(device_id="ghost")
    room = _room_with(_participant("ghost", '{"kind":"device","owner_id":"owner-1"}'))
    resolve = make_device_token_resolver(
        room=room,
        runtime=admin,
        mounts=mounts,
        jwt_secret=SECRET,
    )
    with pytest.raises(DeviceTokenResolverError, match="ghost"):
        await resolve()


async def test_resolver_no_participant_raises():
    """Defensive: resolver called before any participant connected.
    Shouldn't happen in production (chat() implies user spoke) but
    we want a clear message if entrypoint ordering changes."""
    admin = _fake_admin()
    room = _room_with()  # no participants
    resolve = make_device_token_resolver(
        room=room,
        runtime=admin,
        jwt_secret=SECRET,
    )
    with pytest.raises(DeviceTokenResolverError, match="no remote participant"):
        await resolve()


async def test_resolver_failure_does_not_cache():
    """If first call fails, second call retries (not cached as None).
    Matters because admin might come back online mid-session."""
    admin = _fake_admin()
    admin.resolve_companion.side_effect = [
        SystemDataNotFound("transient"),
        _ctx(device_id="dev-1"),
    ]
    mounts = _mounts(device_id="dev-1")
    room = _room_with(_participant("dev-1", '{"kind":"device","owner_id":"owner-1"}'))
    resolve = make_device_token_resolver(
        room=room,
        runtime=admin,
        mounts=mounts,
        jwt_secret=SECRET,
    )
    with pytest.raises(DeviceTokenResolverError):
        await resolve()
    # Second call succeeds.
    token = await resolve()
    assert jwt.decode(token, SECRET, algorithms=["HS256"])["owner_id"] == "owner-1"
    assert admin.resolve_companion.await_count == 2
