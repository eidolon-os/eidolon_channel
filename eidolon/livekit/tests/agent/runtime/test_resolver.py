"""Device token resolver for LiveKit sessions.

Verifies the closure correctly:
  - dispatches participant.metadata.kind=device/owner
  - caches the resolved token (one HTTP + one sign per session)
  - propagates admin errors as DeviceTokenResolverError
  - fails clearly when no participant is connected yet
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import jwt
import pytest

from eidolon_sdk.biz.admin import (
    AdminResolveClient,
    AdminResolveNotFound,
    ResolvedContext,
)
from eidolon.livekit.agent.runtime.resolver import (
    DeviceTokenResolverError,
    make_device_token_resolver,
)


def _fake_admin() -> AsyncMock:
    return AsyncMock(spec=AdminResolveClient)


def _participant(identity: str, metadata: str = "") -> SimpleNamespace:
    return SimpleNamespace(identity=identity, metadata=metadata)


def _room_with(*participants) -> SimpleNamespace:
    # LiveKit Room exposes ``remote_participants`` as a dict keyed by
    # identity. Match that shape so the resolver's defensive code works.
    return SimpleNamespace(
        remote_participants={p.identity: p for p in participants},
    )


pytestmark = pytest.mark.asyncio
SECRET = "test-secret-with-enough-entropy-32b"


async def test_resolver_dispatches_to_device_for_kind_device():
    admin = _fake_admin()
    admin.resolve_device.return_value = ResolvedContext(
        owner_id="owner-1",
        companion_id="companion-1",
        memory_realm_id="realm-1",
        genome_id="genome-1",
        device_id="esp32-007",
    )
    room = _room_with(
        _participant("esp32-007", '{"kind": "device", "device_id": "esp32-007"}')
    )
    resolve = make_device_token_resolver(
        room=room, admin=admin, jwt_secret=SECRET,
    )
    token = await resolve()
    payload = jwt.decode(token, SECRET, algorithms=["HS256"])
    assert payload["owner_id"] == "owner-1"
    assert payload["companion_id"] == "companion-1"
    assert payload["memory_realm_id"] == "realm-1"
    assert payload["genome_id"] == "genome-1"
    assert payload["device_id"] == "esp32-007"
    assert payload["actor_kind"] == "device"
    assert payload["actor_id"] == "esp32-007"
    admin.resolve_device.assert_awaited_once_with("esp32-007")


async def test_resolver_dispatches_to_owner_for_kind_owner():
    admin = _fake_admin()
    admin.resolve_owner.return_value = ResolvedContext(
        owner_id="owner-1",
        companion_id="companion-1",
        memory_realm_id="realm-1",
        genome_id="genome-1",
        device_id=None,
    )
    room = _room_with(
        _participant("owner-1", '{"kind": "owner", "owner_id": "owner-1"}')
    )
    resolve = make_device_token_resolver(
        room=room, admin=admin, jwt_secret=SECRET,
    )
    token = await resolve()
    payload = jwt.decode(token, SECRET, algorithms=["HS256"])
    assert payload["owner_id"] == "owner-1"
    assert payload["companion_id"] == "companion-1"
    assert payload["memory_realm_id"] == "realm-1"
    assert payload["genome_id"] == "genome-1"
    assert payload["actor_kind"] == "owner"
    assert payload["actor_id"] == "owner-1"
    assert "device_id" not in payload
    admin.resolve_owner.assert_awaited_once_with("owner-1")
    admin.resolve_device.assert_not_called()


async def test_resolver_raises_when_metadata_missing_kind():
    """Phase 33.A5 tightening: no silent device-fallback for missing
    or unknown ``kind``. Supported clients tag kind explicitly. A participant with no
    ``kind`` is a misconfigured / pre-32-era client — surface it
    loudly rather than guess."""
    from eidolon.livekit.agent.runtime.resolver import DeviceTokenResolverError

    admin = _fake_admin()
    room = _room_with(_participant("old-esp", metadata=""))
    resolve = make_device_token_resolver(
        room=room, admin=admin, jwt_secret=SECRET,
    )
    with pytest.raises(DeviceTokenResolverError) as exc_info:
        await resolve()
    # message mentions the missing-kind problem so ops can grep it
    assert "kind" in str(exc_info.value)
    admin.resolve_device.assert_not_called()


async def test_resolver_raises_when_kind_unknown():
    """Same strictness for ``kind=anonymous`` or other bogus values."""
    from eidolon.livekit.agent.runtime.resolver import DeviceTokenResolverError

    admin = _fake_admin()
    room = _room_with(_participant("x", '{"kind": "anonymous"}'))
    resolve = make_device_token_resolver(
        room=room, admin=admin, jwt_secret=SECRET,
    )
    with pytest.raises(DeviceTokenResolverError):
        await resolve()
    admin.resolve_device.assert_not_called()


async def test_resolver_caches_token_across_calls():
    """Once resolved, subsequent invocations return the same cached
    token without hitting admin again. One LK session = one HTTP +
    one sign — the whole point of the cache."""
    admin = _fake_admin()
    admin.resolve_device.return_value = ResolvedContext(
        "owner-1", "companion-1", "realm-1", "genome-1", "dev-1"
    )
    room = _room_with(_participant("dev-1", '{"kind": "device"}'))
    resolve = make_device_token_resolver(
        room=room, admin=admin, jwt_secret=SECRET,
    )
    t1 = await resolve()
    t2 = await resolve()
    assert t1 == t2
    admin.resolve_device.assert_awaited_once_with("dev-1")


async def test_resolver_propagates_admin_404_as_resolver_error():
    """Chat-time, not session-construction. Closure wraps admin errors
    in DeviceTokenResolverError so the gRPC LLM raises a clean
    APIConnectionError up to the LK pipeline."""
    admin = _fake_admin()
    admin.resolve_device.side_effect = AdminResolveNotFound("device 'ghost' not found")
    room = _room_with(_participant("ghost", '{"kind": "device"}'))
    resolve = make_device_token_resolver(
        room=room, admin=admin, jwt_secret=SECRET,
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
        room=room, admin=admin, jwt_secret=SECRET,
    )
    with pytest.raises(DeviceTokenResolverError, match="no remote participant"):
        await resolve()


async def test_resolver_failure_does_not_cache():
    """If first call fails, second call retries (not cached as None).
    Matters because admin might come back online mid-session."""
    admin = _fake_admin()
    admin.resolve_device.side_effect = [
        AdminResolveNotFound("transient"),
        ResolvedContext("owner-1", "companion-1", "realm-1", "genome-1", "dev-1"),
    ]
    room = _room_with(_participant("dev-1", '{"kind": "device"}'))
    resolve = make_device_token_resolver(
        room=room, admin=admin, jwt_secret=SECRET,
    )
    with pytest.raises(DeviceTokenResolverError):
        await resolve()
    # Second call succeeds.
    token = await resolve()
    assert jwt.decode(token, SECRET, algorithms=["HS256"])["owner_id"] == "owner-1"
    assert admin.resolve_device.await_count == 2
