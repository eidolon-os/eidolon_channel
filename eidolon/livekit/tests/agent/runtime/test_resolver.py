"""Phase 32.B: device_token_resolver — the heart of plan D.

Verifies the closure correctly:
  - reads participant.metadata.kind to dispatch user vs device
  - caches the resolved token (one HTTP + one sign per session)
  - propagates admin errors as DeviceTokenResolverError
  - fails clearly when no participant is connected yet
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import jwt
import pytest

from eidolon.livekit.agent.runtime.admin_client import (
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


async def test_resolver_dispatches_to_user_for_kind_user():
    admin = _fake_admin()
    admin.resolve_user.return_value = ResolvedContext(
        tenant_id="default",
        user_id="manson",
        agent_id="ag-1",
        template_id="caretaker",
        memory_mcp_url="http://127.0.0.1:8030/mcp",
        device_id=None,
    )
    room = _room_with(_participant("manson", '{"kind": "user"}'))
    resolve = make_device_token_resolver(
        room=room, admin=admin, jwt_secret="s",
    )
    token = await resolve()
    payload = jwt.decode(token, "s", algorithms=["HS256"])
    assert payload["user_id"] == "manson"
    assert payload["tenant_id"] == "default"
    assert payload["template_id"] == "caretaker"
    admin.resolve_user.assert_awaited_once_with("manson")
    admin.resolve_device.assert_not_called()


async def test_resolver_dispatches_to_device_for_kind_device():
    admin = _fake_admin()
    admin.resolve_device.return_value = ResolvedContext(
        tenant_id="default",
        user_id="alice",
        agent_id="ag-2",
        template_id="kid",
        memory_mcp_url="http://127.0.0.1:8030/mcp",
        device_id="esp32-007",
    )
    room = _room_with(
        _participant("esp32-007", '{"kind": "device", "device_id": "esp32-007"}')
    )
    resolve = make_device_token_resolver(
        room=room, admin=admin, jwt_secret="s",
    )
    token = await resolve()
    payload = jwt.decode(token, "s", algorithms=["HS256"])
    assert payload["user_id"] == "alice"  # admin lookup populated user
    admin.resolve_device.assert_awaited_once_with("esp32-007")
    admin.resolve_user.assert_not_called()


async def test_resolver_defaults_to_device_when_metadata_missing():
    """Legacy ESP32 firmware sends NO metadata. Resolver assumes device
    (the Phase 25 path) — preserves backward compat without forcing a
    firmware update."""
    admin = _fake_admin()
    admin.resolve_device.return_value = ResolvedContext(
        "default", "u", "ag", "t", "http://x", "old-esp"
    )
    room = _room_with(_participant("old-esp", metadata=""))
    resolve = make_device_token_resolver(
        room=room, admin=admin, jwt_secret="s",
    )
    await resolve()
    admin.resolve_device.assert_awaited_once_with("old-esp")


async def test_resolver_caches_token_across_calls():
    """Once resolved, subsequent invocations return the same cached
    token without hitting admin again. One LK session = one HTTP +
    one sign — the whole point of the cache."""
    admin = _fake_admin()
    admin.resolve_user.return_value = ResolvedContext(
        "default", "manson", "ag-1", "t", "http://x", None
    )
    room = _room_with(_participant("manson", '{"kind": "user"}'))
    resolve = make_device_token_resolver(
        room=room, admin=admin, jwt_secret="s",
    )
    t1 = await resolve()
    t2 = await resolve()
    assert t1 == t2
    admin.resolve_user.assert_awaited_once()  # exactly once


async def test_resolver_propagates_admin_404_as_resolver_error():
    """Chat-time, not session-construction. Closure wraps admin errors
    in DeviceTokenResolverError so the gRPC LLM raises a clean
    APIConnectionError up to the LK pipeline."""
    admin = _fake_admin()
    admin.resolve_user.side_effect = AdminResolveNotFound("user 'ghost' not found")
    room = _room_with(_participant("ghost", '{"kind": "user"}'))
    resolve = make_device_token_resolver(
        room=room, admin=admin, jwt_secret="s",
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
        room=room, admin=admin, jwt_secret="s",
    )
    with pytest.raises(DeviceTokenResolverError, match="no remote participant"):
        await resolve()


async def test_resolver_failure_does_not_cache():
    """If first call fails, second call retries (not cached as None).
    Matters because admin might come back online mid-session."""
    admin = _fake_admin()
    admin.resolve_user.side_effect = [
        AdminResolveNotFound("transient"),
        ResolvedContext("default", "manson", "ag", "t", "http://x", None),
    ]
    room = _room_with(_participant("manson", '{"kind": "user"}'))
    resolve = make_device_token_resolver(
        room=room, admin=admin, jwt_secret="s",
    )
    with pytest.raises(DeviceTokenResolverError):
        await resolve()
    # Second call succeeds.
    token = await resolve()
    assert jwt.decode(token, "s", algorithms=["HS256"])["user_id"] == "manson"
    assert admin.resolve_user.await_count == 2
