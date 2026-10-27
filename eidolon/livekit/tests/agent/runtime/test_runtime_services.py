from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from eidolon_sdk.biz.persona import ResolvedRuntimeIdentity
from eidolon.livekit.agent.runtime.services import ChannelRuntimeServices


def _context() -> ResolvedRuntimeIdentity:
    return ResolvedRuntimeIdentity(
        owner_id="owner-a",
        companion_id="companion-a",
        memory_realm_id="realm-a",
        genome_id="genome-a",
        schema_version="eidolon.persona_genome",
        genome_hash="pg_hash",
        realizer_version="eidolon.persona_realizer",
        device_id=None,
    )


@pytest.mark.asyncio
async def test_runtime_services_cache_one_context_and_close_once() -> None:
    runtime = AsyncMock()
    runtime.resolve_owner.return_value = _context()
    room = SimpleNamespace(
        remote_participants={
            "owner-a": SimpleNamespace(
                identity="owner-a",
                metadata='{"kind":"owner","owner_id":"owner-a"}',
            )
        }
    )
    services = ChannelRuntimeServices(runtime=runtime, mounts=AsyncMock())

    assert await services.resolve_room(room) == _context()
    assert await services.resolve_room(room) == _context()
    runtime.resolve_owner.assert_awaited_once_with("owner-a")

    await services.aclose()
    await services.aclose()
    runtime.aclose.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_runtime_authority_reuses_one_lazy_session_http_adapter(monkeypatch):
    from eidolon.livekit.agent.runtime import system_data

    snapshot = SimpleNamespace(runtime_identity=lambda **kwargs: _context())
    client = SimpleNamespace(get_companion_runtime=AsyncMock(return_value=snapshot),
        get_companion_face=AsyncMock(return_value=b'face'))
    http = SimpleNamespace(aclose=AsyncMock())
    adapter = system_data.SystemDataRuntimeResolver(client, http_client=http)
    opened = []
    def create(settings):
        opened.append(settings)
        return adapter
    monkeypatch.setattr(system_data, '_open_system_data_runtime', create)
    settings = object()
    deferred = system_data.DeferredSystemDataRuntimeResolver(settings)
    assert opened == []
    assert await deferred.resolve_companion('companion', device_id=None) == _context()
    assert await deferred.get_companion_face('companion') == b'face'
    client.get_companion_runtime.assert_awaited_once_with('companion')
    client.get_companion_face.assert_awaited_once_with('companion')
    assert opened == [settings]
    await deferred.aclose()
    http.aclose.assert_awaited_once()
