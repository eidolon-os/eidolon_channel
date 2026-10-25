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
async def test_voice_authority_reuses_one_lazy_session_http_adapter(monkeypatch):
    from eidolon.livekit.agent.runtime import system_data

    client = SimpleNamespace(get_voiceprint_snapshot=AsyncMock(return_value='snapshot'),
        admit_voiceprint=AsyncMock(return_value=True))
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
    assert await deferred.get_voiceprint_snapshot('companion') == 'snapshot'
    assert await deferred.admit_voiceprint('companion', 7, 'profile') is True
    client.get_voiceprint_snapshot.assert_awaited_once_with('companion')
    client.admit_voiceprint.assert_awaited_once_with('companion', 7, 'profile')
    assert opened == [settings]
    await deferred.aclose()
    http.aclose.assert_awaited_once()
