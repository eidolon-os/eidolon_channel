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
