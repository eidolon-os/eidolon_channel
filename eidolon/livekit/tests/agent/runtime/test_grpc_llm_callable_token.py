"""Phase 32.B: ``EidolonAgentGrpcLlm.device_token`` accepts a callable.

The callable is invoked at first ``_get_session`` (which happens on
first chat()). We use a stub session class so we can assert the
resolved token is what gets passed to EidolonAgentSession, without
opening a real gRPC channel.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from eidolon.livekit.agent.eidolon_agent_rpc.grpc_llm import EidolonAgentGrpcLlm


pytestmark = pytest.mark.asyncio


async def test_static_device_token_path_unchanged():
    """Legacy path: device_token=str → eagerly stored, _resolve returns it
    without calling anything fancy. This pins backward compat with
    Phase 25 callers (and the fallback path in factory)."""
    llm = EidolonAgentGrpcLlm(
        target="127.0.0.1:1",
        device_token="static-token-abc",
        conversation_id="livekit:test",
    )
    resolved = await llm._resolve_device_token()
    assert resolved == "static-token-abc"


async def test_async_callable_device_token_resolved_lazily():
    """plan D path: callable invoked on demand. Returns token; cached
    after first call so subsequent calls don't re-invoke."""
    calls: list[int] = []

    async def resolver() -> str:
        calls.append(1)
        return "fresh-token-xyz"

    llm = EidolonAgentGrpcLlm(
        target="127.0.0.1:1",
        device_token=resolver,
        conversation_id="livekit:test",
    )
    t1 = await llm._resolve_device_token()
    t2 = await llm._resolve_device_token()
    assert t1 == "fresh-token-xyz"
    assert t1 == t2
    assert len(calls) == 1, "resolver should be invoked exactly once"


async def test_sync_callable_device_token_supported():
    """Resolver returning a plain string (not awaitable) works too —
    helps tests + alternate implementations avoid coroutine plumbing."""
    llm = EidolonAgentGrpcLlm(
        target="127.0.0.1:1",
        device_token=lambda: "sync-token",
        conversation_id="livekit:test",
    )
    assert await llm._resolve_device_token() == "sync-token"


async def test_empty_static_token_still_raises_at_init():
    """Backward compat: an explicit empty string is still a programming
    error. Phase 25's eager validation contract preserved."""
    import pytest as _pytest

    with _pytest.raises(ValueError, match="device_token is required"):
        EidolonAgentGrpcLlm(
            target="127.0.0.1:1",
            device_token="",
            conversation_id="livekit:test",
        )


async def test_resolver_returning_empty_raises_at_session_open():
    """Callable that returns empty string → APIConnectionError at first
    session open (not at __init__, since we can't introspect callables
    in advance)."""
    from livekit.agents._exceptions import APIConnectionError

    async def bad_resolver() -> str:
        return "   "

    llm = EidolonAgentGrpcLlm(
        target="127.0.0.1:1",
        device_token=bad_resolver,
        conversation_id="livekit:test",
    )
    with pytest.raises(APIConnectionError, match="returned empty"):
        await llm._resolve_device_token()
