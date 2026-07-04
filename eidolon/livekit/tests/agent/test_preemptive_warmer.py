"""PreemptiveWarmer — speculative brain warm-up on partial transcripts."""

from __future__ import annotations

import asyncio

import pytest

from eidolon.livekit.agent.eidolon_agent_rpc.preemptive import PreemptiveWarmer


class _FakeSession:
    """Records start_turn(speculative=...) + cancel_turn calls."""

    def __init__(self) -> None:
        self.started: list[dict] = []
        self.cancelled: list[str] = []
        self._n = 0

    async def start_turn(self, *, text, conversation_id, trace_id=None, speculative=False):
        self._n += 1
        turn_id = f"spec-{self._n}"
        self.started.append(
            {"turn_id": turn_id, "text": text, "speculative": speculative}
        )

        async def _payloads():
            if False:  # pragma: no cover - empty async iterator
                yield None

        return turn_id, _payloads()

    async def cancel_turn(self, turn_id):
        self.cancelled.append(turn_id)

    def spawn(self, coro, *, name=None):
        return asyncio.ensure_future(coro)


def _warmer(session, **kw):
    return PreemptiveWarmer(session, spawn=session.spawn, **kw)


@pytest.mark.asyncio
async def test_warm_fires_speculative_turn():
    s = _FakeSession()
    w = _warmer(s)
    await w.warm("帮我查一下明天的天气", conversation_id="c1")
    assert len(s.started) == 1
    assert s.started[0]["speculative"] is True
    assert s.started[0]["text"] == "帮我查一下明天的天气"


@pytest.mark.asyncio
async def test_short_interim_is_gated():
    s = _FakeSession()
    w = _warmer(s, min_chars=6)
    await w.warm("嗯", conversation_id="c1")
    assert s.started == []


@pytest.mark.asyncio
async def test_identical_text_deduped():
    s = _FakeSession()
    w = _warmer(s)
    await w.warm("帮我订一张机票", conversation_id="c1")
    await w.warm("帮我订一张机票", conversation_id="c1")
    assert len(s.started) == 1


@pytest.mark.asyncio
async def test_growing_interim_supersedes_previous():
    s = _FakeSession()
    w = _warmer(s)
    await w.warm("帮我订一张机票", conversation_id="c1")
    await w.warm("帮我订一张去上海的机票", conversation_id="c1")
    # New warm cancels the prior speculative and starts a fresh one.
    assert len(s.started) == 2
    assert s.cancelled == ["spec-1"]


@pytest.mark.asyncio
async def test_discard_cancels_in_flight():
    s = _FakeSession()
    w = _warmer(s)
    await w.warm("帮我查一下天气", conversation_id="c1")
    await w.discard()
    assert s.cancelled == ["spec-1"]
    # Idempotent: a second discard is a no-op.
    await w.discard()
    assert s.cancelled == ["spec-1"]


@pytest.mark.asyncio
async def test_warm_never_raises_on_session_error():
    class _Boom(_FakeSession):
        async def start_turn(self, **kw):
            raise RuntimeError("stream down")

    w = _warmer(_Boom())
    await w.warm("帮我查天气", conversation_id="c1")  # must not raise
