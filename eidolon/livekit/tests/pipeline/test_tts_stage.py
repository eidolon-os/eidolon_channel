"""TTS stage lifecycle tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from eidolon.livekit.agent.providers.tts import TtsStage


class _FakeStream:
    def __init__(self) -> None:
        self.closed = False
        self.exception_read = False
        self.done = True
        self.exception = RuntimeError("already consumed")
        self._sent = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._sent:
            raise StopAsyncIteration
        self._sent = True
        return SimpleNamespace(frame="frame-1")

    async def aclose(self) -> None:
        self.closed = True


class _FakeTTS:
    def __init__(self) -> None:
        self.stream = _FakeStream()

    def synthesize(self, _text: str) -> _FakeStream:
        return self.stream


@pytest.mark.asyncio
async def test_synthesize_closes_underlying_stream_when_consumer_stops() -> None:
    fake = _FakeTTS()
    stage = TtsStage(fake)  # type: ignore[arg-type]
    gen = stage.synthesize("hello")

    assert await gen.__anext__() == "frame-1"
    await gen.aclose()

    assert fake.stream.closed is True


@pytest.mark.asyncio
async def test_tts_failures_do_not_count_towards_closing_multimodal_session():
    import time
    from unittest.mock import Mock
    from livekit.agents import tts
    from livekit.agents.voice import AgentSession
    from livekit.agents import APIConnectionError
    from eidolon.livekit.agent.providers.tts import OutputScopedTTS
    class Fake(tts.TTS):
        def __init__(self):
            super().__init__(capabilities=tts.TTSCapabilities(streaming=True),
                             sample_rate=16000, num_channels=1)
        def synthesize(self, text, **kwargs):
            raise NotImplementedError
        def stream(self, **kwargs):
            return sentinel
    sentinel = object()
    inner = Fake()
    scoped = OutputScopedTTS(inner)
    session = AgentSession()
    scoped.on("error", session._on_error)
    observer = Mock()
    scoped.on("output_error", observer)
    for _ in range(5):
        inner.emit("error", tts.TTSError(timestamp=time.time(), label="fake",
            error=APIConnectionError("tts offline", retryable=False), recoverable=False))
    assert observer.call_count == 5
    assert session._tts_error_counts == 0 and session._closing_task is None
    assert scoped.stream() is sentinel  # original synthesis implementation
    assert scoped.capabilities is inner.capabilities
    await scoped.aclose()
