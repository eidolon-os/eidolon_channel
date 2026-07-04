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
