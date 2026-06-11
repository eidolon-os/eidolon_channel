"""Bailian TTS shutdown lifecycle tests."""

from __future__ import annotations

import pytest
from livekit.agents.types import APIConnectOptions

from eidolon.livekit.plugins.tts.bailian.config import BailianTTSConfig
from eidolon.livekit.plugins.tts.bailian.tts import BailianTTS
from eidolon.livekit.plugins.tts.bailian.tts_client import BailianTTSError


class _FakeEmitter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def initialize(self, **_kwargs) -> None:
        self.calls.append("initialize")

    def start_segment(self, **_kwargs) -> None:
        self.calls.append("start_segment")

    def end_segment(self) -> None:
        self.calls.append("end_segment")

    def end_input(self) -> None:
        self.calls.append("end_input")

    def push(self, _data: bytes) -> None:
        self.calls.append("push")


class _FakeBailianClient:
    def __init__(self) -> None:
        self.connected = False
        self.started = 0
        self.disconnected = 0

    async def connect(self) -> bool:
        self.connected = True
        return True

    async def start_task(self) -> None:
        self.started += 1

    async def disconnect(self) -> None:
        self.disconnected += 1


def _tts() -> BailianTTS:
    return BailianTTS(
        BailianTTSConfig(
            api_key="test",
            pool_size=1,
            pool_size_bootstrap=1,
        )
    )


@pytest.mark.asyncio
async def test_acquire_after_shutdown_is_not_recoverable() -> None:
    tts = _tts()
    await tts.shutdown()

    with pytest.raises(BailianTTSError) as exc_info:
        await tts._acquire_conn()

    assert exc_info.value.recoverable is False
    assert "shutting down" in str(exc_info.value)


@pytest.mark.asyncio
async def test_stream_started_during_shutdown_ends_without_pool_acquire() -> None:
    tts = _tts()
    await tts.shutdown()
    stream = tts.stream(conn_options=APIConnectOptions())
    emitter = _FakeEmitter()

    await stream._run(emitter)  # type: ignore[arg-type]

    assert emitter.calls == [
        "initialize",
        "start_segment",
        "end_segment",
        "end_input",
    ]


@pytest.mark.asyncio
async def test_warmup_preconnects_without_starting_provider_task() -> None:
    tts = _tts()
    fake = _FakeBailianClient()
    tts._create_connection = lambda: fake  # type: ignore[method-assign]

    await tts.warmup()
    await tts.shutdown()

    assert fake.connected is True
    assert fake.started == 0
    assert fake.disconnected == 1
