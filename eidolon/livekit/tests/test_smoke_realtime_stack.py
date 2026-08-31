from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmark.provider_smoke import (
    DEFAULT_STT_AUDIO,
    check_stt,
    check_tts,
    load_pcm16_mono,
)


def test_default_stt_smoke_audio_is_supported_pcm16_mono() -> None:
    pcm = load_pcm16_mono(DEFAULT_STT_AUDIO)

    assert len(pcm) == 91_520


@pytest.mark.asyncio
async def test_stt_smoke_rejects_empty_transcript() -> None:
    class EmptyStt:
        async def recognize_streaming(self, audio: bytes) -> str:
            assert audio
            return ""

    factory = SimpleNamespace(stt=EmptyStt())

    with pytest.raises(RuntimeError, match="empty transcript"):
        await check_stt(factory)


@pytest.mark.asyncio
async def test_tts_smoke_consumes_complete_stream_and_accumulates_duration() -> None:
    class CompleteTts:
        def __init__(self) -> None:
            self.completed = False

        async def warmup(self) -> None:
            return None

        async def synthesize(self, text: str):
            assert text
            yield SimpleNamespace(sample_rate=16_000, samples_per_channel=1_600)
            yield SimpleNamespace(sample_rate=16_000, samples_per_channel=3_200)
            self.completed = True

    tts = CompleteTts()
    result = await check_tts(SimpleNamespace(tts=tts))

    assert tts.completed is True
    assert result == {"frames": 2, "sample_rate": 16_000, "duration_ms": 300}
