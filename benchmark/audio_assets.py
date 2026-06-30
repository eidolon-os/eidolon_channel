"""Audio asset helpers for voice benchmarks."""

from __future__ import annotations

import wave
from collections.abc import AsyncIterable, Callable, Sequence
from pathlib import Path
from typing import Any

from eidolon.livekit.tests._harness.audio import DEFAULT_CHANNELS, SAMPLE_WIDTH_BYTES, wav_load

# A composite clip is TTS speech segments separated by trailing silence:
# [(text, silence_ms_after), ...]. Used to synthesize utterances with natural
# internal pauses (turn-merge and hesitation scenarios).
CompositeParts = Sequence[tuple[str, int]]


def write_wav(
    path: str | Path,
    pcm: bytes,
    *,
    sample_rate: int = 16_000,
    channels: int = DEFAULT_CHANNELS,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(SAMPLE_WIDTH_BYTES)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def wav_duration_ms(path: str | Path) -> int:
    pcm, sample_rate = wav_load(path)
    samples = len(pcm) // SAMPLE_WIDTH_BYTES
    return round(samples / sample_rate * 1000)


def load_clip_pcm(path: str | Path) -> tuple[bytes, int]:
    return wav_load(path)


def silence_pcm(duration_ms: int, *, sample_rate: int) -> bytes:
    samples = int(sample_rate * duration_ms / 1000)
    return b"\x00" * (samples * SAMPLE_WIDTH_BYTES)


async def synthesize_pcm(
    synthesize: Callable[[str], AsyncIterable[Any]],
    text: str,
) -> tuple[bytes, int]:
    """Collect a TTS stream (e.g. ``factory.tts.synthesize``) into raw PCM."""

    frames: list[bytes] = []
    sample_rate = 16_000
    async for frame in synthesize(text):
        frames.append(bytes(frame.data))
        sample_rate = int(frame.sample_rate)
    if not frames:
        raise RuntimeError(f"TTS returned no audio for {text!r}")
    return b"".join(frames), sample_rate


async def synthesize_composite_pcm(
    synthesize: Callable[[str], AsyncIterable[Any]],
    parts: CompositeParts,
) -> tuple[bytes, int]:
    """Synthesize speech segments joined by their trailing silences."""

    chunks: list[bytes] = []
    sample_rate = 16_000
    for text, silence_ms in parts:
        pcm, sample_rate = await synthesize_pcm(synthesize, text)
        chunks.append(pcm)
        if silence_ms > 0:
            chunks.append(silence_pcm(silence_ms, sample_rate=sample_rate))
    if not chunks:
        raise RuntimeError("composite clip has no parts")
    return b"".join(chunks), sample_rate
