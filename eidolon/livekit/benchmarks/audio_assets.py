"""Audio asset helpers for voice benchmarks."""

from __future__ import annotations

import wave
from pathlib import Path

from eidolon.livekit.tests._harness.audio import DEFAULT_CHANNELS, SAMPLE_WIDTH_BYTES, wav_load


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
