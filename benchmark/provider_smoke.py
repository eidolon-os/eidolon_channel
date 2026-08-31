"""Shared validity checks for real STT/TTS provider smoke tests."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any


DEFAULT_STT_AUDIO = (
    Path(__file__).resolve().parent
    / "audio"
    / "generated"
    / "normal_followup.wav"
)


def load_pcm16_mono(path: Path, *, expected_sample_rate: int = 16_000) -> bytes:
    if not path.is_file():
        raise FileNotFoundError(f"STT smoke audio missing: {path}")
    with wave.open(str(path), "rb") as audio:
        actual = {
            "channels": audio.getnchannels(),
            "sample_width": audio.getsampwidth(),
            "sample_rate": audio.getframerate(),
        }
        expected = {
            "channels": 1,
            "sample_width": 2,
            "sample_rate": expected_sample_rate,
        }
        if actual != expected:
            raise ValueError(
                f"STT smoke audio must be PCM16 mono {expected_sample_rate}Hz; got {actual}"
            )
        pcm = audio.readframes(audio.getnframes())
    if not pcm:
        raise ValueError(f"STT smoke audio is empty: {path}")
    return pcm


async def check_stt(factory: Any, audio_path: Path = DEFAULT_STT_AUDIO) -> dict[str, object]:
    text = await factory.stt.recognize_streaming(load_pcm16_mono(audio_path))
    if not text.strip():
        raise RuntimeError("STT returned an empty transcript for speech audio")
    return {"chars": len(text), "text": text[:80]}


async def check_tts(factory: Any, text: str = "你好，测试。") -> dict[str, object]:
    await factory.tts.warmup()
    frame_count = 0
    sample_rate: int | None = None
    total_samples = 0
    async for frame in factory.tts.synthesize(text):
        if sample_rate is None:
            sample_rate = frame.sample_rate
        elif frame.sample_rate != sample_rate:
            raise RuntimeError(
                f"TTS sample rate changed within one stream: {sample_rate} -> {frame.sample_rate}"
            )
        frame_count += 1
        total_samples += frame.samples_per_channel
    if frame_count == 0 or sample_rate is None:
        raise RuntimeError("TTS returned no audio frames")
    duration_ms = round(total_samples * 1000 / sample_rate)
    if duration_ms < 100:
        raise RuntimeError(f"TTS returned implausibly short audio: {duration_ms}ms")
    return {
        "frames": frame_count,
        "sample_rate": sample_rate,
        "duration_ms": duration_ms,
    }
