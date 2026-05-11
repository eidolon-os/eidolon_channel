# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Synthetic audio generators + LiveKit AudioFrame helpers.

All generators return raw 16-bit signed little-endian PCM (the
universal interchange format for our STT/TTS plugins). Sample rate
defaults to 16 kHz, mono — match the canonical FunASR / SenseTime
production setup.

Helpers convert between PCM bytes ↔ ``rtc.AudioFrame`` chunks for
feeding through the livekit-agents AudioInput / AudioOutput interfaces.

Zero-dependency: uses only ``math``, ``struct``, ``pathlib``, ``wave``,
plus the already-installed ``livekit.rtc`` package.
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path
from typing import Iterable, Iterator

from livekit import rtc

DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2  # 16-bit signed PCM


def _samples_for_duration(duration_sec: float, sample_rate: int) -> int:
    """Number of PCM samples for a given duration."""
    return max(0, int(round(duration_sec * sample_rate)))


def synth_silence(
    duration_sec: float, *, sample_rate: int = DEFAULT_SAMPLE_RATE
) -> bytes:
    """Generate ``duration_sec`` of silence as 16-bit PCM bytes."""
    n = _samples_for_duration(duration_sec, sample_rate)
    return b"\x00\x00" * n


def synth_tone(
    freq_hz: float,
    duration_sec: float,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    amplitude: float = 0.5,
) -> bytes:
    """Pure sine tone at ``freq_hz``. ``amplitude`` ∈ [0, 1] of int16 max.

    Useful as a "VAD-loud" signal that any reasonable VAD will treat
    as speech-active. Not voice-shaped — for that use ``synth_voiced``.
    """
    n = _samples_for_duration(duration_sec, sample_rate)
    peak = int(round(max(0.0, min(1.0, amplitude)) * 32_767))
    out = bytearray(n * SAMPLE_WIDTH_BYTES)
    two_pi_f = 2.0 * math.pi * freq_hz
    for i in range(n):
        sample = int(round(peak * math.sin(two_pi_f * i / sample_rate)))
        struct.pack_into("<h", out, i * SAMPLE_WIDTH_BYTES, sample)
    return bytes(out)


def synth_voiced(
    duration_sec: float,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    fundamental_hz: float = 140.0,
    amplitude: float = 0.4,
) -> bytes:
    """Speech-shaped buzz: fundamental + 2 harmonics with a slow envelope.

    This is **not** real speech — it's a deterministic signal that
    passes plausibility checks of typical VADs (energy in voice band,
    quasi-periodic). For tests that need a "user is speaking" stimulus
    without reaching for an actual WAV file.
    """
    n = _samples_for_duration(duration_sec, sample_rate)
    peak = int(round(max(0.0, min(1.0, amplitude)) * 32_767))
    out = bytearray(n * SAMPLE_WIDTH_BYTES)
    f0 = 2.0 * math.pi * fundamental_hz
    f1 = 2.0 * math.pi * fundamental_hz * 2.0
    f2 = 2.0 * math.pi * fundamental_hz * 3.0
    # Envelope: smooth attack/decay so we don't get pops at boundaries.
    attack = max(1, int(0.02 * sample_rate))  # 20 ms attack
    decay = max(1, int(0.02 * sample_rate))  # 20 ms decay
    for i in range(n):
        if i < attack:
            env = i / attack
        elif i > n - decay:
            env = max(0.0, (n - i) / decay)
        else:
            env = 1.0
        t = i / sample_rate
        s = (
            0.6 * math.sin(f0 * t)
            + 0.3 * math.sin(f1 * t)
            + 0.1 * math.sin(f2 * t)
        )
        sample = int(round(peak * env * s))
        # Clamp to int16 range.
        if sample > 32_767:
            sample = 32_767
        elif sample < -32_768:
            sample = -32_768
        struct.pack_into("<h", out, i * SAMPLE_WIDTH_BYTES, sample)
    return bytes(out)


def wav_load(path: str | Path) -> tuple[bytes, int]:
    """Load a WAV file and return (pcm_bytes, sample_rate).

    Only 16-bit PCM mono WAVs supported (the only format we feed the
    pipeline today). Raises ``ValueError`` otherwise so tests fail fast
    on misconfigured fixtures.
    """
    p = Path(path)
    with wave.open(str(p), "rb") as wf:
        if wf.getsampwidth() != SAMPLE_WIDTH_BYTES:
            raise ValueError(
                f"{p}: expected 16-bit PCM, got {wf.getsampwidth() * 8}-bit"
            )
        if wf.getnchannels() != DEFAULT_CHANNELS:
            raise ValueError(
                f"{p}: expected mono, got {wf.getnchannels()} channels"
            )
        return wf.readframes(wf.getnframes()), wf.getframerate()


def frames_from_pcm(
    pcm: bytes,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    frame_ms: int = 20,
    num_channels: int = DEFAULT_CHANNELS,
) -> Iterator[rtc.AudioFrame]:
    """Chunk PCM bytes into ``rtc.AudioFrame`` of ``frame_ms`` duration each.

    20 ms is the LiveKit / WebRTC convention; both VAD plugins and the
    AgentSession audio chain assume 10–30 ms frames. Trailing partial
    frame (if PCM length not divisible) is dropped to keep frame size
    consistent — tests generating arbitrary durations should use
    durations that are multiples of ``frame_ms``.
    """
    if frame_ms <= 0:
        raise ValueError("frame_ms must be > 0")
    samples_per_frame = sample_rate * frame_ms // 1000
    bytes_per_frame = samples_per_frame * SAMPLE_WIDTH_BYTES * num_channels
    total = len(pcm) // bytes_per_frame
    for i in range(total):
        start = i * bytes_per_frame
        chunk = pcm[start : start + bytes_per_frame]
        yield rtc.AudioFrame(
            data=chunk,
            sample_rate=sample_rate,
            num_channels=num_channels,
            samples_per_channel=samples_per_frame,
        )


def pcm_from_frames(frames: Iterable[rtc.AudioFrame]) -> bytes:
    """Concatenate raw PCM bytes from a sequence of AudioFrame.

    Useful for asserting on captured TTS output: feed the
    ``RecordingAudioOutput`` buffer in, get back contiguous PCM you can
    measure (length, RMS, etc.).
    """
    return b"".join(bytes(f.data) for f in frames)


def pcm_duration_sec(pcm: bytes, *, sample_rate: int = DEFAULT_SAMPLE_RATE) -> float:
    """Compute duration in seconds of 16-bit mono PCM."""
    samples = len(pcm) // SAMPLE_WIDTH_BYTES
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    return samples / sample_rate


def pcm_rms(pcm: bytes) -> float:
    """Root-mean-square energy ∈ [0, 1] of 16-bit PCM.

    Useful to assert "audio output is non-silent" without inspecting
    the waveform shape. Returns 0.0 for silence, ~0.7 for full-scale
    sine, ~1.0 for full-scale square.
    """
    n = len(pcm) // SAMPLE_WIDTH_BYTES
    if n == 0:
        return 0.0
    # Unpack into ints, square, mean, sqrt.
    samples = struct.unpack(f"<{n}h", pcm)
    total = 0
    for s in samples:
        total += s * s
    mean_sq = total / n
    return math.sqrt(mean_sq) / 32_768.0
