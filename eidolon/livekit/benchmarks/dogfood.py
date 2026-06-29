"""Human+device dogfood helpers for voice benchmarks.

This module belongs to the benchmark layer. It translates declarative dogfood
case metadata into runner behavior: device audio-state cadence and synthetic
mic rendering. Runtime channel code must not depend on it.
"""

from __future__ import annotations

import hashlib
import math
import random
import struct
from typing import Iterable

from eidolon.livekit.tests._harness.audio import (
    SAMPLE_WIDTH_BYTES,
    pcm_duration_sec,
    pcm_rms,
    synth_silence,
    synth_voiced,
)

from .schema import BenchmarkCase, UserStep


def audio_state_interval_sec(case: BenchmarkCase, *, default_sec: float = 0.5) -> float:
    """Return the device audio_state refresh interval for a benchmark case."""

    if not case.dogfood.enabled:
        return default_sec
    hz = float(case.dogfood.device.audio_state_hz or 0.0)
    if hz <= 0:
        return default_sec
    return max(0.05, 1.0 / hz)


def dogfood_metrics(case: BenchmarkCase) -> dict[str, object]:
    """Stable metrics/report fields describing the simulated device envelope."""

    if not case.dogfood.enabled:
        return {"dogfood_enabled": False}
    return {
        "dogfood_enabled": True,
        "dogfood_device_model": case.dogfood.device.model,
        "dogfood_device_mode": case.dogfood.device.mode,
        "dogfood_audio_state_hz": case.dogfood.device.audio_state_hz,
        "dogfood_playback_ack": case.dogfood.device.playback_ack,
        "dogfood_echo_enabled": case.dogfood.acoustics.echo.enabled,
        "dogfood_noise_enabled": case.dogfood.acoustics.noise.enabled,
    }


def render_dogfood_mic_pcm(
    case: BenchmarkCase,
    step: UserStep,
    near_end_pcm: bytes,
    *,
    sample_rate: int,
) -> bytes:
    """Render mic input for a human+device dogfood case.

    The default is a no-op. When dogfood acoustics are enabled, the rendered mic
    contains deterministic near-end user audio plus synthetic agent echo/noise.
    This is intentionally lightweight; real HIL tests should replace the
    synthetic echo with captured playback reference audio.
    """

    if not case.dogfood.enabled:
        return near_end_pcm

    rendered = near_end_pcm
    echo = case.dogfood.acoustics.echo
    if echo.enabled:
        rendered = mix_pcm(
            rendered,
            _synthetic_agent_echo(
                case,
                step,
                base_duration_sec=pcm_duration_sec(near_end_pcm, sample_rate=sample_rate),
                sample_rate=sample_rate,
            ),
            delay_ms=echo.delay_ms,
            gain=db_to_gain(echo.attenuation_db),
            sample_rate=sample_rate,
        )

    noise = case.dogfood.acoustics.noise
    if noise.enabled:
        rendered = mix_pcm(
            rendered,
            _noise_pcm(
                len(rendered),
                sample_rate=sample_rate,
                amplitude=_noise_amplitude(rendered, noise.snr_db, noise.amplitude),
                seed=f"{case.case_id}:{step.start_ms}:{step.text}",
            ),
            sample_rate=sample_rate,
        )

    return rendered


def db_to_gain(db: float) -> float:
    return math.pow(10.0, float(db) / 20.0)


def mix_pcm(
    primary: bytes,
    secondary: bytes,
    *,
    delay_ms: int = 0,
    gain: float = 1.0,
    sample_rate: int,
) -> bytes:
    """Mix two mono int16 PCM buffers, clipping safely to int16."""

    primary_samples = list(_samples(primary))
    delay_samples = max(0, int(sample_rate * delay_ms / 1000))
    secondary_samples = [0] * delay_samples + [
        int(round(sample * gain)) for sample in _samples(secondary)
    ]
    total = max(len(primary_samples), len(secondary_samples))
    if len(primary_samples) < total:
        primary_samples.extend([0] * (total - len(primary_samples)))
    if len(secondary_samples) < total:
        secondary_samples.extend([0] * (total - len(secondary_samples)))
    mixed = [
        _clip_i16(a + b)
        for a, b in zip(primary_samples, secondary_samples, strict=True)
    ]
    # Preserve the original capture duration; delayed echo beyond the capture
    # window is not available to the mic input for this benchmark step.
    original_len = len(primary) // SAMPLE_WIDTH_BYTES
    return _pack_samples(mixed[:original_len])


def _synthetic_agent_echo(
    case: BenchmarkCase,
    step: UserStep,
    *,
    base_duration_sec: float,
    sample_rate: int,
) -> bytes:
    duration_sec = max(
        base_duration_sec,
        float(case.dogfood.agent.tts_duration_ms or 0) / 1000.0,
        float(step.duration_ms or 0) / 1000.0,
    )
    # Low-pitched voice-shaped signal: deterministic and VAD-plausible, but
    # clearly marked as synthetic by its generator and benchmark metadata.
    return synth_voiced(
        duration_sec,
        sample_rate=sample_rate,
        fundamental_hz=115.0,
        amplitude=0.25,
    )


def _noise_pcm(
    byte_len: int,
    *,
    sample_rate: int,
    amplitude: float,
    seed: str,
) -> bytes:
    del sample_rate  # Kept for call-site symmetry with other generators.
    samples = byte_len // SAMPLE_WIDTH_BYTES
    rnd = random.Random(_stable_seed(seed))
    peak = int(round(max(0.0, min(1.0, amplitude)) * 32767))
    return _pack_samples(rnd.randint(-peak, peak) for _ in range(samples))


def _noise_amplitude(pcm: bytes, snr_db: float | None, configured: float) -> float:
    if configured > 0:
        return configured
    if snr_db is None:
        return 0.01
    signal = max(pcm_rms(pcm), 0.01)
    return max(0.0, min(1.0, signal / db_to_gain(float(snr_db))))


def _stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _samples(pcm: bytes) -> Iterable[int]:
    count = len(pcm) // SAMPLE_WIDTH_BYTES
    return struct.unpack(f"<{count}h", pcm[: count * SAMPLE_WIDTH_BYTES])


def _pack_samples(samples: Iterable[int]) -> bytes:
    values = list(samples)
    if not values:
        return b""
    return struct.pack(f"<{len(values)}h", *(_clip_i16(value) for value in values))


def _clip_i16(value: int) -> int:
    return max(-32768, min(32767, int(value)))


def dogfood_silence(duration_sec: float, *, sample_rate: int) -> bytes:
    """Named wrapper used by tests and future runners for dogfood timelines."""

    return synth_silence(duration_sec, sample_rate=sample_rate)
