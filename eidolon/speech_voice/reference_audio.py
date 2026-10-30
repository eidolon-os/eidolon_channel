"""A clone reference — recorded on the phone or picked from a file — normalized and measured.

Any common audio file is decoded with PyAV, made 24 kHz mono PCM16, and judged
the way the voiceprint enrollment judges speech: FireRed VAD over 16 kHz
windows, plus clipping and level. A long file is cut to the 20 seconds with the
most speech, or to the window the Owner chose. Only that window is returned;
the original bytes are not kept anywhere.
"""

from __future__ import annotations

import base64
import io
import threading

import numpy as np
from eidolon_sdk.biz.speech_voice import (
    RECORDING_MAX_MS,
    RECORDING_MIN_MS,
    REFERENCE_MIN_SPEECH_MS,
    REFERENCE_SAMPLE_RATE,
    REFERENCE_WINDOW_MS,
    UPLOAD_MAX_BYTES,
    UPLOAD_MAX_MS,
    ReferenceCheck,
)

from .wav import wav_bytes

VAD_RATE = 16_000
CLIPPING_LIMIT = 0.02
QUIET_RMS = 0.003

_vad_lock = threading.Lock()
_vad = None


def _speech_flags(samples_16k: np.ndarray) -> tuple[np.ndarray, int]:
    """Per-window speech decisions and the window length in samples."""
    global _vad
    with _vad_lock:
        if _vad is None:
            from eidolon.livekit.plugins.vad.firered.processor import PvadProcessor

            _vad = PvadProcessor(force_cpu=True)
        _vad.reset()
        size = _vad.window_size_samples
        flags = np.array([
            _vad(samples_16k[i:i + size]) >= 0.5 for i in range(0, len(samples_16k) - size + 1, size)
        ], dtype=bool)
    return flags, size


def _decode(data: bytes) -> np.ndarray:
    import av

    resampler = av.AudioResampler(format="s16", layout="mono", rate=REFERENCE_SAMPLE_RATE)
    parts: list[np.ndarray] = []
    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.audio[0]
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                parts.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            parts.append(out.to_ndarray().reshape(-1))
    if not parts:
        raise ValueError("no audio")
    return np.concatenate(parts).astype(np.int16)


def check_reference(data: bytes, *, start_ms: int | None = None) -> ReferenceCheck:
    def rejected(problem: str, duration_ms: int = 0) -> ReferenceCheck:
        return ReferenceCheck(accepted=False, problems=[problem], duration_ms=duration_ms,
                              window_start_ms=0, window_ms=0)

    if len(data) > UPLOAD_MAX_BYTES:
        return rejected("too_large")
    try:
        pcm = _decode(data)
    except Exception:  # noqa: BLE001 - whatever a file does to the decoder, it is unreadable
        return rejected("unreadable")
    duration_ms = len(pcm) * 1000 // REFERENCE_SAMPLE_RATE
    if duration_ms < RECORDING_MIN_MS:
        return rejected("too_short", duration_ms)
    if duration_ms > UPLOAD_MAX_MS:
        return rejected("too_long", duration_ms)

    samples = pcm.astype(np.float32) / 32768
    # The VAD judges 16 kHz windows, as in voiceprint enrollment.
    positions = np.arange(0, len(samples), REFERENCE_SAMPLE_RATE / VAD_RATE)
    samples_16k = np.interp(positions, np.arange(len(samples)), samples).astype(np.float32)
    flags, size = _speech_flags(samples_16k)
    window_ms_each = size * 1000 / VAD_RATE

    if duration_ms <= RECORDING_MAX_MS:
        start, length = 0, duration_ms
    else:
        length = REFERENCE_WINDOW_MS
        count = int(length / window_ms_each)
        if start_ms is None:
            sums = np.convolve(flags.astype(int), np.ones(count, dtype=int), mode="valid")
            start = int(np.argmax(sums) * window_ms_each) if len(sums) else 0
        else:
            start = start_ms
        start = max(0, min(start, duration_ms - length))
    first, last = int(start / window_ms_each), int((start + length) / window_ms_each)
    speech_ms = int(flags[first:last].sum() * window_ms_each)
    cut = pcm[start * REFERENCE_SAMPLE_RATE // 1000:(start + length) * REFERENCE_SAMPLE_RATE // 1000]
    window = cut.astype(np.float32) / 32768
    rms = float(np.sqrt(np.mean(window * window))) if len(window) else 0.0
    clipping = float(np.mean(np.abs(window) >= 0.995)) if len(window) else 0.0

    problems = []
    if speech_ms < REFERENCE_MIN_SPEECH_MS:
        problems.append("too_little_speech")
    if rms < QUIET_RMS:
        problems.append("too_quiet")
    if clipping > CLIPPING_LIMIT:
        problems.append("clipping")
    return ReferenceCheck(
        accepted=not problems, problems=problems, duration_ms=duration_ms, window_start_ms=start,
        window_ms=length, speech_ms=speech_ms, rms=round(rms, 4), clipping=round(clipping, 4),
        wav_base64=base64.b64encode(wav_bytes(cut.tobytes(), REFERENCE_SAMPLE_RATE)).decode(),
    )
