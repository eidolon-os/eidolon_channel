"""Filler word manager — pre-cached latency masking audio.

Synthesizes short filler phrases ("嗯...", "好的...") at pipeline warmup
and injects them into the audio output between user-done and agent-starts
to mask the ASR-finalization + LLM-first-token latency gap (~1-2 s).

The filler is pushed as a background task and cancelled the moment
the real agent speech begins. This brings perceived response latency
from ~1.5 s down to ~200 ms.

Audio processing notes:
  * Filler clips are synthesized by the TTS plugin, which may produce
    audio at a different sample rate than the room output chain (e.g.
    BailianTTS → 16 kHz, RoomIO → 24 kHz). We resample to the target
    rate once via ``prepare_for_output(target_sr)`` rather than on
    every inject.
  * A fade envelope (30 ms in, 80 ms out) is applied to avoid the
    "click" artefact from instantaneous onset / cut-off.
  * A short silence lead-in (~120 ms) creates a natural micro-pause
    between the user's last word and the filler — without it the
    filler sounds like it "steps on" the end of the user's speech.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import TYPE_CHECKING

import numpy as np
from livekit import rtc

from livekit.agents.voice import io as lk_io

if TYPE_CHECKING:
    from ..pipeline.tts import TtsStage

logger = logging.getLogger("agent.output.filler")

DEFAULT_PHRASES = ["嗯...", "好的...", "让我想想..."]

# Envelope durations (milliseconds).
_FADE_IN_MS = 30
_FADE_OUT_MS = 80
_SILENCE_LEAD_IN_MS = 120


def _resample_pcm(
    samples: np.ndarray,
    src_rate: int,
    dst_rate: int,
) -> np.ndarray:
    """Linear-interpolation resample for int16 mono PCM.

    Good enough for short filler clips (< 1 s).  We don't pull in
    scipy/librosa just for this — the quality difference is inaudible
    for human-speech clips at 16→24 kHz.
    """
    if src_rate == dst_rate:
        return samples
    ratio = dst_rate / src_rate
    src_len = len(samples)
    dst_len = int(round(src_len * ratio))
    # Float64 indices into the source array.
    indices = np.arange(dst_len, dtype=np.float64) * (src_len - 1) / max(dst_len - 1, 1)
    floor_idx = np.floor(indices).astype(np.int64)
    ceil_idx = np.minimum(floor_idx + 1, src_len - 1)
    frac = (indices - floor_idx).astype(np.float32)
    resampled = (
        samples[floor_idx].astype(np.float32) * (1.0 - frac)
        + samples[ceil_idx].astype(np.float32) * frac
    )
    return np.clip(resampled, -32768, 32767).astype(np.int16)


def _apply_fade_envelope(
    samples: np.ndarray,
    sample_rate: int,
    fade_in_ms: int = _FADE_IN_MS,
    fade_out_ms: int = _FADE_OUT_MS,
) -> np.ndarray:
    """Apply a linear fade-in and fade-out to *samples* (int16 mono).

    Returns a new array — *samples* is not mutated.
    """
    n = len(samples)
    fade_in_n = min(int(sample_rate * fade_in_ms / 1000), n)
    fade_out_n = min(int(sample_rate * fade_out_ms / 1000), n - fade_in_n)

    out = samples.astype(np.float32)
    if fade_in_n > 0:
        ramp_in = np.linspace(0.0, 1.0, fade_in_n, dtype=np.float32)
        out[:fade_in_n] *= ramp_in
    if fade_out_n > 0:
        ramp_out = np.linspace(1.0, 0.0, fade_out_n, dtype=np.float32)
        out[-fade_out_n:] *= ramp_out
    return np.clip(out, -32768, 32767).astype(np.int16)


def _frames_to_samples(frames: list[rtc.AudioFrame]) -> tuple[np.ndarray, int, int]:
    """Concatenate a list of AudioFrames into one int16 array.

    Returns ``(samples, sample_rate, num_channels)``.
    """
    if not frames:
        return np.array([], dtype=np.int16), 0, 1
    sr = frames[0].sample_rate
    nc = frames[0].num_channels
    parts = [np.frombuffer(f.data, dtype=np.int16) for f in frames]
    return np.concatenate(parts), sr, nc


def _samples_to_frames(
    samples: np.ndarray,
    sample_rate: int,
    num_channels: int,
    frame_duration_ms: int = 10,
) -> list[rtc.AudioFrame]:
    """Split a contiguous int16 array back into 10 ms AudioFrames."""
    samples_per_frame = sample_rate * frame_duration_ms // 1000 * num_channels
    frames: list[rtc.AudioFrame] = []
    for offset in range(0, len(samples), samples_per_frame):
        chunk = samples[offset: offset + samples_per_frame]
        if len(chunk) < samples_per_frame:
            # Pad the last frame with silence to avoid partial-frame errors.
            chunk = np.pad(chunk, (0, samples_per_frame - len(chunk)))
        frames.append(
            rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=sample_rate,
                num_channels=num_channels,
                samples_per_channel=len(chunk) // num_channels,
            )
        )
    return frames


class FillerManager:
    """Pre-cache and inject filler audio clips for latency masking.

    Usage::

        mgr = FillerManager(tts_stage, phrases=["嗯...", "好的..."])
        await mgr.warmup()              # pre-synthesize during pipeline startup
        mgr.prepare_for_output(24000)   # resample + envelope once output rate known
        mgr.inject(audio_sink)          # fire-and-forget async push
        mgr.cancel()                    # stop filler when real speech starts
    """

    def __init__(
        self,
        tts_stage: TtsStage,
        *,
        phrases: list[str] | None = None,
    ) -> None:
        self._tts = tts_stage
        self._phrases = phrases or DEFAULT_PHRASES
        # Raw clips as synthesized by TTS (original sample rate).
        self._raw_clips: list[list[rtc.AudioFrame]] = []
        # Clips after resample + envelope (target sample rate).
        self._clips: list[list[rtc.AudioFrame]] = []
        self._inject_task: asyncio.Task | None = None
        self._playing = False

    @property
    def is_playing(self) -> bool:
        return self._playing

    async def warmup(self) -> None:
        """Pre-synthesize all filler phrases into cached AudioFrame lists."""
        for phrase in self._phrases:
            try:
                frames = await self._tts.synthesize_all(phrase)
                if frames:
                    self._raw_clips.append(frames)
                    logger.info(
                        "[FillerManager] cached %r → %d frames (sr=%d)",
                        phrase, len(frames), frames[0].sample_rate,
                    )
            except Exception:
                logger.warning(
                    "[FillerManager] failed to synthesize %r, skipping",
                    phrase, exc_info=True,
                )
        logger.info(
            "[FillerManager] warmup complete — %d clips cached",
            len(self._raw_clips),
        )
        # If prepare_for_output is never called, fall back to raw clips.
        self._clips = self._raw_clips

    def prepare_for_output(self, target_sample_rate: int) -> None:
        """Resample all cached clips to *target_sample_rate* and apply
        fade envelope + silence lead-in.

        Call once after the audio output chain's sample rate is known
        (typically right after ``_install_duck_mixer``).
        """
        if not self._raw_clips:
            return

        processed: list[list[rtc.AudioFrame]] = []
        for raw_clip in self._raw_clips:
            samples, src_sr, nc = _frames_to_samples(raw_clip)
            if len(samples) == 0:
                continue

            # 1. Resample if needed.
            if src_sr != target_sample_rate:
                samples = _resample_pcm(samples, src_sr, target_sample_rate)
                logger.debug(
                    "[FillerManager] resampled clip %d→%d Hz (%d samples)",
                    src_sr, target_sample_rate, len(samples),
                )

            # 2. Apply fade envelope for smooth onset/offset.
            samples = _apply_fade_envelope(samples, target_sample_rate)

            # 3. Prepend silence lead-in (natural micro-pause).
            silence_n = int(target_sample_rate * _SILENCE_LEAD_IN_MS / 1000) * nc
            silence = np.zeros(silence_n, dtype=np.int16)
            samples = np.concatenate([silence, samples])

            # 4. Split back into 10 ms frames at the target rate.
            frames = _samples_to_frames(samples, target_sample_rate, nc)
            processed.append(frames)
            logger.info(
                "[FillerManager] prepared clip → %d frames @ %d Hz "
                "(lead-in=%dms, fade_in=%dms, fade_out=%dms)",
                len(frames), target_sample_rate,
                _SILENCE_LEAD_IN_MS, _FADE_IN_MS, _FADE_OUT_MS,
            )

        self._clips = processed or self._raw_clips
        logger.info(
            "[FillerManager] prepare_for_output complete — %d clips ready @ %d Hz",
            len(self._clips), target_sample_rate,
        )

    def inject(self, audio_sink: lk_io.AudioOutput) -> None:
        """Start pushing a random filler clip to *audio_sink*.

        Non-blocking: spawns an async task. Call :meth:`cancel` when
        the real agent speech starts.
        """
        if not self._clips:
            return
        self.cancel()
        clip = random.choice(self._clips)
        self._inject_task = asyncio.create_task(
            self._push_clip(audio_sink, clip)
        )

    async def _push_clip(
        self,
        sink: lk_io.AudioOutput,
        clip: list[rtc.AudioFrame],
    ) -> None:
        self._playing = True
        try:
            for frame in clip:
                await sink.capture_frame(frame)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.warning("[FillerManager] error pushing filler frame", exc_info=True)
        finally:
            self._playing = False

    def cancel(self) -> None:
        """Stop any in-progress filler playback."""
        if self._inject_task is not None and not self._inject_task.done():
            self._inject_task.cancel()
        self._inject_task = None
        self._playing = False
