# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for tests/_harness/mocks/mock_tts.py."""

from __future__ import annotations

import asyncio
import time

import pytest

from eidolon.livekit.tests._harness.audio import (
    SAMPLE_WIDTH_BYTES,
    pcm_duration_sec,
    pcm_rms,
)
from eidolon.livekit.tests._harness.mocks import MockTTS


async def _collect_audio_bytes(stream) -> bytes:
    """Drain a TTS ChunkedStream and concatenate raw PCM."""
    buf = bytearray()
    async for ev in stream:
        buf.extend(bytes(ev.frame.data))
    return bytes(buf)


class TestSynthesize:
    async def test_pcm_duration_proportional_to_text(self):
        tts = MockTTS(char_seconds=0.05)
        text = "abcdefghij"  # 10 chars × 0.05 = 500 ms expected
        pcm = await _collect_audio_bytes(tts.synthesize(text))
        dur = pcm_duration_sec(pcm)
        assert 0.4 < dur < 0.6, f"got {dur}s for 10 chars × 0.05s"

    async def test_pcm_is_non_silent(self):
        tts = MockTTS()
        pcm = await _collect_audio_bytes(tts.synthesize("hello"))
        assert pcm_rms(pcm) > 0.05

    async def test_records_synth_count(self):
        tts = MockTTS()
        await _collect_audio_bytes(tts.synthesize("a"))
        await _collect_audio_bytes(tts.synthesize("b"))
        assert tts.synth_count == 2
        assert tts.synth_texts == ["a", "b"]

    async def test_empty_text_yields_minimal_pcm(self):
        tts = MockTTS()
        pcm = await _collect_audio_bytes(tts.synthesize(""))
        # Even empty text gets a small non-zero buffer (so the pipeline
        # has something to emit). Should be < 100ms.
        assert 0 < len(pcm) < 16_000 * 2 // 10


class TestErrorMode:
    async def test_synthesize_raises(self):
        tts = MockTTS.errors_with(RuntimeError("tts down"))
        with pytest.raises(RuntimeError, match="tts down"):
            await _collect_audio_bytes(tts.synthesize("hi"))


class TestDelays:
    async def test_first_chunk_delay(self):
        tts = MockTTS(char_seconds=0.02, first_chunk_delay_ms=80)
        t0 = time.monotonic()
        await _collect_audio_bytes(tts.synthesize("ab"))
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.07


class TestProtocolBehavior:
    async def test_emits_at_least_one_frame(self):
        tts = MockTTS(char_seconds=0.02)
        frame_count = 0
        async for ev in tts.synthesize("ab"):
            frame_count += 1
            assert ev.frame.sample_rate == tts.sample_rate
            assert ev.frame.num_channels == tts.num_channels
        assert frame_count >= 1

    async def test_each_frame_is_proper_pcm(self):
        tts = MockTTS()
        async for ev in tts.synthesize("hi"):
            data = bytes(ev.frame.data)
            # 16-bit aligned.
            assert len(data) % SAMPLE_WIDTH_BYTES == 0
            # Frame.samples_per_channel matches data length.
            assert (
                len(data)
                == ev.frame.samples_per_channel * SAMPLE_WIDTH_BYTES * ev.frame.num_channels
            )
