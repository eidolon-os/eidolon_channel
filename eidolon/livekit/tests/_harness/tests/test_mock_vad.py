# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for tests/_harness/mocks/mock_vad.py."""

from __future__ import annotations

import asyncio

import pytest
from livekit.agents.vad import VADEventType

from eidolon.livekit.tests._harness.audio import (
    frames_from_pcm,
    synth_silence,
    synth_voiced,
)
from eidolon.livekit.tests._harness.mocks import MockVAD, MockVADEvent


async def _drain(stream, *, max_events: int, timeout: float = 2.0):
    events = []

    async def runner():
        async for ev in stream:
            events.append(ev)
            if len(events) >= max_events:
                break

    try:
        await asyncio.wait_for(runner(), timeout=timeout)
    except asyncio.TimeoutError:
        pass
    return events


class TestScriptedMode:
    async def test_scripted_start_then_end(self):
        vad = MockVAD.scripted(
            [
                MockVADEvent("start", at_ms=30, probability=0.9),
                MockVADEvent("end", at_ms=80, probability=0.1),
            ]
        )
        stream = vad.stream()
        # Need to push some frames to keep the consume loop alive.
        for f in frames_from_pcm(synth_voiced(0.15), frame_ms=20):
            stream.push_frame(f)
        # Expect: INFERENCE_DONE → START → INFERENCE_DONE → END
        events = await _drain(stream, max_events=4, timeout=0.5)
        await stream.aclose()
        types = [ev.type for ev in events]
        assert VADEventType.START_OF_SPEECH in types
        assert VADEventType.END_OF_SPEECH in types
        # At least 2 INFERENCE_DONE for the 2 scripted moments.
        assert types.count(VADEventType.INFERENCE_DONE) >= 2

    async def test_probabilities_in_inference_events(self):
        vad = MockVAD.scripted(
            [
                MockVADEvent("start", at_ms=20, probability=0.85),
                MockVADEvent("end", at_ms=80, probability=0.15),
            ]
        )
        stream = vad.stream()
        for f in frames_from_pcm(synth_voiced(0.15), frame_ms=20):
            stream.push_frame(f)
        events = await _drain(stream, max_events=4, timeout=0.5)
        await stream.aclose()
        infs = [ev for ev in events if ev.type == VADEventType.INFERENCE_DONE]
        probs = [ev.probability for ev in infs]
        # First inference should match the start probability.
        assert probs[0] == pytest.approx(0.85)


class TestFromAudioMode:
    async def test_voiced_audio_triggers_start(self):
        vad = MockVAD.from_audio(speech_threshold=0.05)
        stream = vad.stream()
        for f in frames_from_pcm(synth_voiced(0.1), frame_ms=20):
            stream.push_frame(f)
        events = await _drain(stream, max_events=10, timeout=0.5)
        await stream.aclose()
        types = [ev.type for ev in events]
        assert VADEventType.START_OF_SPEECH in types

    async def test_silence_yields_low_probability(self):
        vad = MockVAD.from_audio(speech_threshold=0.05)
        stream = vad.stream()
        for f in frames_from_pcm(synth_silence(0.1), frame_ms=20):
            stream.push_frame(f)
        events = await _drain(stream, max_events=8, timeout=0.5)
        await stream.aclose()
        infs = [ev for ev in events if ev.type == VADEventType.INFERENCE_DONE]
        # All probabilities should be near 0 for pure silence.
        assert all(ev.probability < 0.1 for ev in infs)
        assert VADEventType.START_OF_SPEECH not in [ev.type for ev in events]

    async def test_voiced_then_silence_emits_end(self):
        vad = MockVAD.from_audio(speech_threshold=0.05)
        stream = vad.stream()
        for f in frames_from_pcm(synth_voiced(0.06), frame_ms=20):
            stream.push_frame(f)
        for f in frames_from_pcm(synth_silence(0.06), frame_ms=20):
            stream.push_frame(f)
        events = await _drain(stream, max_events=15, timeout=0.5)
        await stream.aclose()
        types = [ev.type for ev in events]
        assert VADEventType.START_OF_SPEECH in types
        assert VADEventType.END_OF_SPEECH in types
        # Order: START before END
        start_idx = types.index(VADEventType.START_OF_SPEECH)
        end_idx = types.index(VADEventType.END_OF_SPEECH)
        assert start_idx < end_idx


class TestSilentMode:
    async def test_no_start_or_end_events(self):
        vad = MockVAD.silent()
        stream = vad.stream()
        for f in frames_from_pcm(synth_voiced(0.1), frame_ms=20):
            stream.push_frame(f)
        events = await _drain(stream, max_events=10, timeout=0.3)
        await stream.aclose()
        types = [ev.type for ev in events]
        assert VADEventType.START_OF_SPEECH not in types
        assert VADEventType.END_OF_SPEECH not in types
        # Inference events still present.
        assert VADEventType.INFERENCE_DONE in types

    async def test_inference_probability_low(self):
        vad = MockVAD.silent()
        stream = vad.stream()
        for f in frames_from_pcm(synth_voiced(0.06), frame_ms=20):
            stream.push_frame(f)
        events = await _drain(stream, max_events=5, timeout=0.3)
        await stream.aclose()
        infs = [ev for ev in events if ev.type == VADEventType.INFERENCE_DONE]
        assert all(ev.probability < 0.1 for ev in infs)


class TestBookkeeping:
    async def test_streams_opened_and_frames(self):
        vad = MockVAD.from_audio()
        s = vad.stream()
        for f in frames_from_pcm(synth_voiced(0.04), frame_ms=20):
            s.push_frame(f)
        await asyncio.sleep(0.05)
        await s.aclose()
        assert vad.streams_opened == 1
        # 2 frames pushed (40ms / 20ms).
        assert vad.frames_received == 2
