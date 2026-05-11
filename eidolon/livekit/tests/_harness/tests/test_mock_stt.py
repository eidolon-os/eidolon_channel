# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for tests/_harness/mocks/mock_stt.py."""

from __future__ import annotations

import asyncio

import pytest
from livekit.agents.stt import SpeechEventType

from eidolon.livekit.tests._harness.audio import (
    frames_from_pcm,
    synth_voiced,
)
from eidolon.livekit.tests._harness.mocks import MockSTT, ScriptedTranscript


async def _drain(stream, *, max_events: int, timeout: float = 2.0):
    """Collect up to max_events from a RecognizeStream."""
    events = []

    async def _runner():
        async for ev in stream:
            events.append(ev)
            if len(events) >= max_events:
                break

    try:
        await asyncio.wait_for(_runner(), timeout=timeout)
    except asyncio.TimeoutError:
        pass
    return events


def _push_some_audio(stream, duration_sec: float = 0.06):
    for f in frames_from_pcm(synth_voiced(duration_sec), frame_ms=20):
        stream.push_frame(f)


class TestScriptedTimedTranscript:
    async def test_emits_start_final_end_in_order(self):
        stt = MockSTT.scripted([("hello world", 50)])
        stream = stt.stream()
        _push_some_audio(stream)
        # Need to flush so input loop completes naturally.
        events = await _drain(stream, max_events=3)
        await stream.aclose()
        assert len(events) >= 3
        assert events[0].type == SpeechEventType.START_OF_SPEECH
        assert events[1].type == SpeechEventType.FINAL_TRANSCRIPT
        assert events[1].alternatives[0].text == "hello world"
        assert events[2].type == SpeechEventType.END_OF_SPEECH

    async def test_interim_then_final(self):
        stt = MockSTT.scripted(
            [
                ScriptedTranscript(
                    text="你好世界",
                    interims=["你", "你好", "你好世"],
                    interim_gap_ms=10,
                    trigger_after_ms=20,
                )
            ]
        )
        stream = stt.stream()
        _push_some_audio(stream, duration_sec=0.2)
        # Expect: START + 3 INTERIM + FINAL + END = 6 events
        events = await _drain(stream, max_events=6)
        await stream.aclose()
        types = [ev.type for ev in events]
        assert types[0] == SpeechEventType.START_OF_SPEECH
        # Three interims
        interim_count = types.count(SpeechEventType.INTERIM_TRANSCRIPT)
        assert interim_count == 3
        assert types.count(SpeechEventType.FINAL_TRANSCRIPT) == 1
        assert types.count(SpeechEventType.END_OF_SPEECH) == 1

    async def test_no_final_when_disabled(self):
        stt = MockSTT(
            scripts=[
                ScriptedTranscript(
                    text="never lands",
                    interims=["partial"],
                    interim_gap_ms=10,
                    trigger_after_ms=20,
                    final=False,
                )
            ]
        )
        stream = stt.stream()
        _push_some_audio(stream, duration_sec=0.2)
        events = await _drain(stream, max_events=3, timeout=0.4)
        await stream.aclose()
        types = [ev.type for ev in events]
        assert SpeechEventType.START_OF_SPEECH in types
        assert SpeechEventType.INTERIM_TRANSCRIPT in types
        assert SpeechEventType.FINAL_TRANSCRIPT not in types
        assert SpeechEventType.END_OF_SPEECH not in types

    async def test_multiple_segments(self):
        stt = MockSTT.scripted([("first", 30), ("second", 60)])
        stream = stt.stream()
        _push_some_audio(stream, duration_sec=0.2)
        events = await _drain(stream, max_events=6)
        await stream.aclose()
        finals = [
            ev.alternatives[0].text
            for ev in events
            if ev.type == SpeechEventType.FINAL_TRANSCRIPT
        ]
        assert finals == ["first", "second"]


class TestPCMByteTrigger:
    async def test_fires_after_n_bytes(self):
        # 3200 bytes = 100ms PCM. Trigger waits until that's pushed.
        stt = MockSTT(
            scripts=[
                ScriptedTranscript(
                    text="ok", trigger_after_ms=0, after_pcm_bytes=3200
                )
            ]
        )
        stream = stt.stream()
        # Push 60ms first → still under threshold.
        for f in frames_from_pcm(synth_voiced(0.06), frame_ms=20):
            stream.push_frame(f)
        # Yield briefly so the consume loop catches up.
        await asyncio.sleep(0.05)
        # No final yet — push more (40ms more = 1280 bytes; total >3200).
        for f in frames_from_pcm(synth_voiced(0.08), frame_ms=20):
            stream.push_frame(f)
        events = await _drain(stream, max_events=3, timeout=1.0)
        await stream.aclose()
        finals = [ev for ev in events if ev.type == SpeechEventType.FINAL_TRANSCRIPT]
        assert len(finals) == 1
        assert finals[0].alternatives[0].text == "ok"


class TestErrorPath:
    async def test_stream_raises_configured_exc(self):
        stt = MockSTT.errors_with(RuntimeError("stt down"))
        stream = stt.stream()
        with pytest.raises(RuntimeError, match="stt down"):
            await _drain(stream, max_events=1)
        await stream.aclose()


class TestBatchRecognize:
    async def test_recognize_returns_first_script(self):
        stt = MockSTT.scripted([("batch text", 0)])
        from livekit.agents.stt import SpeechEvent

        # Real STT.recognize wants an AudioBuffer (single AudioFrame
        # or list of frames). 100ms voiced is enough.
        frames = list(frames_from_pcm(synth_voiced(0.1), frame_ms=20))
        ev = await stt.recognize(frames)
        assert isinstance(ev, SpeechEvent)
        assert ev.type == SpeechEventType.FINAL_TRANSCRIPT
        assert ev.alternatives[0].text == "batch text"


class TestBookkeeping:
    async def test_streams_opened_increments(self):
        stt = MockSTT.scripted([("x", 0)])
        s1 = stt.stream()
        s2 = stt.stream()
        await s1.aclose()
        await s2.aclose()
        assert stt.streams_opened == 2

    async def test_bytes_pushed_tracked(self):
        stt = MockSTT.scripted([("x", 9999)])  # never fires
        stream = stt.stream()
        for f in frames_from_pcm(synth_voiced(0.04), frame_ms=20):
            stream.push_frame(f)
        await asyncio.sleep(0.05)
        await stream.aclose()
        # 40ms @ 16kHz × 2 bytes = 1280 bytes
        assert stt.bytes_pushed == 1280
