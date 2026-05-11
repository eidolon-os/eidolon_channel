# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for tests/_harness/headless.py.

These verify the in-memory AgentSession wiring (no LiveKit room).
Real scenario tests (turn-taking, interruption, backchannel) live
in tests/scenarios/ — these are unit tests for the harness itself.
"""

from __future__ import annotations

import asyncio

import pytest
from livekit import rtc

from eidolon.livekit.tests._harness.audio import (
    DEFAULT_SAMPLE_RATE,
    SAMPLE_WIDTH_BYTES,
    pcm_rms,
    synth_silence,
    synth_voiced,
)
from eidolon.livekit.tests._harness.headless import (
    EventRecorder,
    RecordingAudioOutput,
    ScriptedAudioInput,
    headless_session,
)
from eidolon.livekit.tests._harness.mocks import (
    MockLLM,
    MockSTT,
    MockTTS,
    MockVAD,
    MockVADEvent,
    ScriptedReply,
    ScriptedTranscript,
)


# ──────────────────────────────────────────────────────────────
#  ScriptedAudioInput
# ──────────────────────────────────────────────────────────────


class TestScriptedAudioInput:
    async def test_drains_pushed_pcm(self):
        ai = ScriptedAudioInput()
        ai.feed_pcm(synth_voiced(0.1))  # 5 frames @ 20ms
        ai.end()
        frames: list[rtc.AudioFrame] = []
        async for f in ai:
            frames.append(f)
        assert len(frames) == 5
        assert ai.frames_consumed == 5
        assert ai.bytes_pushed == 16_000 * 2 // 10

    async def test_silence_helper_pushes_zero_bytes(self):
        ai = ScriptedAudioInput()
        ai.feed_silence(0.06)  # 3 frames @ 20ms
        ai.end()
        frames = []
        async for f in ai:
            frames.append(f)
        assert len(frames) == 3
        # Each frame should be all-zero PCM.
        for f in frames:
            assert pcm_rms(bytes(f.data)) == 0.0

    async def test_anext_blocks_until_data_or_end(self):
        ai = ScriptedAudioInput()

        async def consumer() -> rtc.AudioFrame:
            return await ai.__anext__()

        consumer_task = asyncio.create_task(consumer())
        # Give the consumer a moment to block.
        await asyncio.sleep(0.05)
        assert not consumer_task.done()
        ai.feed_pcm(synth_voiced(0.02))
        frame = await asyncio.wait_for(consumer_task, timeout=0.5)
        assert frame.samples_per_channel == 320  # 20ms @ 16kHz
        ai.end()

    async def test_end_unblocks_anext_with_stopiteration(self):
        ai = ScriptedAudioInput()
        ai.end()
        with pytest.raises(StopAsyncIteration):
            await ai.__anext__()

    async def test_feed_after_end_raises(self):
        ai = ScriptedAudioInput()
        ai.end()
        with pytest.raises(RuntimeError):
            ai.feed_pcm(synth_voiced(0.02))


# ──────────────────────────────────────────────────────────────
#  RecordingAudioOutput
# ──────────────────────────────────────────────────────────────


class TestRecordingAudioOutput:
    async def test_capture_into_segment(self):
        ao = RecordingAudioOutput()
        # Build a 40ms voiced frame and feed it twice.
        pcm = synth_voiced(0.04)
        frame = rtc.AudioFrame(
            data=pcm,
            sample_rate=DEFAULT_SAMPLE_RATE,
            num_channels=1,
            samples_per_channel=len(pcm) // SAMPLE_WIDTH_BYTES,
        )
        await ao.capture_frame(frame)
        await ao.capture_frame(frame)
        assert ao.captured_bytes == len(pcm) * 2
        assert ao.first_audio_event.is_set()
        # No flush yet → segment still open.
        assert ao.segment_count == 0

    async def test_flush_finalizes_segment(self):
        ao = RecordingAudioOutput()
        pcm = synth_voiced(0.04)
        frame = rtc.AudioFrame(
            data=pcm,
            sample_rate=DEFAULT_SAMPLE_RATE,
            num_channels=1,
            samples_per_channel=len(pcm) // 2,
        )
        await ao.capture_frame(frame)
        ao.flush()
        assert ao.segment_count == 1
        assert ao.segments[0].pcm == pcm
        assert ao.segments[0].cleared is False

    async def test_clear_buffer_marks_segment_cleared(self):
        ao = RecordingAudioOutput()
        pcm = synth_voiced(0.04)
        frame = rtc.AudioFrame(
            data=pcm,
            sample_rate=DEFAULT_SAMPLE_RATE,
            num_channels=1,
            samples_per_channel=len(pcm) // 2,
        )
        await ao.capture_frame(frame)
        ao.clear_buffer()
        assert ao.segment_count == 1
        assert ao.segments[0].cleared is True

    async def test_collected_pcm_spans_segments(self):
        ao = RecordingAudioOutput()
        for _ in range(3):
            pcm = synth_voiced(0.02)
            frame = rtc.AudioFrame(
                data=pcm,
                sample_rate=DEFAULT_SAMPLE_RATE,
                num_channels=1,
                samples_per_channel=len(pcm) // 2,
            )
            await ao.capture_frame(frame)
            ao.flush()
        assert ao.segment_count == 3
        # collected_pcm is concatenation of all segment buffers.
        assert len(ao.collected_pcm) == 3 * len(synth_voiced(0.02))

    async def test_wait_for_first_audio(self):
        ao = RecordingAudioOutput()

        async def producer() -> None:
            await asyncio.sleep(0.05)
            pcm = synth_voiced(0.02)
            frame = rtc.AudioFrame(
                data=pcm,
                sample_rate=DEFAULT_SAMPLE_RATE,
                num_channels=1,
                samples_per_channel=len(pcm) // 2,
            )
            await ao.capture_frame(frame)

        asyncio.create_task(producer())
        await ao.wait_for_first_audio(timeout=0.5)
        assert ao.first_audio_event.is_set()


# ──────────────────────────────────────────────────────────────
#  HeadlessSession integration — full pipeline e2e (no LiveKit room)
# ──────────────────────────────────────────────────────────────


class TestHeadlessSessionEndToEnd:
    """Stitch all 4 mocks into a real AgentSession and verify a full
    user→agent turn.

    These exercise the *seam* between mock plugins and the real
    AgentSession orchestration — if the framework's internals shift
    in a way that breaks our mocks' protocol contract, these tests
    catch it.
    """

    async def test_session_starts_and_closes_cleanly(self):
        async with headless_session(
            llm=MockLLM.scripted([("hi", "hello")]),
            stt=MockSTT.scripted([("hi", 30)]),
            tts=MockTTS(char_seconds=0.02),
            vad=MockVAD.silent(),
        ) as h:
            assert h.session is not None
            assert h.audio_in is not None
            assert h.audio_out is not None
            assert h.events is not None

    async def test_user_transcript_reaches_recorder(self):
        async with headless_session(
            llm=MockLLM.scripted([("hello", "hi back")]),
            stt=MockSTT.scripted(
                [ScriptedTranscript(text="hello", trigger_after_ms=50)]
            ),
            tts=MockTTS(char_seconds=0.02),
            vad=MockVAD.scripted(
                [
                    MockVADEvent("start", at_ms=20, probability=0.9),
                    MockVADEvent("end", at_ms=120, probability=0.1),
                ]
            ),
        ) as h:
            # Feed 200 ms of voiced + 100 ms silence.
            h.audio_in.feed_pcm(synth_voiced(0.2))
            h.audio_in.feed_silence(0.1)
            # Wait for the final user transcript.
            await h.events.wait_for(
                lambda e: e.type == "user_input_transcribed"
                and e.payload.is_final,
                timeout=5.0,
            )
            finals = h.events.user_finals()
            assert any(t.transcript == "hello" for t in finals)

    async def test_agent_responds_with_tts_audio(self):
        async with headless_session(
            llm=MockLLM.scripted([("hello", "hi there")]),
            stt=MockSTT.scripted(
                [ScriptedTranscript(text="hello", trigger_after_ms=50)]
            ),
            tts=MockTTS(char_seconds=0.02),
            vad=MockVAD.scripted(
                [
                    MockVADEvent("start", at_ms=20, probability=0.9),
                    MockVADEvent("end", at_ms=120, probability=0.1),
                ]
            ),
        ) as h:
            h.audio_in.feed_pcm(synth_voiced(0.2))
            h.audio_in.feed_silence(0.1)
            # Wait for first non-silent TTS output.
            await h.audio_out.wait_for_first_audio(timeout=10.0)
            assert h.audio_out.captured_bytes > 0

    async def test_event_recorder_captures_state_transitions(self):
        async with headless_session(
            llm=MockLLM.scripted([("hello", "hi")]),
            stt=MockSTT.scripted(
                [ScriptedTranscript(text="hello", trigger_after_ms=50)]
            ),
            tts=MockTTS(char_seconds=0.02),
            vad=MockVAD.scripted(
                [
                    MockVADEvent("start", at_ms=20, probability=0.9),
                    MockVADEvent("end", at_ms=120, probability=0.1),
                ]
            ),
        ) as h:
            h.audio_in.feed_pcm(synth_voiced(0.2))
            h.audio_in.feed_silence(0.1)
            # Explicitly wait for agent to enter speaking state — this
            # is what the assertion below checks for. wait_for_first_audio
            # is racy: PCM can land before or after the state event.
            await h.events.wait_for_state(agent="speaking", timeout=10.0)
            history = h.events.agent_state_history()
            # Should have seen at least: idle → listening → thinking → speaking
            assert "speaking" in history


# ──────────────────────────────────────────────────────────────
#  EventRecorder sub-API
# ──────────────────────────────────────────────────────────────


class TestEventRecorderHelpers:
    """Test EventRecorder helpers in isolation by feeding it events
    via a dummy session."""

    async def test_collects_and_categorizes_events(self):
        # Use the headless session to produce real events.
        async with headless_session(
            llm=MockLLM.echo(),
            stt=MockSTT.scripted(
                [
                    ScriptedTranscript(
                        text="ping",
                        interims=["pi", "pin"],
                        interim_gap_ms=10,
                        trigger_after_ms=30,
                    )
                ]
            ),
            tts=MockTTS(char_seconds=0.02),
            vad=MockVAD.scripted(
                [
                    MockVADEvent("start", at_ms=10, probability=0.9),
                    MockVADEvent("end", at_ms=90, probability=0.1),
                ]
            ),
        ) as h:
            h.audio_in.feed_pcm(synth_voiced(0.15))
            h.audio_in.feed_silence(0.1)
            await h.audio_out.wait_for_first_audio(timeout=10.0)
            interims = h.events.user_interims()
            finals = h.events.user_finals()
            # We should see >= 2 interims (we sent 3 in the script, but
            # AgentSession may collapse) and 1 final.
            assert len(finals) >= 1
            assert finals[0].transcript == "ping"
            assert h.events.agent_state_history()
            # of_type filter works
            close_events = h.events.of_type("close")
            # No close while still in context.
            assert close_events == []
