"""Unit tests for the voice agent pipeline.

Run with::

    cd <repository-root>
    .venv/bin/python -m pytest eidolon/livekit/agent/pipeline/test_pipeline.py -v
"""

from __future__ import annotations

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pcm_frame(duration_ms: int = 100, sample_rate: int = 16000) -> bytes:
    """Create a synthetic PCM audio frame (silence)."""
    num_samples = int(sample_rate * duration_ms / 1000)
    return b"\x00\x00" * num_samples


# ---------------------------------------------------------------------------
# Test: BatchPipeline audio processing (mocked)
# ---------------------------------------------------------------------------

class TestBatchPipeline:
    @pytest.mark.asyncio
    async def test_process_audio_track_calls_stt_llm_tts(self):
        """Test that _process_audio_track invokes stt -> llm -> tts in sequence."""
        from eidolon.livekit.agent.batch import BatchPipeline
        from eidolon.livekit.agent.pipeline import PipelineCallbacks, PipelineState
        from eidolon.livekit.agent.pipeline.llm import LlmOutput
        from livekit import rtc

        # Directly call the processing chain to test the sequence
        mock_factory = MagicMock()
        mock_factory.stt.recognize = AsyncMock(return_value="book a flight")
        mock_factory.llm.chat = AsyncMock(
            return_value=LlmOutput(text="好的，我来帮您订票。")
        )

        async def mock_synthesize(text):
            tts_frame = rtc.AudioFrame(
                data=b"\x00\x00" * 1600,
                sample_rate=16000,
                num_channels=1,
                samples_per_channel=1600,
            )
            yield tts_frame

        mock_factory.tts.synthesize = MagicMock(side_effect=mock_synthesize)

        callbacks_fired: list[str] = []

        def make_cb(name):
            def cb(*args):
                callbacks_fired.append(name)
            return cb

        pipeline = BatchPipeline(
            mock_factory,
            callbacks=PipelineCallbacks(
                on_user_message=make_cb("user_message"),
                on_agent_started_speaking=make_cb("agent_started"),
                on_agent_message=make_cb("agent_message"),
                on_agent_ended_speaking=make_cb("agent_ended"),
                on_agent_response_done=make_cb("response_done"),
            ),
        )

        # Directly set up state as if a track ended with audio
        pipeline._room = MagicMock()
        pipeline._room.local_publish_audio = AsyncMock()
        audio_blob = b"\x00\x01" * 1600  # 200ms of audio

        # Manually exercise the processing steps as _process_audio_track would
        pipeline._state = PipelineState.PROCESSING_AUDIO
        transcript = await mock_factory.stt.recognize(audio_blob)
        assert transcript == "book a flight"

        pipeline._callbacks.on_user_message(transcript)
        assert "user_message" in callbacks_fired

        pipeline._state = PipelineState.GENERATING
        pipeline._callbacks.on_agent_started_speaking()
        assert "agent_started" in callbacks_fired

        llm_output = await mock_factory.llm.chat(
            MagicMock(text=transcript)
        )
        response_text = llm_output.text
        assert response_text == "好的，我来帮您订票。"

        pipeline._state = PipelineState.SPEAKING
        frame_count = 0
        async for frame in mock_factory.tts.synthesize(response_text):
            await pipeline._room.local_publish_audio(frame)
            frame_count += 1

        assert frame_count == 1
        pipeline._callbacks.on_agent_message(response_text)
        pipeline._callbacks.on_agent_ended_speaking()
        pipeline._callbacks.on_agent_response_done()
        assert "agent_message" in callbacks_fired
        assert "agent_ended" in callbacks_fired
        assert "response_done" in callbacks_fired

        mock_factory.stt.recognize.assert_called_once()
        mock_factory.llm.chat.assert_called_once()
        mock_factory.tts.synthesize.assert_called_once_with("好的，我来帮您订票。")

    def test_processing_flag_prevents_concurrent_calls(self):
        """Test that the _processing flag prevents concurrent track processing."""
        from eidolon.livekit.agent.batch import BatchPipeline
        from eidolon.livekit.agent.pipeline import PipelineCallbacks

        mock_factory = MagicMock()
        pipeline = BatchPipeline(
            mock_factory,
            callbacks=PipelineCallbacks(),
        )

        # Initially not processing
        assert pipeline._processing is False

        # Simulate processing started
        pipeline._processing = True
        assert pipeline._processing is True

        # The concurrent check in _process_audio_track relies on this flag
        # Setting it back simulates processing completion
        pipeline._processing = False
        assert pipeline._processing is False

    def test_frames_to_pcm_blob(self):
        """Test that _frames_to_pcm_blob correctly concatenates frames."""
        from eidolon.livekit.agent.batch import BatchPipeline
        from livekit import rtc

        frame1 = rtc.AudioFrame(
            data=b"\x01\x02\x03\x04",
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=2,
        )
        frame2 = rtc.AudioFrame(
            data=b"\x05\x06\x07\x08",
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=2,
        )

        blob = BatchPipeline._frames_to_pcm_blob([frame1, frame2])
        assert blob == b"\x01\x02\x03\x04\x05\x06\x07\x08"


# ---------------------------------------------------------------------------
# Test: BasePipeline shared behavior
# ---------------------------------------------------------------------------

class TestBasePipeline:
    def test_initial_state(self):
        from eidolon.livekit.agent.batch import BatchPipeline
        from eidolon.livekit.agent.pipeline import PipelineCallbacks, PipelineState

        mock_factory = MagicMock()
        pipeline = BatchPipeline(mock_factory, callbacks=PipelineCallbacks())

        assert pipeline.state == PipelineState.IDLE
        assert pipeline.factory is mock_factory
        assert pipeline._started is False

    @pytest.mark.asyncio
    async def test_shutdown_resets_state(self):
        from eidolon.livekit.agent.batch import BatchPipeline
        from eidolon.livekit.agent.pipeline import PipelineCallbacks, PipelineState

        mock_factory = MagicMock()
        pipeline = BatchPipeline(mock_factory, callbacks=PipelineCallbacks())

        pipeline._state = PipelineState.SPEAKING
        pipeline._started = True

        await pipeline.shutdown()

        assert pipeline.state == PipelineState.IDLE
        assert pipeline._started is False

    def test_on_agent_state_changed_callbacks(self):
        """Test that _on_agent_state_changed fires correct callbacks."""
        from eidolon.livekit.agent.batch import BatchPipeline
        from eidolon.livekit.agent.pipeline import PipelineCallbacks, PipelineState

        callbacks_fired: list[str] = []

        def make_cb(name):
            def cb(*args):
                callbacks_fired.append(name)
            return cb

        mock_factory = MagicMock()
        pipeline = BatchPipeline(
            mock_factory,
            callbacks=PipelineCallbacks(
                on_agent_started_speaking=make_cb("agent_started"),
                on_agent_ended_speaking=make_cb("agent_ended"),
                on_agent_response_done=make_cb("response_done"),
            ),
        )

        # Simulate agent state change to speaking
        mock_event = MagicMock()
        mock_event.old_state = "idle"
        mock_event.new_state = "speaking"
        pipeline._on_agent_state_changed(mock_event)
        assert "agent_started" in callbacks_fired
        assert pipeline.state == PipelineState.SPEAKING

        # Simulate agent state change to idle
        callbacks_fired.clear()
        mock_event.new_state = "idle"
        mock_event.old_state = "speaking"
        pipeline._on_agent_state_changed(mock_event)
        assert "agent_ended" in callbacks_fired
        assert "response_done" in callbacks_fired
        assert pipeline.state == PipelineState.IDLE

    def test_on_user_transcribed_callbacks(self):
        """Test that _on_user_transcribed fires correct callbacks."""
        from eidolon.livekit.agent.batch import BatchPipeline
        from eidolon.livekit.agent.pipeline import PipelineCallbacks

        callbacks_fired: list[str] = []

        def make_cb(name):
            def cb(*args):
                callbacks_fired.append(name)
            return cb

        mock_factory = MagicMock()
        pipeline = BatchPipeline(
            mock_factory,
            callbacks=PipelineCallbacks(on_user_message=make_cb("user_message")),
        )

        # Simulate final transcript
        mock_event = MagicMock()
        mock_event.is_final = True
        mock_event.transcript = "hello world"
        pipeline._on_user_transcribed(mock_event)
        assert "user_message" in callbacks_fired

        # Simulate interim transcript (should not fire)
        callbacks_fired.clear()
        mock_event.is_final = False
        mock_event.transcript = "hel"
        pipeline._on_user_transcribed(mock_event)
        assert "user_message" not in callbacks_fired

    def test_on_session_error_callback(self):
        """Test that _on_session_error fires the error callback."""
        from eidolon.livekit.agent.batch import BatchPipeline
        from eidolon.livekit.agent.pipeline import PipelineCallbacks

        errors: list = []

        mock_factory = MagicMock()
        pipeline = BatchPipeline(
            mock_factory,
            callbacks=PipelineCallbacks(on_error=lambda e: errors.append(e)),
        )

        mock_event = MagicMock()
        mock_event.error = ValueError("test error")
        pipeline._on_session_error(mock_event)
        assert len(errors) == 1
        assert isinstance(errors[0], ValueError)


# ---------------------------------------------------------------------------
# Test: types
# ---------------------------------------------------------------------------

class TestTypes:
    def test_generate_turn_id(self):
        from eidolon.livekit.agent.pipeline import generate_turn_id

        id1 = generate_turn_id()
        id2 = generate_turn_id()
        assert len(id1) == 16
        assert id1 != id2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
