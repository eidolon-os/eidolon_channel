"""Unit tests for the voice agent pipeline.

Run with::

    cd <repository-root>
    .venv/bin/python -m pytest eidolon/livekit/agent/pipeline/test_pipeline.py -v
"""

from __future__ import annotations

import asyncio

import pytest
from unittest.mock import MagicMock

from eidolon.livekit.agent.shared.pipeline import BasePipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pcm_frame(duration_ms: int = 100, sample_rate: int = 16000) -> bytes:
    """Create a synthetic PCM audio frame (silence)."""
    num_samples = int(sample_rate * duration_ms / 1000)
    return b"\x00\x00" * num_samples


class _TestPipeline(BasePipeline):
    async def run(self, room) -> None:
        self._room = room
        self._started = True


class TestSttStage:
    @pytest.mark.asyncio
    async def test_recognize_streaming_returns_on_end_of_speech(self):
        from eidolon.livekit.agent.providers.stt import SttStage
        from eidolon.livekit.tests._harness.mocks import (
            MockSTT,
            ScriptedTranscript,
        )

        pcm = make_pcm_frame(duration_ms=300)
        stt = MockSTT.scripted(
            [ScriptedTranscript(text="你好", after_pcm_bytes=len(pcm))]
        )
        stage = SttStage(stt)

        transcript = await asyncio.wait_for(
            stage.recognize_streaming(pcm),
            timeout=1.0,
        )

        assert transcript == "你好"
        assert stt.bytes_pushed == len(pcm)


# ---------------------------------------------------------------------------
# Test: BasePipeline shared behavior
# ---------------------------------------------------------------------------

class TestBasePipeline:
    def test_initial_state(self):
        from eidolon.livekit.agent.shared import PipelineCallbacks, PipelineState

        mock_factory = MagicMock()
        pipeline = _TestPipeline(mock_factory, callbacks=PipelineCallbacks())

        assert pipeline.state == PipelineState.IDLE
        assert pipeline.factory is mock_factory
        assert pipeline._started is False

    @pytest.mark.asyncio
    async def test_shutdown_resets_state(self):
        from eidolon.livekit.agent.shared import PipelineCallbacks, PipelineState

        mock_factory = MagicMock()
        pipeline = _TestPipeline(mock_factory, callbacks=PipelineCallbacks())

        pipeline._state = PipelineState.SPEAKING
        pipeline._started = True

        await pipeline.shutdown()

        assert pipeline.state == PipelineState.IDLE
        assert pipeline._started is False

    def test_on_agent_state_changed_callbacks(self):
        """Test that _on_agent_state_changed fires correct callbacks."""
        from eidolon.livekit.agent.shared import PipelineCallbacks, PipelineState

        callbacks_fired: list[str] = []

        def make_cb(name):
            def cb(*args):
                callbacks_fired.append(name)
            return cb

        mock_factory = MagicMock()
        pipeline = _TestPipeline(
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
        from eidolon.livekit.agent.shared import PipelineCallbacks

        callbacks_fired: list[str] = []

        def make_cb(name):
            def cb(*args):
                callbacks_fired.append(name)
            return cb

        mock_factory = MagicMock()
        pipeline = _TestPipeline(
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
        from eidolon.livekit.agent.shared import PipelineCallbacks

        errors: list = []

        mock_factory = MagicMock()
        pipeline = _TestPipeline(
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
        from eidolon.livekit.agent.shared import generate_turn_id

        id1 = generate_turn_id()
        id2 = generate_turn_id()
        assert len(id1) == 16
        assert id1 != id2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
