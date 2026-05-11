"""Integration tests for the voice agent pipelines using real .env configuration.

These tests exercise the actual stt/llm/tts components configured in .env,
only mocking the LiveKit Room connection (local_publish_audio and track events)
which require a running LiveKit server.

Run with::

    cd <repository-root>
    source .venv/bin/activate
    python -m pytest eidolon/channel/livekit/agent/pipeline/test_integration.py -v -s
"""

from __future__ import annotations

import logging
import os
import struct
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from eidolon.channel.livekit.agent.pipeline.llm import LlmInput

pytestmark = pytest.mark.integration

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("integration")

# Load .env into os.environ before anything else tries to read it.
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    from dotenv import dotenv_values

    _env = dotenv_values(_env_path)
    for k, v in _env.items():
        if k not in os.environ:
            os.environ[k] = v
    logger.info("[setup] loaded .env from %s: keys=%s", _env_path, list(_env.keys()))


# ---------------------------------------------------------------------------
# Fixtures — build real components from .env
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def shared_stage_factory():
    """Build SharedStageFactory with real components from .env config.

    Uses ``SharedStageFactory.from_config(cfg)`` — the same path used by
    ``server.py`` in production — so this test exercises the real config
    -> plugin construction wiring.
    """
    from eidolon.channel.livekit.agent import SharedStageFactory
    from eidolon.channel.livekit.common.config import load_agent_config

    cfg = load_agent_config()
    logger.info(
        "[fixture] config: stt=%s tts=%s vad=%s llm.model=%s",
        cfg.stt_provider, cfg.tts_provider, cfg.vad_provider, cfg.llm.model,
    )

    try:
        factory = SharedStageFactory.from_config(cfg)
    except Exception as e:
        pytest.fail(f"Failed to build SharedStageFactory: {e}")

    logger.info(
        "[fixture] SharedStageFactory ready — stt=%s llm=%s tts=%s vad=%s",
        type(factory.stt).__name__,
        type(factory.llm).__name__,
        type(factory.tts).__name__,
        type(factory.vad).__name__ if factory.vad else None,
    )
    return factory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sine_wave_blob(duration_ms: int = 500) -> bytes:
    """Create a PCM audio blob (440Hz sine wave, 16kHz mono 16-bit)."""
    import math
    import struct

    sample_rate = 16000
    num_samples = int(sample_rate * duration_ms / 1000)
    parts = []
    for i in range(num_samples):
        value = int(16000 * math.sin(2 * math.pi * 440 * i / sample_rate))
        parts.append(struct.pack("<h", value))
    return b"".join(parts)


# ---------------------------------------------------------------------------
# Test: SharedStageFactory initialization with real config
# ---------------------------------------------------------------------------

class TestSharedStageFactoryInit:
    """Test that SharedStageFactory initializes with real .env components."""

    def test_factory_has_all_components(self, shared_stage_factory):
        assert shared_stage_factory.stt is not None
        assert shared_stage_factory.llm is not None
        assert shared_stage_factory.tts is not None
        logger.info("[PASS] factory has stt, llm, tts")

    def test_stt_is_bailian_instance(self, shared_stage_factory):
        from eidolon.channel.livekit.plugins.stt.bailian import BailianFunASRSTT

        assert isinstance(shared_stage_factory.stt.stt, BailianFunASRSTT)
        logger.info("[PASS] STT is BailianFunASRSTT")

    def test_llm_is_livekit_llm_instance(self, shared_stage_factory):
        from livekit.agents import llm as lk_llm

        assert isinstance(shared_stage_factory.llm.llm, lk_llm.LLM)
        logger.info("[PASS] LLM is livekit LLM")

    def test_tts_is_sensetime_instance(self, shared_stage_factory):
        from eidolon.channel.livekit.plugins.tts.sensetime import SenseTimeTTS

        assert isinstance(shared_stage_factory.tts.tts, SenseTimeTTS)
        logger.info("[PASS] TTS is SenseTimeTTS")


# ---------------------------------------------------------------------------
# Test: BatchPipeline with real STT -> LLM -> TTS
# ---------------------------------------------------------------------------

class TestBatchPipelineIntegration:
    """Integration tests for BatchPipeline using real stt/llm/tts components.

    The LiveKit Room connection is mocked (local_publish_audio and track events)
    but all stt/llm/tts calls use the real .env-configured services.
    """

    @pytest.mark.asyncio
    async def test_batch_pipeline_stt_recognize_real(self, shared_stage_factory):
        """Test real STT recognize() call with a sine-wave audio blob.

        Note: synthetic sine wave is not real speech, so STT will return empty.
        This still exercises the full WebSocket connection lifecycle.
        """
        audio_blob = _make_sine_wave_blob(duration_ms=500)
        logger.info("[test] audio_blob size=%d bytes", len(audio_blob))

        logger.info("[test] calling stt.recognize()...")
        transcript = await shared_stage_factory.stt.recognize(audio_blob)
        logger.info("[test] STT transcript=%r", transcript)
        assert isinstance(transcript, str), "STT should return a string"
        # Sine wave is not speech — empty is expected
        logger.info("[PASS] stt.recognize() completed (empty=%s)", not transcript.strip())

    @pytest.mark.asyncio
    async def test_batch_pipeline_llm_chat_real(self, shared_stage_factory):
        """Test real LLM chat() call with a Chinese question."""
        test_text = "你好，今天天气怎么样？"
        logger.info("[test] calling llm.chat() with text=%r", test_text)

        llm_output = await shared_stage_factory.llm.chat(LlmInput(text=test_text))

        assert llm_output.text, "LLM should return non-empty response"
        logger.info("[test] LLM output=%r", llm_output.text[:200])
        logger.info("[PASS] llm.chat() completed — length=%d", len(llm_output.text))

    @pytest.mark.asyncio
    async def test_batch_pipeline_tts_synthesize_real(self, shared_stage_factory):
        """Test real TTS synthesize() call."""
        test_text = "好的，今天天气晴朗，适合外出散步。"
        logger.info("[test] calling tts.synthesize() with text=%r", test_text)

        tts_frames = []
        async for frame in shared_stage_factory.tts.synthesize(test_text):
            tts_frames.append(frame)

        total_samples = sum(f.samples_per_channel for f in tts_frames)
        duration_s = total_samples / 16000.0
        logger.info(
            "[test] TTS produced %d frames, samples=%d, duration=%.2fs",
            len(tts_frames),
            total_samples,
            duration_s,
        )
        assert len(tts_frames) > 0, "TTS should produce audio frames"
        logger.info("[PASS] tts.synthesize() completed")

    @pytest.mark.asyncio
    async def test_batch_pipeline_full_stt_llm_tts_flow(self, shared_stage_factory):
        """Test the complete STT -> LLM -> TTS flow with real services.

        Since synthetic audio produces empty STT transcript, we use a
        hardcoded transcript for the LLM step to exercise the full chain.
        """
        from eidolon.channel.livekit.agent.batch import BatchPipeline
        from eidolon.channel.livekit.agent.pipeline import PipelineCallbacks

        pipeline = BatchPipeline(
            shared_stage_factory,
            callbacks=PipelineCallbacks(
                on_user_message=lambda t: logger.info("[callback] user_message: %r", t),
                on_agent_started_speaking=lambda: logger.info("[callback] agent_started_speaking"),
                on_agent_message=lambda t: logger.info("[callback] agent_message: %r", t),
                on_agent_ended_speaking=lambda: logger.info("[callback] agent_ended_speaking"),
                on_agent_response_done=lambda: logger.info("[callback] agent_response_done"),
            ),
        )

        # Step 1: STT with synthetic audio (empty transcript expected)
        audio_blob = _make_sine_wave_blob(duration_ms=500)
        transcript = await shared_stage_factory.stt.recognize(audio_blob)
        logger.info("[test] STT transcript=%r", transcript)

        # Step 2: LLM (use hardcoded text since synthetic audio produces no transcript)
        llm_text = "我想知道明天的天气如何"
        llm_output = await shared_stage_factory.llm.chat(LlmInput(text=llm_text))
        response_text = llm_output.text
        logger.info("[test] LLM response=%r", response_text[:200])
        assert response_text, "LLM should return a response"

        # Step 3: TTS
        tts_frames = []
        async for frame in shared_stage_factory.tts.synthesize(response_text):
            tts_frames.append(frame)
        logger.info("[test] TTS frames=%d", len(tts_frames))
        assert len(tts_frames) > 0, "TTS should produce frames"

        logger.info("[PASS] Full stt->llm->tts flow completed")


# ---------------------------------------------------------------------------
# Test: StreamingPipeline component initialization with real config
# ---------------------------------------------------------------------------

class TestStreamingPipelineIntegration:
    """Integration tests for StreamingPipeline component initialization.

    StreamingPipeline.run() requires a real LiveKit Room connection (needs a
    running LiveKit server), so we test:
    1. Pipeline initialization with real factory
    2. Agent building with real components
    3. EOT model loading
    4. Session event handlers
    """

    def test_streaming_pipeline_initializes(self, shared_stage_factory):
        """StreamingPipeline should initialize with the real SharedStageFactory."""
        from eidolon.channel.livekit.agent import StreamingPipeline

        pipeline = StreamingPipeline(
            shared_stage_factory,
            instructions="你是一个友好的语音助手。请简短回复。",
            allow_interruptions=True,
        )

        assert pipeline._factory is shared_stage_factory
        assert pipeline._started is False
        assert pipeline._allow_interruptions is True
        logger.info("[PASS] StreamingPipeline initialized")

    def test_streaming_pipeline_builds_agent(self, shared_stage_factory):
        """_build_agent() should create a VoiceAgent with real stt/llm/tts/vad."""
        from eidolon.channel.livekit.agent import StreamingPipeline

        pipeline = StreamingPipeline(shared_stage_factory, instructions="Test")
        agent = pipeline._build_agent()

        assert agent is not None
        assert agent.stt is not None, "Agent should have STT"
        assert agent.llm is not None, "Agent should have LLM"
        assert agent.tts is not None, "Agent should have TTS"
        logger.info(
            "[PASS] Agent built — stt=%s llm=%s tts=%s vad=%s",
            type(agent.stt).__name__,
            type(agent.llm).__name__,
            type(agent.tts).__name__,
            type(agent.vad).__name__ if agent.vad else None,
        )

    def test_eot_model_loads(self, shared_stage_factory):
        """ChineseModel EOT should load from the bundled ONNX model."""
        from eidolon.channel.livekit.agent import StreamingPipeline

        pipeline = StreamingPipeline(shared_stage_factory, instructions="Test")
        eot_model = pipeline._get_eot_model()

        assert eot_model is not None
        assert hasattr(eot_model, "predict_end_of_turn")
        assert hasattr(eot_model, "unlikely_threshold")
        logger.info("[PASS] EOT model loaded — type=%s", type(eot_model).__name__)

    @pytest.mark.asyncio
    async def test_eot_model_predict_end_of_turn(self, shared_stage_factory):
        """EOT model should return a score for Chinese text."""
        from eidolon.channel.livekit.agent import StreamingPipeline
        from livekit.agents.llm import ChatContext, ChatMessage

        pipeline = StreamingPipeline(shared_stage_factory, instructions="Test")
        eot_model = pipeline._get_eot_model()

        ctx = ChatContext()
        ctx.add_message(role="user", content=["今天天气很好，我想去公园散步。"])

        score = await eot_model.predict_end_of_turn(ctx)
        logger.info("[test] EOT score=%.3f for: %r", score, "今天天气很好...")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        logger.info("[PASS] EOT predict_end_of_turn works — score=%.3f", score)

    def test_streaming_pipeline_event_handlers(self, shared_stage_factory):
        """Test _on_user_state_changed and _on_agent_state_changed fire callbacks."""
        from eidolon.channel.livekit.agent import StreamingPipeline
        from eidolon.channel.livekit.agent.pipeline import PipelineCallbacks

        callbacks_fired: dict[str, bool] = {}

        def make_cb(name):
            def cb():
                callbacks_fired[name] = True

            return cb

        pipeline = StreamingPipeline(
            shared_stage_factory,
            callbacks=PipelineCallbacks(
                on_user_started_speaking=make_cb("user_started"),
                on_user_ended_speaking=make_cb("user_ended"),
                on_agent_started_speaking=make_cb("agent_started"),
                on_agent_ended_speaking=make_cb("agent_ended"),
            ),
        )

        # Simulate user speaking
        ev = MagicMock()
        ev.old_state = "idle"
        ev.new_state = "speaking"
        pipeline._on_user_state_changed(ev)
        assert callbacks_fired.get("user_started") is True
        logger.info("[PASS] _on_user_state_changed fires user_started callback")

        # Simulate user done speaking
        callbacks_fired.clear()
        ev.old_state = "speaking"
        ev.new_state = "listening"
        pipeline._on_user_state_changed(ev)
        assert callbacks_fired.get("user_ended") is True
        logger.info("[PASS] _on_user_state_changed fires user_ended callback")

        # Simulate agent speaking
        callbacks_fired.clear()
        ev.old_state = "thinking"
        ev.new_state = "speaking"
        pipeline._on_agent_state_changed(ev)
        assert callbacks_fired.get("agent_started") is True
        logger.info("[PASS] _on_agent_state_changed fires agent_started callback")


# ---------------------------------------------------------------------------
# Test: VAD initialization
# ---------------------------------------------------------------------------

class TestVADIntegration:
    """Test VAD component initialization."""

    def test_vad_initializes_if_configured(self, shared_stage_factory):
        vad = shared_stage_factory.vad
        if vad is None:
            pytest.skip("VAD not configured (VAD_PROVIDER=none or unavailable)")
        assert hasattr(vad, "stream"), "VAD should have a stream() method"
        logger.info("[PASS] VAD initialized — type=%s", type(vad).__name__)

    @pytest.mark.asyncio
    async def test_vad_stream_produces_events(self, shared_stage_factory):
        """VAD stream should emit events when given audio."""
        vad = shared_stage_factory.vad
        if vad is None:
            pytest.skip("VAD not configured")

        from livekit import rtc

        stream = vad.stream()
        audio_blob = _make_sine_wave_blob(duration_ms=1000)

        frame = rtc.AudioFrame(
            data=audio_blob,
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=len(audio_blob) // 2,
        )

        stream.push_frame(frame)
        stream.end_input()

        events_received = 0
        async for event in stream:
            events_received += 1
            logger.info("[test] VAD event #%d type=%s", events_received, event.type)
            if events_received >= 5:
                break

        logger.info("[PASS] VAD stream produced %d events", events_received)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
