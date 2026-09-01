"""AgentSession turn_handling config for the full-duplex StreamingPipeline."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from eidolon_sdk.biz.contracts import INTERACTION_MODE_HALF_DUPLEX

from eidolon.livekit.agent.full_duplex import StreamingPipeline
from eidolon.livekit.agent.full_duplex.agent_builder import build_full_duplex_agent
from eidolon.livekit.agent.full_duplex.lifecycle import FullDuplexSessionLifecycle
from eidolon.livekit.agent.factory import SharedStageFactory
from eidolon.livekit.common.config import EotPolicyConfig, TurnPolicyConfig


def _pipe(*, allow_interruptions: bool) -> StreamingPipeline:
    p = StreamingPipeline.__new__(StreamingPipeline)
    p._allow_interruptions = allow_interruptions
    p._false_interruption_timeout = 6.0
    p._turn_policy = TurnPolicyConfig()
    p._avatar_enabled = False
    return p


def test_channel_owned_turn_handling_disables_framework_auto_interrupt() -> None:
    th = _pipe(allow_interruptions=True)._build_turn_handling()
    intr = th["interruption"]
    assert intr["enabled"] is False
    # Channel policy still needs overlapping speech to reach STT.
    assert intr["discard_audio_if_uninterruptible"] is False
    assert intr["false_interruption_timeout"] == 6.0


def test_preemptive_passthrough() -> None:
    p = _pipe(allow_interruptions=True)
    th = p._build_turn_handling()
    assert th["preemptive_generation"]["enabled"] == p._turn_policy.preemptive.enabled
    assert (
        th["preemptive_generation"]["preemptive_tts"]
        == p._turn_policy.preemptive.preemptive_tts
    )


def test_livekit_17_agent_uses_structured_turn_handling_and_public_stt_node() -> None:
    pipeline = SimpleNamespace(
        _instructions="test",
        _factory=SimpleNamespace(
            stt=SimpleNamespace(stt=None),
            llm=SimpleNamespace(llm=None),
            tts=SimpleNamespace(tts=None),
            vad=None,
        ),
        _turn_detection=lambda: "vad",
    )

    agent = build_full_duplex_agent(pipeline)

    assert agent.turn_detection == "vad"
    assert inspect.isasyncgenfunction(agent.stt_node)


@pytest.mark.asyncio
async def test_session_uses_livekit_transcription_timeout_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from eidolon.livekit.agent.full_duplex import lifecycle as lifecycle_module
    from livekit.agents import voice as livekit_voice

    captured: dict[str, object] = {}

    class SessionStarted(Exception):
        pass

    class FakeAgentSession:
        def __init__(self, **kwargs):
            captured["options"] = kwargs
            self.output = SimpleNamespace(audio=None)

        def on(self, event_name, callback):
            captured.setdefault("events", {})[event_name] = callback

        async def start(self, **kwargs):
            raise SessionStarted

    async def runtime_identity(_room):
        return "owner-1"

    monkeypatch.setattr(livekit_voice, "AgentSession", FakeAgentSession)
    monkeypatch.setattr(
        lifecycle_module,
        "wait_for_runtime_participant_identity",
        runtime_identity,
    )

    pipeline = SimpleNamespace(
        _room=None,
        _started=False,
        _runtime_participant_identity="",
        _ensure_turn_event_sink=lambda: SimpleNamespace(start=AsyncMock()),
        _build_agent=lambda: object(),
        _build_turn_handling=lambda: {"interruption": {"enabled": False}},
        _aec_warmup_duration=0.4,
        _stt_commit_transcript_timeout=1.5,
        _on_user_state_changed=MagicMock(),
        _on_agent_state_changed=MagicMock(),
        _on_user_transcribed=MagicMock(),
        _on_user_transcription_timeout=MagicMock(),
        _on_session_error=MagicMock(),
        _session_signals=SimpleNamespace(register_vad_inference_callback=MagicMock()),
        _warmup_stages=AsyncMock(),
        _filler=None,
        _ensure_room_data_bridge=lambda: SimpleNamespace(install=MagicMock()),
        _voiceprint_turns=SimpleNamespace(install=MagicMock()),
        _audio_sample_rate=16000,
        _avatar_enabled=False,
    )
    room = SimpleNamespace(name="room-1", on=MagicMock())

    with pytest.raises(SessionStarted):
        await FullDuplexSessionLifecycle(pipeline).run(room)

    assert captured["options"] == {
        "turn_handling": {"interruption": {"enabled": False}},
        "aec_warmup_duration": 0.4,
        "transcription_timeout": 1.5,
    }
    assert captured["events"]["user_transcription_timeout"] is (
        pipeline._on_user_transcription_timeout
    )


@pytest.mark.asyncio
async def test_public_tts_node_taps_text_without_reimplementing_synthesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.agents.voice import Agent

    forwarded: list[str] = []
    recorded: list[tuple[int, str]] = []

    async def fake_default_tts_node(self, text, model_settings):
        async for chunk in text:
            forwarded.append(chunk)
        yield "audio-frame"

    monkeypatch.setattr(Agent, "tts_node", fake_default_tts_node)
    pipeline = SimpleNamespace(
        _instructions="test",
        _factory=SimpleNamespace(
            stt=SimpleNamespace(stt=None),
            llm=SimpleNamespace(llm=None),
            tts=SimpleNamespace(tts=None),
            vad=None,
        ),
        _turn_detection=lambda: "vad",
        _begin_streamed_assistant_speech=lambda: 17,
        _append_streamed_assistant_speech=lambda stream_id, text: recorded.append(
            (stream_id, text)
        ),
        _abort_streamed_assistant_speech=lambda stream_id: None,
    )
    agent = build_full_duplex_agent(pipeline)

    async def text_source():
        yield "第一段"
        yield "第二段"

    frames = [frame async for frame in agent.tts_node(text_source(), None)]

    assert frames == ["audio-frame"]
    assert forwarded == ["第一段", "第二段"]
    assert recorded == [(17, "第一段"), (17, "第二段")]


def test_turn_policy_speech_merge_grace_reaches_coordinator() -> None:
    p = _pipe(allow_interruptions=True)
    p._turn_policy = TurnPolicyConfig(
        eot=EotPolicyConfig(speech_merge_grace_ms=650),
    )
    coordinator = p._build_user_turn_coordinator()
    coordinator.start_speech(timeline=None, now=0.0)
    coordinator.note_speech_stopped(eot_score=0.0, now=0.1)

    assert coordinator.can_merge_new_speech(now=0.75) is True
    assert coordinator.can_merge_new_speech(now=0.751) is False


def test_streaming_pipeline_accepts_half_duplex_rejects_ptt() -> None:
    from eidolon_sdk.biz.contracts import INTERACTION_MODE_PTT

    factory = SharedStageFactory.__new__(SharedStageFactory)
    # ptt is served by the button-driven segment pipeline, not StreamingPipeline.
    with pytest.raises(ValueError, match="HalfDuplexPttPipeline"):
        StreamingPipeline(factory, interaction_mode=INTERACTION_MODE_PTT)
    # half_duplex IS served by StreamingPipeline (streaming EOT commit, no
    # barge-in). The mode guard must not reject it; deeper __init__ may fail on
    # the bare test factory, but never with the guard's ValueError.
    try:
        StreamingPipeline(factory, interaction_mode=INTERACTION_MODE_HALF_DUPLEX)
    except ValueError as exc:
        assert "HalfDuplexPttPipeline" not in str(exc)
    except Exception:
        pass
