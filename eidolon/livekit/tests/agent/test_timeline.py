"""Turn timeline tests."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest

from eidolon_sdk.biz.contracts import CLIENT_AUDIO_STATE_TOPIC

from eidolon.livekit.agent.integration.client_audio_state import parse_client_audio_state
from eidolon.livekit.agent.full_duplex.interruption_effects import (
    FullDuplexInterruptionEffects,
)
from eidolon.livekit.agent.output.ducking import OutputDuckingController
from eidolon.livekit.agent.shared.types import PipelineState
from eidolon.livekit.agent.turn_policy import TurnPolicyRuntime
from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.common.config import ObservabilityConfig, TurnPolicyConfig


def test_timeline_marks_and_durations() -> None:
    timeline = TurnTimeline("turn-1")
    timeline.mark("speech_started_at")
    timeline.mark("interrupt_resolved_at")
    timeline.set_attr("decision", "cancel")
    snap = timeline.snapshot()
    assert snap["turn_id"] == "turn-1"
    assert snap["attrs"]["decision"] == "cancel"
    assert snap["durations_ms"]["vad_start_to_interrupt_resolved"] is not None


def test_timeline_records_normalized_decision_attrs() -> None:
    timeline = TurnTimeline("turn-2")

    timeline.record_decision(
        action="rollback",
        reason="intent:backchannel",
        rollback_drop_buffered=True,
        intent="backchannel",
        intent_source="lexicon",
        intent_confidence=0.95,
        tier="tier3_backchannel_noise",
        tier_reason="intent:backchannel",
        source="turn_policy",
        resolved_reason="timeout",
        eot_score=0.1,
        transcript_preview="嗯",
        vad_active=False,
    )

    snap = timeline.snapshot()
    assert snap["attrs"]["interrupt_action"] == "rollback"
    assert snap["attrs"]["decision_reason"] == "intent:backchannel"
    assert snap["attrs"]["rollback_drop_buffered"] is True
    assert snap["attrs"]["decision"]["intent"] == "backchannel"
    assert snap["attrs"]["decision"]["tier"] == "tier3_backchannel_noise"


def test_timeline_provider_latency_snapshot() -> None:
    timeline = TurnTimeline("turn-3")
    for mark in (
        "speech_started_at",
        "transcript_final_at",
        "speech_stopped_at",
        "turn_committed_at",
        "llm_started_at",
        "tts_first_audio_at",
    ):
        timeline.mark(mark)

    snap = timeline.snapshot()
    provider_latency = snap["attrs"]["provider_latency_ms"]
    provider_segments = snap["attrs"]["provider_segments"]

    assert "stt_final_ms" in provider_latency
    assert "commit_to_llm_started_ms" in provider_latency
    assert "commit_to_tts_first_audio_ms" in provider_latency
    assert snap["durations_ms"]["commit_to_tts_first_audio"] is not None
    assert {segment["name"] for segment in provider_segments} >= {
        "stt_final",
        "brain_start",
        "commit_to_first_audio",
    }
    assert all("stage" in segment for segment in provider_segments)


def test_timeline_interrupt_latency_breakdown() -> None:
    timeline = TurnTimeline("turn-interrupt")
    timeline.mark_at("speech_started_at", 10.0)
    timeline.mark_at("interrupt_started_at", 10.02)
    timeline.mark_at("transcript_interim_first_at", 10.34)
    timeline.mark_at("transcript_actionable_first_at", 10.37)
    timeline.record_decision(
        action="cancel",
        reason="intent:hard_stop",
        rollback_drop_buffered=False,
        intent="hard_stop",
    )
    timeline.mark_at("interrupt_intent_admitted_at", 10.39)
    timeline.mark_at("interrupt_resolved_at", 10.45)

    snap = timeline.snapshot()
    provider_latency = snap["attrs"]["provider_latency_ms"]
    durations = snap["durations_ms"]

    assert round(provider_latency["interrupt_speech_to_started_ms"]) == 20
    assert round(provider_latency["interrupt_speech_to_first_transcript_ms"]) == 340
    assert round(provider_latency["interrupt_started_to_first_transcript_ms"]) == 320
    assert round(provider_latency["stt_speech_to_actionable_transcript_ms"]) == 370
    assert round(
        provider_latency["stt_first_transcript_to_actionable_transcript_ms"]
    ) == 30
    assert round(provider_latency["interrupt_first_transcript_to_intent_admitted_ms"]) == 50
    assert round(provider_latency["interrupt_actionable_transcript_to_resolved_ms"]) == 80
    assert round(provider_latency["interrupt_intent_admitted_to_resolved_ms"]) == 60
    assert round(provider_latency["interrupt_speech_to_resolved_ms"]) == 450
    assert round(durations["stt_speech_to_actionable_transcript"]) == 370
    assert round(durations["interrupt_actionable_transcript_to_resolved"]) == 80
    assert round(durations["interrupt_first_transcript_to_resolved"]) == 110


def test_timeline_keeps_cancel_resolution_distinct_from_rollback() -> None:
    timeline = TurnTimeline("turn-rollback-then-cancel")
    timeline.mark_at("speech_started_at", 10.0)
    timeline.mark_at("interrupt_started_at", 10.02)
    timeline.mark_interrupt_resolved("rollback", timestamp=10.45)
    timeline.mark_at("transcript_actionable_first_at", 11.2)
    timeline.mark_interrupt_resolved("cancel", timestamp=11.5)

    snap = timeline.snapshot()
    provider_latency = snap["attrs"]["provider_latency_ms"]
    durations = snap["durations_ms"]

    assert snap["timestamps"]["interrupt_resolved_at"] == 10.45
    assert snap["timestamps"]["interrupt_rollback_resolved_at"] == 10.45
    assert snap["timestamps"]["interrupt_cancel_resolved_at"] == 11.5
    assert round(provider_latency["interrupt_speech_to_resolved_ms"]) == 450
    assert round(provider_latency["interrupt_speech_to_rollback_resolved_ms"]) == 450
    assert round(provider_latency["interrupt_speech_to_cancel_resolved_ms"]) == 1500
    assert round(
        provider_latency["interrupt_actionable_transcript_to_cancel_resolved_ms"]
    ) == 300
    assert round(durations["vad_start_to_interrupt_rollback_resolved"]) == 450
    assert round(durations["vad_start_to_interrupt_cancel_resolved"]) == 1500


def test_timeline_does_not_mark_noise_as_actionable_transcript() -> None:
    timeline = TurnTimeline("turn-noise")
    timeline.mark_at("speech_started_at", 10.0)
    timeline.mark_at("transcript_interim_first_at", 10.2)

    timeline.record_decision(
        action="rollback",
        reason="intent:backchannel",
        rollback_drop_buffered=False,
        intent="backchannel",
    )

    snap = timeline.snapshot()
    assert "transcript_actionable_first_at" not in snap["timestamps"]
    assert (
        snap["attrs"]["provider_latency_ms"][
            "stt_speech_to_actionable_transcript_ms"
        ]
        is None
    )


def test_timeline_marks_semantic_score_wait_as_actionable_transcript() -> None:
    timeline = TurnTimeline("turn-actionable-hold")
    timeline.mark_at("speech_started_at", 10.0)
    timeline.mark_at("transcript_interim_first_at", 10.1)

    timeline.record_decision(
        action="hold",
        reason="semantic_score_wait score=0.35 evidence=enough_cjk_interim",
        rollback_drop_buffered=False,
        intent="uncertain",
    )

    snap = timeline.snapshot()
    assert "transcript_actionable_first_at" in snap["timestamps"]
    assert (
        snap["attrs"]["provider_latency_ms"][
            "stt_speech_to_actionable_transcript_ms"
        ]
        is not None
    )


def test_timeline_records_hold_recheck_ms() -> None:
    timeline = TurnTimeline("turn-stable-hold")

    timeline.record_decision(
        action="hold",
        reason="stable_signal_wait intent=topic_switch age_ms=80 window_ms=120",
        rollback_drop_buffered=False,
        intent="uncertain",
        topic_switch_hint=True,
        hold_recheck_ms=40,
    )

    assert timeline.snapshot()["attrs"]["decision"]["hold_recheck_ms"] == 40


def test_pipeline_records_semantic_interrupt_gate_events() -> None:
    from eidolon.livekit.agent.full_duplex import StreamingPipeline

    pipeline = StreamingPipeline.__new__(StreamingPipeline)
    pipeline._timeline = TurnTimeline("turn-semantic-gate")

    for index in range(18):
        pipeline._record_transcript_admission_event(
            {
                "accepted": False,
                "reason": "agent_echo",
                "transcript_preview": f"echo-{index}",
            }
        )

    for index in range(18):
        pipeline._record_semantic_gate_event(
            {
                "stage": "initial",
                "action": "inactive",
                "reason": "no_interrupt_window",
                "transcript_preview": f"early-{index}",
            }
        )

    admissions = pipeline._timeline.attrs["transcript_admission_events"]
    assert len(admissions) == 16
    assert admissions[0]["transcript_preview"] == "echo-2"
    assert pipeline._timeline.attrs["transcript_admission_last_event"] == {
        "accepted": False,
        "reason": "agent_echo",
        "transcript_preview": "echo-17",
    }

    events = pipeline._timeline.attrs["semantic_interrupt_gate_events"]
    assert len(events) == 16
    assert events[0]["transcript_preview"] == "early-2"
    assert pipeline._timeline.attrs["semantic_interrupt_gate_last_event"] == {
        "stage": "initial",
        "action": "inactive",
        "reason": "no_interrupt_window",
        "transcript_preview": "early-17",
    }


def test_timeline_mark_after_sets_synthetic_llm_first_delta() -> None:
    timeline = TurnTimeline("turn-4")
    timeline.mark("turn_committed_at")
    timeline.mark("llm_started_at")
    timeline.mark_after("llm_first_delta_at", "llm_started_at", 0.25)

    snap = timeline.snapshot()
    assert snap["attrs"]["provider_latency_ms"]["commit_to_llm_first_delta_ms"] is not None
    assert snap["durations_ms"]["commit_to_llm_first_delta"] is not None


def test_streaming_pipeline_records_llm_metrics_into_timeline() -> None:
    from eidolon.livekit.agent.full_duplex import StreamingPipeline

    class _FakeLlm:
        def __init__(self) -> None:
            self.handlers = {}

        def on(self, name: str, handler) -> None:
            self.handlers[name] = handler

    fake_llm = _FakeLlm()
    pipeline = StreamingPipeline.__new__(StreamingPipeline)
    pipeline._factory = SimpleNamespace(llm=SimpleNamespace(llm=fake_llm))
    pipeline._timeline = TurnTimeline("turn-llm")
    pipeline._timeline.mark("turn_committed_at")
    pipeline._timeline.mark("llm_started_at")

    pipeline._ensure_provider_event_observer()
    pipeline._provider_events.install_llm_metrics_observer()
    fake_llm.handlers["metrics_collected"](
        SimpleNamespace(
            request_id="req-1",
            ttft=0.12,
            duration=0.4,
            completion_tokens=3,
            prompt_tokens=7,
            total_tokens=10,
            cancelled=False,
        )
    )

    snap = pipeline._timeline.snapshot()
    assert snap["timestamps"]["llm_first_delta_at"] >= snap["timestamps"]["llm_started_at"]
    assert snap["attrs"]["llm_metrics"]["ttft_ms"] == 120.0
    assert snap["attrs"]["provider_latency_ms"]["commit_to_llm_first_delta_ms"] is not None


def test_streaming_pipeline_records_brain_provider_events_into_timeline() -> None:
    from eidolon.livekit.agent.full_duplex import StreamingPipeline

    class _FakeLlm:
        def __init__(self) -> None:
            self.handlers = {}

        def on(self, name: str, handler) -> None:
            self.handlers[name] = handler

    fake_llm = _FakeLlm()
    pipeline = StreamingPipeline.__new__(StreamingPipeline)
    pipeline._factory = SimpleNamespace(llm=SimpleNamespace(llm=fake_llm))
    pipeline._timeline = TurnTimeline("turn-brain")
    pipeline._timeline.mark("turn_committed_at")

    pipeline._ensure_provider_event_observer()
    pipeline._provider_events.install_brain_provider_event_observer()
    fake_llm.handlers["provider_event"](
        {
            "provider": "eidolon_agent_rpc",
            "event": "brain_request_started",
            "timestamp": 100.0,
            "conversation_id": "livekit:test",
        }
    )
    fake_llm.handlers["provider_event"](
        {
            "provider": "eidolon_agent_rpc",
            "event": "brain_request_sent",
            "timestamp": 100.2,
            "turn_id": "t",
            "request_id": "eidolon-t",
        }
    )
    fake_llm.handlers["provider_event"](
        {
            "provider": "eidolon_agent_rpc",
            "event": "brain_first_delta",
            "timestamp": 100.5,
            "turn_id": "t",
            "request_id": "eidolon-t",
        }
    )

    snap = pipeline._timeline.snapshot()
    assert snap["timestamps"]["brain_request_started_at"] == 100.0
    assert snap["timestamps"]["brain_request_sent_at"] == 100.2
    assert snap["timestamps"]["brain_first_delta_at"] == 100.5
    assert snap["timestamps"]["llm_first_delta_at"] == 100.5
    assert snap["attrs"]["brain_rpc"]["request_id"] == "eidolon-t"
    assert round(
        snap["attrs"]["provider_latency_ms"]["brain_request_to_first_delta_ms"]
    ) == 300


def test_streaming_pipeline_records_stt_provider_events_into_timeline() -> None:
    from eidolon.livekit.agent.full_duplex import StreamingPipeline

    class _FakeStt:
        def __init__(self) -> None:
            self.handlers = {}

        def on(self, name: str, handler) -> None:
            self.handlers[name] = handler

    fake_stt = _FakeStt()
    pipeline = StreamingPipeline.__new__(StreamingPipeline)
    pipeline._factory = SimpleNamespace(stt=SimpleNamespace(stt=fake_stt))
    pipeline._timeline = TurnTimeline("turn-stt")
    pipeline._timeline.mark_at("speech_started_at", 10.0)

    pipeline._ensure_provider_event_observer()
    pipeline._provider_events.install_stt_provider_event_observer()
    fake_stt.handlers["provider_event"](
        {
            "provider": "bailian",
            "model": "fun-asr",
            "event": "stt_turn_first_audio_sent",
            "timestamp": 10.1,
            "turn_id": "turn-stt",
            "stream_id": "s",
            "language": "zh",
        }
    )
    fake_stt.handlers["provider_event"](
        {
            "provider": "bailian",
            "model": "fun-asr",
            "event": "stt_provider_first_partial",
            "timestamp": 10.35,
            "stream_id": "s",
            "text_preview": "你好",
        }
    )
    pipeline._timeline.mark_at("transcript_interim_first_at", 10.4)

    snap = pipeline._timeline.snapshot()
    assert snap["timestamps"]["stt_first_audio_sent_at"] == 10.1
    assert snap["timestamps"]["stt_provider_first_partial_at"] == 10.35
    assert snap["attrs"]["stt_stream"]["stream_id"] == "s"
    assert snap["attrs"]["stt_stream"]["last_text_preview"] == "你好"
    assert round(snap["attrs"]["provider_latency_ms"]["stt_first_audio_sent_ms"]) == 100
    assert round(
        snap["attrs"]["provider_latency_ms"][
            "stt_first_audio_to_provider_partial_ms"
        ]
    ) == 250
    assert round(
        snap["attrs"]["provider_latency_ms"][
            "stt_provider_partial_to_livekit_interim_ms"
        ]
    ) == 50
    assert round(
        snap["attrs"]["provider_latency_ms"]["stt_speech_to_provider_partial_ms"]
    ) == 350


def test_streaming_pipeline_replays_pending_stt_provider_events() -> None:
    from eidolon.livekit.agent.full_duplex import StreamingPipeline

    class _FakeStt:
        def __init__(self) -> None:
            self.handlers = {}

        def on(self, name: str, handler) -> None:
            self.handlers[name] = handler

    fake_stt = _FakeStt()
    pipeline = StreamingPipeline.__new__(StreamingPipeline)
    pipeline._factory = SimpleNamespace(stt=SimpleNamespace(stt=fake_stt))
    pipeline._timeline = None

    pipeline._ensure_provider_event_observer()
    pipeline._provider_events.install_stt_provider_event_observer()
    fake_stt.handlers["provider_event"](
        {
            "provider": "bailian",
            "model": "fun-asr",
            "event": "stt_turn_first_audio_sent",
            "timestamp": 10.05,
            "turn_id": "turn-stt-pending",
            "stream_id": "pending-stream",
            "language": "zh",
        }
    )

    pipeline._timeline = TurnTimeline("turn-stt-pending")
    pipeline._timeline.mark_at("speech_started_at", 10.1)
    pipeline._provider_events.apply_pending_stt_provider_events()

    snap = pipeline._timeline.snapshot()
    assert snap["timestamps"]["stt_first_audio_sent_at"] == 10.05
    assert snap["attrs"]["stt_stream"]["stream_id"] == "pending-stream"
    assert pipeline._provider_events.pending_stt_provider_events == []


def test_streaming_pipeline_observes_next_stt_audio_for_turn() -> None:
    from eidolon.livekit.agent.full_duplex import StreamingPipeline

    class _FakeStt:
        def __init__(self) -> None:
            self.observed = None

        def observe_next_audio_for_turn(
            self,
            *,
            turn_id: str,
            speech_started_at: float,
        ) -> bool:
            self.observed = {
                "turn_id": turn_id,
                "speech_started_at": speech_started_at,
            }
            return True

    fake_stt = _FakeStt()
    pipeline = StreamingPipeline.__new__(StreamingPipeline)
    pipeline._factory = SimpleNamespace(stt=SimpleNamespace(_stt=fake_stt))
    pipeline._timeline = TurnTimeline("turn-observe-audio")
    pipeline._timeline.mark_at("speech_started_at", 20.0)

    pipeline._ensure_provider_event_observer()
    pipeline._provider_events.observe_stt_turn_audio()
    pipeline._provider_events.record_stt_provider_event(
        {
            "provider": "bailian",
            "model": "fun-asr",
            "event": "stt_turn_first_audio_sent",
            "timestamp": 20.08,
            "turn_id": "turn-observe-audio",
            "stream_id": "s",
        }
    )

    snap = pipeline._timeline.snapshot()
    assert fake_stt.observed == {
        "turn_id": "turn-observe-audio",
        "speech_started_at": 20.0,
    }
    assert snap["attrs"]["stt_turn_audio_observer_installed"] is True
    assert round(snap["attrs"]["provider_latency_ms"]["stt_first_audio_sent_ms"]) == 80


def test_streaming_pipeline_flushes_timeline_on_agent_playback_done(tmp_path) -> None:
    from eidolon.livekit.agent.full_duplex import StreamingPipeline

    debug_path = tmp_path / "timeline.jsonl"
    pipeline = StreamingPipeline.__new__(StreamingPipeline)
    pipeline._state = PipelineState.SPEAKING
    pipeline._callbacks = MagicMock()
    pipeline._ducking = OutputDuckingController()
    pipeline._ducking.mixer = None
    pipeline._filler = None
    pipeline._timeline = TurnTimeline("turn-normal")
    pipeline._timeline_debug_flushed = False
    pipeline._observability = ObservabilityConfig(timeline_debug_path=str(debug_path))

    pipeline._on_agent_state_changed(
        SimpleNamespace(old_state="speaking", new_state="listening")
    )

    rows = [json.loads(line) for line in debug_path.read_text().splitlines()]
    assert len(rows) == 1
    assert "agent_audio_playback_done_at" in rows[0]["timestamps"]
    assert rows[0]["attrs"]["timeline_flush_reason"] == "agent_audio_playback_done"
    assert pipeline._timeline is None


def test_streaming_pipeline_snapshot_does_not_clear_timeline(tmp_path) -> None:
    from eidolon.livekit.agent.full_duplex import StreamingPipeline

    debug_path = tmp_path / "timeline.jsonl"
    pipeline = StreamingPipeline.__new__(StreamingPipeline)
    pipeline._timeline = TurnTimeline("turn-snapshot")
    pipeline._timeline_debug_flushed = False
    pipeline._observability = ObservabilityConfig(timeline_debug_path=str(debug_path))

    pipeline._append_turn_timeline_snapshot(
        pipeline._timeline,
        "agent_output_first_delta_timeout",
    )
    assert pipeline._timeline is not None
    assert pipeline._timeline_debug_flushed is False

    pipeline._flush_turn_timeline(pipeline._timeline, "agent_audio_playback_done")

    rows = [json.loads(line) for line in debug_path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["attrs"]["timeline_snapshot_reason"] == (
        "agent_output_first_delta_timeout"
    )
    assert rows[0]["attrs"]["timeline_flush_reason"] == (
        "agent_output_first_delta_timeout"
    )
    assert rows[1]["attrs"]["timeline_flush_reason"] == "agent_audio_playback_done"
    assert pipeline._timeline is None


def test_streaming_pipeline_flushes_unfinished_timeline_on_session_close(
    tmp_path,
) -> None:
    from eidolon.livekit.agent.full_duplex import StreamingPipeline
    from eidolon.livekit.agent.full_duplex.lifecycle import FullDuplexSessionLifecycle

    debug_path = tmp_path / "timeline.jsonl"
    pipeline = StreamingPipeline.__new__(StreamingPipeline)
    pipeline._state = PipelineState.SPEAKING
    pipeline._callbacks = MagicMock()
    pipeline._ducking = OutputDuckingController()
    pipeline._ducking.mixer = None
    pipeline._timeline = TurnTimeline("turn-close")
    pipeline._timeline_debug_flushed = False
    pipeline._observability = ObservabilityConfig(timeline_debug_path=str(debug_path))
    pipeline._session_closed_event = MagicMock()
    pipeline._get_eot_model = MagicMock()

    FullDuplexSessionLifecycle(pipeline)._on_session_close(
        SimpleNamespace(reason="participant_left", error=None)
    )

    rows = [json.loads(line) for line in debug_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["attrs"]["timeline_flush_reason"] == "session_closed"
    pipeline._session_closed_event.set.assert_called_once()


def test_streaming_pipeline_does_not_flush_cancelled_timeline_on_session_close(
    tmp_path,
) -> None:
    from eidolon.livekit.agent.full_duplex import StreamingPipeline
    from eidolon.livekit.agent.full_duplex.lifecycle import FullDuplexSessionLifecycle

    debug_path = tmp_path / "timeline.jsonl"
    pipeline = StreamingPipeline.__new__(StreamingPipeline)
    pipeline._timeline = TurnTimeline("turn-cancel")
    pipeline._timeline_debug_flushed = False
    pipeline._observability = ObservabilityConfig(timeline_debug_path=str(debug_path))
    pipeline._session_closed_event = MagicMock()
    pipeline._get_eot_model = MagicMock()
    pipeline._ducking = MagicMock()
    pipeline._ducking.get_metrics.return_value = None

    pipeline._append_timeline_debug("interrupt_cancel", clear=True)
    FullDuplexSessionLifecycle(pipeline)._on_session_close(
        SimpleNamespace(reason="participant_left", error=None)
    )

    rows = [json.loads(line) for line in debug_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["attrs"]["timeline_flush_reason"] == "interrupt_cancel"
    pipeline._session_closed_event.set.assert_called_once()


def test_parse_client_audio_state_sanitizes_payload() -> None:
    state = parse_client_audio_state(
        b'{"type":"client.audio_state","input_mode":"auto","ptt":false,'
        b'"manual_interrupt":true,"playback_state":"agent_speaking",'
        b'"mic_muted":false,"rms":1.3,"snr_hint":"0.4","client_ts_ms":42}',
        participant_identity="alice",
        received_at=10.0,
    )

    assert state.participant_identity == "alice"
    assert state.input_mode == "auto"
    assert state.manual_interrupt is True
    assert state.playback_state == "agent_speaking"
    assert state.rms == 1.0
    assert state.snr_hint == 0.4
    assert state.is_fresh(now=11.0)


def test_streaming_pipeline_observes_client_audio_state() -> None:
    from eidolon.livekit.agent.full_duplex import StreamingPipeline
    from eidolon.livekit.agent.session.room_data import RoomDataHandler

    pipeline = StreamingPipeline.__new__(StreamingPipeline)
    pipeline._timeline = TurnTimeline("turn-1")
    pipeline._room_data = RoomDataHandler(get_timeline=lambda: pipeline._timeline)

    packet = SimpleNamespace(
        topic=CLIENT_AUDIO_STATE_TOPIC,
        data=(
            b'{"type":"client.audio_state","input_mode":"auto",'
            b'"playback_state":"idle","mic_muted":true}'
        ),
        participant=SimpleNamespace(identity="alice"),
    )

    pipeline._room_data.handle_packet(packet)

    state = pipeline._room_data.client_audio_states["alice"]
    assert state.mic_muted is True
    assert pipeline._timeline.attrs["client_audio_state"][
        "participant_identity"
    ] == "alice"
    assert pipeline._timeline.attrs["room_data_events"][-1] == {
        "topic": CLIENT_AUDIO_STATE_TOPIC,
        "participant_identity": "alice",
        "bytes": len(packet.data),
    }


def test_streaming_pipeline_ptt_data_force_cancels() -> None:
    # PTT (deliberate button press / tap-to-stop) is the only explicit client
    # interrupt — hard-cut immediately with force=True so it cuts through
    # half_duplex's allow_interruptions=False.
    from eidolon.livekit.agent.full_duplex import StreamingPipeline
    from eidolon.livekit.agent.session.room_data import RoomDataHandler

    pipeline = StreamingPipeline.__new__(StreamingPipeline)
    pipeline._timeline = TurnTimeline("turn-1")
    pipeline._turn_policy = TurnPolicyConfig()
    pipeline._turn_runtime = TurnPolicyRuntime(pipeline._turn_policy)
    pipeline._room_data = RoomDataHandler(get_timeline=lambda: pipeline._timeline)
    pipeline._state = PipelineState.SPEAKING
    pipeline._ducking = SimpleNamespace(is_cancelled=False)
    pipeline._decision_effects = MagicMock()
    pipeline._ensure_decision_effect_applier = MagicMock()
    effects = SimpleNamespace(
        cancel_and_interrupt=MagicMock(),
        cancel_silent_generation_for_explicit_preempt=MagicMock(),
        rollback_if_suspended=MagicMock(),
        handle_hold_decision=MagicMock(),
    )
    pipeline._client_audio_state = SimpleNamespace(
        latest_state=lambda participant_identity=None: (
            pipeline._room_data.latest_client_audio_state(
                participant_identity=participant_identity,
            )
        ),
        agent_output_active_for_interrupts=lambda participant_identity=None: True,
    )
    pipeline._ensure_client_audio_state_view = MagicMock(
        return_value=pipeline._client_audio_state,
    )
    pipeline._ensure_interruption_effects = MagicMock(return_value=effects)
    pipeline._client_preempts = pipeline._build_client_preempt_handler()

    packet = SimpleNamespace(
        topic=CLIENT_AUDIO_STATE_TOPIC,
        data=(
            b'{"type":"client.audio_state","input_mode":"ptt",'
            b'"playback_state":"agent_speaking","mic_muted":false,'
            b'"ptt":true}'
        ),
        participant=SimpleNamespace(identity="alice"),
    )

    pipeline._room_data.handle_packet(packet)
    pipeline._client_preempts.on_client_room_packet(packet)

    effects.cancel_and_interrupt.assert_called_once_with(force=True)
    assert pipeline._timeline.attrs["explicit_client_interrupt"][
        "participant_identity"
    ] == "alice"


@pytest.mark.asyncio
async def test_streaming_pipeline_publishes_client_playback_stop_control() -> None:
    from eidolon.livekit.agent.full_duplex import StreamingPipeline

    pipeline = StreamingPipeline.__new__(StreamingPipeline)
    pipeline._timeline = TurnTimeline("turn-playback-stop")
    pipeline._room = SimpleNamespace(local_participant=SimpleNamespace())
    pipeline._room.local_participant.publish_data = AsyncMock()

    pipeline._publish_client_control("playback.stop", reason="interrupt_cancel")
    await asyncio.sleep(0)

    pipeline._room.local_participant.publish_data.assert_awaited_once()
    args = pipeline._room.local_participant.publish_data.await_args.args
    kwargs = pipeline._room.local_participant.publish_data.await_args.kwargs
    payload = json.loads(args[0])

    assert kwargs == {"reliable": True, "topic": "eidolon.control"}
    assert payload["v"] == 1
    assert payload["kind"] == "cmd"
    assert payload["op"] == "playback.stop"
    assert payload["src"] == {"type": "channel", "id": "eidolon_channel"}
    assert payload["payload"] == {
        "reason": "interrupt_cancel",
        "turn_id": "turn-playback-stop",
    }
    assert pipeline._timeline.attrs["client_control_events"][-1] == {
        "op": "playback.stop",
        "reason": "interrupt_cancel",
        "turn_id": "turn-playback-stop",
    }


@pytest.mark.asyncio
async def test_client_playback_stop_before_timeline_attaches_to_next_turn() -> None:
    from eidolon.livekit.agent.full_duplex import StreamingPipeline

    pipeline = StreamingPipeline.__new__(StreamingPipeline)
    pipeline._timeline = None
    pipeline._pending_client_control_events = []
    pipeline._room = SimpleNamespace(local_participant=SimpleNamespace())
    pipeline._room.local_participant.publish_data = AsyncMock()

    pipeline._publish_client_control("playback.stop", reason="interrupt_cancel")
    await asyncio.sleep(0)

    pipeline._room.local_participant.publish_data.assert_awaited_once()
    payload = json.loads(pipeline._room.local_participant.publish_data.await_args.args[0])
    assert payload["payload"] == {
        "reason": "interrupt_cancel",
        "turn_id": "",
    }
    assert pipeline._pending_client_control_events == [
        {
            "op": "playback.stop",
            "reason": "interrupt_cancel",
            "turn_id": "",
        }
    ]

    timeline = TurnTimeline("turn-after-ptt-press")
    pipeline._apply_pending_client_control_events(timeline)

    assert pipeline._pending_client_control_events == []
    assert timeline.attrs["client_control_events"] == [
        {
            "op": "playback.stop",
            "reason": "interrupt_cancel",
            "turn_id": "turn-after-ptt-press",
        }
    ]


@pytest.mark.asyncio
async def test_duck_cancel_publishes_playback_stop_control() -> None:
    from eidolon.livekit.agent.full_duplex import StreamingPipeline

    class FakeDucking:
        is_cancelled = False

        def __init__(self) -> None:
            self.cancelled = False

        def stats(self) -> SimpleNamespace:
            return SimpleNamespace(suspend_ms=0.0, buffered_frames=0, buffered_sec=0.0)

        def cancel_output(self) -> None:
            self.cancelled = True

    pipeline = StreamingPipeline.__new__(StreamingPipeline)
    pipeline._ensure_runtime_defaults = lambda: None
    pipeline._timeline = TurnTimeline("turn-ptt-stop")
    pipeline._room = SimpleNamespace(local_participant=SimpleNamespace())
    pipeline._room.local_participant.publish_data = AsyncMock()
    pipeline._ducking = FakeDucking()
    pipeline._session = MagicMock()
    pipeline._allow_interruptions = False
    pipeline._get_eot_model = MagicMock(return_value=MagicMock())
    pipeline._interruption_orchestrator = SimpleNamespace(
        should_collect_after_confirmed_cancel=lambda: False,
        should_commit_after_confirmed_cancel=lambda: False,
        current_transcript="",
        resolve=MagicMock(),
    )
    snapshot_interrupted_context = MagicMock()
    pipeline._callbacks = MagicMock()
    pipeline._cancel_residual_commit_suppress_sec = lambda: 2.0
    effects = FullDuplexInterruptionEffects(
        ducking=pipeline._ducking,
        callbacks=pipeline._callbacks,
        get_session=lambda: pipeline._session,
        allow_interruptions=lambda: pipeline._allow_interruptions,
        get_eot_model=lambda: pipeline._get_eot_model(),
        get_timeline=lambda: pipeline._timeline,
        get_latest_asr_text=lambda: "",
        get_state_label=lambda: "SPEAKING",
        get_interruption_orchestrator=lambda: pipeline._interruption_orchestrator,
        publish_playback_stop=lambda reason: pipeline._publish_client_control(
            "playback.stop",
            reason=reason,
        ),
        snapshot_interrupted_context=snapshot_interrupted_context,
        commit_post_speech_interruption_candidate=MagicMock(return_value=False),
        reject_post_speech_interruption_candidate=MagicMock(),
        cancel_residual_commit_suppress_sec=(
            pipeline._cancel_residual_commit_suppress_sec
        ),
        semantic_interrupt_run=MagicMock(),
        correction_topic_stability_window_ms=lambda: 120,
        set_interrupt_cancel_suppression=MagicMock(),
        soft_interrupt_timeout_sec=lambda: 0.5,
    )

    effects.cancel_and_interrupt(force=True)
    await asyncio.sleep(0)

    pipeline._room.local_participant.publish_data.assert_awaited_once()
    payload = json.loads(pipeline._room.local_participant.publish_data.await_args.args[0])
    assert payload["op"] == "playback.stop"
    assert payload["payload"] == {
        "reason": "interrupt_cancel",
        "turn_id": "turn-ptt-stop",
    }
    assert pipeline._timeline.attrs["client_control_events"][-1]["op"] == "playback.stop"
    assert pipeline._ducking.cancelled is True
    pipeline._session.interrupt.assert_called_once_with(force=True)


def test_streaming_pipeline_ignores_duplicate_duck_cancel() -> None:
    callbacks = MagicMock()
    session = MagicMock()
    effects = FullDuplexInterruptionEffects(
        ducking=SimpleNamespace(is_cancelled=True),
        callbacks=callbacks,
        get_session=lambda: session,
        allow_interruptions=lambda: True,
        get_eot_model=lambda: MagicMock(),
        get_timeline=lambda: TurnTimeline("turn-duplicate-cancel"),
        get_latest_asr_text=lambda: "",
        get_state_label=lambda: "SPEAKING",
        get_interruption_orchestrator=MagicMock(),
        publish_playback_stop=MagicMock(),
        snapshot_interrupted_context=MagicMock(),
        commit_post_speech_interruption_candidate=MagicMock(return_value=False),
        reject_post_speech_interruption_candidate=MagicMock(),
        cancel_residual_commit_suppress_sec=lambda: 0.0,
        semantic_interrupt_run=MagicMock(),
        correction_topic_stability_window_ms=lambda: 120,
        set_interrupt_cancel_suppression=MagicMock(),
        soft_interrupt_timeout_sec=lambda: 0.5,
    )

    effects.cancel_and_interrupt()

    callbacks.on_duck_resolved.assert_not_called()
    session.interrupt.assert_not_called()


def test_streaming_pipeline_cancels_when_playback_evidence_outlives_duck_state() -> None:
    class CancelledDucking:
        is_cancelled = True
        is_suspended = False

        def __init__(self) -> None:
            self.cancelled = False

        def cancel_output(self) -> None:
            self.cancelled = True

    callbacks = MagicMock()
    session = MagicMock()
    ducking = CancelledDucking()
    timeline = TurnTimeline("turn-playback-evidence-cancel")
    orchestrator = SimpleNamespace(
        should_collect_after_confirmed_cancel=MagicMock(return_value=False),
        should_commit_after_confirmed_cancel=MagicMock(return_value=False),
        current_transcript="换个话题",
        resolve=MagicMock(),
    )
    publish_playback_stop = MagicMock()
    effects = FullDuplexInterruptionEffects(
        ducking=ducking,
        callbacks=callbacks,
        get_session=lambda: session,
        allow_interruptions=lambda: True,
        get_eot_model=lambda: MagicMock(),
        get_timeline=lambda: timeline,
        get_latest_asr_text=lambda: "换个话题",
        get_state_label=lambda: "SPEAKING",
        get_interruption_orchestrator=lambda: orchestrator,
        publish_playback_stop=publish_playback_stop,
        snapshot_interrupted_context=MagicMock(),
        commit_post_speech_interruption_candidate=MagicMock(return_value=False),
        reject_post_speech_interruption_candidate=MagicMock(),
        cancel_residual_commit_suppress_sec=lambda: 0.0,
        semantic_interrupt_run=MagicMock(),
        correction_topic_stability_window_ms=lambda: 120,
        set_interrupt_cancel_suppression=MagicMock(),
        soft_interrupt_timeout_sec=lambda: 0.5,
        playback_evidence_active=lambda: True,
    )

    effects.cancel_and_interrupt()

    publish_playback_stop.assert_called_once_with("interrupt_cancel")
    assert ducking.cancelled is True
    callbacks.on_duck_resolved.assert_called_once_with("cancel")
    orchestrator.resolve.assert_called_once_with(action="cancel", reason="eot_cancel")
    session.interrupt.assert_called_once_with(force=False)
    assert "interrupt_cancel_resolved_at" in timeline.timestamps
    assert timeline.attrs["duck_events"][-1]["event"] == "duck_cancelled"


def test_cancel_collects_confirmed_semantic_turn_without_resolving_candidate() -> None:
    class FakeDucking:
        is_cancelled = False

        def __init__(self) -> None:
            self.cancelled = False

        def stats(self) -> SimpleNamespace:
            return SimpleNamespace(suspend_ms=123.0, buffered_frames=3, buffered_sec=0.6)

        def cancel_output(self) -> None:
            self.cancelled = True
            self.is_cancelled = True

    timeline = TurnTimeline("turn-confirmed-semantic")
    callbacks = MagicMock()
    session = MagicMock()
    eot_model = MagicMock()
    ducking = FakeDucking()
    orchestrator = SimpleNamespace(
        should_collect_after_confirmed_cancel=MagicMock(return_value=True),
        should_commit_after_confirmed_cancel=MagicMock(return_value=False),
        mark_confirmed_cancel_collecting_turn=MagicMock(),
        resolve=MagicMock(),
        current_transcript="不是",
    )
    publish_playback_stop = MagicMock()
    snapshot_context = MagicMock()
    commit_candidate = MagicMock(return_value=True)
    set_suppression = MagicMock()
    effects = FullDuplexInterruptionEffects(
        ducking=ducking,
        callbacks=callbacks,
        get_session=lambda: session,
        allow_interruptions=lambda: True,
        get_eot_model=lambda: eot_model,
        get_timeline=lambda: timeline,
        get_latest_asr_text=lambda: "不是",
        get_state_label=lambda: "SPEAKING",
        get_interruption_orchestrator=lambda: orchestrator,
        publish_playback_stop=publish_playback_stop,
        snapshot_interrupted_context=snapshot_context,
        commit_post_speech_interruption_candidate=commit_candidate,
        reject_post_speech_interruption_candidate=MagicMock(),
        cancel_residual_commit_suppress_sec=lambda: 2.0,
        semantic_interrupt_run=MagicMock(),
        correction_topic_stability_window_ms=lambda: 120,
        set_interrupt_cancel_suppression=set_suppression,
        soft_interrupt_timeout_sec=lambda: 0.5,
    )

    effects.cancel_and_interrupt()

    snapshot_context.assert_called_once_with()
    publish_playback_stop.assert_called_once_with("interrupt_cancel")
    assert ducking.cancelled is True
    orchestrator.mark_confirmed_cancel_collecting_turn.assert_called_once_with()
    orchestrator.resolve.assert_not_called()
    orchestrator.should_commit_after_confirmed_cancel.assert_not_called()
    commit_candidate.assert_not_called()
    callbacks.on_duck_resolved.assert_called_once_with("cancel")
    session.interrupt.assert_called_once_with(force=False)
    eot_model.update_vad.assert_called_once_with(False)
    set_suppression.assert_called_once_with(True, ANY)
    assert timeline.attrs["cancel_reason"] == "eot_cancel"
    assert timeline.attrs["duck_events"][-1]["event"] == "duck_cancelled"


def test_rollback_if_suspended_records_timeline_action() -> None:
    class FakeDucking:
        installed = True
        is_suspended = True

        def __init__(self) -> None:
            self.unducked = False

        def stats(self) -> SimpleNamespace:
            return SimpleNamespace(
                suspend_ms=123.0,
                buffered_frames=2,
                buffered_sec=0.4,
            )

        def unduck_if_suspended(self, *, drop_buffered: bool) -> None:
            self.unducked = True
            self.drop_buffered = drop_buffered

    timeline = TurnTimeline("turn-rollback")
    callbacks = MagicMock()
    ducking = FakeDucking()
    orchestrator = SimpleNamespace(
        awaiting_post_speech_evidence=False,
        resolve=MagicMock(),
    )
    effects = FullDuplexInterruptionEffects(
        ducking=ducking,
        callbacks=callbacks,
        get_session=lambda: MagicMock(),
        allow_interruptions=lambda: True,
        get_eot_model=lambda: MagicMock(),
        get_timeline=lambda: timeline,
        get_latest_asr_text=lambda: "",
        get_state_label=lambda: "SPEAKING",
        get_interruption_orchestrator=lambda: orchestrator,
        publish_playback_stop=MagicMock(),
        snapshot_interrupted_context=MagicMock(),
        commit_post_speech_interruption_candidate=MagicMock(return_value=False),
        reject_post_speech_interruption_candidate=MagicMock(),
        cancel_residual_commit_suppress_sec=lambda: 0.0,
        semantic_interrupt_run=MagicMock(),
        correction_topic_stability_window_ms=lambda: 120,
        set_interrupt_cancel_suppression=MagicMock(),
        soft_interrupt_timeout_sec=lambda: 0.5,
    )

    effects.rollback_if_suspended(reason="agent_echo", drop_buffered=False)

    assert ducking.unducked is True
    assert timeline.attrs["interrupt_action"] == "rollback"
    assert timeline.attrs["rollback_reason"] == "agent_echo"
    assert timeline.attrs["rollback_drop_buffered"] is False
    orchestrator.resolve.assert_called_once_with(action="rollback", reason="agent_echo")
