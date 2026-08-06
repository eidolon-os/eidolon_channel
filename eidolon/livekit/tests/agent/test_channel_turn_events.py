from __future__ import annotations

import json
from types import SimpleNamespace

from eidolon.livekit.agent.observability import turn_events
from eidolon.livekit.agent.observability import (
    ChannelEventContext,
    ChannelTurnEventSink,
    TurnTimeline,
)


def _enabled_sink() -> ChannelTurnEventSink:
    observed = []
    sink = ChannelTurnEventSink(observer=observed.append)
    sink._test_observed = observed  # type: ignore[attr-defined]
    sink._context = ChannelEventContext(  # type: ignore[attr-defined]
        owner_id="owner-1",
        companion_id="companion-1",
        device_id="device-1",
        room_name="room-1",
        session_flow_id=None,
    )
    return sink


def test_phase_projection_stays_in_the_telemetry_lane() -> None:
    sink = _enabled_sink()
    timeline = TurnTimeline("channel-turn-1")
    timeline.mark_at("speech_started_at", 10.0)

    sink.phase_changed(
        timeline=timeline,
        previous_phase="idle",
        phase="user_speech_open",
        event="speech_started",
        reason="new_speech_started",
        side_effect="none",
        occurred_at=10.125,
        details={"eot_score": 0.7, "transcript": "must not leave Channel"},
    )

    assert len(sink._test_observed) == 1  # type: ignore[attr-defined]
    assert sink.telemetry_observed_count == 1


def test_terminal_projection_is_deduped_and_classifies_rejection() -> None:
    sink = _enabled_sink()
    timeline = TurnTimeline("channel-turn-rejected")
    timeline.mark("speech_started_at")
    timeline.set_attr(
        "full_duplex_state",
        {"phase": "user_turn_rejected", "transition_count": 2},
    )

    sink.terminal(timeline, "voiceprint_commit_blocked")
    sink.terminal(timeline, "duplicate_flush")

    assert len(sink._test_observed) == 1  # type: ignore[attr-defined]
    pending = sink._test_observed[0]  # type: ignore[attr-defined]
    assert pending.event_type == "channel.turn.rejected"
    assert pending.outcome == "denied"
    assert pending.payload["status"] == "rejected"
    assert "turn_committed" in pending.payload["missing_milestones"]


def test_terminal_projection_distinguishes_interrupted_response_from_failed_tts() -> None:
    sink = _enabled_sink()
    interrupted = TurnTimeline("channel-turn-interrupted")
    interrupted.mark("turn_committed_at")
    interrupted.mark("tts_first_audio_at")
    failed = TurnTimeline("channel-turn-tts-failed")
    failed.mark("turn_committed_at")
    failed.mark("tts_first_audio_at")
    failed.mark("tts_error_at")

    sink.terminal(interrupted, "interrupted_by_user")
    sink.terminal(failed, "nonrecoverable_tts_error")

    interrupted_event, failed_event = sink._test_observed  # type: ignore[attr-defined]
    assert interrupted_event is not None
    assert interrupted_event.event_type == "channel.turn.completed"
    assert interrupted_event.payload["status"] == "interrupted"
    assert interrupted_event.payload["terminal_reason"] == "interrupted_by_user"
    assert failed_event is not None
    assert failed_event.event_type == "channel.turn.failed"
    assert failed_event.outcome == "failure"
    assert failed_event.payload["status"] == "failed"


def test_telemetry_adapter_failure_does_not_escape_into_voice_work() -> None:
    def _fail(_event) -> None:
        raise RuntimeError("metrics backend unavailable")

    sink = ChannelTurnEventSink(observer=_fail)
    sink._context = ChannelEventContext(  # type: ignore[attr-defined]
        owner_id="owner-1",
        companion_id="companion-1",
        device_id=None,
        room_name="room-1",
        session_flow_id=None,
    )
    sink.terminal(TurnTimeline("turn-1"), "agent_playback_done")

    assert sink.dropped_count == 1


async def test_event_context_supports_companion_without_device() -> None:
    participant = SimpleNamespace(
        identity="companion-1",
        metadata=json.dumps(
            {
                "kind": "companion",
                "owner_id": "owner-1",
                "companion_id": "companion-1",
            }
        ),
    )
    room = SimpleNamespace(
        name="room-virtual",
        remote_participants={"companion-1": participant},
    )

    context = await turn_events._resolve_event_context(room)

    assert context.owner_id == "owner-1"
    assert context.companion_id == "companion-1"
    assert context.device_id is None


async def test_owner_event_context_requires_explicit_companion_selection() -> None:
    participant = SimpleNamespace(
        identity="owner-a",
        metadata=json.dumps({"kind": "owner", "owner_id": "owner-a"}),
    )
    room = SimpleNamespace(
        name="room-owner",
        remote_participants={"owner-a": participant},
    )

    try:
        await turn_events._resolve_event_context(room)
    except RuntimeError as exc:
        assert "explicitly selected companion" in str(exc)
    else:
        raise AssertionError("Channel must not guess a Companion from Owner scope")


async def test_sink_keeps_session_and_turn_chain_in_telemetry_lane() -> None:
    participant = SimpleNamespace(
        identity="device-1",
        metadata=json.dumps(
            {
                "kind": "device",
                "device_id": "device-1",
                "owner_id": "owner-1",
                "companion_id": "companion-1",
            }
        ),
    )
    room = SimpleNamespace(
        name="room-1",
        remote_participants={"device-1": participant},
    )
    observed = []
    sink = ChannelTurnEventSink(observer=observed.append)
    await sink.start(room)
    assert sink.enabled

    timeline = TurnTimeline("channel-turn-1")
    timeline.mark_at("speech_started_at", 10.0)
    timeline.set_attr("full_duplex_state", {"phase": "turn_finished"})
    sink.phase_changed(
        timeline=timeline,
        previous_phase="idle",
        phase="user_speech_open",
        event="speech_started",
        reason="new_speech_started",
        side_effect="none",
        occurred_at=10.05,
    )
    sink.milestone(timeline, "brain_request_sent")
    sink.terminal(timeline, "agent_playback_done")
    await sink.close()

    event_types = [event.event_type for event in observed]
    assert event_types == [
        "channel.session.started",
        "channel.turn.phase_changed",
        "channel.turn.milestone",
        "channel.turn.completed",
        "channel.session.ended",
    ]
    turn_events_by_trace = [event for event in observed if event.trace_id == "channel-turn-1"]
    assert len(turn_events_by_trace) == 3
    assert all(event.payload.get("device_id") == "device-1" for event in observed)


def test_channel_telemetry_has_no_eidolon_data_dependency() -> None:
    source = turn_events.__file__
    assert source is not None
    assert "eidolon_data" not in open(source, encoding="utf-8").read()
