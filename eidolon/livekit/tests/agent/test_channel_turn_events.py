from __future__ import annotations

import json
from types import SimpleNamespace

from eidolon_data import DataSettings, DataStore

from eidolon.livekit.agent.observability import turn_events
from eidolon.livekit.agent.observability import (
    ChannelEventContext,
    ChannelTurnEventSink,
    TurnTimeline,
)


def _enabled_sink(*, queue_max: int = 16) -> ChannelTurnEventSink:
    sink = ChannelTurnEventSink(queue_max=queue_max)
    sink._context = ChannelEventContext(  # type: ignore[attr-defined]
        owner_id="owner-1",
        companion_id="companion-1",
        device_id="device-1",
        room_name="room-1",
    )
    sink._writer = object()  # type: ignore[assignment,attr-defined]
    return sink


def test_phase_projection_is_non_blocking_safe_and_ordered() -> None:
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

    pending = sink._queue.get_nowait()  # type: ignore[attr-defined]
    assert pending is not None
    assert pending.event_type == "channel.turn.phase_changed"
    assert pending.trace_id == "channel-turn-1"
    assert pending.payload["transition_seq"] == 1
    assert pending.payload["elapsed_ms"] == 125.0
    assert pending.payload["details"] == {"eot_score": 0.7}
    assert "transcript" not in str(pending.payload)


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

    assert sink._queue.qsize() == 1  # type: ignore[attr-defined]
    pending = sink._queue.get_nowait()  # type: ignore[attr-defined]
    assert pending is not None
    assert pending.event_type == "channel.turn.rejected"
    assert pending.outcome == "denied"
    assert pending.payload["status"] == "rejected"
    assert "turn_committed" in pending.payload["missing_milestones"]


def test_queue_pressure_drops_observability_not_voice_work() -> None:
    sink = _enabled_sink(queue_max=1)
    first = TurnTimeline("turn-1")
    second = TurnTimeline("turn-2")

    sink.milestone(first, "generating")
    sink.milestone(second, "generating")

    assert sink._queue.qsize() == 1  # type: ignore[attr-defined]
    assert sink.dropped_count == 1


async def test_sink_persists_replayable_session_and_turn_chain(tmp_path, monkeypatch) -> None:
    settings = DataSettings(
        sqlite_path=str(tmp_path / "eidolon.sqlite3"),
        object_store_path=str(tmp_path / "objects"),
    )
    bootstrap = DataStore.open(settings)
    await bootstrap.init_schema()
    await bootstrap.owners.create(owner_id="owner-1", display_name="Owner")
    await bootstrap.companions.create(
        companion_id="companion-1",
        owner_id="owner-1",
        display_name="Companion",
    )
    await bootstrap.devices.create_device(
        device_id="device-1",
        owner_id="owner-1",
        bound_companion_id="companion-1",
        kind="voice_body",
    )
    await bootstrap.close()

    monkeypatch.setattr(turn_events, "load_data_settings", lambda: settings)
    participant = SimpleNamespace(
        identity="device-1",
        metadata=json.dumps({"kind": "device", "device_id": "device-1"}),
    )
    room = SimpleNamespace(
        name="room-1",
        remote_participants={"device-1": participant},
    )
    sink = ChannelTurnEventSink()
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

    reader = DataStore.open(settings)
    try:
        events = await reader.events.list_for_owner("owner-1", limit=20)
    finally:
        await reader.close()

    event_types = [event.event_type for event in reversed(events)]
    assert event_types == [
        "channel.session.started",
        "channel.turn.phase_changed",
        "channel.turn.milestone",
        "channel.turn.completed",
        "channel.session.ended",
    ]
    turn_events_by_trace = [event for event in events if event.trace_id == "channel-turn-1"]
    assert len(turn_events_by_trace) == 3
    assert all(event.companion_id == "companion-1" for event in events)
    assert all(event.payload_json.get("device_id") == "device-1" for event in events)
