"""Turn timeline tests."""

from __future__ import annotations

from eidolon.livekit.agent.observability import TurnTimeline


def test_timeline_marks_and_durations() -> None:
    timeline = TurnTimeline("turn-1")
    timeline.mark("speech_started_at")
    timeline.mark("interrupt_resolved_at")
    timeline.set_attr("decision", "cancel")
    snap = timeline.snapshot()
    assert snap["turn_id"] == "turn-1"
    assert snap["attrs"]["decision"] == "cancel"
    assert snap["durations_ms"]["vad_start_to_interrupt_resolved"] is not None

