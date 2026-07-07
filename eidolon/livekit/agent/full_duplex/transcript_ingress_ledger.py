"""Pipeline-level transcript ingress facts for full-duplex sessions."""

from __future__ import annotations

import time
from typing import Any

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.shared.types import PipelineState


class FullDuplexTranscriptIngressLedger:
    """Keep recent transcript callbacks independent of turn timeline lifetime."""

    def __init__(self, *, max_events: int = 32, timeline_events: int = 16) -> None:
        self._max_events = max_events
        self._timeline_events = timeline_events
        self._events: list[dict[str, Any]] = []
        self._sequence = 0

    def record(
        self,
        payload: dict[str, object],
        *,
        timeline: TurnTimeline | None,
        user_speaking_active: bool,
        pipeline_state: PipelineState | str | None,
    ) -> dict[str, Any]:
        self._sequence += 1
        event = dict(payload)
        event["sequence"] = self._sequence
        event["observed_at"] = time.monotonic()
        event["timeline_present"] = timeline is not None
        event["user_speaking_active"] = user_speaking_active
        state_value = self._state_value(pipeline_state)
        if state_value:
            event["pipeline_state"] = state_value
        if timeline is not None:
            event["timeline_turn_id"] = timeline.turn_id

        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events :]
        self.attach_to_timeline(timeline, reason="record")
        return event

    def attach_to_timeline(
        self,
        timeline: TurnTimeline | None,
        *,
        reason: str,
    ) -> None:
        if timeline is None:
            return
        recent = [dict(event) for event in self._events[-self._timeline_events :]]
        if not recent:
            return
        pre_timeline = [
            dict(event) for event in recent if event.get("timeline_present") is False
        ]
        cross_turn = [
            dict(event)
            for event in recent
            if event.get("timeline_turn_id") not in (None, timeline.turn_id)
        ]
        timeline.set_attr("transcript_ingress_recent_events", recent)
        timeline.set_attr("transcript_ingress_recent_event_count", len(recent))
        timeline.set_attr("transcript_ingress_recent_attach_reason", reason)
        timeline.set_attr(
            "transcript_ingress_pre_timeline_event_count",
            len(pre_timeline),
        )
        if pre_timeline:
            timeline.set_attr("transcript_ingress_pre_timeline_events", pre_timeline)
        timeline.set_attr(
            "transcript_ingress_recent_cross_turn_event_count",
            len(cross_turn),
        )
        if cross_turn:
            timeline.set_attr("transcript_ingress_recent_cross_turn_events", cross_turn)

    @staticmethod
    def _state_value(state: PipelineState | str | None) -> str | None:
        if state is None:
            return None
        if isinstance(state, PipelineState):
            return state.name.lower()
        value = getattr(state, "value", state)
        return str(value) if value is not None else None
