"""Side-effect-free full-duplex turn contract state machine."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from eidolon.livekit.agent.observability import TurnTimeline

FullDuplexSideEffect = Literal["none", "reversible", "irreversible"]

TIMELINE_TEXT_PREVIEW_MAX_CHARS = 120
TIMELINE_TRANSITION_MAX_ITEMS = 48


class FullDuplexPhase(str, Enum):
    """High-level contract phase for one full-duplex turn timeline."""

    IDLE = "idle"
    USER_SPEECH_OPEN = "user_speech_open"
    PROVISIONAL_DUCK = "provisional_duck"
    EVIDENCE_ARBITRATION = "evidence_arbitration"
    ACCEPTED_INTERRUPTION = "accepted_interruption"
    REJECTED_INTERRUPTION = "rejected_interruption"
    USER_TURN_PENDING = "user_turn_pending"
    USER_TURN_COMMITTED = "user_turn_committed"
    USER_TURN_REJECTED = "user_turn_rejected"


@dataclass(frozen=True)
class FullDuplexTransition:
    """One observable transition in the full-duplex contract."""

    phase: FullDuplexPhase
    event: str
    reason: str
    at: float
    side_effect: FullDuplexSideEffect = "none"
    transcript_preview: str = ""
    details: dict[str, object] = field(default_factory=dict)

    def snapshot(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "phase": self.phase.value,
            "event": self.event,
            "reason": self.reason,
            "side_effect": self.side_effect,
            "at": self.at,
            "transcript_preview": self.transcript_preview,
        }
        if self.details:
            payload["details"] = dict(self.details)
        return payload


class FullDuplexStateMachine:
    """Record the unified full-duplex contract without applying side effects."""

    def __init__(self, *, clock: Any | None = None) -> None:
        self._clock = clock or time.monotonic
        self._phase = FullDuplexPhase.IDLE
        self._transitions: list[FullDuplexTransition] = []

    @property
    def phase(self) -> FullDuplexPhase:
        return self._phase

    @property
    def transitions(self) -> tuple[FullDuplexTransition, ...]:
        return tuple(self._transitions)

    def transition(
        self,
        phase: FullDuplexPhase,
        *,
        event: str,
        reason: str,
        timeline: TurnTimeline | None = None,
        side_effect: FullDuplexSideEffect = "none",
        transcript: str = "",
        details: dict[str, object] | None = None,
        now: float | None = None,
    ) -> FullDuplexTransition:
        transition = FullDuplexTransition(
            phase=phase,
            event=event,
            reason=reason,
            side_effect=side_effect,
            at=self._now(now),
            transcript_preview=transcript[:TIMELINE_TEXT_PREVIEW_MAX_CHARS],
            details=dict(details or {}),
        )
        self._phase = phase
        self._transitions.append(transition)
        self._record_timeline(timeline, transition)
        return transition

    def snapshot(self) -> dict[str, object]:
        last = self._transitions[-1].snapshot() if self._transitions else None
        return {
            "phase": self._phase.value,
            "transition_count": len(self._transitions),
            "last": last,
        }

    def _now(self, value: float | None) -> float:
        return float(self._clock() if value is None else value)

    def _record_timeline(
        self,
        timeline: TurnTimeline | None,
        transition: FullDuplexTransition,
    ) -> None:
        if timeline is None:
            return
        events = list(timeline.attrs.get("full_duplex_state_transitions") or ())
        events.append(transition.snapshot())
        timeline.set_attr(
            "full_duplex_state_transitions",
            events[-TIMELINE_TRANSITION_MAX_ITEMS:],
        )
        timeline.set_attr(
            "full_duplex_state",
            {
                "phase": self._phase.value,
                "last": transition.snapshot(),
                "transition_count": len(events),
            },
        )
