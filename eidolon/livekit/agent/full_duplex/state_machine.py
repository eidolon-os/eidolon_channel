"""Side-effect-free full-duplex turn contract state machine."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from eidolon.livekit.agent.observability import TurnTimeline

logger = logging.getLogger("agent.full_duplex.state_machine")

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


# F1 guardrail (2026-07): expected edges of the contract, derived from the real
# transition call sites. A product turn can contain several acoustic segments,
# so speech, interruption arbitration, and the pending framework boundary are
# deliberately re-entrant until the product turn reaches a terminal phase.
# This is a *heuristic* graph used only to flag anomalous sequences as an
# observability signal — the recorder never enforces or blocks. Reset (IDLE)
# and product rejection (USER_TURN_REJECTED) may happen from any phase, so they
# are always allowed (see ``_is_expected_transition``).
_ALWAYS_ALLOWED_PHASES: frozenset[FullDuplexPhase] = frozenset(
    {FullDuplexPhase.IDLE, FullDuplexPhase.USER_TURN_REJECTED}
)
_EXPECTED_NEXT_PHASES: dict[FullDuplexPhase, frozenset[FullDuplexPhase]] = {
    FullDuplexPhase.IDLE: frozenset({FullDuplexPhase.USER_SPEECH_OPEN}),
    FullDuplexPhase.USER_SPEECH_OPEN: frozenset(
        {
            FullDuplexPhase.PROVISIONAL_DUCK,
            FullDuplexPhase.EVIDENCE_ARBITRATION,
            FullDuplexPhase.ACCEPTED_INTERRUPTION,
            FullDuplexPhase.USER_TURN_PENDING,
            FullDuplexPhase.USER_TURN_COMMITTED,
        }
    ),
    FullDuplexPhase.PROVISIONAL_DUCK: frozenset(
        {
            FullDuplexPhase.EVIDENCE_ARBITRATION,
            FullDuplexPhase.ACCEPTED_INTERRUPTION,
            FullDuplexPhase.REJECTED_INTERRUPTION,
            FullDuplexPhase.USER_SPEECH_OPEN,
            FullDuplexPhase.USER_TURN_PENDING,
        }
    ),
    FullDuplexPhase.EVIDENCE_ARBITRATION: frozenset(
        {
            FullDuplexPhase.ACCEPTED_INTERRUPTION,
            FullDuplexPhase.REJECTED_INTERRUPTION,
            FullDuplexPhase.PROVISIONAL_DUCK,
            FullDuplexPhase.USER_SPEECH_OPEN,
            FullDuplexPhase.USER_TURN_PENDING,
        }
    ),
    FullDuplexPhase.ACCEPTED_INTERRUPTION: frozenset(
        {
            FullDuplexPhase.EVIDENCE_ARBITRATION,
            FullDuplexPhase.REJECTED_INTERRUPTION,
            FullDuplexPhase.USER_SPEECH_OPEN,
            FullDuplexPhase.USER_TURN_PENDING,
            FullDuplexPhase.USER_TURN_COMMITTED,
        }
    ),
    FullDuplexPhase.REJECTED_INTERRUPTION: frozenset(
        {
            FullDuplexPhase.PROVISIONAL_DUCK,
            FullDuplexPhase.EVIDENCE_ARBITRATION,
            FullDuplexPhase.ACCEPTED_INTERRUPTION,
            FullDuplexPhase.USER_SPEECH_OPEN,
            FullDuplexPhase.USER_TURN_PENDING,
        }
    ),
    FullDuplexPhase.USER_TURN_PENDING: frozenset(
        {
            FullDuplexPhase.USER_SPEECH_OPEN,
            FullDuplexPhase.PROVISIONAL_DUCK,
            FullDuplexPhase.EVIDENCE_ARBITRATION,
            FullDuplexPhase.ACCEPTED_INTERRUPTION,
            FullDuplexPhase.REJECTED_INTERRUPTION,
            FullDuplexPhase.USER_TURN_COMMITTED,
        }
    ),
    FullDuplexPhase.USER_TURN_COMMITTED: frozenset({FullDuplexPhase.USER_SPEECH_OPEN}),
    FullDuplexPhase.USER_TURN_REJECTED: frozenset({FullDuplexPhase.USER_SPEECH_OPEN}),
}


def _is_expected_transition(
    from_phase: FullDuplexPhase,
    to_phase: FullDuplexPhase,
) -> bool:
    if to_phase == from_phase or to_phase in _ALWAYS_ALLOWED_PHASES:
        return True
    return to_phase in _EXPECTED_NEXT_PHASES.get(from_phase, frozenset())


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
    """Record the unified full-duplex contract without applying side effects.

    Validation is scoped to the supplied ``TurnTimeline``. Provider callbacks
    for an old turn can legally interleave with a newly-opened turn; comparing
    both against one session-global phase creates false anomalies and makes the
    observable contract depend on callback scheduling.
    """

    def __init__(self, *, clock: Any | None = None) -> None:
        self._clock = clock or time.monotonic
        self._phase = FullDuplexPhase.IDLE
        self._transitions: list[FullDuplexTransition] = []
        self._unexpected_count = 0

    @property
    def phase(self) -> FullDuplexPhase:
        return self._phase

    @property
    def transitions(self) -> tuple[FullDuplexTransition, ...]:
        return tuple(self._transitions)

    @property
    def unexpected_transition_count(self) -> int:
        """Count of transitions outside the expected graph (observability)."""
        return self._unexpected_count

    def phase_for(self, timeline: TurnTimeline | None) -> FullDuplexPhase:
        """Return the last phase owned by one timeline (or the session fallback)."""

        if timeline is None:
            return self._phase
        state = timeline.attrs.get("full_duplex_state") or {}
        value = state.get("phase") if isinstance(state, dict) else None
        try:
            return FullDuplexPhase(value) if value else FullDuplexPhase.IDLE
        except (TypeError, ValueError):
            return FullDuplexPhase.IDLE

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
        previous_phase = self.phase_for(timeline)
        if not _is_expected_transition(previous_phase, phase):
            self._note_unexpected_transition(
                timeline, previous_phase, phase, event=event, reason=reason
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

    def _note_unexpected_transition(
        self,
        timeline: TurnTimeline | None,
        from_phase: FullDuplexPhase,
        to_phase: FullDuplexPhase,
        *,
        event: str,
        reason: str,
    ) -> None:
        # F1 guardrail: never blocks — the transition still applies. This only
        # surfaces sequences outside the expected graph so anomalies show up in
        # the timeline and quantify how fragmented the real turn state is.
        # Heuristic: treat entries as leads, not proof.
        self._unexpected_count += 1
        logger.warning(
            "[FullDuplexStateMachine] unexpected transition %s -> %s (event=%s reason=%s)",
            from_phase.value,
            to_phase.value,
            event,
            reason,
        )
        if timeline is None:
            return
        entry = {
            "from": from_phase.value,
            "to": to_phase.value,
            "event": event,
            "reason": reason,
        }
        events = list(timeline.attrs.get("full_duplex_unexpected_transitions") or ())
        events.append(entry)
        timeline.set_attr(
            "full_duplex_unexpected_transitions",
            events[-TIMELINE_TRANSITION_MAX_ITEMS:],
        )
        timeline.set_attr(
            "full_duplex_unexpected_transition_count",
            int(timeline.attrs.get("full_duplex_unexpected_transition_count") or 0) + 1,
        )

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
