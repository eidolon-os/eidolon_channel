"""Pure playback-overlap evidence contract for full-duplex turns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..turn_policy import Action, InterruptIntent


@dataclass(frozen=True)
class PlaybackTurnEvidenceResolution:
    """Decision-level result for a completed turn that overlaps playback.

    This object deliberately contains no LiveKit/session side effects. Runtime
    adapters can use it to decide whether to apply an interruption decision and
    whether a semantic redirect should continue to the LLM as the next user turn.
    """

    should_apply: bool
    reason: str
    continue_to_llm: bool = False


def resolve_playback_turn_decision(
    decision: Any | None,
) -> PlaybackTurnEvidenceResolution:
    """Classify a turn-policy decision for playback-overlap evidence."""

    if decision is None:
        return PlaybackTurnEvidenceResolution(
            should_apply=False,
            reason="no_decision",
        )
    if not playback_turn_decision_can_resolve(decision):
        action = getattr(getattr(decision, "action", None), "value", "")
        return PlaybackTurnEvidenceResolution(
            should_apply=False,
            reason=f"decision_not_resolvable:{action or 'unknown'}",
        )
    return PlaybackTurnEvidenceResolution(
        should_apply=True,
        reason=str(getattr(decision, "reason", "") or "resolved"),
        continue_to_llm=playback_turn_decision_continues_to_llm(decision),
    )


def playback_turn_decision_continues_to_llm(decision: Any) -> bool:
    """True for confirmed normal interruptions that become a user turn."""

    return (
        decision.action is Action.CANCEL
        and decision.intent is InterruptIntent.NORMAL_INTERRUPT
    )


def playback_turn_decision_can_resolve(decision: Any) -> bool:
    """True when playback-overlap evidence may terminally resolve a turn."""

    if decision.action is Action.ROLLBACK:
        return True
    if decision.action is not Action.CANCEL:
        return False
    if decision.intent is InterruptIntent.HARD_STOP:
        return True
    return decision.intent is InterruptIntent.NORMAL_INTERRUPT
