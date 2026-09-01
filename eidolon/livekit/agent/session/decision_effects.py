"""Apply turn-policy decisions to LiveKit session side effects."""

from __future__ import annotations

from collections.abc import Callable

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.turn_policy import Action, Decision, TurnPolicyRuntime


class DecisionEffectApplier:
    """Execute side effects implied by a turn-policy ``Decision``.

    The policy layer decides *what* should happen. This class owns the
    session-facing effects that follow from that decision:

    - timeline decision attributes,
    - cancel / rollback callbacks supplied by ``StreamingPipeline``.
    """

    def __init__(
        self,
        *,
        turn_runtime: TurnPolicyRuntime,
        get_timeline: Callable[[], TurnTimeline | None],
        on_cancel: Callable[[], None],
        on_rollback: Callable[[str, bool], None],
        on_hold: Callable[[Decision, str, float | None, bool | None], None] | None = None,
        on_decision: Callable[..., object] | None = None,
        record_full_duplex_transition: Callable[..., object] | None = None,
    ) -> None:
        self._turn_runtime = turn_runtime
        self._get_timeline = get_timeline
        self._on_cancel = on_cancel
        self._on_rollback = on_rollback
        self._on_hold = on_hold
        self._on_decision = on_decision
        self._record_full_duplex_transition = record_full_duplex_transition

    def apply(
        self,
        decision: Decision,
        *,
        resolved_reason: str | None = None,
        eot_score: float | None = None,
        transcript: str = "",
        vad_active: bool | None = None,
    ) -> None:
        self.record_decision_attrs(
            decision,
            resolved_reason=resolved_reason,
            eot_score=eot_score,
            transcript=transcript,
            vad_active=vad_active,
        )
        if self._on_decision is not None:
            self._on_decision(
                decision,
                source=resolved_reason or "turn_policy",
                transcript=transcript,
                vad_active=vad_active,
                eot_score=eot_score,
            )
        if self._record_full_duplex_transition is not None:
            self._record_full_duplex_transition(
                decision,
                source=resolved_reason or "turn_policy",
                transcript=transcript,
                vad_active=vad_active,
                eot_score=eot_score,
            )
        if decision.action is Action.CANCEL:
            self._on_cancel()
            return
        if decision.action is Action.ROLLBACK:
            self._on_rollback(
                resolved_reason or decision.reason,
                decision.rollback_drop_buffered,
            )
            return
        if decision.action is Action.HOLD and self._on_hold is not None:
            self._on_hold(decision, transcript, eot_score, vad_active)

    def record_decision_attrs(
        self,
        decision: Decision,
        *,
        source: str = "turn_policy",
        resolved_reason: str | None = None,
        eot_score: float | None = None,
        transcript: str = "",
        vad_active: bool | None = None,
    ) -> None:
        timeline = self._get_timeline()
        if timeline is None:
            return
        timeline.record_decision(
            action=decision.action.value,
            reason=decision.reason,
            rollback_drop_buffered=decision.rollback_drop_buffered,
            intent=decision.intent.value if decision.intent is not None else None,
            intent_source=decision.intent_source,
            intent_confidence=decision.intent_confidence,
            topic_switch_hint=decision.topic_switch_hint,
            correction_hint=decision.correction_hint,
            tier=decision.tier or None,
            tier_reason=decision.tier_reason or None,
            source=source,
            resolved_reason=resolved_reason,
            eot_score=eot_score,
            transcript_preview=transcript[:120],
            vad_active=vad_active,
            hold_recheck_ms=decision.hold_recheck_ms,
        )
