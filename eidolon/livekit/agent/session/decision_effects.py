"""Apply turn-policy decisions to LiveKit session side effects."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.turn_policy import Action, Decision, TurnPolicyRuntime

logger = logging.getLogger("agent.session.decision_effects")


class DecisionEffectApplier:
    """Execute side effects implied by a turn-policy ``Decision``.

    The policy layer decides *what* should happen. This class owns the
    session-facing effects that follow from that decision:

    - timeline decision attributes,
    - remote-brain turn-control metadata,
    - cancel / rollback callbacks supplied by ``StreamingPipeline``.
    """

    def __init__(
        self,
        *,
        factory: Any,
        turn_runtime: TurnPolicyRuntime,
        get_timeline: Callable[[], TurnTimeline | None],
        on_cancel: Callable[[], None],
        on_rollback: Callable[[str, bool], None],
    ) -> None:
        self._factory = factory
        self._turn_runtime = turn_runtime
        self._get_timeline = get_timeline
        self._on_cancel = on_cancel
        self._on_rollback = on_rollback

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
        if decision.action is Action.CANCEL:
            self._publish_control_signal(decision)
            self._on_cancel()
            return
        if decision.action is Action.ROLLBACK:
            self._publish_control_signal(decision)
            self._on_rollback(
                resolved_reason or decision.reason,
                decision.rollback_drop_buffered,
            )
            return

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
        )

    def publish_turn_control(self, metadata: dict[str, object]) -> None:
        """Attach control hints to the next remote-brain turn when supported."""
        try:
            llm_plugin = getattr(self._factory.llm, "llm", None)
            setter = getattr(llm_plugin, "set_turn_control_metadata", None)
            if setter is not None:
                setter(metadata)
        except Exception:
            logger.debug(
                "[DecisionEffectApplier] failed to publish turn_control metadata",
                exc_info=True,
            )

    def _publish_control_signal(self, decision: Decision) -> None:
        signal = self._turn_runtime.control_signal_from_decision(decision)
        metadata = signal.as_metadata()
        self.publish_turn_control(metadata)
        timeline = self._get_timeline()
        if timeline is not None:
            timeline.set_attr("turn_control", metadata)
