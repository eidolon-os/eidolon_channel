"""Semantic interrupt handling for streaming sessions."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.output import DuckingStats
from eidolon.livekit.agent.turn_policy import (
    Decision,
    InterruptIntent,
    TurnPolicyRuntime,
)

logger = logging.getLogger("agent.session.semantic_interrupt")


class SemanticInterruptHandler:
    """Resolve STT transcript updates into interrupt side effects.

    ``turn_policy`` decides the tier, intent and action. This handler owns the
    hot-path wiring around that decision: strong-stop fast path, duck-active
    resolution, fallback soft-interrupt staging, timeline attributes and
    remote-brain turn-control metadata.
    """

    def __init__(
        self,
        *,
        get_eot_model: Callable[[], Any],
        turn_runtime: TurnPolicyRuntime,
        get_timeline: Callable[[], TurnTimeline | None],
        get_duck_active: Callable[[], bool],
        get_duck_stats: Callable[[], DuckingStats],
        get_vad_active: Callable[[], bool],
        soft_interrupt_active: Callable[[], bool],
        soft_interrupt_timeout: Callable[[], float],
        apply_decision: Callable[..., None],
        interrupt_current_turn: Callable[[], None],
        enter_soft_interrupt: Callable[[], None],
        decide_from_transcript: Callable[..., Decision] | None = None,
    ) -> None:
        self._get_eot_model = get_eot_model
        self._turn_runtime = turn_runtime
        self._get_timeline = get_timeline
        self._get_duck_active = get_duck_active
        self._get_duck_stats = get_duck_stats
        self._get_vad_active = get_vad_active
        self._soft_interrupt_active = soft_interrupt_active
        self._soft_interrupt_timeout = soft_interrupt_timeout
        self._apply_decision = apply_decision
        self._interrupt_current_turn = interrupt_current_turn
        self._enter_soft_interrupt = enter_soft_interrupt
        self._decide_from_transcript = decide_from_transcript or turn_runtime.decide_from_transcript

    def run(self, text: str, *, is_final: bool = False) -> None:
        """Run the synchronous EOT semantic check for one transcript update."""
        if not text or not text.strip():
            return

        eot_model = self._get_eot_model()
        duck_active = self._get_duck_active()
        vad_active = self._get_vad_active()
        score = eot_model.current_eot_score

        should_cut = eot_model.should_interrupt(
            text,
            vad_active=vad_active,
            is_final=is_final,
        )

        if duck_active:
            self._handle_duck_active(
                text,
                score=score,
                vad_active=vad_active,
                is_final=is_final,
            )
            return

        if self._handle_fallback_semantic(
            text,
            score=score,
            vad_active=vad_active,
            is_final=is_final,
        ):
            return

        if self._soft_interrupt_active():
            logger.info(
                "[SemanticInterruptHandler] EOT: already in soft interrupt, waiting. text=%r",
                text[:80],
            )
            return

        if should_cut:
            self._handle_fallback_eot_score(
                text,
                score=score,
                vad_active=vad_active,
                hard_threshold=eot_model.hard_interrupt_score_threshold,
            )
        else:
            logger.info(
                "[SemanticInterruptHandler] EOT: should_interrupt=False, text=%r",
                text[:80],
            )

    def _handle_duck_active(
        self,
        text: str,
        *,
        score: float,
        vad_active: bool,
        is_final: bool,
    ) -> None:
        decision = self._decide_from_transcript(
            text,
            score,
            vad_active=vad_active,
            agent_speaking=True,
            is_final=is_final,
        )
        stats = self._get_duck_stats()
        logger.info(
            "[SemanticInterruptHandler] EOT(duck): decision=%s reason=%s "
            "suspend_ms=%.0f score=%.2f text=%r",
            decision.action.value,
            decision.reason,
            stats.suspend_ms,
            score,
            text[:80],
        )
        self._apply_decision(
            decision,
            eot_score=score,
            transcript=text,
            vad_active=vad_active,
        )

    def _handle_fallback_semantic(
        self,
        text: str,
        *,
        score: float,
        vad_active: bool,
        is_final: bool,
    ) -> bool:
        semantic_decision = self._decide_from_transcript(
            text,
            score,
            vad_active=vad_active,
            agent_speaking=True,
            is_final=is_final,
        )
        if not self._is_fallback_semantic_decision(semantic_decision):
            return False

        logger.info(
            "[SemanticInterruptHandler] EOT(fallback semantic): decision=%s "
            "reason=%s score=%.2f text=%r",
            semantic_decision.action.value,
            semantic_decision.reason,
            score,
            text[:80],
        )
        self._apply_decision(
            semantic_decision,
            eot_score=score,
            transcript=text,
            vad_active=vad_active,
        )
        return True

    @staticmethod
    def _is_fallback_semantic_decision(decision: Decision) -> bool:
        if decision.topic_switch_hint or decision.correction_hint:
            return True
        return decision.intent in (
            InterruptIntent.HARD_STOP,
            InterruptIntent.TOPIC_SWITCH,
            InterruptIntent.CORRECTION,
            InterruptIntent.BACKCHANNEL,
            InterruptIntent.NOISE,
        )

    def _handle_fallback_eot_score(
        self,
        text: str,
        *,
        score: float,
        vad_active: bool,
        hard_threshold: float,
    ) -> None:
        timeline = self._get_timeline()
        if score >= hard_threshold:
            logger.info(
                "[SemanticInterruptHandler] EOT: score=%.2f >= %.2f -> "
                "hard interrupt (skip soft stage). text=%r",
                score,
                hard_threshold,
                text[:80],
            )
            if timeline is not None:
                timeline.record_decision(
                    action="cancel",
                    reason="fallback_eot_hard_score",
                    rollback_drop_buffered=False,
                    source="eot_fallback",
                    eot_score=score,
                    transcript_preview=text[:120],
                    vad_active=vad_active,
                )
                timeline.mark_interrupt_resolved("cancel")
                timeline.set_attr("cancel_reason", "fallback_eot_hard_score")
            self._interrupt_current_turn()
            return

        logger.info(
            "[SemanticInterruptHandler] EOT: score=%.2f -> soft interrupt (timeout=%.2fs). text=%r",
            score,
            self._soft_interrupt_timeout(),
            text[:80],
        )
        if timeline is not None:
            timeline.record_decision(
                action="hold",
                reason="fallback_eot_soft_interrupt",
                rollback_drop_buffered=False,
                source="eot_fallback",
                eot_score=score,
                transcript_preview=text[:120],
                vad_active=vad_active,
            )
        self._enter_soft_interrupt()
