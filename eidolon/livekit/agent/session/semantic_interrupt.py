"""Semantic interrupt handling for streaming sessions."""

from __future__ import annotations

import logging
import asyncio
from collections.abc import Callable
from typing import Any

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.output import DuckingStats
from eidolon.livekit.agent.turn_policy import (
    Decision,
    InterruptIntent,
    InterruptIntentResult,
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
        intent_classifier: Any = None,
        get_candidate_scope: Callable[[], Any] = lambda: None,
        get_final_transcript: Callable[[], str] = lambda: "",
        get_assistant_text: Callable[[], str] = lambda: "",
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
        self._intent_classifier = intent_classifier
        self._get_candidate_scope = get_candidate_scope
        self._get_final_transcript = get_final_transcript
        self._get_assistant_text = get_assistant_text
        self._intent_task: asyncio.Task | None = None
        self._intent_tasks: set[asyncio.Task] = set()
        self._intent_key: Any = None
        self._intent_result: InterruptIntentResult | None = None
        self._closed = False

    @property
    def uses_model_intent(self) -> bool:
        return self._turn_runtime.config.interrupt.intent_provider == "llm"

    async def wait_for_pending_intent(self) -> None:
        # Endpointing cancellation must not cancel the candidate's shared
        # inference. A revised final owns a new task and its own timeout.
        # Superseded candidates must not delay the current endpoint.
        task = self._intent_task
        if task is not None and not task.done() and self._owns_current_candidate(self._intent_key):
            await asyncio.shield(task)

    def _owns_current_candidate(self, key: Any) -> bool:
        return (
            key is not None and not self._closed and self._intent_key == key
            and self._get_candidate_scope() == key[0]
            and self._get_final_transcript() == key[1]
            and self._get_duck_active()
        )

    async def aclose(self) -> None:
        self._closed = True
        tasks = list(self._intent_tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._intent_task = None

    def run(self, text: str, *, is_final: bool = False) -> None:
        """Route one transcript through the configured interruption evidence path."""
        if not text or not text.strip():
            return

        eot_model = self._get_eot_model()
        duck_active = self._get_duck_active()
        vad_active = self._get_vad_active()
        score = eot_model.current_eot_score

        if self.uses_model_intent:
            if duck_active and not self._closed:
                self._handle_model_intent(text, score=score, vad_active=vad_active, is_final=is_final)
            return

        if duck_active:
            self._handle_duck_active(
                text,
                score=score,
                vad_active=vad_active,
                is_final=is_final,
            )
            return

        # The suspended-output path already has a policy owner. Legacy EOT
        # evaluation can mutate its score and cooldown, so run it only where
        # its verdict is actually consumed.
        should_cut = eot_model.should_interrupt(
            text,
            vad_active=vad_active,
            is_final=is_final,
        )
        score = eot_model.current_eot_score

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
                text, score=score, vad_active=vad_active,
                hard_threshold=eot_model.hard_interrupt_score_threshold,
            )
        else:
            logger.info("[SemanticInterruptHandler] EOT: should_interrupt=False, text=%r", text[:80])

    def _handle_model_intent(self, text: str, *, score: float, vad_active: bool, is_final: bool) -> None:
        scope = self._get_candidate_scope()
        if scope is None:
            return
        key = (scope, text)
        if key != self._intent_key:
            if self._intent_task is not None:
                self._intent_task.cancel()
            self._intent_key = key
            self._intent_result = None
            self._intent_task = None
        if is_final and self._intent_task is None and self._intent_result is None:
            self._intent_task = asyncio.create_task(
                self._classify_final(key, text, self._get_assistant_text()),
                name="interrupt_intent",
            )
            self._intent_tasks.add(self._intent_task)
            self._intent_task.add_done_callback(self._intent_tasks.discard)
        decision = self._decide_from_transcript(
            text, score, vad_active=vad_active, agent_speaking=True,
            is_final=is_final, intent_result=self._intent_result,
        )
        self._apply_decision(decision, transcript=text, vad_active=vad_active, eot_score=score)

    async def _classify_final(self, key: Any, text: str, assistant_text: str) -> None:
        try:
            timeout = max(.001, self._turn_runtime.config.interrupt.intent_timeout_ms / 1000)
            async with asyncio.timeout(timeout):
                result = await self._intent_classifier.classify(text, assistant_text=assistant_text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("[SemanticInterruptHandler] intent unavailable: %s", type(exc).__name__)
            result = InterruptIntentResult(InterruptIntent.UNCERTAIN, 0.0, "unavailable", type(exc).__name__)
        # Cancellation alone is insufficient: a provider can finish after
        # supersession. Recheck the acoustic generation, response and final text.
        if not self._owns_current_candidate(key):
            return
        self._intent_result = result
        self.run(text, is_final=True)

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
