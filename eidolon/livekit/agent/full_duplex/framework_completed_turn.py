"""LiveKit framework completed-turn gate for full-duplex sessions."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from ..observability import TurnTimeline
from ..session.messages import message_text
from ..session.voiceprint_reasons import (
    voiceprint_blocked_reason,
    voiceprint_error_reason,
)
from ..turn_policy import Action, InterruptIntent

if TYPE_CHECKING:
    from .pipeline import StreamingPipeline
    from .session_turn_boundary import FullDuplexSessionTurnBoundary
    from .turn_completion import FullDuplexTurnCompletion

logger = logging.getLogger("agent")


class FullDuplexFrameworkCompletedTurnGate:
    """Decide whether LiveKit's completed-turn hook may reach the LLM."""

    def __init__(
        self,
        pipeline: StreamingPipeline,
        *,
        completion: FullDuplexTurnCompletion,
        session_turns: FullDuplexSessionTurnBoundary,
    ) -> None:
        self._pipeline = pipeline
        self._completion = completion
        self._session_turns = session_turns

    async def allows_completed_turn(self, *, new_message: Any) -> bool:
        owner = self._pipeline
        completion = self._completion
        owner._ensure_runtime_defaults()
        task = getattr(owner, "_completed_turn_voiceprint_task", None)
        result = getattr(owner, "_completed_turn_voiceprint_result", None)
        timeline = getattr(owner, "_completed_turn_voiceprint_timeline", None) or getattr(
            owner, "_timeline", None
        )
        completed_transcript = message_text(new_message)
        if timeline is not None:
            timeline.set_attr(
                "framework_completed_turn",
                {
                    "text_preview": completed_transcript[:120],
                    "text_length": len(completed_transcript),
                },
            )
        if self._stop_active_interruption_framework_completed_turn(
            completed_transcript,
            timeline=timeline,
        ):
            return False
        if task is None and result is None:
            if self._stop_non_semantic_framework_completed_turn(
                completed_transcript,
                timeline=timeline,
            ):
                return False
            if self._should_defer_framework_completed_turn(completed_transcript):
                self._defer_framework_completed_turn(
                    completed_transcript=completed_transcript,
                    timeline=timeline,
                    voiceprint_reason="",
                )
                return False
            self._align_framework_completed_turn(
                completed_transcript,
                timeline=timeline,
                voiceprint_reason="",
            )
            return True
        if result is None and task is not None:
            try:
                result = await task
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                completion._record_voiceprint_commit_gate(
                    timeline,
                    allowed=False,
                    reason=voiceprint_error_reason(type(exc).__name__),
                )
                completion.clear_session_user_turn(
                    voiceprint_error_reason(type(exc).__name__)
                )
                owner._flush_turn_timeline(timeline, "voiceprint_commit_blocked")
                logger.exception("[StreamingPipeline] voiceprint gate failed in turn hook")
                return False
            owner._completed_turn_voiceprint_result = result

        allowed = bool(getattr(result, "commit_allowed", False))
        reason = str(getattr(result, "commit_reason", "") or "unknown")
        completion._record_voiceprint_commit_gate(
            timeline,
            allowed=allowed,
            reason=reason,
        )
        if allowed:
            if self._stop_active_interruption_framework_completed_turn(
                completed_transcript,
                timeline=timeline,
            ):
                return False
            if self._stop_non_semantic_framework_completed_turn(
                completed_transcript,
                timeline=timeline,
            ):
                return False
            if self._should_defer_framework_completed_turn(completed_transcript):
                self._defer_framework_completed_turn(
                    completed_transcript=completed_transcript,
                    timeline=timeline,
                    voiceprint_reason=reason,
                )
                return False
            self._align_framework_completed_turn(
                completed_transcript,
                timeline=timeline,
                voiceprint_reason=reason,
            )
            return True

        if completion._should_keep_waiting_merge_after_inconclusive_voiceprint(
            result,
            transcript=completed_transcript,
        ):
            completion._defer_inconclusive_voiceprint_result(
                transcript=completed_transcript,
                timeline=timeline,
                reason=reason,
            )
            return False

        owner._suppress_transcripts_until_next_speech = True
        completion.clear_session_user_turn(voiceprint_blocked_reason(reason))
        owner._flush_turn_timeline(timeline, "voiceprint_commit_blocked")
        logger.info(
            "[StreamingPipeline] voiceprint gate stopped completed turn reason=%s transcript=%r",
            reason,
            completed_transcript[:80],
        )
        return False

    def _eot_thinks_turn_complete(self) -> bool:
        owner = self._pipeline
        eot_model = owner._get_eot_model()
        if eot_model is None:
            return True
        score = float(
            getattr(
                eot_model,
                "current_eot_score",
                getattr(eot_model, "_current_eot_score", 1.0),
            )
            or 0.0
        )
        return score >= float(owner._turn_policy.eot.eot_unlikely_threshold)

    def _should_defer_framework_completed_turn(self, transcript: str) -> bool:
        owner = self._pipeline
        owner._ensure_user_turn_coordinator()
        candidate = owner._user_turns.active
        if candidate is None:
            return False
        if candidate.state in {"committed", "rejected"}:
            return False
        if owner._user_turns.should_wait_for_deferred_voiceprint_merge():
            return True
        if owner._user_turns.should_wait_for_statement_sequence_merge():
            return True
        if self._eot_thinks_turn_complete():
            return False
        selected = candidate.selected_text or transcript
        return self._completion._looks_like_short_statement_continuation(selected)

    def _defer_framework_completed_turn(
        self,
        *,
        completed_transcript: str,
        timeline: TurnTimeline | None,
        voiceprint_reason: str,
    ) -> None:
        owner = self._pipeline
        completion = self._completion
        completion.clear_session_user_turn("framework_completed_wait_for_continuation")
        owner._ensure_user_turn_coordinator()
        decision = owner._user_turns.defer_framework_completed(
            transcript=completed_transcript,
            reason="framework_completed_wait_for_continuation",
            timeline=timeline,
            voiceprint_reason=voiceprint_reason,
        )
        if timeline is not None:
            timeline.set_attr(
                "framework_completed_deferred",
                {
                    "reason": decision.reason,
                    "state": "waiting_merge",
                    "text_preview": decision.transcript[:120],
                    "text_length": len(decision.transcript),
                },
            )
            owner._append_turn_timeline_snapshot(
                timeline,
                "framework_completed_waiting_merge",
            )
        completion.schedule_deferred_low_eot_commit(
            verify_task=None,
            eot_model=owner._get_eot_model(),
            transcript=decision.transcript or completed_transcript,
            timeline=timeline,
            delay_sec=decision.delay_sec,
        )
        owner._completed_turn_voiceprint_task = None
        owner._completed_turn_voiceprint_result = None
        owner._completed_turn_voiceprint_timeline = None
        logger.info(
            "[StreamingPipeline] deferred framework completed turn "
            "for continuation transcript=%r voiceprint_reason=%s",
            completed_transcript[:80],
            voiceprint_reason,
        )

    def _align_framework_completed_turn(
        self,
        completed_transcript: str,
        *,
        timeline: TurnTimeline | None,
        voiceprint_reason: str,
    ) -> None:
        owner = self._pipeline
        self._completion.cancel_deferred_low_eot_commit("framework_completed_turn")
        owner._ensure_user_turn_coordinator()
        decision = owner._user_turns.mark_framework_completed(
            transcript=completed_transcript,
            reason="framework_completed_turn",
            timeline=timeline,
            voiceprint_reason=voiceprint_reason,
        )
        canonical = decision.transcript or completed_transcript
        self._session_turns.publish_canonical_user_text(
            canonical,
            source="framework_completed_turn",
            timeline=timeline,
        )

    @staticmethod
    def _non_semantic_completed_turn_reason(
        timeline: TurnTimeline | None,
    ) -> str:
        if timeline is None:
            return ""
        decision = timeline.attrs.get("decision")
        if not isinstance(decision, dict):
            return ""
        action = str(decision.get("action") or "")
        intent = str(decision.get("intent") or "")
        if action == "rollback":
            return f"non_semantic_completed_turn:{intent or action}"
        if intent in {"backchannel", "noise", "hard_stop"}:
            return f"non_semantic_completed_turn:{intent}"
        return ""

    def _stop_non_semantic_framework_completed_turn(
        self,
        completed_transcript: str,
        *,
        timeline: TurnTimeline | None,
    ) -> bool:
        owner = self._pipeline
        completion = self._completion
        stop_reason = self._non_semantic_completed_turn_reason(timeline)
        if not stop_reason:
            return False
        completion.cancel_deferred_low_eot_commit(stop_reason)
        owner._ensure_user_turn_coordinator()
        owner._user_turns.reject_active(stop_reason)
        completion.clear_session_user_turn(stop_reason)
        owner._flush_turn_timeline(timeline, stop_reason)
        logger.info(
            "[StreamingPipeline] stopped framework completed turn reason=%s transcript=%r",
            stop_reason,
            completed_transcript[:80],
        )
        return True

    def _stop_active_interruption_framework_completed_turn(
        self,
        completed_transcript: str,
        *,
        timeline: TurnTimeline | None,
    ) -> bool:
        owner = self._pipeline
        completion = self._completion
        interruption_owner = getattr(owner, "_interruption_orchestrator", None)
        if (
            interruption_owner is None
            or not interruption_owner.blocks_framework_completed_turn()
        ):
            return False
        decision = self._decide_from_completed_turn_evidence(
            completed_transcript,
            timeline=timeline,
        )
        if decision is not None and self._completed_turn_can_resolve(decision):
            owner._ensure_decision_effect_applier()
            owner._decision_effects.apply(
                decision,
                resolved_reason="framework_completed_interruption_evidence",
                eot_score=self._current_eot_score(),
                transcript=completed_transcript,
                vad_active=False,
            )
            completion.clear_session_user_turn(
                f"interruption_owner_resolved:{decision.action.value}"
            )
            return True
        reason = "interruption_owner_waiting_for_evidence"
        completion.cancel_deferred_low_eot_commit(reason)
        completion.clear_session_user_turn(reason)
        if timeline is not None:
            timeline.set_attr(
                "framework_completed_blocked_by_interruption_owner",
                {
                    "reason": reason,
                    "state": interruption_owner.state.value,
                    "text_preview": completed_transcript[:120],
                    "text_length": len(completed_transcript),
                },
            )
            owner._append_turn_timeline_snapshot(timeline, reason)
        logger.info(
            "[StreamingPipeline] blocked framework completed turn while "
            "interruption owner waits reason=%s state=%s transcript=%r",
            reason,
            interruption_owner.state.value,
            completed_transcript[:80],
        )
        return True

    @staticmethod
    def _completed_turn_can_resolve(decision: Any) -> bool:
        if decision.action is Action.ROLLBACK:
            return True
        if decision.action is not Action.CANCEL:
            return False
        if decision.intent is InterruptIntent.HARD_STOP:
            return True
        return bool(decision.topic_switch_hint or decision.correction_hint)

    def _decide_from_completed_turn_evidence(
        self,
        completed_transcript: str,
        *,
        timeline: TurnTimeline | None,
    ):
        text = completed_transcript.strip()
        if not text:
            return None
        owner = self._pipeline
        interruption_owner = getattr(owner, "_interruption_orchestrator", None)
        if interruption_owner is None:
            return None
        decision = interruption_owner.decide_from_transcript(
            owner._turn_runtime,
            text,
            self._current_eot_score(),
            vad_active=False,
            agent_speaking=True,
            is_final=True,
        )
        if timeline is not None:
            timeline.set_attr(
                "framework_completed_interruption_evidence",
                {
                    "action": decision.action.value,
                    "reason": decision.reason,
                    "intent": (
                        decision.intent.value if decision.intent is not None else None
                    ),
                    "text_preview": text[:120],
                    "text_length": len(text),
                },
            )
        return decision

    def _current_eot_score(self) -> float:
        owner = self._pipeline
        eot_model = owner._get_eot_model()
        return float(
            getattr(
                eot_model,
                "current_eot_score",
                getattr(eot_model, "_current_eot_score", 0.0),
            )
            or 0.0
        )
