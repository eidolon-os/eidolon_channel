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
from .playback_turn_evidence import resolve_playback_turn_decision
from .state_machine import FullDuplexPhase

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

    async def allows_completed_turn(
        self,
        *,
        turn_ctx: Any,
        new_message: Any,
    ) -> bool:
        owner = self._pipeline
        completion = self._completion
        owner._ensure_runtime_defaults()
        voiceprint_turn = completion.completed_voiceprint_turn(
            fallback_timeline=getattr(owner, "_timeline", None)
        )
        task = voiceprint_turn.task
        result = voiceprint_turn.result
        timeline = voiceprint_turn.timeline
        completed_transcript = message_text(new_message)
        if timeline is not None:
            timeline.mark("framework_completed_turn_at")
            timeline.set_attr(
                "framework_completed_turn",
                {
                    "text_preview": completed_transcript[:120],
                    "text_length": len(completed_transcript),
                },
            )
            self._record_completed_gate_event(
                timeline,
                stage="received",
                action="observe",
                reason="framework_completed_turn",
                transcript=completed_transcript,
            )
        if task is None and result is None:
            return self._route_allowed_framework_completed_turn(
                turn_ctx=turn_ctx,
                completed_transcript=completed_transcript,
                new_message=new_message,
                timeline=timeline,
                voiceprint_reason="",
            )

        if result is None and task is not None:
            try:
                result = await task
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                reject_reason = voiceprint_error_reason(type(exc).__name__)
                completion._record_voiceprint_commit_gate(
                    timeline,
                    allowed=False,
                    reason=reject_reason,
                )
                self._reject_framework_completed_candidate(
                    reject_reason,
                    transcript=completed_transcript,
                    timeline=timeline,
                    event="voiceprint_gate_error",
                    flush_reason="voiceprint_commit_blocked",
                )
                logger.exception("[StreamingPipeline] voiceprint gate failed in turn hook")
                return False
            completion.remember_completed_voiceprint_result(result)

        allowed = bool(getattr(result, "commit_allowed", False))
        reason = str(getattr(result, "commit_reason", "") or "unknown")
        completion._record_voiceprint_commit_gate(
            timeline,
            allowed=allowed,
            reason=reason,
        )
        if allowed:
            return self._route_allowed_framework_completed_turn(
                turn_ctx=turn_ctx,
                completed_transcript=completed_transcript,
                new_message=new_message,
                timeline=timeline,
                voiceprint_reason=reason,
            )

        if "context_error" in reason:
            self._session_turns.notify_context_error_once(reason)
        owner._set_suppress_transcripts_until_next_speech(True, reason="voiceprint_commit_blocked")
        self._reject_framework_completed_candidate(
            voiceprint_blocked_reason(reason),
            transcript=completed_transcript,
            timeline=timeline,
            event="voiceprint_gate_rejected",
            flush_reason="voiceprint_commit_blocked",
        )
        logger.info(
            "[StreamingPipeline] voiceprint gate stopped completed turn reason=%s transcript=%r",
            reason,
            completed_transcript[:80],
        )
        return False

    def _reject_framework_completed_candidate(
        self,
        reason: str,
        *,
        transcript: str,
        timeline: TurnTimeline | None,
        event: str,
        flush_reason: str,
    ) -> None:
        owner = self._pipeline
        owner._ensure_user_turn_coordinator()
        decision = owner._user_turns.reject_active(reason)
        self._finish_rejected_turn(
            decision=decision,
            reason=reason,
            transcript=transcript,
            timeline=timeline,
            event=event,
            flush_reason=flush_reason,
        )

    def _route_allowed_framework_completed_turn(
        self,
        *,
        turn_ctx: Any,
        completed_transcript: str,
        new_message: Any,
        timeline: TurnTimeline | None,
        voiceprint_reason: str,
    ) -> bool:
        owner = self._pipeline
        owner._ensure_user_turn_coordinator()
        readiness = owner._user_turns.framework_completion_readiness(completed_transcript)
        if not readiness.ready:
            if timeline is not None:
                timeline.set_attr(
                    "framework_completed_deferred",
                    {
                        "reason": readiness.reason,
                        "framework_text_preview": completed_transcript[:120],
                        "pending_text_preview": readiness.pending_transcript[:120],
                    },
                )
                self._record_completed_gate_event(
                    timeline,
                    stage="candidate_readiness",
                    action="defer",
                    reason=readiness.reason,
                    transcript=completed_transcript,
                    pending_text_preview=readiness.pending_transcript[:120],
                )
            logger.info(
                "[StreamingPipeline] deferred stale framework completion "
                "reason=%s framework=%r pending=%r",
                readiness.reason,
                completed_transcript[:80],
                readiness.pending_transcript[:80],
            )
            return False
        candidate_transcript = owner._user_turns.prepare_framework_completed(
            transcript=completed_transcript,
            timeline=timeline,
        )
        if timeline is not None:
            timeline.set_attr(
                "framework_completed_candidate",
                {
                    "framework_text_preview": completed_transcript[:120],
                    "framework_text_length": len(completed_transcript),
                    "canonical_text_preview": candidate_transcript[:120],
                    "canonical_text_length": len(candidate_transcript),
                },
            )
            self._record_completed_gate_event(
                timeline,
                stage="candidate_assembled",
                action="evaluate",
                reason="canonical_user_turn_candidate",
                transcript=candidate_transcript,
                framework_text_preview=completed_transcript[:120],
                framework_text_length=len(completed_transcript),
            )
        interruption_allowed = self._resolve_completed_turn_interruption_evidence(
            completed_transcript=candidate_transcript,
            timeline=timeline,
        )
        if interruption_allowed is False:
            return False
        return self._align_framework_completed_turn(
            turn_ctx=turn_ctx,
            completed_transcript=completed_transcript,
            new_message=new_message,
            timeline=timeline,
            voiceprint_reason=voiceprint_reason,
        )

    def _resolve_completed_turn_interruption_evidence(
        self,
        *,
        completed_transcript: str,
        timeline: TurnTimeline | None,
    ) -> bool | None:
        active_resolution = self._resolve_active_interruption_framework_completed_turn(
            completed_transcript,
            timeline=timeline,
        )
        if active_resolution is not None:
            return active_resolution
        return self._resolve_recorded_interruption_verdict(timeline=timeline)

    def _resolve_recorded_interruption_verdict(
        self,
        *,
        timeline: TurnTimeline | None,
    ) -> bool | None:
        """Consume only the interruption owner's typed terminal result."""

        if timeline is None:
            return None
        owner = self._pipeline
        interruption_owner = getattr(owner, "_interruption_orchestrator", None)
        if interruption_owner is None:
            return None
        verdict = interruption_owner.verdict_for(timeline.turn_id)
        if verdict is None:
            return None
        self._record_completed_gate_event(
            timeline,
            stage="interruption_verdict",
            action="continue" if verdict.continue_to_llm else "reject",
            reason=verdict.reason,
            transcript=verdict.transcript,
            verdict=verdict.action.value,
            intent=verdict.intent,
            turn_policy_action=verdict.turn_policy_action,
        )
        if verdict.continue_to_llm:
            return True
        owner._ensure_user_turn_coordinator()
        reason = f"interruption_verdict:{verdict.action.value}:{verdict.reason}"
        decision = owner._user_turns.reject_active(reason)
        self._finish_rejected_turn(
            decision=decision,
            reason=reason,
            transcript=verdict.transcript,
            timeline=timeline,
            event="interruption_verdict_rejected",
            flush_reason=f"interruption_{verdict.action.value}",
        )
        return False

    def _align_framework_completed_turn(
        self,
        completed_transcript: str,
        *,
        turn_ctx: Any,
        new_message: Any,
        timeline: TurnTimeline | None,
        voiceprint_reason: str,
    ) -> bool:
        owner = self._pipeline
        owner._ensure_user_turn_coordinator()
        decision = owner._user_turns.mark_framework_completed(
            transcript=completed_transcript,
            reason="framework_completed_turn",
            timeline=timeline,
            voiceprint_reason=voiceprint_reason,
        )
        if decision.action == "none":
            if timeline is not None:
                timeline.set_attr(
                    "framework_completed_duplicate",
                    {
                        "reason": decision.reason,
                        "text_preview": (decision.transcript or completed_transcript)[:120],
                        "text_length": len(decision.transcript or completed_transcript),
                    },
                )
                self._record_completed_gate_event(
                    timeline,
                    stage="framework_completed_turn",
                    action="skip",
                    reason=decision.reason,
                    transcript=decision.transcript or completed_transcript,
                )
            logger.info(
                "[StreamingPipeline] skipped duplicate framework completed turn "
                "reason=%s transcript=%r",
                decision.reason,
                completed_transcript[:80],
            )
            return False
        if decision.action == "reject":
            self._finish_rejected_turn(
                decision=decision,
                reason=decision.reason,
                transcript=decision.transcript or completed_transcript,
                timeline=timeline,
                event="framework_completed_rejected",
                flush_reason=decision.reason,
            )
            logger.info(
                "[StreamingPipeline] rejected framework completed turn reason=%s transcript=%r",
                decision.reason,
                completed_transcript[:80],
            )
            return False
        canonical = decision.transcript or completed_transcript
        self._session_turns.publish_canonical_user_text(
            new_message,
            canonical,
            source="framework_completed_turn",
            timeline=timeline,
        )
        self._session_turns.consume_interrupted_context(
            turn_ctx,
            timeline=timeline,
        )
        _record_contract_transition(
            owner,
            FullDuplexPhase.USER_TURN_COMMITTED,
            event="framework_completed_turn",
            reason=decision.reason,
            side_effect="irreversible",
            transcript=canonical,
            timeline=timeline,
        )
        owner._claim_agent_output_timeline(timeline)
        return True

    def _finish_rejected_turn(
        self,
        *,
        decision: Any,
        reason: str,
        transcript: str,
        timeline: TurnTimeline | None,
        event: str,
        flush_reason: str,
    ) -> None:
        """Close every rejected product turn through one terminal boundary."""

        owner = self._pipeline
        if decision.action == "reject":
            _record_contract_transition(
                owner,
                FullDuplexPhase.USER_TURN_REJECTED,
                event=event,
                reason=reason,
                side_effect="irreversible",
                transcript=decision.transcript or transcript,
                timeline=timeline,
            )
        owner._flush_turn_timeline(timeline, flush_reason)

    def _resolve_active_interruption_framework_completed_turn(
        self,
        completed_transcript: str,
        *,
        timeline: TurnTimeline | None,
    ) -> bool | None:
        owner = self._pipeline
        interruption_owner = getattr(owner, "_interruption_orchestrator", None)
        if (
            interruption_owner is None
            or not owner._barge_in_enabled
            or not interruption_owner.blocks_framework_completed_turn()
        ):
            return None
        decision = self._decide_from_completed_turn_evidence(
            completed_transcript,
            timeline=timeline,
        )
        resolution = resolve_playback_turn_decision(decision)
        if resolution.should_apply:
            owner._ensure_decision_effect_applier()
            owner._decision_effects.apply(
                decision,
                resolved_reason="framework_completed_interruption_evidence",
                eot_score=self._current_eot_score(),
                transcript=completed_transcript,
                vad_active=False,
            )
            verdict_resolution = self._resolve_recorded_interruption_verdict(timeline=timeline)
            if verdict_resolution is not None:
                return verdict_resolution
            if interruption_owner.active:
                interruption_owner.resolve(
                    action=decision.action.value,
                    reason="framework_completed_interruption_evidence",
                )
            return self._resolve_recorded_interruption_verdict(timeline=timeline)
        reason = "interruption_owner_waiting_for_evidence"
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
        return False

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
                    "intent": (decision.intent.value if decision.intent is not None else None),
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

    @staticmethod
    def _record_completed_gate_event(
        timeline: TurnTimeline,
        *,
        stage: str,
        action: str,
        reason: str,
        transcript: str,
        **fields: Any,
    ) -> None:
        payload = {
            "stage": stage,
            "action": action,
            "reason": reason,
            "transcript_preview": transcript[:120],
            "text_length": len(transcript),
            **fields,
        }
        events = list(timeline.attrs.get("framework_completed_gate_events") or ())
        events.append(payload)
        events = events[-16:]
        timeline.set_attr("framework_completed_gate_events", events)
        timeline.set_attr("framework_completed_gate_last_event", payload)


def _record_contract_transition(
    owner: Any,
    phase: FullDuplexPhase,
    *,
    event: str,
    reason: str,
    transcript: str = "",
    side_effect: str = "none",
    timeline: TurnTimeline | None = None,
) -> None:
    recorder = getattr(owner, "_record_full_duplex_transition", None)
    if recorder is None:
        return
    recorder(
        phase,
        event=event,
        reason=reason,
        side_effect=side_effect,
        transcript=transcript,
        timeline=timeline,
    )
