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
from .playback_turn_evidence import (
    non_semantic_completed_turn_reason,
    resolve_playback_turn_decision,
)
from .turn_completion_policy import (
    eot_thinks_turn_complete,
    looks_like_short_statement_continuation,
)
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

    async def allows_completed_turn(self, *, new_message: Any) -> bool:
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
                completed_transcript=completed_transcript,
                timeline=timeline,
                voiceprint_reason="",
            )

        interruption_allowed = self._resolve_completed_turn_interruption_evidence(
            completed_transcript=completed_transcript,
            timeline=timeline,
            voiceprint_reason="",
        )
        if interruption_allowed is not None:
            return interruption_allowed

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
                completed_transcript=completed_transcript,
                timeline=timeline,
                voiceprint_reason=reason,
            )

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

    def _route_allowed_framework_completed_turn(
        self,
        *,
        completed_transcript: str,
        timeline: TurnTimeline | None,
        voiceprint_reason: str,
    ) -> bool:
        interruption_allowed = self._resolve_completed_turn_interruption_evidence(
            completed_transcript=completed_transcript,
            timeline=timeline,
            voiceprint_reason=voiceprint_reason,
        )
        if interruption_allowed is not None:
            return interruption_allowed
        if self._stop_non_semantic_framework_completed_turn(
            completed_transcript,
            timeline=timeline,
        ):
            return False
        if self._should_defer_framework_completed_turn(completed_transcript):
            self._defer_framework_completed_turn(
                completed_transcript=completed_transcript,
                timeline=timeline,
                voiceprint_reason=voiceprint_reason,
            )
            return False
        return self._align_framework_completed_turn(
            completed_transcript,
            timeline=timeline,
            voiceprint_reason=voiceprint_reason,
        )

    def _resolve_completed_turn_interruption_evidence(
        self,
        *,
        completed_transcript: str,
        timeline: TurnTimeline | None,
        voiceprint_reason: str,
    ) -> bool | None:
        if self._stop_active_interruption_framework_completed_turn(
            completed_transcript,
            timeline=timeline,
        ):
            return False
        return self._resolve_playback_completed_turn_evidence(
            completed_transcript,
            timeline=timeline,
            voiceprint_reason=voiceprint_reason,
        )

    def _resolve_playback_completed_turn_evidence(
        self,
        completed_transcript: str,
        *,
        timeline: TurnTimeline | None,
        voiceprint_reason: str,
    ) -> bool | None:
        """Resolve late playback-overlap evidence from LiveKit completed-turn.

        Some real-room paths produce only an early interim transcript before
        client playback state reaches Channel. When the final framework turn
        arrives, the user may still be interrupting audible assistant playback.
        In that case the completed turn itself is valid evidence: redirect
        intents should cancel current playback and continue to the LLM, while
        non-semantic intents should stop before they become a user turn.
        """

        continue_to_llm = self._apply_playback_turn_evidence(
            completed_transcript,
            timeline=timeline,
            resolved_reason="framework_completed_playback_evidence",
            timeline_attr="framework_completed_playback_evidence",
            cancel_deferred=True,
        )
        if continue_to_llm is None:
            return None

        if continue_to_llm:
            return self._align_framework_completed_turn(
                completed_transcript,
                timeline=timeline,
                voiceprint_reason=voiceprint_reason,
            )

        return False

    def resolve_deferred_playback_commit_evidence(
        self,
        transcript: str,
        *,
        timeline: TurnTimeline | None,
    ) -> bool | None:
        """Resolve playback-overlap evidence before a low-EOT deferred commit.

        Real room timing can deliver client playback state after LiveKit's
        completed-turn hook has already deferred a low-EOT fragment. This gate
        keeps the delayed commit path under the same interruption owner.
        """

        return self._apply_playback_turn_evidence(
            transcript,
            timeline=timeline,
            resolved_reason="deferred_low_eot_playback_evidence",
            timeline_attr="deferred_low_eot_playback_evidence",
            cancel_deferred=False,
        )

    def _apply_playback_turn_evidence(
        self,
        transcript: str,
        *,
        timeline: TurnTimeline | None,
        resolved_reason: str,
        timeline_attr: str,
        cancel_deferred: bool,
    ) -> bool | None:
        playback_active = self._playback_active_for_completed_turn(timeline=timeline)
        if timeline is not None:
            self._record_completed_gate_event(
                timeline,
                stage="playback_check",
                action="continue" if playback_active else "skip",
                reason="playback_active" if playback_active else "no_playback_evidence",
                transcript=transcript,
                playback_active=playback_active,
            )
        if not playback_active:
            return None
        if self._client_state_blocks_playback_evidence(timeline=timeline):
            if timeline is not None:
                timeline.set_attr(
                    timeline_attr,
                    {
                        "action": "ignore",
                        "continue_to_llm": False,
                        "intent": None,
                        "reason": "client_mic_muted",
                        "text_preview": transcript[:120],
                        "text_length": len(transcript),
                    },
                )
                self._record_completed_gate_event(
                    timeline,
                    stage=timeline_attr,
                    action="ignore",
                    reason="client_mic_muted",
                    transcript=transcript,
                    playback_active=True,
                    continue_to_llm=False,
                )
            owner = self._pipeline
            owner._ensure_user_turn_coordinator()
            owner._user_turns.reject_active("client_mic_muted")
            self._completion.clear_session_user_turn("client_mic_muted")
            return False
        decision = self._decide_from_completed_turn_evidence(
            transcript,
            timeline=timeline,
        )
        resolution = resolve_playback_turn_decision(decision)
        if not resolution.should_apply:
            if timeline is not None:
                self._record_completed_gate_event(
                    timeline,
                    stage=timeline_attr,
                    action="skip",
                    reason=resolution.reason,
                    transcript=transcript,
                    playback_active=True,
                    intent=(
                        decision.intent.value
                        if decision is not None and decision.intent is not None
                        else None
                    ),
                    decision_reason=decision.reason if decision is not None else None,
                )
            return None

        owner = self._pipeline
        completion = self._completion
        continue_to_llm = resolution.continue_to_llm
        if timeline is not None:
            timeline.mark("framework_completed_playback_evidence_at")
            timeline.set_attr(
                timeline_attr,
                {
                    "action": decision.action.value,
                    "continue_to_llm": continue_to_llm,
                    "intent": (
                        decision.intent.value if decision.intent is not None else None
                    ),
                    "reason": decision.reason,
                    "text_preview": transcript[:120],
                    "text_length": len(transcript),
                },
            )
            self._record_completed_gate_event(
                timeline,
                stage=timeline_attr,
                action=decision.action.value,
                reason=decision.reason,
                transcript=transcript,
                playback_active=True,
                continue_to_llm=continue_to_llm,
                intent=decision.intent.value if decision.intent is not None else None,
                topic_switch_hint=decision.topic_switch_hint,
                correction_hint=decision.correction_hint,
            )
        owner._ensure_decision_effect_applier()
        owner._decision_effects.apply(
            decision,
            resolved_reason=resolved_reason,
            eot_score=self._current_eot_score(),
            transcript=transcript,
            vad_active=False,
        )
        if cancel_deferred:
            completion.cancel_deferred_low_eot_commit(resolved_reason)
        if continue_to_llm:
            return True

        owner._ensure_user_turn_coordinator()
        owner._user_turns.reject_active(
            f"interruption_owner_resolved:{decision.action.value}"
        )
        completion.clear_session_user_turn(
            f"interruption_owner_resolved:{decision.action.value}"
        )
        return False

    def _client_state_blocks_playback_evidence(
        self,
        *,
        timeline: TurnTimeline | None,
    ) -> bool:
        owner = self._pipeline
        try:
            client = owner._ensure_client_audio_state_view().latest_state()
        except Exception:  # noqa: BLE001 - completed-turn gate must fail closed
            logger.debug(
                "[StreamingPipeline] completed-turn client-state check failed",
                exc_info=True,
            )
            return False
        timeline_client = _timeline_client_audio_state(timeline)
        mic_muted = bool(getattr(client, "mic_muted", False)) or bool(
            timeline_client.get("mic_muted")
        )
        if not owner._turn_policy.attention.ignore_when_mic_muted or not mic_muted:
            return False
        if timeline is not None:
            timeline.set_attr(
                "framework_completed_playback_evidence_ignored",
                {
                    "reason": "client_mic_muted",
                    "participant_identity": (
                        getattr(client, "participant_identity", "")
                        or str(timeline_client.get("participant_identity") or "")
                    ),
                    "playback_state": (
                        getattr(client, "playback_state", "")
                        or str(timeline_client.get("playback_state") or "")
                    ),
                },
            )
        return True

    def _playback_active_for_completed_turn(
        self,
        *,
        timeline: TurnTimeline | None,
    ) -> bool:
        owner = self._pipeline
        try:
            active = bool(
                owner._ensure_client_audio_state_view().agent_output_active_for_interrupts()
            )
        except Exception:  # noqa: BLE001 - completed-turn gate must fail closed
            logger.debug(
                "[StreamingPipeline] completed-turn playback activity check failed",
                exc_info=True,
            )
            active = False
        if active:
            return True
        timeline_client = _timeline_client_audio_state(timeline)
        return timeline_client.get("playback_state") == "agent_speaking"

    def _eot_thinks_turn_complete(self) -> bool:
        owner = self._pipeline
        return eot_thinks_turn_complete(
            owner._get_eot_model(),
            unlikely_threshold=float(owner._turn_policy.eot.eot_unlikely_threshold),
        )

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
        return looks_like_short_statement_continuation(
            selected,
            max_cjk_chars=owner._turn_policy.eot.short_statement_defer_max_cjk_chars,
        )

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
        if decision.action == "reject":
            _record_contract_transition(
                owner,
                FullDuplexPhase.USER_TURN_REJECTED,
                event="framework_completed_rejected",
                reason=decision.reason,
                transcript=decision.transcript or completed_transcript,
                timeline=timeline,
            )
            completion.clear_session_user_turn(decision.reason)
            owner._flush_turn_timeline(timeline, decision.reason)
            logger.info(
                "[StreamingPipeline] rejected framework completed turn "
                "reason=%s transcript=%r",
                decision.reason,
                completed_transcript[:80],
            )
            return
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
        _record_contract_transition(
            owner,
            FullDuplexPhase.USER_TURN_PENDING,
            event="framework_completed_deferred",
            reason=decision.reason,
            transcript=decision.transcript or completed_transcript,
            timeline=timeline,
        )
        completion.schedule_deferred_low_eot_commit(
            verify_task=None,
            eot_model=owner._get_eot_model(),
            transcript=decision.transcript or completed_transcript,
            timeline=timeline,
            delay_sec=decision.delay_sec,
        )
        completion.clear_completed_voiceprint_turn()
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
    ) -> bool:
        owner = self._pipeline
        self._completion.cancel_deferred_low_eot_commit("framework_completed_turn")
        owner._ensure_user_turn_coordinator()
        decision = owner._user_turns.mark_framework_completed(
            transcript=completed_transcript,
            reason="framework_completed_turn",
            timeline=timeline,
            voiceprint_reason=voiceprint_reason,
        )
        if decision.action == "reject":
            _record_contract_transition(
                owner,
                FullDuplexPhase.USER_TURN_REJECTED,
                event="framework_completed_rejected",
                reason=decision.reason,
                transcript=decision.transcript or completed_transcript,
                timeline=timeline,
            )
            self._completion.clear_session_user_turn(decision.reason)
            owner._flush_turn_timeline(timeline, decision.reason)
            logger.info(
                "[StreamingPipeline] rejected framework completed turn "
                "reason=%s transcript=%r",
                decision.reason,
                completed_transcript[:80],
            )
            return False
        canonical = decision.transcript or completed_transcript
        self._session_turns.publish_canonical_user_text(
            canonical,
            source="framework_completed_turn",
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
        return True

    @staticmethod
    def _non_semantic_completed_turn_reason(
        timeline: TurnTimeline | None,
    ) -> str:
        if timeline is None:
            return ""
        decision = timeline.attrs.get("decision")
        if not isinstance(decision, dict):
            return ""
        return non_semantic_completed_turn_reason(decision)

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


def _timeline_client_audio_state(timeline: TurnTimeline | None) -> dict[str, Any]:
    if timeline is None:
        return {}
    value = timeline.attrs.get("client_audio_state")
    return value if isinstance(value, dict) else {}


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
