"""Full-duplex VAD speech segment lifecycle."""

from __future__ import annotations

import logging
import time
from typing import Any

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.shared.types import generate_turn_id

from .state_machine import FullDuplexPhase

logger = logging.getLogger("agent")


class FullDuplexSpeechLifecycle:
    """Own VAD start/stop lifecycle for full-duplex streaming turns.

    This component coordinates one open-mic speech segment. Terminal decisions
    still belong to the interruption owner, turn runtime, user-turn coordinator,
    and voiceprint commit path on the pipeline.
    """

    def __init__(self, owner: Any) -> None:
        self._owner = owner

    def handle_started(self) -> None:
        owner = self._owner
        turn_completion = owner._ensure_turn_completion()
        owner._ensure_user_turn_coordinator()
        merge_continuation = owner._user_turns.can_merge_new_speech()
        owner._set_interrupt_cancel_suppression(False, 0.0, reason="new_speech_started")
        owner._set_suppress_transcripts_until_next_speech(False, reason="new_speech_started")
        turn_completion.cancel_deferred_low_eot_commit("new_speech_started")
        if not merge_continuation:
            turn_completion.cancel_pending_voiceprint_commits("new_speech_started")
            turn_completion.clear_completed_voiceprint_turn()
            turn_completion.reset_candidate_voiceprint_tasks()
        replaced_unmerged_timeline = owner._timeline if not merge_continuation else None
        replaced_candidate = (
            owner._user_turns.active if not merge_continuation else None
        )
        superseding_pending_candidate = (
            replaced_unmerged_timeline is not None
            and replaced_candidate is not None
            and getattr(replaced_candidate, "state", None)
            not in {"committed", "rejected"}
        )

        owner._callbacks.on_user_started_speaking()
        owner._user_speaking_start_time = time.monotonic()
        if not merge_continuation or owner._timeline is None:
            owner._timeline = TurnTimeline(generate_turn_id())
            owner._timeline_debug_flushed = False
            if owner._room is not None:
                owner._timeline.set_attr("room_name", owner._room.name or "")

        owner._user_turns.start_speech(timeline=owner._timeline)
        if replaced_unmerged_timeline is not None:
            if superseding_pending_candidate:
                _record_contract_transition(
                    owner,
                    FullDuplexPhase.USER_TURN_REJECTED,
                    event="user_turn_superseded_by_new_speech",
                    reason="superseded_by_new_speech",
                    timeline=replaced_unmerged_timeline,
                )
            owner._append_turn_timeline_snapshot(
                replaced_unmerged_timeline,
                "speech_started_replaced_unmerged_timeline",
            )
        _record_contract_transition(
            owner,
            FullDuplexPhase.USER_SPEECH_OPEN,
            event="speech_started",
            reason="merge_continuation" if merge_continuation else "new_speech_started",
        )
        owner._timeline.mark("speech_started_at")
        owner._attach_transcript_ingress_recent_events("speech_started")
        owner._apply_pending_explicit_client_preempt(owner._timeline)
        owner._apply_pending_client_control_events(owner._timeline)
        owner._voiceprint_turns.start_turn(timeline=owner._timeline)
        owner._ensure_provider_event_observer()
        owner._provider_events.apply_pending_stt_provider_events()
        owner._provider_events.observe_stt_turn_audio()
        owner._latest_asr_text = ""

        owner._get_eot_model().update_vad(True)

        if owner._uses_livekit_native_adaptive_interruption():
            if owner._timeline is not None:
                owner._timeline.set_attr("interruption_owner", "livekit_native_adaptive")
            return

        interrupt_window_started = owner._attention_effects.handle_speaking_started()
        if interrupt_window_started:
            owner._interruption_orchestrator.start_candidate(timeline=owner._timeline)

    def handle_stopped(self) -> None:
        owner = self._owner
        turn_completion = owner._ensure_turn_completion()
        owner._user_speaking_start_time = None
        if owner._timeline is not None:
            owner._timeline.mark("speech_stopped_at")
        voiceprint_task = owner._voiceprint_turns.finish_turn()
        turn_completion.remember_completed_voiceprint_turn(
            voiceprint_task,
            timeline=owner._timeline,
        )

        eot_model = owner._get_eot_model()
        eot_model.update_vad(False)

        committed_confirmed_cancel = self._commit_confirmed_cancel_on_stop(
            voiceprint_task
        )
        defer_post_speech_evidence = (
            False
            if committed_confirmed_cancel
            else self._resolve_interruption_candidate_on_stop()
        )
        owner._callbacks.on_user_ended_speaking()
        # Clear only the skip flag; leave the residual-commit-suppress window.
        owner._set_interrupt_cancel_suppression(False, reason="speech_stopped")
        if committed_confirmed_cancel:
            owner._latest_asr_text = ""
            return
        if defer_post_speech_evidence:
            turn_completion.remember_candidate_voiceprint_task(voiceprint_task)
            return
        if owner._session is None:
            owner._latest_asr_text = ""
            return

        transcript = owner._user_turns.selected_text or owner._latest_asr_text
        attention_reject_reason = turn_completion.attention_admission_reject_reason(
            transcript=transcript
        )
        if attention_reject_reason:
            logger.info(
                "[StreamingPipeline] rejecting attention-ignored turn reason=%s "
                "transcript=%r",
                attention_reject_reason,
                transcript[:80],
            )
            if owner._user_turns.active is not None:
                owner._user_turns.reject_active(attention_reject_reason)
            if _restore_superseded_candidate_after_reject(
                owner,
                turn_completion,
                eot_model,
                attention_reject_reason,
            ):
                owner._latest_asr_text = ""
                return
            _record_contract_transition(
                owner,
                FullDuplexPhase.USER_TURN_REJECTED,
                event="attention_admission_rejected",
                reason=attention_reject_reason,
                transcript=transcript,
            )
            eot_model.reset()
            turn_completion.clear_session_user_turn(attention_reject_reason)
            turn_completion.reset_candidate_voiceprint_tasks()
            owner._latest_asr_text = ""
            return
        if transcript:
            turn_completion.remember_candidate_voiceprint_task(voiceprint_task)
        if owner._user_turns.active is None and transcript:
            if owner._timeline is None:
                owner._timeline = TurnTimeline(generate_turn_id())
                owner._timeline_debug_flushed = False
            owner._user_turns.start_speech(timeline=owner._timeline)
            owner._apply_pending_explicit_client_preempt(owner._timeline)
            owner._apply_pending_client_control_events(owner._timeline)
            owner._user_turns.add_transcript(transcript, is_final=True)

        low_evidence_reason = turn_completion.playback_low_evidence_reject_reason(
            transcript=transcript,
            eot_model=eot_model,
        )
        if low_evidence_reason:
            logger.info(
                "[StreamingPipeline] rejecting playback low-evidence turn reason=%s "
                "transcript=%r",
                low_evidence_reason,
                transcript[:80],
            )
            owner._user_turns.reject_active(low_evidence_reason)
            if _restore_superseded_candidate_after_reject(
                owner,
                turn_completion,
                eot_model,
                low_evidence_reason,
            ):
                owner._latest_asr_text = ""
                return
            _record_contract_transition(
                owner,
                FullDuplexPhase.USER_TURN_REJECTED,
                event="playback_low_evidence_rejected",
                reason=low_evidence_reason,
                transcript=transcript,
            )
            eot_model.reset()
            turn_completion.clear_session_user_turn(low_evidence_reason)
            turn_completion.reset_candidate_voiceprint_tasks()
            owner._latest_asr_text = ""
            return

        should_defer = turn_completion.should_defer_low_eot_commit(
            transcript=transcript,
            eot_model=eot_model,
        )
        decision = owner._user_turns.finish_speech(
            eot_score=getattr(
                eot_model,
                "current_eot_score",
                getattr(eot_model, "_current_eot_score", None),
            ),
            should_defer=should_defer,
        )
        if decision.action == "reject":
            if _restore_superseded_candidate_after_reject(
                owner,
                turn_completion,
                eot_model,
                decision.reason,
            ):
                owner._latest_asr_text = ""
                return
            _record_contract_transition(
                owner,
                FullDuplexPhase.USER_TURN_REJECTED,
                event="user_turn_rejected",
                reason=decision.reason,
                transcript=decision.transcript or transcript,
            )
            eot_model.reset()
            turn_completion.clear_session_user_turn(decision.reason)
            turn_completion.reset_candidate_voiceprint_tasks()
            owner._latest_asr_text = ""
        elif decision.action == "defer":
            _record_contract_transition(
                owner,
                FullDuplexPhase.USER_TURN_PENDING,
                event="user_turn_deferred",
                reason=decision.reason,
                transcript=decision.transcript,
            )
            turn_completion.schedule_deferred_low_eot_commit(
                verify_task=None,
                eot_model=eot_model,
                transcript=decision.transcript,
                timeline=owner._timeline,
                delay_sec=decision.delay_sec,
            )
        else:
            _record_contract_transition(
                owner,
                FullDuplexPhase.USER_TURN_PENDING,
                event="user_turn_voiceprint_pending",
                reason=decision.reason,
                transcript=decision.transcript or transcript,
            )
            turn_completion.schedule_voiceprint_gated_commit(
                verify_task=turn_completion.candidate_voiceprint_gate_task(),
                eot_model=eot_model,
                transcript=decision.transcript or transcript,
                timeline=owner._timeline,
            )
            owner._latest_asr_text = ""

    def _commit_confirmed_cancel_on_stop(self, voiceprint_task: Any) -> bool:
        owner = self._owner
        interruption_owner = getattr(owner, "_interruption_orchestrator", None)
        if interruption_owner is None:
            return False
        transcript = owner._user_turns.selected_text or owner._latest_asr_text
        if interruption_owner.finish_confirmed_cancel_speech(transcript) is not True:
            return False
        turn_completion = owner._ensure_turn_completion()
        turn_completion.remember_candidate_voiceprint_task(voiceprint_task)
        committed = turn_completion.commit_post_speech_interruption_candidate(
            "confirmed_cancel_speech_end",
            transcript_override=transcript,
        )
        if committed:
            interruption_owner.resolve(
                action="cancel",
                reason="confirmed_cancel_turn_committed",
            )
            owner._set_interrupt_cancel_suppression(False, 0.0)
        return committed

    def _resolve_interruption_candidate_on_stop(self) -> bool:
        owner = self._owner
        interruption_effects = owner._ensure_interruption_effects()
        if interruption_effects.soft_interrupt_active():
            logger.info(
                "[StreamingPipeline] user fell silent during soft interrupt; "
                "false interruption, cancelling"
            )
            interruption_effects.cancel_soft_interrupt()

        if (
            owner._ducking.is_suspended
            and not owner._uses_livekit_native_adaptive_interruption()
        ):
            should_defer = (
                owner._interruption_orchestrator.defer_false_resume_after_speech_end(
                    transcript=owner._latest_asr_text,
                    duck_suspended=True,
                )
            )
            if not should_defer:
                decision = owner._turn_runtime.user_silent_decision(
                    owner._latest_asr_text
                )
                owner._decision_effects.apply(
                    decision,
                    resolved_reason="user_silent",
                    transcript=owner._latest_asr_text,
                    vad_active=False,
                )
            return should_defer
        return False


def _restore_superseded_candidate_after_reject(
    owner: Any,
    turn_completion: Any,
    eot_model: Any,
    reject_reason: str,
) -> bool:
    restore = owner._user_turns.restore_superseded_candidate_if_replacement_rejected(
        reject_reason
    )
    if restore.action != "commit":
        return False
    restored_candidate = owner._user_turns.active
    restored_timeline = getattr(restored_candidate, "timeline", None)
    _record_contract_transition(
        owner,
        FullDuplexPhase.USER_TURN_PENDING,
        event="superseded_candidate_restored",
        reason=restore.reason,
        transcript=restore.transcript,
        timeline=restored_timeline,
    )
    turn_completion.clear_session_user_turn(reject_reason)
    turn_completion.reset_candidate_voiceprint_tasks()
    turn_completion.schedule_voiceprint_gated_commit(
        verify_task=None,
        eot_model=eot_model,
        transcript=restore.transcript,
        timeline=restored_timeline,
    )
    return True


def _record_contract_transition(
    owner: Any,
    phase: FullDuplexPhase,
    *,
    event: str,
    reason: str,
    transcript: str = "",
    timeline: TurnTimeline | None = None,
) -> None:
    recorder = getattr(owner, "_record_full_duplex_transition", None)
    if recorder is None:
        return
    recorder(
        phase,
        event=event,
        reason=reason,
        transcript=transcript,
        timeline=timeline,
    )
