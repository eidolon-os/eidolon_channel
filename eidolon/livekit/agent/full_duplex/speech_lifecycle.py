"""Full-duplex VAD speech segment lifecycle."""

from __future__ import annotations

import logging
import time
from typing import Any

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.pipeline.types import generate_turn_id

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
        owner._ensure_user_turn_coordinator()
        merge_continuation = owner._user_turns.can_merge_new_speech()
        owner._skip_commit_after_interrupt_cancel = False
        owner._suppress_transcripts_until_next_speech = False
        owner._cancel_deferred_low_eot_commit("new_speech_started")
        if not merge_continuation:
            owner._cancel_pending_voiceprint_commits("new_speech_started")
            owner._completed_turn_voiceprint_task = None
            owner._completed_turn_voiceprint_result = None
            owner._completed_turn_voiceprint_timeline = None
            owner._reset_candidate_voiceprint_tasks()

        owner._callbacks.on_user_started_speaking()
        owner._user_speaking_start_time = time.monotonic()
        if not merge_continuation or owner._timeline is None:
            owner._timeline = TurnTimeline(generate_turn_id())
            owner._timeline_debug_flushed = False
            if owner._room is not None:
                owner._timeline.set_attr("room_name", owner._room.name or "")

        owner._user_turns.start_speech(timeline=owner._timeline)
        owner._timeline.mark("speech_started_at")
        owner._apply_pending_explicit_client_preempt(owner._timeline)
        owner._apply_pending_client_control_events(owner._timeline)
        owner._voiceprint_turns.start_turn(timeline=owner._timeline)
        owner._apply_pending_stt_provider_events()
        owner._observe_stt_turn_audio()
        owner._latest_asr_text = ""

        owner._get_eot_model().update_vad(True)

        if owner._uses_livekit_native_adaptive_interruption():
            if owner._timeline is not None:
                owner._timeline.set_attr("interruption_owner", "livekit_native_adaptive")
            return

        owner._attention_effects.handle_speaking_started()
        if owner._ducking.is_suspended:
            owner._interruption_orchestrator.start_candidate(timeline=owner._timeline)

    def handle_stopped(self) -> None:
        owner = self._owner
        owner._user_speaking_start_time = None
        if owner._timeline is not None:
            owner._timeline.mark("speech_stopped_at")
        voiceprint_task = owner._voiceprint_turns.finish_turn()
        owner._completed_turn_voiceprint_task = voiceprint_task
        owner._completed_turn_voiceprint_result = None
        owner._completed_turn_voiceprint_timeline = owner._timeline

        eot_model = owner._get_eot_model()
        eot_model.update_vad(False)

        defer_post_speech_evidence = self._resolve_interruption_candidate_on_stop()
        owner._callbacks.on_user_ended_speaking()
        owner._skip_commit_after_interrupt_cancel = False
        if defer_post_speech_evidence:
            owner._remember_candidate_voiceprint_task(voiceprint_task)
            return
        if owner._session is None:
            owner._latest_asr_text = ""
            return

        transcript = owner._user_turns.selected_text or owner._latest_asr_text
        if transcript:
            owner._remember_candidate_voiceprint_task(voiceprint_task)
        if owner._user_turns.active is None and transcript:
            if owner._timeline is None:
                owner._timeline = TurnTimeline(generate_turn_id())
                owner._timeline_debug_flushed = False
            owner._user_turns.start_speech(timeline=owner._timeline)
            owner._apply_pending_explicit_client_preempt(owner._timeline)
            owner._apply_pending_client_control_events(owner._timeline)
            owner._user_turns.add_transcript(transcript, is_final=True)

        low_evidence_reason = owner._playback_low_evidence_reject_reason(
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
            eot_model.reset()
            owner._clear_session_user_turn(low_evidence_reason)
            owner._reset_candidate_voiceprint_tasks()
            owner._latest_asr_text = ""
            return

        should_defer = owner._should_defer_low_eot_commit(
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
            eot_model.reset()
            owner._clear_session_user_turn(decision.reason)
            owner._reset_candidate_voiceprint_tasks()
            owner._latest_asr_text = ""
        elif decision.action == "defer":
            owner._schedule_deferred_low_eot_commit(
                verify_task=None,
                eot_model=eot_model,
                transcript=decision.transcript,
                timeline=owner._timeline,
                delay_sec=decision.delay_sec,
            )
        else:
            owner._schedule_voiceprint_gated_commit(
                verify_task=owner._candidate_voiceprint_gate_task(),
                eot_model=eot_model,
                transcript=decision.transcript or transcript,
                timeline=owner._timeline,
            )
            owner._latest_asr_text = ""

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
