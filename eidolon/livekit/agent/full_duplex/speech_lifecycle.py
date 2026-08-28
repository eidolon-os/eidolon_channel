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

    This component coordinates one open-mic speech segment. It records acoustic
    boundaries and interruption evidence, but never commits, rejects, or clears
    a product user turn. Product completion belongs to the framework-completed
    turn gate.
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
        if not merge_continuation:
            turn_completion.cancel_completed_voiceprint_turn()
            turn_completion.reset_candidate_voiceprint_tasks()
        replaced_unmerged_timeline = owner._timeline if not merge_continuation else None
        replaced_candidate = owner._user_turns.active if not merge_continuation else None
        superseding_pending_candidate = (
            replaced_unmerged_timeline is not None
            and replaced_candidate is not None
            and getattr(replaced_candidate, "state", None) not in {"committed", "rejected"}
        )

        owner._callbacks.on_user_started_speaking()
        owner._user_speaking_start_time = time.monotonic()
        if not merge_continuation or owner._timeline is None:
            owner._timeline = TurnTimeline(generate_turn_id())
            llm_plugin = getattr(
                getattr(getattr(owner, "_factory", None), "llm", None),
                "llm",
                None,
            )
            set_trace_id = getattr(llm_plugin, "set_turn_trace_id", None)
            if set_trace_id is not None:
                set_trace_id(owner._timeline.turn_id)
            owner._timeline_debug_flushed = False
            if owner._room is not None:
                owner._timeline.set_attr("room_name", owner._room.name or "")
            owner._timeline.set_attr(
                "participant_identity",
                getattr(owner, "_runtime_participant_identity", "") or "",
            )

        owner._user_turns.start_speech(timeline=owner._timeline)
        owner._ensure_agent_output_coordinator().link_interruption_candidate(owner._timeline)
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
            if superseding_pending_candidate:
                owner._flush_turn_timeline(
                    replaced_unmerged_timeline,
                    "superseded_by_new_speech",
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

        if not owner._barge_in_enabled:
            # half_duplex: no barge-in. VAD-start must NOT arm a duck window or an
            # interruption candidate — the mic is closed while the agent speaks, so
            # there is no barge-in to detect, and the user turn commits through the
            # framework's normal endpointing path. Arming it here (the pre-fix
            # behaviour) routed every idle-turn utterance into the interruption
            # evidence window, which then rolled back on timeout and never committed.
            if owner._timeline is not None:
                owner._timeline.set_attr("interruption_owner", "disabled_no_barge_in")
            return

        interrupt_window_started = owner._attention_effects.handle_speaking_started()
        if interrupt_window_started:
            owner._interruption_orchestrator.start_candidate(
                timeline=owner._timeline,
                generation_id=owner._user_turns.current_generation_id,
            )

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
        owner._ensure_user_turn_coordinator()
        owner._user_turns.note_speech_stopped(
            eot_score=getattr(
                eot_model,
                "current_eot_score",
                getattr(eot_model, "_current_eot_score", None),
            )
        )

        confirmed_cancel_stopped = self._finish_confirmed_cancel_on_stop()
        defer_post_speech_evidence = (
            False if confirmed_cancel_stopped else self._resolve_interruption_candidate_on_stop()
        )
        owner._callbacks.on_user_ended_speaking()
        # Clear only the skip flag; leave the residual-commit-suppress window.
        owner._set_interrupt_cancel_suppression(False, reason="speech_stopped")
        if defer_post_speech_evidence or confirmed_cancel_stopped:
            turn_completion.remember_candidate_voiceprint_task(voiceprint_task)
        elif owner._user_turns.selected_text or owner._latest_asr_text:
            turn_completion.remember_candidate_voiceprint_task(voiceprint_task)
        transcript = owner._user_turns.selected_text or owner._latest_asr_text
        if transcript:
            _record_contract_transition(
                owner,
                FullDuplexPhase.USER_TURN_PENDING,
                event="speech_stopped_waiting_framework",
                reason="automatic_turn_lifecycle",
                transcript=transcript,
            )

    def _finish_confirmed_cancel_on_stop(self) -> bool:
        owner = self._owner
        if not owner._barge_in_enabled:
            return False
        interruption_owner = getattr(owner, "_interruption_orchestrator", None)
        if interruption_owner is None:
            return False
        transcript = owner._user_turns.selected_text or owner._latest_asr_text
        if interruption_owner.finish_confirmed_cancel_speech(transcript) is not True:
            return False
        return True

    def _resolve_interruption_candidate_on_stop(self) -> bool:
        owner = self._owner
        if not owner._barge_in_enabled:
            return False
        interruption_effects = owner._ensure_interruption_effects()
        if interruption_effects.soft_interrupt_active():
            logger.info(
                "[StreamingPipeline] user fell silent during soft interrupt; "
                "false interruption, cancelling"
            )
            interruption_effects.cancel_soft_interrupt()

        if owner._ducking.is_suspended and not owner._uses_livekit_native_adaptive_interruption():
            should_defer = owner._interruption_orchestrator.defer_false_resume_after_speech_end(
                transcript=owner._latest_asr_text,
                duck_suspended=True,
            )
            if not should_defer:
                decision = owner._turn_runtime.user_silent_decision(owner._latest_asr_text)
                owner._decision_effects.apply(
                    decision,
                    resolved_reason="user_silent",
                    transcript=owner._latest_asr_text,
                    vad_active=False,
                )
            return should_defer
        return False


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
