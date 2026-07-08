"""Commit or reject post-speech interruption candidates."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from ..observability import TurnTimeline
from ..shared.types import generate_turn_id
from .state_machine import FullDuplexPhase

if TYPE_CHECKING:
    from .pipeline import StreamingPipeline

logger = logging.getLogger("agent")


class FullDuplexPostSpeechInterruptionCommitter:
    """Finalize the transcript collected while interruption evidence resolves."""

    def __init__(
        self,
        pipeline: StreamingPipeline,
        *,
        cancel_deferred_low_eot_commit: Callable[[str], None],
        clear_session_user_turn: Callable[[str], None],
        candidate_voiceprint_gate_task: Callable[[], asyncio.Task | None],
        schedule_voiceprint_gated_commit: Callable[..., None],
        cancel_completed_voiceprint_turn: Callable[[], None],
        reset_candidate_voiceprint_tasks: Callable[[], None],
    ) -> None:
        self._pipeline = pipeline
        self._cancel_deferred_low_eot_commit = cancel_deferred_low_eot_commit
        self._clear_session_user_turn = clear_session_user_turn
        self._candidate_voiceprint_gate_task = candidate_voiceprint_gate_task
        self._schedule_voiceprint_gated_commit = schedule_voiceprint_gated_commit
        self._cancel_completed_voiceprint_turn = cancel_completed_voiceprint_turn
        self._reset_candidate_voiceprint_tasks = reset_candidate_voiceprint_tasks

    def commit_candidate(
        self,
        reason: str,
        *,
        transcript_override: str = "",
    ) -> bool:
        pipeline = self._pipeline
        interruption_owner = getattr(pipeline, "_interruption_orchestrator", None)
        pipeline._ensure_user_turn_coordinator()
        owner_transcript = (
            interruption_owner.current_transcript if interruption_owner is not None else ""
        )
        transcript = (
            transcript_override
            or owner_transcript
            or pipeline._user_turns.selected_text
            or pipeline._latest_asr_text
        ).strip()
        if not transcript:
            return False
        timeline = getattr(pipeline, "_timeline", None)
        self._cancel_deferred_low_eot_commit(reason)
        if pipeline._user_turns.active is None:
            if pipeline._timeline is None:
                pipeline._timeline = TurnTimeline(generate_turn_id())
                pipeline._timeline_debug_flushed = False
                timeline = pipeline._timeline
            pipeline._user_turns.start_speech(timeline=pipeline._timeline)
            pipeline._apply_pending_explicit_client_preempt(pipeline._timeline)
            pipeline._apply_pending_client_control_events(pipeline._timeline)
        pipeline._user_turns.add_transcript(transcript, is_final=True)
        eot_model = pipeline._get_eot_model()
        decision = pipeline._user_turns.finish_speech(
            eot_score=getattr(
                eot_model,
                "current_eot_score",
                getattr(eot_model, "_current_eot_score", None),
            ),
            should_defer=False,
        )
        if decision.action == "reject":
            self._clear_session_user_turn(decision.reason)
            return False
        committed_text = decision.transcript or transcript
        _record_contract_transition(
            pipeline,
            FullDuplexPhase.USER_TURN_PENDING,
            event="post_speech_interruption_candidate_pending",
            reason=reason,
            transcript=committed_text,
            timeline=timeline,
        )
        if timeline is not None:
            timeline.set_attr(
                "post_speech_interruption_candidate_committed",
                {
                    "reason": reason,
                    "transcript_preview": committed_text[:120],
                    "text_length": len(committed_text),
                },
            )
        self._schedule_voiceprint_gated_commit(
            verify_task=self._candidate_voiceprint_gate_task(),
            eot_model=eot_model,
            transcript=committed_text,
            timeline=timeline,
        )
        pipeline._latest_asr_text = ""
        logger.info(
            "[StreamingPipeline] committed post-speech interruption candidate "
            "reason=%s transcript=%r",
            reason,
            committed_text[:80],
        )
        return True

    def reject_candidate(self, reason: str) -> None:
        pipeline = self._pipeline
        timeline = getattr(pipeline, "_timeline", None)
        self._cancel_deferred_low_eot_commit(reason)
        pipeline._ensure_user_turn_coordinator()
        decision = pipeline._user_turns.reject_active(reason)
        eot_model = pipeline._get_eot_model()
        try:
            eot_model.reset()
        except Exception:
            logger.debug(
                "[StreamingPipeline] EOT reset failed while rejecting "
                "post-speech interruption candidate",
                exc_info=True,
            )
        self._cancel_completed_voiceprint_turn()
        self._reset_candidate_voiceprint_tasks()
        self._clear_session_user_turn(reason)
        pipeline._latest_asr_text = ""
        _record_contract_transition(
            pipeline,
            FullDuplexPhase.USER_TURN_REJECTED,
            event="post_speech_interruption_candidate_rejected",
            reason=reason,
            transcript=decision.transcript,
            timeline=timeline,
        )
        if timeline is not None:
            timeline.set_attr(
                "post_speech_interruption_candidate_rejected",
                {
                    "reason": reason,
                    "transcript_preview": decision.transcript[:120],
                    "text_length": len(decision.transcript),
                },
            )
            pipeline._flush_turn_timeline(timeline, reason)
        logger.info(
            "[StreamingPipeline] rejected post-speech interruption candidate "
            "reason=%s transcript=%r",
            reason,
            decision.transcript[:80],
        )


def _record_contract_transition(
    pipeline: StreamingPipeline,
    phase: FullDuplexPhase,
    *,
    event: str,
    reason: str,
    transcript: str = "",
    timeline: TurnTimeline | None = None,
) -> None:
    recorder = getattr(pipeline, "_record_full_duplex_transition", None)
    if recorder is None:
        return
    recorder(
        phase,
        event=event,
        reason=reason,
        transcript=transcript,
        timeline=timeline,
    )
