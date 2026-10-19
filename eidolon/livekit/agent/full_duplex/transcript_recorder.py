"""Accepted transcript side effects for full-duplex sessions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .transcript_event import FullDuplexTranscriptEvent
from .transcript_hypothesis_reconciler import TranscriptHypothesisReconciler

if TYPE_CHECKING:
    from .pipeline import StreamingPipeline

logger = logging.getLogger("agent")


class FullDuplexTranscriptRecorder:
    """Record accepted STT events into turn state, timeline, and EOT state."""

    def __init__(
        self,
        pipeline: StreamingPipeline,
        *,
        transcript_revision_min_normalized_chars: int,
    ) -> None:
        self._pipeline = pipeline
        self._hypotheses = TranscriptHypothesisReconciler(
            min_normalized_chars=transcript_revision_min_normalized_chars,
        )

    def record(self, transcript_event: FullDuplexTranscriptEvent) -> str:
        if not transcript_event.has_transcript:
            return ""

        pipeline = self._pipeline
        # Real recognized speech (interim or final) keeps the session alive.
        # Empty/noise transcripts deliberately don't, so a silent room still
        # trips the idle watchdog.
        pipeline._mark_activity()
        pipeline._ensure_user_turn_coordinator()
        evidence = transcript_event.evidence
        if pipeline._user_turns.transcript_starts_new_turn(evidence):
            pipeline._ensure_speech_lifecycle().handle_transcript_started()
        pipeline._latest_asr_text = transcript_event.transcript
        if evidence.available:
            receipt = pipeline._user_turns.add_transcript(
                transcript_event.transcript,
                is_final=transcript_event.is_final,
                evidence=evidence,
            )
        else:
            receipt = pipeline._user_turns.add_transcript(
                transcript_event.transcript,
                is_final=transcript_event.is_final,
            )
        if (receipt is not None and transcript_event.is_final
                and pipeline._timeline is not None
                and pipeline._timeline.attrs.get("candidate_origin") == "provider_transcript_without_vad"
                and getattr(pipeline, "_user_speaking_start_time", None) is None):
            pipeline._user_turns.note_speech_stopped(eot_score=None)
        if receipt is not None:
            covered_segments = self._hypotheses.observe(
                receipt,
                text=transcript_event.transcript,
                is_final=transcript_event.is_final,
            )
            if covered_segments:
                pipeline._user_turns.cover_pending_transcript_segments(
                    candidate_id=receipt.candidate_id,
                    segment_indexes=covered_segments,
                    covered_by_generation_id=receipt.generation_id,
                    covered_by_segment_index=receipt.segment_index,
                    reason=(
                        "provider_final_same_revision"
                        if receipt.evidence is not None
                        and receipt.evidence.revision_key
                        else "provider_final_equivalent_hypothesis"
                    ),
                )
        selected_policy_text = (
            getattr(pipeline._user_turns, "selected_policy_text", "")
            if receipt is not None
            else ""
        )
        policy_transcript = (
            selected_policy_text
            if isinstance(selected_policy_text, str) and selected_policy_text.strip()
            else transcript_event.transcript
        )
        if pipeline._timeline is not None:
            pipeline._timeline.mark(transcript_event.timeline_mark)
        self.sync_policy_state(policy_transcript, is_final=transcript_event.is_final)
        return policy_transcript

    def sync_policy_state(self, text: str, *, is_final: bool) -> None:
        """Publish canonical evidence after either an update or a withdrawal."""
        pipeline = self._pipeline
        pipeline._latest_asr_text = text
        orchestrator = getattr(pipeline, "_interruption_orchestrator", None)
        if (
            orchestrator is not None
            and pipeline._barge_in_enabled
            and not pipeline._uses_livekit_native_adaptive_interruption()
        ):
            # Use the same assembled evidence as the semantic decision. A raw
            # provider segment is not a revision of the whole candidate.
            orchestrator.note_transcript(
                text,
                is_final=is_final,
            )
        try:
            pipeline._get_eot_model().update_asr(
                text,
                is_final=is_final,
            )
        except Exception:
            logger.exception("[StreamingPipeline] eot_model.update_asr failed")
