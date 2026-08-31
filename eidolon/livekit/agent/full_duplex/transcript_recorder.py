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

    def record(self, transcript_event: FullDuplexTranscriptEvent) -> None:
        if not transcript_event.has_transcript:
            return

        pipeline = self._pipeline
        # Real recognized speech (interim or final) keeps the session alive.
        # Empty/noise transcripts deliberately don't, so a silent room still
        # trips the idle watchdog.
        pipeline._mark_activity()
        pipeline._latest_asr_text = transcript_event.transcript
        orchestrator = getattr(pipeline, "_interruption_orchestrator", None)
        if (
            orchestrator is not None
            and pipeline._barge_in_enabled
            and not pipeline._uses_livekit_native_adaptive_interruption()
        ):
            orchestrator.note_transcript(
                transcript_event.transcript,
                is_final=transcript_event.is_final,
            )
        pipeline._ensure_user_turn_coordinator()
        buffered_evidence = pipeline._ensure_transcript_evidence_buffer().take(
            transcript_event.transcript,
            is_final=transcript_event.is_final,
        )
        evidence = (
            buffered_evidence.merge(transcript_event.evidence)
            if buffered_evidence is not None
            else transcript_event.evidence
        )
        transcript_kwargs = {"evidence": evidence} if evidence.available else {}
        receipt = pipeline._user_turns.add_transcript(
            transcript_event.transcript,
            is_final=transcript_event.is_final,
            **transcript_kwargs,
        )
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
        if pipeline._timeline is not None:
            pipeline._timeline.mark(transcript_event.timeline_mark)
        try:
            pipeline._get_eot_model().update_asr(
                transcript_event.transcript,
                is_final=transcript_event.is_final,
            )
        except Exception:
            logger.exception("[StreamingPipeline] eot_model.update_asr failed")
