"""Accepted transcript side effects for full-duplex sessions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .transcript_event import FullDuplexTranscriptEvent

if TYPE_CHECKING:
    from .pipeline import StreamingPipeline

logger = logging.getLogger("agent")


class FullDuplexTranscriptRecorder:
    """Record accepted STT events into turn state, timeline, and EOT state."""

    def __init__(self, pipeline: StreamingPipeline) -> None:
        self._pipeline = pipeline

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
        pipeline._user_turns.add_transcript(
            transcript_event.transcript,
            is_final=transcript_event.is_final,
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
