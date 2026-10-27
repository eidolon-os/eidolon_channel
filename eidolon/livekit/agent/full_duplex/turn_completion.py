"""Single product-turn completion boundary for full-duplex sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..observability import TurnTimeline
from .framework_completed_turn import FullDuplexFrameworkCompletedTurnGate
from .session_turn_boundary import FullDuplexSessionTurnBoundary
from .state_machine import FullDuplexPhase

if TYPE_CHECKING:
    from .pipeline import StreamingPipeline


class FullDuplexTurnCompletion:
    """Own the only normal full-duplex boundary before a turn reaches the LLM.

    LiveKit's ``on_user_turn_completed`` hook remains the only normal commit
    boundary.  Transcript-less liveness is driven by LiveKit's public
    ``user_transcription_timeout`` event, so this owner applies the product
    rejection and recovery effects without duplicating the framework timer.
    """

    def __init__(self, pipeline: StreamingPipeline) -> None:
        self._pipeline = pipeline
        self._session_turns = FullDuplexSessionTurnBoundary(pipeline)
        self._framework_completed_turn = FullDuplexFrameworkCompletedTurnGate(
            pipeline,
            completion=self,
            session_turns=self._session_turns,
        )

    def handle_transcription_timeout(self, event: Any) -> bool:
        """Reject one text-free candidate after LiveKit's public timeout event."""

        coordinator = self._pipeline._user_turns
        candidate = coordinator.active
        if (
            candidate is None
            or candidate.state != "open"
            or coordinator.selected_text.strip()
        ):
            return False

        pipeline = self._pipeline
        timeline = candidate.timeline
        reason = "speech_stopped_without_transcript_deadline"
        if timeline is not None:
            timeline.set_attr(
                "transcriptless_candidate_expiry",
                {
                    "outcome": "rejected",
                    "reason": reason,
                    "source": "livekit_user_transcription_timeout",
                    "speech_duration": float(
                        getattr(event, "speech_duration", 0.0) or 0.0
                    ),
                    "vad_speech_started_at": float(
                        getattr(event, "vad_speech_started_at", 0.0) or 0.0
                    ),
                },
            )
        decision = coordinator.reject_active(reason)
        if decision.action != "reject":
            return False
        pipeline._record_full_duplex_transition(
            FullDuplexPhase.USER_TURN_REJECTED,
            event="livekit_user_transcription_timeout",
            reason=reason,
            side_effect="irreversible",
            timeline=timeline,
        )
        # VAD without any transcript is not evidence of a user request or a
        # provider failure. Reject quietly: speaking here turns ambient noise
        # into assistant activity and perpetually renews the room idle window.
        pipeline._flush_turn_timeline(timeline, reason)
        return True


    def notify_silent_output_failure_once(
        self,
        *,
        timeline: TurnTimeline | None,
        error_type: str,
    ) -> bool:
        return self._session_turns.notify_silent_output_failure_once(
            timeline=timeline,
            error_type=error_type,
        )

    async def allows_completed_turn(
        self,
        *,
        turn_ctx: Any,
        new_message: Any,
    ) -> bool:
        return await self._framework_completed_turn.allows_completed_turn(
            turn_ctx=turn_ctx,
            new_message=new_message,
        )
