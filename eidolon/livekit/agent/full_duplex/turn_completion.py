"""Single product-turn completion boundary for full-duplex sessions."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from ..observability import TurnTimeline
from .framework_completed_turn import FullDuplexFrameworkCompletedTurnGate
from .session_turn_boundary import FullDuplexSessionTurnBoundary
from .state_machine import FullDuplexPhase
from .turn_completion_policy import select_combined_voiceprint_result
from .voiceprint_commit_state import (
    CompletedVoiceprintTurnState,
    FullDuplexVoiceprintCommitState,
)

if TYPE_CHECKING:
    from .pipeline import StreamingPipeline


class FullDuplexTurnCompletion:
    """Own the only normal full-duplex boundary before a turn reaches the LLM.

    LiveKit's ``on_user_turn_completed`` hook remains the only normal commit
    boundary.  This owner also enforces a transcript-less liveness deadline:
    after VAD-stop, a candidate that receives neither transcript nor framework
    completion is explicitly rejected instead of remaining open until session
    teardown.
    """

    def __init__(self, pipeline: StreamingPipeline) -> None:
        self._pipeline = pipeline
        self._voiceprint_state = FullDuplexVoiceprintCommitState(pipeline)
        self._session_turns = FullDuplexSessionTurnBoundary(pipeline)
        self._framework_completed_turn = FullDuplexFrameworkCompletedTurnGate(
            pipeline,
            completion=self,
            session_turns=self._session_turns,
        )
        self._transcriptless_expiry_task: asyncio.Task[None] | None = None

    def cancel_transcriptless_expiry(self) -> None:
        task = self._transcriptless_expiry_task
        self._transcriptless_expiry_task = None
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()

    def arm_transcriptless_expiry(self) -> None:
        """Reject one stopped, text-free candidate at the product deadline."""

        self.cancel_transcriptless_expiry()
        coordinator = self._pipeline._user_turns
        candidate = coordinator.active
        if (
            candidate is None
            or candidate.state != "open"
            or coordinator.selected_text.strip()
        ):
            return
        self._transcriptless_expiry_task = asyncio.create_task(
            self._expire_transcriptless_candidate(
                candidate_id=candidate.candidate_id,
                timeline=candidate.timeline,
            )
        )

    async def _expire_transcriptless_candidate(
        self,
        *,
        candidate_id: str,
        timeline: TurnTimeline | None,
    ) -> None:
        pipeline = self._pipeline
        coordinator = pipeline._user_turns
        loop = asyncio.get_running_loop()
        started_at = loop.time()
        deadline = started_at + max(
            0.0,
            float(pipeline._stt_commit_transcript_timeout),
        )
        evidence_updates = 0
        try:
            while True:
                active = coordinator.active
                if (
                    active is None
                    or active.candidate_id != candidate_id
                    or active.state != "open"
                ):
                    return
                if coordinator.selected_text.strip():
                    return
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                version = coordinator.transcript_change_version
                changed = await coordinator.wait_for_transcript_change(
                    after_version=version,
                    timeout_sec=remaining,
                )
                if not changed:
                    break
                evidence_updates += 1

            active = coordinator.active
            if (
                active is None
                or active.candidate_id != candidate_id
                or active.state != "open"
                or coordinator.selected_text.strip()
            ):
                return
            reason = "speech_stopped_without_transcript_deadline"
            elapsed_ms = (loop.time() - started_at) * 1000.0
            if timeline is not None:
                timeline.set_attr(
                    "transcriptless_candidate_expiry",
                    {
                        "outcome": "rejected",
                        "reason": reason,
                        "elapsed_ms": elapsed_ms,
                        "evidence_updates": evidence_updates,
                    },
                )
            decision = coordinator.reject_active(reason)
            if decision.action != "reject":
                return
            pipeline._record_full_duplex_transition(
                FullDuplexPhase.USER_TURN_REJECTED,
                event="transcriptless_candidate_deadline",
                reason=reason,
                side_effect="irreversible",
                timeline=timeline,
            )
            self.cancel_completed_voiceprint_turn()
            self.reset_candidate_voiceprint_tasks()
            pipeline._flush_turn_timeline(timeline, reason)
        except asyncio.CancelledError:
            return
        finally:
            if self._transcriptless_expiry_task is asyncio.current_task():
                self._transcriptless_expiry_task = None

    def reset_candidate_voiceprint_tasks(self) -> None:
        self._voiceprint_state.reset_candidate_tasks()

    def remember_candidate_voiceprint_task(self, task: asyncio.Task | None) -> None:
        self._voiceprint_state.remember_candidate_task(task)

    def candidate_voiceprint_gate_task(self) -> asyncio.Task | None:
        tasks = self._voiceprint_state.pop_candidate_tasks()
        if not tasks:
            return None
        if len(tasks) == 1:
            return tasks[0]
        return asyncio.create_task(self._combine_candidate_voiceprint_results(tasks))

    async def _combine_candidate_voiceprint_results(
        self,
        tasks: list[asyncio.Task],
    ) -> Any:
        results = await asyncio.gather(*tasks)
        return select_combined_voiceprint_result(results)

    def remember_completed_voiceprint_turn(
        self,
        task: asyncio.Task | None,
        *,
        timeline: TurnTimeline | None,
    ) -> None:
        self._voiceprint_state.remember_completed_turn(task, timeline=timeline)

    def remember_completed_voiceprint_result(self, result: Any) -> None:
        self._voiceprint_state.remember_completed_result(result)

    def completed_voiceprint_turn(
        self,
        *,
        fallback_timeline: TurnTimeline | None,
    ) -> CompletedVoiceprintTurnState:
        return self._voiceprint_state.completed_turn(fallback_timeline=fallback_timeline)

    def clear_completed_voiceprint_turn(self) -> None:
        self._voiceprint_state.clear_completed_turn()

    def cancel_completed_voiceprint_turn(self) -> None:
        self._voiceprint_state.cancel_completed_turn()

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

    async def voiceprint_allows_completed_turn(
        self,
        *,
        turn_ctx: Any,
        new_message: Any,
    ) -> bool:
        self.cancel_transcriptless_expiry()
        return await self._framework_completed_turn.allows_completed_turn(
            turn_ctx=turn_ctx,
            new_message=new_message,
        )

    def _record_voiceprint_commit_gate(
        self,
        timeline: TurnTimeline | None,
        *,
        allowed: bool,
        reason: str,
    ) -> None:
        if timeline is None:
            return
        timeline.set_attr(
            "voiceprint_commit_gate",
            {"allowed": allowed, "reason": reason},
        )
