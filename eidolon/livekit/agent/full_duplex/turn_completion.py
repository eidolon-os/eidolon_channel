"""Single product-turn completion boundary for full-duplex sessions."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from ..observability import TurnTimeline
from .framework_completed_turn import FullDuplexFrameworkCompletedTurnGate
from .session_turn_boundary import FullDuplexSessionTurnBoundary
from .turn_completion_policy import select_combined_voiceprint_result
from .voiceprint_commit_state import (
    CompletedVoiceprintTurnState,
    FullDuplexVoiceprintCommitState,
)

if TYPE_CHECKING:
    from .pipeline import StreamingPipeline


class FullDuplexTurnCompletion:
    """Own the only normal full-duplex boundary before a turn reaches the LLM.

    VAD-stop, interruption effects, timers, and voiceprint callbacks may record
    evidence, but none of them can commit or clear a user turn. LiveKit's
    ``on_user_turn_completed`` hook enters through :meth:`voiceprint_allows_completed_turn`
    and the framework-completed gate performs the product terminal decision.
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
