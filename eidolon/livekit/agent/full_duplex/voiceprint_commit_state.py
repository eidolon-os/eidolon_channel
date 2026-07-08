"""Runtime state adapter for full-duplex voiceprint commit tasks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..observability import TurnTimeline


@dataclass(frozen=True)
class CompletedVoiceprintTurnState:
    """Snapshot of the framework-completed voiceprint gate state."""

    task: asyncio.Task | None
    result: Any | None
    timeline: TurnTimeline | None


class FullDuplexVoiceprintCommitState:
    """Own voiceprint task bookkeeping while preserving pipeline storage."""

    def __init__(self, pipeline: Any) -> None:
        self._pipeline = pipeline

    def pending_commit_tasks(self) -> set[asyncio.Task]:
        tasks = getattr(self._pipeline, "_pending_voiceprint_commit_tasks", None)
        if tasks is None:
            tasks = set()
            self._pipeline._pending_voiceprint_commit_tasks = tasks
        return tasks

    def add_pending_commit_task(self, task: asyncio.Task) -> None:
        self.pending_commit_tasks().add(task)
        task.add_done_callback(self.pending_commit_tasks().discard)

    def reset_candidate_tasks(self) -> None:
        self._pipeline._candidate_voiceprint_tasks = []

    def remember_candidate_task(self, task: asyncio.Task | None) -> None:
        if task is None:
            return
        if not hasattr(self._pipeline, "_candidate_voiceprint_tasks"):
            self._pipeline._candidate_voiceprint_tasks = []
        self._pipeline._candidate_voiceprint_tasks.append(task)

    def pop_candidate_tasks(self) -> list[asyncio.Task]:
        tasks = list(getattr(self._pipeline, "_candidate_voiceprint_tasks", []))
        self._pipeline._candidate_voiceprint_tasks = []
        return tasks

    def remember_completed_turn(
        self,
        task: asyncio.Task | None,
        *,
        timeline: TurnTimeline | None,
    ) -> None:
        self._pipeline._completed_turn_voiceprint_task = task
        self._pipeline._completed_turn_voiceprint_result = None
        self._pipeline._completed_turn_voiceprint_timeline = timeline

    def remember_completed_result(self, result: Any) -> None:
        self._pipeline._completed_turn_voiceprint_result = result

    def completed_turn(
        self,
        *,
        fallback_timeline: TurnTimeline | None,
    ) -> CompletedVoiceprintTurnState:
        return CompletedVoiceprintTurnState(
            task=getattr(self._pipeline, "_completed_turn_voiceprint_task", None),
            result=getattr(self._pipeline, "_completed_turn_voiceprint_result", None),
            timeline=(
                getattr(self._pipeline, "_completed_turn_voiceprint_timeline", None)
                or fallback_timeline
            ),
        )

    def clear_completed_turn(self) -> None:
        self._pipeline._completed_turn_voiceprint_task = None
        self._pipeline._completed_turn_voiceprint_result = None
        self._pipeline._completed_turn_voiceprint_timeline = None

    def cancel_completed_turn(self) -> None:
        task = getattr(self._pipeline, "_completed_turn_voiceprint_task", None)
        if task is not None and not task.done():
            task.cancel()
        self.clear_completed_turn()
