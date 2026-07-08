import asyncio
from contextlib import suppress
from types import SimpleNamespace

import pytest

from eidolon.livekit.agent.full_duplex.voiceprint_commit_state import (
    FullDuplexVoiceprintCommitState,
)
from eidolon.livekit.agent.observability import TurnTimeline


def test_candidate_voiceprint_tasks_pop_and_reset() -> None:
    owner = SimpleNamespace()
    state = FullDuplexVoiceprintCommitState(owner)
    first = SimpleNamespace()
    second = SimpleNamespace()

    state.remember_candidate_task(None)
    state.remember_candidate_task(first)
    state.remember_candidate_task(second)

    assert state.pop_candidate_tasks() == [first, second]
    assert state.pop_candidate_tasks() == []

    state.remember_candidate_task(first)
    state.reset_candidate_tasks()
    assert state.pop_candidate_tasks() == []


def test_completed_voiceprint_turn_snapshot_and_clear() -> None:
    owner = SimpleNamespace()
    state = FullDuplexVoiceprintCommitState(owner)
    task = SimpleNamespace()
    timeline = TurnTimeline("voiceprint-turn")
    fallback = TurnTimeline("fallback-turn")

    state.remember_completed_turn(task, timeline=timeline)
    snapshot = state.completed_turn(fallback_timeline=fallback)
    assert snapshot.task is task
    assert snapshot.result is None
    assert snapshot.timeline is timeline

    result = SimpleNamespace(commit_allowed=True)
    state.remember_completed_result(result)
    snapshot = state.completed_turn(fallback_timeline=fallback)
    assert snapshot.result is result

    state.clear_completed_turn()
    snapshot = state.completed_turn(fallback_timeline=fallback)
    assert snapshot.task is None
    assert snapshot.result is None
    assert snapshot.timeline is fallback


@pytest.mark.asyncio
async def test_cancel_completed_voiceprint_turn_cancels_pending_task() -> None:
    owner = SimpleNamespace()
    state = FullDuplexVoiceprintCommitState(owner)
    task = asyncio.create_task(asyncio.sleep(60))

    state.remember_completed_turn(task, timeline=None)
    state.cancel_completed_turn()

    assert task.cancelled() or task.cancelling()
    snapshot = state.completed_turn(fallback_timeline=None)
    assert snapshot.task is None
    assert snapshot.result is None
    with suppress(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_pending_voiceprint_commit_task_discards_when_done() -> None:
    owner = SimpleNamespace()
    state = FullDuplexVoiceprintCommitState(owner)
    task = asyncio.create_task(asyncio.sleep(0))

    state.add_pending_commit_task(task)
    assert task in state.pending_commit_tasks()
    await task
    await asyncio.sleep(0)

    assert task not in state.pending_commit_tasks()
