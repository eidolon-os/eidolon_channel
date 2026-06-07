"""Focused tests for duck suspend-window deadline handling."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from eidolon.livekit.agent.output import DuckingStats
from eidolon.livekit.agent.session.duck_timeout import DuckSuspendTimeoutHandler
from eidolon.livekit.agent.turn_policy import Action, Decision, InterruptIntent


async def _no_sleep(_timeout_sec: float) -> None:
    return None


def _eot_model(*, score: float = 0.0, max_suspend_sec: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(
        current_eot_score=score,
        _config=SimpleNamespace(duck_buffer_max_sec=max_suspend_sec),
    )


def _handler(
    *,
    runtime: MagicMock,
    duck_suspended: bool = True,
    suspend_start: float | None = None,
    latest_asr_text: str = "",
    vad_active: bool = True,
    eot_model: SimpleNamespace | None = None,
) -> tuple[DuckSuspendTimeoutHandler, SimpleNamespace]:
    calls = SimpleNamespace(
        create_task=MagicMock(),
        set_timeout_task=MagicMock(),
        apply_decision=MagicMock(),
    )
    handler = DuckSuspendTimeoutHandler(
        turn_runtime=runtime,
        sleep=_no_sleep,
        create_task=calls.create_task,
        get_duck_suspended=lambda: duck_suspended,
        get_duck_stats=lambda: DuckingStats(suspend_ms=123.0),
        get_suspend_start=lambda: (
            time.monotonic() if suspend_start is None else suspend_start
        ),
        set_timeout_task=calls.set_timeout_task,
        get_latest_asr_text=lambda: latest_asr_text,
        get_vad_active=lambda: vad_active,
        get_eot_model=lambda: eot_model or _eot_model(),
        apply_decision=calls.apply_decision,
    )
    return handler, calls


@pytest.mark.asyncio
async def test_deadline_noops_when_duck_no_longer_suspended() -> None:
    runtime = MagicMock()
    handler, calls = _handler(runtime=runtime, duck_suspended=False)

    await handler.run(0.01)

    runtime.deadline_decision.assert_not_called()
    calls.apply_decision.assert_not_called()


@pytest.mark.asyncio
async def test_hold_rearms_before_max_suspend_budget() -> None:
    runtime = MagicMock()
    decision = Decision(
        action=Action.HOLD,
        reason="deadline_wait_for_transcript",
        intent=InterruptIntent.UNCERTAIN,
    )
    runtime.deadline_decision.return_value = decision
    handler, calls = _handler(
        runtime=runtime,
        suspend_start=time.monotonic(),
        eot_model=_eot_model(max_suspend_sec=1.0),
    )

    await handler.run(0.01)

    calls.create_task.assert_called_once()
    calls.set_timeout_task.assert_called_once_with(calls.create_task.return_value)
    calls.apply_decision.assert_called_once_with(
        decision,
        resolved_reason="timeout",
        transcript="",
        vad_active=True,
    )
    calls.create_task.call_args.args[0].close()


@pytest.mark.asyncio
async def test_hold_rolls_back_after_max_suspend_budget() -> None:
    runtime = MagicMock()
    runtime.deadline_decision.return_value = Decision(
        action=Action.HOLD,
        reason="deadline_wait_for_transcript",
        intent=InterruptIntent.UNCERTAIN,
    )
    runtime.tiers.annotate_decision.side_effect = lambda decision: decision
    handler, calls = _handler(
        runtime=runtime,
        suspend_start=time.monotonic() - 2.0,
        eot_model=_eot_model(max_suspend_sec=0.01),
    )

    await handler.run(0.01)

    calls.create_task.assert_not_called()
    applied = calls.apply_decision.call_args.args[0]
    assert applied.action is Action.ROLLBACK
    assert applied.rollback_drop_buffered is True
    assert applied.intent is InterruptIntent.UNCERTAIN
