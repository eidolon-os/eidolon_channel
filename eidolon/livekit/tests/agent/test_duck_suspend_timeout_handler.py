"""Focused tests for duck suspend-window deadline handling."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.output import DuckingStats
from eidolon.livekit.agent.session.duck_timeout import DuckSuspendTimeoutHandler
from eidolon.livekit.agent.session.interruption_orchestrator import InterruptionOrchestrator
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
    should_hold_for_evidence: bool = False,
    max_suspend_sec: float = 0.0,
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
        should_hold_for_evidence=lambda: should_hold_for_evidence,
        get_max_suspend_sec=lambda: max_suspend_sec,
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
async def test_owner_no_evidence_budget_caps_generic_duck_buffer() -> None:
    runtime = MagicMock()
    decision = Decision(
        action=Action.HOLD,
        reason="deadline_wait_for_transcript",
        intent=InterruptIntent.UNCERTAIN,
    )
    runtime.deadline_decision.return_value = decision
    handler, calls = _handler(
        runtime=runtime,
        suspend_start=time.monotonic() - 0.45,
        eot_model=_eot_model(max_suspend_sec=2.0),
        max_suspend_sec=0.8,
    )

    await handler.run(0.45)

    rearmed_coro = calls.create_task.call_args.args[0]
    assert rearmed_coro.cr_frame is not None
    assert rearmed_coro.cr_frame.f_locals["timeout_sec"] == pytest.approx(0.35, abs=0.02)
    rearmed_coro.close()


@pytest.mark.asyncio
async def test_hold_rearms_with_policy_recheck_budget() -> None:
    runtime = MagicMock()
    decision = Decision(
        action=Action.HOLD,
        reason="semantic_score_wait score=0.00 evidence=interim_substantive",
        intent=InterruptIntent.UNCERTAIN,
        hold_recheck_ms=120,
    )
    runtime.deadline_decision.return_value = decision
    handler, calls = _handler(
        runtime=runtime,
        suspend_start=time.monotonic(),
        eot_model=_eot_model(max_suspend_sec=2.0),
    )

    await handler.run(0.5)

    calls.create_task.assert_called_once()
    rearmed_coro = calls.create_task.call_args.args[0]
    calls.set_timeout_task.assert_called_once_with(calls.create_task.return_value)
    calls.apply_decision.assert_called_once_with(
        decision,
        resolved_reason="timeout",
        transcript="",
        vad_active=True,
    )
    assert rearmed_coro.cr_frame is not None
    assert rearmed_coro.cr_frame.f_locals["timeout_sec"] == pytest.approx(0.12)
    rearmed_coro.close()


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


@pytest.mark.asyncio
async def test_vad_idle_can_hold_for_post_speech_evidence_window() -> None:
    runtime = MagicMock()
    runtime.deadline_decision.return_value = Decision(
        action=Action.ROLLBACK,
        reason="deadline_vad_idle_drop_stale",
        rollback_drop_buffered=True,
        intent=InterruptIntent.UNCERTAIN,
    )
    handler, calls = _handler(
        runtime=runtime,
        suspend_start=time.monotonic(),
        vad_active=False,
        should_hold_for_evidence=True,
        max_suspend_sec=6.0,
    )

    await handler.run(0.5)

    calls.create_task.assert_called_once()
    applied = calls.apply_decision.call_args.args[0]
    assert applied.action is Action.HOLD
    assert applied.reason == "deadline_wait_for_post_speech_evidence"
    calls.create_task.call_args.args[0].close()


@pytest.mark.asyncio
async def test_active_speech_backchannel_deadline_cannot_terminally_rollback() -> None:
    runtime = MagicMock()
    runtime.deadline_decision.return_value = Decision(
        action=Action.ROLLBACK,
        reason="deadline_intent:backchannel",
        rollback_drop_buffered=False,
        intent=InterruptIntent.BACKCHANNEL,
    )
    handler, calls = _handler(
        runtime=runtime,
        suspend_start=time.monotonic(),
        latest_asr_text="好",
        vad_active=True,
        should_hold_for_evidence=True,
        max_suspend_sec=6.0,
    )

    await handler.run(0.45)

    applied = calls.apply_decision.call_args.args[0]
    assert applied.action is Action.HOLD
    assert applied.reason == "deadline_wait_for_active_speech_evidence"
    assert applied.intent is InterruptIntent.BACKCHANNEL
    calls.create_task.call_args.args[0].close()


# ---------------------------------------------------------------------------
# Integration with a real InterruptionOrchestrator: a no-transcript false
# interrupt must RESUME at the short no-evidence window, not hold the full 6s
# (fix f5aad9f). Complements the orchestrator-level decision test in
# test_false_interrupt_no_evidence_resume.py.
# ---------------------------------------------------------------------------


class _Clock:
    def __init__(self, t: float = 100.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _post_speech_no_transcript_owner(clock: _Clock) -> InterruptionOrchestrator:
    owner = InterruptionOrchestrator(
        evidence_timeout_sec=6.0,
        min_speech_sec=0.25,
        no_evidence_timeout_sec=0.8,
        clock=clock,
    )
    owner.start_candidate(timeline=TurnTimeline("t"))
    clock.t += 0.7  # spoke ~700ms, then VAD end with no transcript at all
    owner.defer_false_resume_after_speech_end(transcript="", duck_suspended=True)
    return owner


def _owner_handler(
    owner: InterruptionOrchestrator,
) -> tuple[DuckSuspendTimeoutHandler, SimpleNamespace]:
    runtime = MagicMock()
    runtime.deadline_decision.return_value = Decision(
        action=Action.ROLLBACK,
        reason="deadline_vad_idle_drop_stale",
        rollback_drop_buffered=True,
        intent=InterruptIntent.UNCERTAIN,
    )
    runtime.tiers.annotate_decision.side_effect = lambda d: d
    calls = SimpleNamespace(
        create_task=MagicMock(),
        set_timeout_task=MagicMock(),
        apply_decision=MagicMock(),
    )
    handler = DuckSuspendTimeoutHandler(
        turn_runtime=runtime,
        sleep=_no_sleep,
        create_task=calls.create_task,
        get_duck_suspended=lambda: True,
        get_duck_stats=lambda: DuckingStats(suspend_ms=1.0),
        get_suspend_start=lambda: time.monotonic(),
        set_timeout_task=calls.set_timeout_task,
        get_latest_asr_text=lambda: "",
        get_vad_active=lambda: False,
        get_eot_model=lambda: _eot_model(max_suspend_sec=1.0),
        apply_decision=calls.apply_decision,
        should_hold_for_evidence=owner.should_hold_deadline,
        get_max_suspend_sec=owner.max_suspend_sec,
    )
    return handler, calls


@pytest.mark.asyncio
async def test_no_transcript_false_interrupt_resumes_after_grace() -> None:
    clock = _Clock()
    owner = _post_speech_no_transcript_owner(clock)
    handler, calls = _owner_handler(owner)

    clock.t += 0.85  # past the 0.8s no-evidence grace, still no transcript

    await handler.run(0.01)

    applied = calls.apply_decision.call_args.args[0]
    assert applied.action is Action.ROLLBACK  # resumed
    calls.create_task.assert_not_called()  # did NOT re-arm / hold to full 6s


@pytest.mark.asyncio
async def test_no_transcript_false_interrupt_holds_within_grace() -> None:
    clock = _Clock()
    owner = _post_speech_no_transcript_owner(clock)
    handler, calls = _owner_handler(owner)

    clock.t += 0.2  # still within the no-evidence grace

    await handler.run(0.01)

    applied = calls.apply_decision.call_args.args[0]
    assert applied.action is Action.HOLD
    assert applied.reason == "deadline_wait_for_post_speech_evidence"
    calls.create_task.call_args.args[0].close()  # close the re-armed coro
