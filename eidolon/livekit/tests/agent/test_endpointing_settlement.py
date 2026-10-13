"""Generation-scoped settlement at the automatic SDK endpoint boundary."""

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from eidolon.livekit.agent.full_duplex.pipeline import StreamingPipeline
from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.session.interruption_orchestrator import InterruptionOrchestrator
from eidolon.livekit.agent.turn_policy import Action, Decision


def make_owner(*, budget=1, clock=None):
    owner = InterruptionOrchestrator(
        evidence_timeout_sec=budget, min_speech_sec=0, clock=clock,
    )
    owner.start_candidate(timeline=TurnTimeline("candidate"), generation_id=1)
    return owner


@pytest.mark.asyncio
async def test_provisional_resume_does_not_release_endpoint_and_waiter_cancel_is_local():
    owner = make_owner()
    first = asyncio.create_task(owner.wait_for_resolution(("candidate", 1)))
    second = asyncio.create_task(owner.wait_for_resolution(("candidate", 1)))
    await asyncio.sleep(0)
    owner.note_turn_policy_decision(Decision(Action.RESUME, "await_more_evidence"))
    await asyncio.sleep(0)
    assert not first.done() and not second.done()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert owner.active
    owner.resolve(action="cancel", reason="confirmed")
    await asyncio.wait_for(second, 1)


@pytest.mark.asyncio
async def test_supersession_wakes_old_waiter_without_waiting_for_new_candidate():
    owner = make_owner()
    waiter = asyncio.create_task(owner.wait_for_resolution(("candidate", 1)))
    await asyncio.sleep(0)
    owner.start_candidate(timeline=TurnTimeline("candidate"), generation_id=2)
    await asyncio.wait_for(waiter, 1)
    assert owner.active_key == ("candidate", 2)


@pytest.mark.asyncio
async def test_wait_uses_remaining_budget_and_does_not_make_policy_decisions():
    now = 10.0
    owner = make_owner(budget=1, clock=lambda: now)
    owner.defer_false_resume_after_speech_end(transcript="有效转写", duck_suspended=True)
    now += 1.1
    with pytest.raises(TimeoutError):
        await owner.wait_for_resolution(("candidate", 1))
    assert owner.active
    assert owner.verdict_for("candidate", generation_id=1) is None


@pytest.mark.asyncio
async def test_session_exit_releases_waiters_and_refuses_new_endpoint_queries():
    owner = make_owner(budget=60)
    waiter = asyncio.create_task(owner.wait_for_resolution(("candidate", 1)))
    await asyncio.sleep(0)
    owner.close()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(waiter, 1)
    with pytest.raises(asyncio.CancelledError):
        await owner.wait_for_resolution(("candidate", 1))
    assert owner.verdict_for("candidate", generation_id=1) is None


def pipeline(owner):
    return NS(
        _interruption_orchestrator=owner,
        _semantic_interrupts=NS(wait_for_pending_intent=AsyncMock()),
        _user_turns=NS(active=NS(candidate_id="candidate"), current_generation_id=1),
        _session=NS(current_speech=NS(interrupted=False, wait_for_playout=AsyncMock())),
        _duck_deadline=NS(run=AsyncMock()),
    )


@pytest.mark.asyncio
async def test_endpoint_waits_for_verdict_then_cancelled_speech_cleanup():
    owner = make_owner()
    p = pipeline(owner)
    cleanup = asyncio.Event()
    p._session.current_speech.wait_for_playout.side_effect = cleanup.wait
    waiter = asyncio.create_task(StreamingPipeline._settle_interrupt_before_endpointing(p))
    await asyncio.sleep(0)
    assert not waiter.done()
    owner.resolve(action="cancel", reason="confirmed")
    p._session.current_speech.interrupted = True
    await asyncio.sleep(0)
    assert not waiter.done()
    cleanup.set()
    await asyncio.wait_for(waiter, 1)
    p._session.current_speech.wait_for_playout.assert_awaited_once()
    p._duck_deadline.run.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("replace_at", ["intent", "verdict", "cleanup"])
async def test_old_endpoint_cannot_complete_replacement_generation(replace_at):
    owner = make_owner()
    p = pipeline(owner)

    async def replace(*args):
        p._user_turns.current_generation_id = 2
        owner.start_candidate(timeline=TurnTimeline("candidate"), generation_id=2)

    if replace_at == "intent":
        p._semantic_interrupts.wait_for_pending_intent.side_effect = replace
    elif replace_at == "cleanup":
        owner.resolve(action="cancel", reason="confirmed")
        p._session.current_speech.interrupted = True
        p._session.current_speech.wait_for_playout.side_effect = replace
    waiter = asyncio.create_task(StreamingPipeline._settle_interrupt_before_endpointing(p))
    if replace_at == "verdict":
        await asyncio.sleep(0)
        await replace()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert owner.active_key == ("candidate", 2)
    p._duck_deadline.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_expired_wait_reuses_existing_deadline_effects_without_cancelling_speech():
    owner = make_owner(budget=0)
    p = pipeline(owner)

    async def deadline(_):
        owner.resolve(action="rollback", reason="deadline_hold_owner_budget_elapsed")

    p._duck_deadline.run.side_effect = deadline
    await StreamingPipeline._settle_interrupt_before_endpointing(p)
    p._duck_deadline.run.assert_awaited_once_with(0)
    p._session.current_speech.wait_for_playout.assert_not_awaited()
    assert not owner.verdict_for("candidate", generation_id=1).continue_to_llm


@pytest.mark.asyncio
async def test_new_speech_rearming_deadline_does_not_release_unresolved_eot():
    owner = make_owner(budget=0)
    p = pipeline(owner)
    with pytest.raises(asyncio.CancelledError):
        await StreamingPipeline._settle_interrupt_before_endpointing(p)
    p._duck_deadline.run.assert_awaited_once_with(0)
    assert owner.active
