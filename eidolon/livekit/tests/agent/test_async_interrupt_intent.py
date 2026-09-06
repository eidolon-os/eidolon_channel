"""Candidate-scoped asynchronous evidence must not bypass the existing owner."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from eidolon.livekit.agent.output import DuckingStats
from eidolon.livekit.agent.session.semantic_interrupt import SemanticInterruptHandler
from eidolon.livekit.agent.turn_policy import Action, InterruptIntent, InterruptIntentResult, TurnPolicyRuntime
from eidolon.livekit.common.config import TurnPolicyConfig


def result(intent):
    return InterruptIntentResult(intent, 0.0, 'model', 'test evidence')


def setup_handler(classifier, *, vad=False, timeout=1500):
    cfg = TurnPolicyConfig()
    cfg = replace(cfg, interrupt=replace(cfg.interrupt, intent_provider='llm', intent_timeout_ms=timeout))
    runtime = TurnPolicyRuntime(cfg)
    state = SimpleNamespace(vad=vad, text='完整转写', scope=(('candidate', 1), 123), active=True)
    decisions = []

    def apply(decision, **kwargs):
        decisions.append(decision)
        if decision.action in {Action.CANCEL, Action.ROLLBACK}:
            state.active = False

    model = SimpleNamespace(current_eot_score=.99, should_interrupt=MagicMock())
    handler = SemanticInterruptHandler(
        get_eot_model=lambda: model, turn_runtime=runtime,
        get_timeline=lambda: None, get_duck_active=lambda: state.active,
        get_duck_stats=lambda: DuckingStats(), get_vad_active=lambda: state.vad,
        soft_interrupt_active=lambda: False, soft_interrupt_timeout=lambda: .45,
        apply_decision=apply, interrupt_current_turn=MagicMock(), enter_soft_interrupt=MagicMock(),
        intent_classifier=classifier, get_candidate_scope=lambda: state.scope if state.active else None,
        get_final_transcript=lambda: state.text, get_assistant_text=lambda: '正在介绍方案',
    )
    return handler, state, decisions, model


@pytest.mark.asyncio
async def test_backchannel_waits_for_vad_end_and_reuses_final_result():
    classifier = SimpleNamespace(classify=AsyncMock(return_value=result(InterruptIntent.BACKCHANNEL)))
    handler, state, decisions, model = setup_handler(classifier, vad=True)
    try:
        handler.run(state.text, is_final=True)
        await handler._intent_task
        assert all(d.action is Action.HOLD for d in decisions)
        state.vad = False
        handler.run(state.text, is_final=True)
        assert decisions[-1].action is Action.ROLLBACK
        assert not decisions[-1].rollback_drop_buffered
        classifier.classify.assert_awaited_once()
        model.should_interrupt.assert_not_called()
    finally:
        await handler.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('intent', [InterruptIntent.CORRECTION, InterruptIntent.TOPIC_SWITCH, InterruptIntent.HARD_STOP])
async def test_final_intent_can_cancel_without_high_eot(intent):
    classifier = SimpleNamespace(classify=AsyncMock(return_value=result(intent)))
    handler, state, decisions, model = setup_handler(classifier)
    model.current_eot_score = 0.0
    try:
        handler.run(state.text, is_final=True)
        assert decisions[-1].action is Action.HOLD
        await handler._intent_task
        assert decisions[-1].action is Action.CANCEL
        assert decisions[-1].intent is intent
    finally:
        await handler.aclose()


@pytest.mark.asyncio
async def test_repeated_final_starts_one_request_and_no_legacy_timer_cut():
    ready = asyncio.Event()
    async def classify(*args, **kwargs):
        await ready.wait()
        return result(InterruptIntent.CORRECTION)
    classifier = SimpleNamespace(classify=AsyncMock(side_effect=classify))
    handler, state, decisions, _ = setup_handler(classifier)
    try:
        for _ in range(5):
            handler.run(state.text, is_final=True)
        await asyncio.sleep(0)
        assert classifier.classify.await_count == 1
        deadline = handler._turn_runtime.deadline_decision(True, has_transcript=True, transcript=state.text, eot_score=.99)
        assert deadline.action is Action.HOLD
        ready.set()
        await handler._intent_task
        assert sum(d.action is Action.CANCEL for d in decisions) == 1
    finally:
        await handler.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['text', 'generation', 'response'])
async def test_late_result_cannot_control_revised_or_replaced_turn(change):
    ready = asyncio.Event()
    async def classify(*args, **kwargs):
        try:
            await ready.wait()
        except asyncio.CancelledError:
            await ready.wait()  # A non-cooperative provider still cannot own the new turn.
        return result(InterruptIntent.HARD_STOP)
    handler, state, decisions, _ = setup_handler(SimpleNamespace(classify=classify))
    try:
        handler.run(state.text, is_final=True)
        task = handler._intent_task
        await asyncio.sleep(0)
        if change == 'text':
            state.text = '后来更正的文本'
            handler.run(state.text, is_final=False)
        elif change == 'generation':
            state.scope = (('candidate', 2), 123)
        else:
            state.scope = (('candidate', 1), 456)
        ready.set()
        await task
        assert all(d.action is Action.HOLD for d in decisions)
    finally:
        await handler.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['timeout', 'error'])
async def test_failed_intent_resumes_after_silence_without_destroying_audio(failure):
    async def classify(*args, **kwargs):
        if failure == 'error':
            raise RuntimeError('unavailable')
        await asyncio.Event().wait()
    handler, state, decisions, _ = setup_handler(SimpleNamespace(classify=classify), timeout=10)
    try:
        handler.run(state.text, is_final=True)
        await handler._intent_task
        assert decisions[-1].action is Action.ROLLBACK
        assert not decisions[-1].rollback_drop_buffered
    finally:
        await handler.aclose()


@pytest.mark.asyncio
async def test_shutdown_cancels_pending_intent_without_late_effect():
    classifier = SimpleNamespace(classify=AsyncMock(side_effect=lambda *args, **kwargs: None))
    async def waiting(*args, **kwargs):
        await asyncio.Event().wait()
    classifier.classify.side_effect = waiting
    handler, state, decisions, _ = setup_handler(classifier)
    handler.run(state.text, is_final=True)
    await asyncio.sleep(0)
    await handler.aclose()
    assert all(d.action is Action.HOLD for d in decisions)


@pytest.mark.asyncio
@pytest.mark.parametrize('superseded_by', ['resolved', 'generation', 'response', 'final_revision'])
async def test_endpointing_does_not_wait_for_superseded_intent(superseded_by):
    pending = asyncio.Event()
    async def classify(*args, **kwargs):
        await pending.wait()
        return result(InterruptIntent.HARD_STOP)

    handler, state, decisions, _ = setup_handler(SimpleNamespace(classify=classify))
    try:
        handler.run(state.text, is_final=True)
        await asyncio.sleep(0)
        old_task = handler._intent_task
        if superseded_by == 'resolved':
            state.active = False
        elif superseded_by == 'generation':
            state.scope = (('candidate', 2), 123)
        elif superseded_by == 'response':
            state.scope = (('candidate', 1), 456)
        else:
            state.text = '当前 final 已被修订'
        # The old provider remains pending. Endpointing the new input must
        # finish independently, without cancelling or trusting that request.
        await asyncio.wait_for(handler.wait_for_pending_intent(), timeout=.05)
        assert not old_task.done()
        pending.set()
        await old_task
        assert all(d.action is Action.HOLD for d in decisions)
    finally:
        await handler.aclose()


@pytest.mark.asyncio
async def test_endpointing_cancellation_does_not_cancel_current_intent():
    pending = asyncio.Event()
    async def classify(*args, **kwargs):
        await pending.wait()
        return result(InterruptIntent.CORRECTION)

    handler, state, decisions, _ = setup_handler(SimpleNamespace(classify=classify))
    try:
        handler.run(state.text, is_final=True)
        waiter = asyncio.create_task(handler.wait_for_pending_intent())
        await asyncio.sleep(0)
        assert not waiter.done()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not handler._intent_task.done()
        pending.set()
        await handler._intent_task
        assert decisions[-1].action is Action.CANCEL
    finally:
        await handler.aclose()
