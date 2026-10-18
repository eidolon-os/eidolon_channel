"""Transport contract and real policy consumption for SystemOne intent evidence."""

import asyncio
import json
from dataclasses import replace
from unittest.mock import MagicMock

import httpx
import pytest

from eidolon.livekit.agent.factory import SharedStageFactory
from eidolon.livekit.agent.turn_policy import Action, InterruptIntent
from eidolon.livekit.agent.turn_policy.intent_classifier import (
    INTENT_LABELS, InterruptIntentClassifier, LlmInterruptClassifier,
)
from eidolon.livekit.agent.turn_policy.systemone_intent_classifier import SystemOneInterruptClassifier
from eidolon.livekit.common.config import load_effective_config
from eidolon.livekit.common.config.validators import validate_effective_config
from eidolon.livekit.tests.agent.test_async_interrupt_intent import setup_handler
from eidolon.livekit.tests.agent.test_interrupt_model_config import config


def payload(label='stop', probability=.97):
    probabilities = {key: (1 - probability) / 3 for key in INTENT_LABELS}
    probabilities[label] = probability
    return {'answers': {'intent': {
        'type': 'choice', 'choice': label, 'probabilities': probabilities,
    }}}


def classifier(client=None, *, timeout_sec=1):
    return SystemOneInterruptClassifier(
        endpoint='http://decider.test/d3/v1/systemone', model='d3',
        client=client, timeout_sec=timeout_sec,
    )


def systemone_config():
    cfg = config(intent_provider='systemone')
    intr = cfg.turn_policy.interrupt
    return replace(cfg, turn_policy=replace(cfg.turn_policy, interrupt=replace(
        intr, systemone=replace(intr.systemone,
            endpoint='http://decider.test/d3/v1/systemone', model='d3'),
    )))


def test_both_implement_the_same_abstract_interface():
    with pytest.raises(TypeError):
        InterruptIntentClassifier()
    assert issubclass(LlmInterruptClassifier, InterruptIntentClassifier)
    assert issubclass(SystemOneInterruptClassifier, InterruptIntentClassifier)


@pytest.mark.asyncio
async def test_exact_request_preserves_context_and_returns_raw_scores():
    seen = []
    def handle(request):
        seen.append(request)
        return httpx.Response(200, json=payload('reply', .703))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        instance = classifier(client)
        result = await instance.classify('他说“别说了”是什么意思？', assistant_text='前' * 2100)
        body = json.loads(seen[0].content)
        assert str(seen[0].url) == 'http://decider.test/d3/v1/systemone'
        assert body['model'] == 'd3'
        assert json.loads(body['state']) == {
            'assistant_text': '前' * 2000, 'user_text': '他说“别说了”是什么意思？',
        }
        assert body['questions']['intent']['instructions'] == LlmInterruptClassifier._INSTRUCTIONS
        assert set(body['questions']) == {'intent'}
        assert result.intent is InterruptIntent.NORMAL_INTERRUPT
        assert result.confidence == .703
        assert result.probabilities == payload('reply', .703)['answers']['intent']['probabilities']
        await instance.aclose()
        assert not client.is_closed, 'injected clients belong to their caller'
        with pytest.raises(RuntimeError):
            await instance.classify('继续', assistant_text='')


@pytest.mark.parametrize('fault', [
    'missing_answers', 'missing_intent', 'missing_type', 'wrong_type', 'missing_probability',
    'extra_label', 'unknown_choice', 'nan', 'inf', 'negative', 'over_one', 'bool', 'string',
    'wrong_sum', 'not_argmax', 'null', 'list',
])
def test_reject_malformed_response(fault):
    data = payload()
    answer = data['answers']['intent']
    probabilities = answer['probabilities']
    if fault == 'missing_answers':
        data = {}
    elif fault == 'missing_intent':
        data['answers'] = {}
    elif fault == 'missing_type':
        answer.pop('type')
    elif fault == 'wrong_type':
        answer['type'] = 'score'
    elif fault == 'missing_probability':
        probabilities.pop('keep')
    elif fault == 'extra_label':
        probabilities['other'] = 0
    elif fault == 'unknown_choice':
        answer['choice'] = 'ready'
    elif fault in ['nan', 'inf', 'negative', 'over_one', 'bool', 'string']:
        probabilities['stop'] = {
            'nan': float('nan'), 'inf': float('inf'), 'negative': -.1,
            'over_one': 1.1, 'bool': True, 'string': '0.97',
        }[fault]
    elif fault == 'wrong_sum':
        probabilities['keep'] = .5
    elif fault == 'not_argmax':
        answer['choice'] = 'keep'
    elif fault == 'null':
        data = None
    elif fault == 'list':
        answer['probabilities'] = []
    with pytest.raises(ValueError):
        SystemOneInterruptClassifier._parse(data)


@pytest.mark.asyncio
@pytest.mark.parametrize('status', [302, 400, 401, 404, 413, 429, 500, 503])
async def test_http_error_has_no_retry_or_redirect(status):
    requests = []
    def handle(request):
        requests.append(request)
        return httpx.Response(status, json=payload(), headers={'location': 'http://elsewhere.test'})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle), follow_redirects=True) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await classifier(client).classify('停止', assistant_text='播报')
        assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', ['network', 'json', 'schema', 'timeout'])
async def test_failure_uses_existing_channel_uncertainty_policy(fault):
    async def handle(request):
        if fault == 'network':
            raise httpx.ConnectError('injected', request=request)
        if fault == 'timeout':
            await asyncio.Event().wait()
        return httpx.Response(200, text='not-json') if fault == 'json' else httpx.Response(200, json={})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        handler, state, decisions, _ = setup_handler(classifier(client), timeout=20, provider='systemone')
        try:
            handler.run(state.text, is_final=True)
            await handler._intent_task
            assert decisions[-1].action is Action.HOLD
            assert decisions[-1].intent_source == 'unavailable'
            assert not decisions[-1].rollback_drop_buffered
        finally:
            await handler.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('label', list(INTENT_LABELS))
@pytest.mark.parametrize('probability', [.97, .4])
@pytest.mark.parametrize('vad', [True, False])
async def test_four_labels_follow_same_policy_without_probability_gate(label, probability, vad):
    from eidolon.livekit.agent.observability import TurnTimeline
    timeline = TurnTimeline('systemone')
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=payload(label, probability)),
    )) as client:
        handler, state, decisions, _ = setup_handler(
            classifier(client), vad=vad, provider='systemone', timeline=timeline,
        )
        try:
            handler.run(state.text, is_final=True)
            await handler._intent_task
            expected = Action.CANCEL if label in ('stop', 'reply') else (
                Action.HOLD if vad or label == 'wait' else Action.ROLLBACK
            )
            assert decisions[-1].action is expected
            assert timeline.attrs['interrupt_intent_observations'][0]['probabilities'][label] == probability
        finally:
            await handler.aclose()


@pytest.mark.asyncio
async def test_cancellation_propagates_and_next_request_is_usable():
    entered = asyncio.Event()
    release = asyncio.Event()
    async def handle(request):
        entered.set()
        await release.wait()
        return httpx.Response(200, json=payload())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        instance = classifier(client)
        task = asyncio.create_task(instance.classify('停止', assistant_text='播报'))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        assert (await instance.classify('停止', assistant_text='播报')).intent is InterruptIntent.HARD_STOP


@pytest.mark.asyncio
async def test_warmup_is_bounded_and_owned_client_closes():
    async def handle(request):
        assert json.loads(json.loads(request.content)['state']) == {'user_text': '', 'assistant_text': ''}
        await asyncio.Event().wait()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(TimeoutError):
            await classifier(client, timeout_sec=.01).warmup()
    instance = classifier()
    await instance.aclose()
    await instance.aclose()
    assert instance._client.is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize('owner', ['channel', 'livekit_native_adaptive'])
async def test_factory_selects_systemone_without_building_chat_model(monkeypatch, owner):
    build = MagicMock()
    monkeypatch.setattr(SharedStageFactory, '_build_llm', build)
    cfg = systemone_config()
    cfg = replace(cfg, turn_policy=replace(cfg.turn_policy, interruption_owner=owner),
        providers=replace(cfg.providers, brain_provider='eidolon_agent'),
        remote_agent_rpc=replace(cfg.remote_agent_rpc, target='127.0.0.1:45051'),
        llm=replace(cfg.llm, model='', base_url=''))
    validate_effective_config(cfg)
    instance = SharedStageFactory.build_interrupt_classifier(cfg)
    build.assert_not_called()
    if owner == 'channel':
        assert isinstance(instance, InterruptIntentClassifier)
        assert isinstance(instance, SystemOneInterruptClassifier)
        await instance.aclose()
    else:
        assert instance is None


@pytest.mark.parametrize('endpoint,model', [
    ('', 'd3'), ('ftp://server/path', 'd3'), ('http://user:pass@server/path', 'd3'),
    ('http://server:wrong/path', 'd3'), ('http://server/path', ''), (None, 'd3'),
    ('http://server/path', None), ('http://bad host/path', 'd3'),
])
def test_invalid_systemone_configuration_rejected(endpoint, model):
    cfg = systemone_config()
    intr = cfg.turn_policy.interrupt
    cfg = replace(cfg, turn_policy=replace(cfg.turn_policy, interrupt=replace(
        intr, systemone=replace(intr.systemone, endpoint=endpoint, model=model),
    )))
    with pytest.raises(ValueError, match='systemone'):
        validate_effective_config(cfg)


@pytest.mark.asyncio
async def test_yaml_switch_and_rollback_only_change_intent_provider(tmp_path, monkeypatch):
    from pathlib import Path
    repo = Path(__file__).resolve().parents[4]
    monkeypatch.setenv('EIDOLON_CHANNEL_SETTINGS_YAML', str(repo / 'config/settings.example.yaml'))
    monkeypatch.delenv('EIDOLON_CHANNEL_SETTINGS_OVERLAY_YAML', raising=False)
    original = load_effective_config()
    monkeypatch.setenv('EIDOLON_CHANNEL_SETTINGS_OVERLAY_YAML', str(repo / 'config/overlays/intent_d3.yaml'))
    changed = load_effective_config()
    assert changed.turn_policy.interrupt.intent_provider == 'systemone'
    assert changed.turn_policy.interrupt.systemone.model == 'd3'
    assert changed.llm == original.llm
    assert changed.turn_policy.eot == original.turn_policy.eot
    assert changed.turn_policy.interrupt.intent_timeout_ms == original.turn_policy.interrupt.intent_timeout_ms
    instance = SharedStageFactory.build_interrupt_classifier(changed)
    assert isinstance(instance, SystemOneInterruptClassifier)
    await instance.aclose()
    monkeypatch.delenv('EIDOLON_CHANNEL_SETTINGS_OVERLAY_YAML')
    assert load_effective_config() == original
