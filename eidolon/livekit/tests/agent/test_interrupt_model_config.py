"""Intent configuration reuses the LLM factory and owns its client lifetime."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from eidolon.livekit.agent.factory import SharedStageFactory
from eidolon.livekit.agent.session.eot_model import InterruptAwareTurnDetector
from eidolon.livekit.common.config.schema import EffectiveAgentConfig
from eidolon.livekit.common.config.validators import validate_effective_config


def config(**kwargs):
    cfg = EffectiveAgentConfig()
    return replace(cfg, turn_policy=replace(cfg.turn_policy, interrupt=replace(cfg.turn_policy.interrupt, **kwargs)))


def test_disabled_intent_does_not_construct_llm(monkeypatch):
    build = MagicMock()
    monkeypatch.setattr(SharedStageFactory, '_build_llm', build)
    assert SharedStageFactory.build_interrupt_classifier(config()) is None
    build.assert_not_called()


def test_native_interruption_owner_does_not_construct_channel_classifier(monkeypatch):
    build = MagicMock()
    monkeypatch.setattr(SharedStageFactory, '_build_llm', build)
    cfg = config(intent_provider='llm')
    cfg = replace(cfg, turn_policy=replace(cfg.turn_policy, interruption_owner='livekit_native_adaptive'))
    assert SharedStageFactory.build_interrupt_classifier(cfg) is None
    build.assert_not_called()


@pytest.mark.parametrize('owner,brain,required', [
    ('channel', 'eidolon_agent', True),
    ('livekit_native_adaptive', 'eidolon_agent', False),
    ('livekit_native_adaptive', 'direct_llm', True),
])
def test_direct_model_requirement_follows_actual_consumer(owner, brain, required):
    cfg = config(intent_provider='llm')
    cfg = replace(cfg, turn_policy=replace(cfg.turn_policy, interruption_owner=owner),
        providers=replace(cfg.providers, brain_provider=brain),
        remote_agent_rpc=replace(cfg.remote_agent_rpc, target='127.0.0.1:45051'),
        llm=replace(cfg.llm, model='', base_url=''))
    if required:
        with pytest.raises(ValueError, match='llm.model is required'):
            validate_effective_config(cfg)
    else:
        validate_effective_config(cfg)


@pytest.mark.asyncio
async def test_enabled_intent_reuses_adapter_with_bounded_request_and_closes(monkeypatch):
    client = SimpleNamespace(aclose=AsyncMock(), chat=MagicMock())
    build = MagicMock(return_value=client)
    monkeypatch.setattr(SharedStageFactory, '_build_llm', build)
    cfg = config(intent_provider='llm', intent_timeout_ms=1500)
    classifier = SharedStageFactory.build_interrupt_classifier(cfg)
    passed = build.call_args.args[0]
    assert passed.llm.model == cfg.llm.model
    assert passed.llm.base_url == cfg.llm.base_url
    assert passed.llm.timeout == 1.5
    assert passed.llm.max_completion_tokens == 24
    client.chat.assert_not_called()
    factory = SharedStageFactory(llm=client, stt=object(), tts=object(), interrupt_classifier=classifier)
    await factory.aclose()
    client.aclose.assert_awaited_once()


@pytest.mark.parametrize('kwargs', [dict(intent_provider='phrases'), dict(intent_timeout_ms=0), dict(intent_timeout_ms=6001)])
def test_invalid_intent_policy_rejected(kwargs):
    with pytest.raises(ValueError, match='turn_policy.interrupt.intent_'):
        validate_effective_config(config(**kwargs))


@pytest.mark.asyncio
async def test_endpointing_preserves_eot_probability_after_settlement():
    order = []
    async def settle():
        order.append('settled')
    async def predict(ctx, **kwargs):
        assert order == ['settled']
        return .27
    detector = SimpleNamespace(model='existing', predict_end_of_turn=AsyncMock(side_effect=predict))
    wrapper = InterruptAwareTurnDetector(detector, settle)
    ctx = object()
    assert await wrapper.predict_end_of_turn(ctx, timeout=1.2) == .27
    detector.predict_end_of_turn.assert_awaited_once_with(ctx, timeout=1.2)


@pytest.mark.asyncio
@pytest.mark.parametrize('score', [.2, .8])
async def test_endpointing_context_is_a_snapshot_and_preserves_model_score(score):
    from livekit.agents.llm import ChatContext

    selected = '不是我刚才说错了。'
    def resolve(text):
        assert text == '我刚才说错了。'
        return selected
    async def settle():
        nonlocal selected
        selected = '下一轮完全不同的请求。'
    detector = SimpleNamespace(predict_end_of_turn=AsyncMock(return_value=score))
    wrapper = InterruptAwareTurnDetector(detector, settle, resolve_transcript=resolve)
    ctx = ChatContext()
    ctx.add_message(role='assistant', content='旧回答。')
    ctx.add_message(role='user', content='我刚才说错了。')
    original = ctx.items[-1]
    assert await wrapper.predict_end_of_turn(ctx, timeout=1.2) == score
    scored = detector.predict_end_of_turn.call_args.args[0]
    assert scored.items[-1].text_content == '不是我刚才说错了。'
    assert scored.items[-1].id == original.id
    assert scored.items[0] is ctx.items[0]
    assert original.text_content == '我刚才说错了。'
    assert ctx.items[-1] is original


@pytest.mark.parametrize('wrapped', [False, True])
def test_complete_turn_fast_path_preserves_session_endpoint_maximum(wrapped):
    from livekit.agents.types import NOT_GIVEN
    from eidolon.livekit.agent.full_duplex.agent_builder import build_full_duplex_agent
    from eidolon.livekit.plugins.eot.models.base import EidolonEOTModel

    detector = MagicMock(spec=EidolonEOTModel)
    detector.get_dynamic_silence_threshold.return_value = .25
    selected = InterruptAwareTurnDetector(detector, AsyncMock()) if wrapped else detector
    pipeline = SimpleNamespace(
        _instructions='test',
        _factory=SimpleNamespace(stt=SimpleNamespace(stt=None), llm=SimpleNamespace(llm=None),
            tts=SimpleNamespace(tts=None), vad=None),
        _turn_detection=lambda: selected,
    )
    agent = build_full_duplex_agent(pipeline)
    assert agent.min_endpointing_delay == .25
    assert agent.max_endpointing_delay is NOT_GIVEN


@pytest.mark.asyncio
async def test_intent_warmup_is_stateless_and_bounded():
    import asyncio
    from eidolon.livekit.agent.turn_policy.intent_classifier import LlmInterruptClassifier

    classifier = LlmInterruptClassifier(object(), timeout_sec=.02)
    classifier.classify = AsyncMock(return_value=None)
    await classifier.warmup()
    classifier.classify.assert_awaited_once_with('', assistant_text='')
    classifier.classify = AsyncMock(side_effect=lambda *a, **kw: None)
    async def stalled(*args, **kwargs):
        await asyncio.sleep(10)
    classifier.classify.side_effect = stalled
    with pytest.raises(TimeoutError):
        await classifier.warmup()


@pytest.mark.parametrize('mode,owner,provider,stage_included,adapter_included', [
    ('full_duplex', 'channel', 'llm', True, True),
    ('half_duplex', 'channel', 'llm', False, False),
    ('full_duplex', 'livekit_native_adaptive', 'llm', False, False),
    ('full_duplex', 'channel', 'none', False, True),
])
def test_intent_stage_and_endpoint_adapter_follow_policy_owner(
    mode, owner, provider, stage_included, adapter_included,
):
    from eidolon.livekit.agent.full_duplex.pipeline import StreamingPipeline

    classifier = object()
    factory = SimpleNamespace(stt=SimpleNamespace(stt=None), tts=SimpleNamespace(tts=None),
        llm=SimpleNamespace(llm=None), vad=None, interrupt_classifier=classifier)
    policy = config(intent_provider=provider).turn_policy
    policy = replace(policy, interruption_owner=owner)
    pipeline = StreamingPipeline(factory, interaction_mode=mode, allow_interruptions=mode == 'full_duplex', turn_policy=policy)
    assert (classifier in pipeline._lifecycle_stages()) is stage_included
    assert isinstance(pipeline._turn_detection(), InterruptAwareTurnDetector) is adapter_included


def test_llm_transport_body_reuses_native_adapter(monkeypatch):
    from livekit.plugins import openai
    client = MagicMock()
    monkeypatch.setattr(openai, 'LLM', client)
    cfg = config()
    cfg = replace(cfg, llm=replace(cfg.llm, extra_body={'thinking': {'type': 'disabled'}}))
    SharedStageFactory._build_llm(cfg)
    assert client.call_args.kwargs['extra_body'] == cfg.llm.extra_body


@pytest.mark.asyncio
@pytest.mark.parametrize('purpose', ['reply', 'intent'])
@pytest.mark.parametrize('base_url,token_field', [
    ('https://api.deepseek.com', 'max_tokens'),
    ('https://api.deepseek.com/v1', 'max_tokens'),
    ('https://api.openai.com/v1', 'max_completion_tokens'),
    ('https://proxy.example/v1', 'max_completion_tokens'),
])
async def test_token_limit_reaches_the_endpoint_in_its_supported_field(monkeypatch, base_url, token_field, purpose):
    """Inspect the actual SDK HTTP body, rather than just constructor kwargs."""
    import json
    import httpx
    from openai import AsyncOpenAI
    from livekit.agents.llm import ChatContext
    from livekit.plugins import openai

    bodies = []
    def respond(request):
        bodies.append(json.loads(request.content))
        chunk = {'id': 'test', 'choices': [{'index': 0, 'delta': {'content': '测试'}, 'finish_reason': 'stop'}]}
        return httpx.Response(200, headers={'content-type': 'text/event-stream'},
            text='data: ' + json.dumps(chunk) + '\n\ndata: [DONE]\n\n')

    adapter = openai.LLM
    async with AsyncOpenAI(api_key='test', base_url=base_url,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))) as client:
        monkeypatch.setattr(openai, 'LLM', lambda **kwargs: adapter(client=client, **kwargs))
        cfg = config(intent_provider='llm')
        body = {'thinking': {'type': 'disabled'}} if purpose == 'reply' else {}
        cfg = replace(cfg, llm=replace(cfg.llm, api_key='test', base_url=base_url,
            model='deepseek-v4-flash', max_completion_tokens=24, extra_body=body))
        model = (SharedStageFactory.build_interrupt_classifier(cfg)._model
            if purpose == 'intent' else SharedStageFactory._build_llm(cfg))
        context = ChatContext()
        context.add_message(role='user', content='测试')
        try:
            async with model.chat(chat_ctx=context) as stream:
                assert ''.join([text async for text in stream.to_str_iterable()]) == '测试'
        finally:
            await model.aclose()
        assert len(bodies) == 1
        assert bodies[0].get(token_field) == 24
        other = 'max_completion_tokens' if token_field == 'max_tokens' else 'max_tokens'
        assert other not in bodies[0]
        if purpose == 'reply' or base_url.startswith('https://api.deepseek.com'):
            assert bodies[0]['thinking'] == {'type': 'disabled'}
        else:
            assert 'thinking' not in bodies[0], 'unknown endpoints must not inherit vendor options'
        assert cfg.llm.extra_body == body


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_model_obeys_configured_output_limit(record_property):
    from eidolon.livekit.common.config import load_effective_config
    from livekit.agents.llm import ChatContext
    from livekit.agents.types import APIConnectOptions

    cfg = load_effective_config()
    cfg = replace(cfg, llm=replace(cfg.llm, max_completion_tokens=24, timeout=15))
    model = SharedStageFactory._build_llm(cfg)
    usage = []
    model.on('metrics_collected', usage.append)
    context = ChatContext()
    context.add_message(role='user', content='请详细介绍全双工语音对话的输入、处理和输出流程，至少写八段，每段两句话。')
    try:
        async with model.chat(chat_ctx=context, conn_options=APIConnectOptions(max_retry=0)) as stream:
            text = ''.join([part async for part in stream.to_str_iterable()])
        record_property('reply_chars', len(text))
        assert usage, 'provider usage is required to verify the token limit'
        record_property('completion_tokens', usage[-1].completion_tokens)
        assert 0 < usage[-1].completion_tokens <= 24
    finally:
        await model.aclose()
