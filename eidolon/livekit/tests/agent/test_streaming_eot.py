"""Native SDK lifecycle must not bypass Channel's single commit owner."""
import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from livekit.agents.voice.turn import TurnDetectionEvent, _StreamingTurnDetector

from eidolon.livekit.agent.session.eot_model import create_session_eot_model
from eidolon.livekit.agent.session.streaming_eot import NativeAudioTurnStream
from eidolon.livekit.common.config.schema import EffectiveAgentConfig, TurnPolicyConfig
from eidolon.livekit.common.config.validators import validate_effective_config


def mini_policy():
    policy = TurnPolicyConfig()
    return replace(policy, eot=replace(policy.eot, provider='livekit_mini'),
                   interrupt=replace(policy.interrupt, intent_provider='systemone',
                       systemone=replace(policy.interrupt.systemone,
                           endpoint='http://localhost:8080/d3/v1/systemone', model='d3')))


def event(score=.8, duration=.01):
    return TurnDetectionEvent(type='eot_prediction', end_of_turn_probability=score,
                              last_speaking_time=1, inference_duration=duration)


def stream_fixture(*, settle=None, timeout=.1, bind_scope=None):
    futures = []
    def predict():
        f = asyncio.get_running_loop().create_future()
        futures.append(f)
        return f
    native = SimpleNamespace(model='mini', provider='livekit', is_fallback=False,
        prediction_timeout=timeout, predict=predict, cancel_inference=Mock(),
        flush=Mock(), end_input=Mock(), aclose=AsyncMock(), push_audio=Mock())
    scores, observations = [], []
    stream = NativeAudioTurnStream(native, settle=settle or AsyncMock(),
        begin_observation=lambda: lambda outcome, score: observations.append((outcome, score)),
        accept_score=scores.append, reset_score=lambda: None, settlement_budget=6,
        bind_scope=bind_scope)
    return stream, native, futures, scores, observations


def test_mini_is_native_protocol_and_does_not_construct_firered(monkeypatch):
    from eidolon.livekit.plugins import eot
    fire = Mock(side_effect=AssertionError('FireRed must not be loaded'))
    monkeypatch.setattr(eot, 'ChineseModel', fire)
    model = create_session_eot_model(mini_policy())
    assert isinstance(model.detector(settle=AsyncMock(), begin_observation=lambda: None),
                      _StreamingTurnDetector)
    model.update_asr('一个完全不同的文本', True)
    assert model.current_eot_score == 0
    fire.assert_not_called()


@pytest.mark.parametrize('provider,threshold,valid', [
    ('firered', None, True), ('livekit_mini', None, True),
    ('livekit_mini', .4, True), ('livekit_mini', 0, False),
    ('livekit_mini', float('nan'), False), ('unknown', None, False),
])
def test_provider_configuration(provider, threshold, valid):
    policy = mini_policy()
    base = EffectiveAgentConfig()
    cfg = replace(base, llm=replace(base.llm, base_url='http://localhost:8080/v1'), turn_policy=replace(policy,
        eot=replace(policy.eot, provider=provider, mini_unlikely_threshold=threshold)))
    if valid:
        validate_effective_config(cfg)
    else:
        with pytest.raises(ValueError, match='turn_policy.eot'):
            validate_effective_config(cfg)


def test_mini_requires_existing_intent_owner():
    policy = mini_policy()
    cfg = replace(EffectiveAgentConfig(), turn_policy=replace(policy,
        interrupt=replace(policy.interrupt, intent_provider='none')))
    with pytest.raises(ValueError, match='Channel-owned'):
        validate_effective_config(cfg)


def test_audio_scores_are_session_local_and_text_cannot_refresh_them():
    first = create_session_eot_model(mini_policy())
    second = create_session_eot_model(mini_policy())
    first.current_eot_score = .83
    second.update_vad(True)
    assert first.update_asr('新文本版本', True) == .83
    assert second.current_eot_score == 0
    first.end_session()
    assert first.current_eot_score == 0


def test_opt_in_yaml_uses_existing_config_loader():
    from pathlib import Path
    import yaml
    from eidolon.livekit.common.config.loader import _load_turn_policy
    root = Path(__file__).resolve().parents[4]
    overlay = yaml.safe_load((root / 'config/overlays/eot_mini.yaml').read_text())
    policy = _load_turn_policy(overlay['turn_policy'])
    assert policy.eot.provider == 'livekit_mini'
    assert policy.eot.mini_unlikely_threshold is None
    assert TurnPolicyConfig().eot.provider == 'firered'


def test_mini_prewarm_does_not_load_firered(monkeypatch):
    from eidolon.livekit.agent.server import _load_prewarm_models
    from eidolon.livekit.common import config
    from eidolon.livekit.plugins import eot
    from eidolon.livekit.plugins.vad.firered import FireredPvadVAD
    from livekit import local_inference
    cfg = replace(EffectiveAgentConfig(), turn_policy=mini_policy())
    cfg = replace(cfg, voiceprint=replace(cfg.voiceprint, enabled=False))
    monkeypatch.setattr(config, 'load_effective_config', lambda: cfg)
    monkeypatch.setattr(FireredPvadVAD, 'load', Mock())
    warm = Mock()
    fire = Mock(side_effect=AssertionError('unexpected FireRed'))
    monkeypatch.setattr(local_inference, 'init_eot', warm)
    monkeypatch.setattr(eot, 'ChineseModel', fire)
    _load_prewarm_models(SimpleNamespace(userdata={}))
    warm.assert_called_once_with()
    fire.assert_not_called()


@pytest.mark.asyncio
async def test_prediction_waits_for_settlement_without_changing_native_evidence():
    release, entered = asyncio.Event(), asyncio.Event()
    async def settle():
        entered.set()
        await release.wait()
    stream, native, futures, scores, observations = stream_fixture(settle=settle)
    task = stream.predict()
    result = event()
    futures[-1].set_result(result)
    await entered.wait()
    assert not task.done() and scores == []
    # The SDK cannot use its one-second inference budget to bypass slow intent.
    assert stream.prediction_timeout == 6.1
    release.set()
    assert await task is result
    assert scores == [.8] and observations == [('returned', .8)]
    await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('invalidate', ['cancel_inference', 'flush', 'end_input'])
async def test_continuation_and_shutdown_cannot_publish_stale_score(invalidate):
    entered, release = asyncio.Event(), asyncio.Event()
    async def settle():
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            # Simulate a provider that completes despite task cancellation.
            await release.wait()
    stream, _, futures, scores, observations = stream_fixture(settle=settle)
    old = stream.predict()
    futures[-1].set_result(event(.9))
    await entered.wait()
    getattr(stream, invalidate)()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await old
    assert scores == [] and observations == [('cancelled', .9)]
    await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('result', [event(1, None), event(0, None), event(float('nan')), event(1.1)])
async def test_sdk_default_or_invalid_probability_is_not_completion_evidence(result):
    settle = AsyncMock()
    stream, _, futures, scores, observations = stream_fixture(settle=settle)
    task = stream.predict()
    futures[-1].set_result(result)
    assert (await task).end_of_turn_probability == 0
    assert observations == [('unavailable', None)]
    settle.assert_awaited_once()
    await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['timeout', 'exception'])
async def test_failure_still_settles_intent_and_uses_long_endpoint(failure):
    settle = AsyncMock()
    stream, native, futures, scores, observations = stream_fixture(settle=settle, timeout=.01)
    task = stream.predict()
    if failure == 'exception':
        futures[-1].set_exception(RuntimeError('inference unavailable'))
    assert (await task).end_of_turn_probability == 0
    settle.assert_awaited_once()
    assert observations == [('timeout' if failure == 'timeout' else 'unavailable', None)]
    if failure == 'timeout':
        native.cancel_inference.assert_called_with(timed_out=True)
    await stream.aclose()


@pytest.mark.asyncio
async def test_supersession_before_task_start_and_close_drain_all_work():
    stream, _, futures, scores, observations = stream_fixture()
    old = stream.predict()
    new = stream.predict()
    futures[-1].set_result(event(.7))
    with pytest.raises(asyncio.CancelledError):
        await old
    assert (await new).end_of_turn_probability == .7
    await stream.aclose()
    closed = stream.predict()
    with pytest.raises(asyncio.CancelledError):
        await closed
    assert scores == [.7]


@pytest.mark.asyncio
@pytest.mark.parametrize('when', ['before_task_start', 'during_inference', 'during_settlement'])
async def test_candidate_change_without_sdk_cancellation_cannot_rebind_old_audio(when):
    identity = ('candidate-1', 1)
    entered, release = asyncio.Event(), asyncio.Event()
    def bind():
        snapshot = identity
        def check():
            if snapshot != identity:
                raise asyncio.CancelledError('candidate superseded')
        return check
    async def settle():
        entered.set()
        if when == 'during_settlement':
            await release.wait()
    stream, _, futures, scores, _ = stream_fixture(settle=settle, bind_scope=bind)
    old = stream.predict()
    if when == 'during_inference':
        await asyncio.sleep(0)
    if when == 'during_settlement':
        futures[-1].set_result(event(.9))
        await entered.wait()
    identity = ('candidate-2', 2)
    if not futures[-1].done():
        futures[-1].set_result(event(.9))
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await old
    assert scores == []
    await stream.aclose()
