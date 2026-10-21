import asyncio
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from eidolon.livekit.agent.full_duplex.lifecycle import FullDuplexSessionLifecycle
from eidolon.livekit.agent.session.speaker_observation import SpeakerTurnObservation
from eidolon.livekit.common.config import VoiceprintConfig
from .test_speaker_turn_observation import capture, commit, turn


def pipeline(**overrides):
    return SimpleNamespace(
        _voiceprint_config=replace(VoiceprintConfig(), observation_companion_id="companion",
            observation_until=time.time() + 60, **overrides),
        _speaker_observation=None, _session_trace=SimpleNamespace(event=MagicMock()),
        _factory=SimpleNamespace(runtime_session_id="session", voiceprint_service=None,
            stt=SimpleNamespace(stt=SimpleNamespace(audio_evidence_capture=None))),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["expired", "other", "missing_token", "legacy"])
async def test_activation_requires_current_explicit_scope_and_no_legacy(monkeypatch, case):
    monkeypatch.setenv("EIDOLON_CHANNEL_PROVIDER_TOKEN", "test")
    p = pipeline()
    context = SimpleNamespace(companion_id="companion", owner_id="owner")
    if case == "expired":
        p._voiceprint_config = replace(p._voiceprint_config, observation_until=time.time() - 1)
    elif case == "other":
        context.companion_id = "other"
    elif case == "missing_token":
        monkeypatch.delenv("EIDOLON_CHANNEL_PROVIDER_TOKEN")
    else:
        p._factory.voiceprint_service = object()
    FullDuplexSessionLifecycle(p)._configure_speaker_observation(context)
    assert p._speaker_observation is None


@pytest.mark.asyncio
async def test_activation_records_per_turn_with_session_scope_and_closes_http(monkeypatch):
    monkeypatch.setenv("EIDOLON_CHANNEL_PROVIDER_TOKEN", "test")
    p = pipeline()
    lifecycle = FullDuplexSessionLifecycle(p)
    lifecycle._configure_speaker_observation(SimpleNamespace(companion_id="companion", owner_id="owner"))
    assert p._speaker_observation is not None
    http = p._speaker_observation_http
    commit(p._speaker_observation, turn("a"))
    commit(p._speaker_observation, turn("b"))
    events = p._session_trace.event.call_args_list
    assert len(events) == 2
    assert all(e.args[0] == "speaker_observation" for e in events)
    assert events[0].args[1]["session_id"] == "session"
    assert events[0].args[1]["companion_id"] == "companion"
    await lifecycle._close_speaker_observation()
    assert http.is_closed


@pytest.mark.asyncio
async def test_deadline_clears_capture_and_prevents_new_or_late_extraction():
    calls = []

    async def extract(pcm):
        calls.append(pcm)
        await asyncio.Future()

    observer = SpeakerTurnObservation(extract=extract, record=lambda _: None, lifetime_sec=.01)
    old = capture()
    observer.bind_capture(old)
    commit(observer, turn())
    await asyncio.sleep(.03)
    assert old.retained_samples == 0 and old.active_source is None
    assert observer.pending is None
    fresh = capture()
    observer.bind_capture(fresh)
    assert fresh.retained_samples == 0
    commit(observer, turn("next"))
    assert len(calls) == 1
    await observer.aclose()


@pytest.mark.parametrize("fields", [
    {"observation_until": float("nan")}, {"observation_until": -1},
    {"observation_until": "tomorrow"}, {"observation_until": True},
    {"observation_until": 100}, {"observation_companion_id": "companion"},
    {"observation_companion_id": "companion", "observation_until": 100, "enabled": True},
])
def test_invalid_or_unbounded_observation_settings_are_rejected(fields):
    from eidolon.livekit.common.config.schema import EffectiveAgentConfig
    from eidolon.livekit.common.config.validators import validate_effective_config

    with pytest.raises(ValueError, match="voiceprint"):
        validate_effective_config(replace(EffectiveAgentConfig(), voiceprint=replace(VoiceprintConfig(), **fields)))
