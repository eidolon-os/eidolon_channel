from __future__ import annotations

import asyncio
import base64
from dataclasses import replace

import numpy as np
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from eidolon_sdk.biz.speech_voice import (
    CatalogVoiceSelection,
    CompanionSpeechVoice,
    VoiceRef,
)
from eidolon_sdk.biz.system_data import SystemDataNotFound

from eidolon.channel_provider.http import _problem_body
from eidolon.channel_provider.speech_voices import register_speech_voices
from eidolon.livekit.agent.session import speech_voice as session_voice
from eidolon.livekit.common.config.schema import EffectiveAgentConfig, ProvidersConfig
from eidolon.livekit.plugins.tts.voice_catalogs import config_speaking
from eidolon.speech_voice import reference_audio
from eidolon.speech_voice.catalog import HostVoiceCatalogService
from eidolon.speech_voice.preview import VoicePreviews

from .fakes import FakeConfig, adapter_class, write_catalog

AUTH = {"Authorization": "Bearer t"}


@pytest.fixture
async def client(tmp_path, monkeypatch):
    monkeypatch.setattr(reference_audio, "_speech_flags",
                        lambda samples, size=512: (np.ones(len(samples) // size, dtype=bool), size))
    cls = adapter_class(write_catalog(tmp_path / "v.yaml"))
    catalog = HostVoiceCatalogService(cls(FakeConfig()), [cls], tmp_path / "s.json")

    async def synthesize(ref, text):
        return b"\x01\x00" * 1600, 16000

    previews = VoicePreviews(tmp_path / "p", synthesize=synthesize, propagator="eidolon-host-test")
    app = web.Application()
    register_speech_voices(app, catalog=catalog, previews=previews,
                           authorized=lambda r: r.headers.get("Authorization") == "Bearer t",
                           problem=_problem_body)
    async with TestClient(TestServer(app)) as c:
        await asyncio.sleep(0.05)  # the start-up refresh
        yield c


async def test_the_catalog_answers_with_an_etag_and_needs_the_token(client) -> None:
    assert (await client.get("/v1/speech-voices")).status == 401
    first = await client.get("/v1/speech-voices", headers=AUTH)
    body = await first.json()
    assert first.status == 200 and body["voices"][0]["is_host_default"]
    again = await client.get("/v1/speech-voices", headers={**AUTH, "If-None-Match": first.headers["ETag"]})
    assert again.status == 304


async def test_resolution_is_served_from_the_same_function(client) -> None:
    record = CompanionSpeechVoice(companion_id="c", revision=1, selection=CatalogVoiceSelection(
        provider="fake", model="m1", voice_id="bright-f", display_name="亮亮"))
    response = await client.post("/v1/speech-voices/resolve", headers=AUTH,
                                 json={"speech_voice": record.model_dump(mode="json")})
    assert (await response.json())["reason"] == "selected"


async def test_previews_are_wav_marked_as_ai_and_only_for_this_model(client) -> None:
    ok = await client.post("/v1/speech-voices/previews", headers=AUTH,
                           json={"ref": {"provider": "fake", "model": "m1", "voice_id": "warm-f"}, "sample": 1})
    data = await ok.read()
    assert ok.status == 200 and ok.content_type == "audio/wav" and b"AIGC" in data
    other = await client.post("/v1/speech-voices/previews", headers=AUTH,
                              json={"ref": {"provider": "fake", "model": "m2", "voice_id": "gentle-f"}, "sample": 1})
    assert other.status == 409


async def test_a_large_reference_file_is_accepted_and_normalized(client) -> None:
    from .test_audio_and_previews import _tone_wav

    data = _tone_wav(30)  # ~2.6 MB, far above the app-wide body limit
    response = await client.post("/v1/speech-voices/references/check?start_ms=2000", headers=AUTH, data=data)
    body = await response.json()
    assert response.status == 200 and body["accepted"] and body["window_start_ms"] == 2000


async def test_making_and_tracking_a_voice(client) -> None:
    designed = await client.post("/v1/speech-voices/enrollments/design", headers=AUTH, json={
        "prompt": "清亮的女声", "preview_text": "你好呀，我是你的新伙伴，今天想聊点什么？"})
    body = await designed.json()
    assert body["binding"]["status"] == "ready"
    assert b"AIGC" in base64.b64decode(body["preview_wav_base64"])
    cloned = await client.post("/v1/speech-voices/enrollments/clone", headers=AUTH,
                               json={"reference_wav_base64": base64.b64encode(b"RIFF").decode()})
    assert (await cloned.json())["binding"]["status"] == "deploying"
    assert (await client.get("/v1/speech-voices/enrollments/m1-e1-c?provider=fake", headers=AUTH)).status == 200
    assert (await client.get("/v1/speech-voices/enrollments/m1-e1-c?provider=other", headers=AUTH)).status == 409
    assert (await client.delete("/v1/speech-voices/enrollments/m1-e1-c?provider=fake", headers=AUTH)).status == 204
    assert await (await client.get("/v1/speech-voices/enrollments?provider=fake", headers=AUTH)).json() == [
        "m1-e1-c", "m1-e1-orphan"]


# ---------------------------------------------------------------- the worker

def _bailian(**tts):
    cfg = replace(EffectiveAgentConfig(), providers=ProvidersConfig(tts_provider="bailian"))
    return replace(cfg, bailian_tts=replace(cfg.bailian_tts, model="cosyvoice-v3-flash", voice="longanyang", **tts))


def test_the_voice_is_written_into_the_plugin_config_only_when_it_fits() -> None:
    cfg = _bailian()
    chosen = config_speaking(cfg, VoiceRef(provider="bailian", model="cosyvoice-v3-flash", voice_id="longwan_v3"))
    assert chosen.bailian_tts.voice == "longwan_v3" and cfg.bailian_tts.voice == "longanyang"
    other_model = config_speaking(cfg, VoiceRef(provider="bailian", model="qwen-audio-3.1-tts-flash", voice_id="x"))
    assert other_model.bailian_tts.voice == "longanyang"


class _Authority:
    def __init__(self, result=None, *, delay=0.0, missing=False):
        self.result, self.delay, self.missing = result, delay, missing

    async def get_companion_speech_voice(self, companion_id):
        await asyncio.sleep(self.delay)
        if self.missing:
            raise SystemDataNotFound("no endpoint")
        return self.result


async def _ready(value):
    return value


@pytest.fixture
async def provider(monkeypatch):
    answers = {"reason": "selected"}

    async def resolve(request):
        body = await request.json()
        answers["seen"] = body
        return web.json_response({"ref": {"provider": "bailian", "model": "cosyvoice-v3-flash",
                                          "voice_id": "longwan_v3"}, "voice_key": "k",
                                  "display_name": "龙婉", "reason": answers["reason"]})

    app = web.Application()
    app.router.add_post("/v1/speech-voices/resolve", resolve)
    server = TestServer(app)
    await server.start_server()
    monkeypatch.setattr(session_voice, "_provider_endpoint", lambda: (str(server.make_url("")).rstrip("/"), "t"))
    yield answers
    await server.close()


async def test_a_session_uses_what_channel_provider_resolved(provider) -> None:
    record = CompanionSpeechVoice(companion_id="c", revision=1)
    result = await session_voice.resolve_session_voice(_bailian(), companion_id=lambda: _ready("c"),
                                                       authority=_Authority(record))
    assert result.resolved.ref.voice_id == "longwan_v3" and result.resolve_ms < 500
    assert provider["seen"]["speech_voice"]["companion_id"] == "c"


async def test_an_authority_without_voices_reads_as_no_choice(provider) -> None:
    await session_voice.resolve_session_voice(_bailian(), companion_id=lambda: _ready("c"),
                                              authority=_Authority(missing=True))
    assert provider["seen"] == {"speech_voice": None}


async def test_an_unreachable_provider_keeps_a_choice_that_fits(monkeypatch) -> None:
    monkeypatch.setattr(session_voice, "_provider_endpoint", lambda: ("http://127.0.0.1:9", "t"))
    record = CompanionSpeechVoice(companion_id="c", revision=1, selection=CatalogVoiceSelection(
        provider="bailian", model="cosyvoice-v3-flash", voice_id="longwan_v3", display_name="龙婉"))
    result = await session_voice.resolve_session_voice(_bailian(), companion_id=lambda: _ready("c"),
                                                       authority=_Authority(record))
    assert (result.resolved.reason, result.resolved.ref.voice_id) == ("selected_unverified", "longwan_v3")


async def test_a_slow_authority_costs_at_most_the_deadline(provider) -> None:
    result = await session_voice.resolve_session_voice(_bailian(), companion_id=lambda: _ready("c"),
                                                       authority=_Authority(delay=2))
    assert result.resolved.reason == "authority_unavailable"
    assert result.resolved.ref.voice_id == "longanyang" and result.resolve_ms < 700


async def test_a_provider_without_an_adapter_keeps_its_configured_voice() -> None:
    cfg = replace(EffectiveAgentConfig(), providers=ProvidersConfig(tts_provider="sensetime"))
    result = await session_voice.resolve_session_voice(cfg, companion_id=lambda: _ready("c"), authority=_Authority())
    assert (result.resolved.reason, result.resolved.ref) == ("host_fixed_voice", None)
