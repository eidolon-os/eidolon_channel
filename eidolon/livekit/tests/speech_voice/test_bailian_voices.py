from __future__ import annotations

import base64
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from eidolon.livekit.plugins.tts.bailian.config import BailianTTSConfig
from eidolon.livekit.plugins.tts.bailian.voice_catalog import (
    BailianVoiceCatalogAdapter,
    host_prefix,
    region_of,
)
from eidolon.speech_voice.adapter import EnrollmentRejected
from eidolon.speech_voice.wav import wav_bytes


def _adapter(model="cosyvoice-v3-flash", url="wss://dashscope.aliyuncs.com/api-ws/v1/inference"):
    return BailianVoiceCatalogAdapter(BailianTTSConfig(api_url=url, api_key="k", model=model, voice="longanyang"))


def test_region_comes_from_the_address() -> None:
    assert region_of("wss://dashscope.aliyuncs.com/api-ws/v1/inference") == "cn-beijing"
    assert region_of("wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference") == "ap-southeast-1"
    assert region_of("wss://ws1.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference") == "cn-beijing"
    assert region_of("wss://proxy.example/api") is None


async def test_list_voices_keeps_the_model_and_the_region() -> None:
    beijing = {v.voice_id for v in (await _adapter().list_voices()).voices}
    assert "longanyang" in beijing and "longwan_v3" in beijing
    assert not any(v.endswith("_v3.1") for v in beijing)
    singapore = {v.voice_id for v in (await _adapter(
        url="wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference").list_voices()).voices}
    assert "longanyang" in singapore and "longwan_v3" in singapore and "longxian_v3" not in singapore
    qwen = {v.voice_id for v in (await _adapter("qwen-audio-3.1-tts-flash").list_voices()).voices}
    assert "longanyang_v3.1" in qwen and "longanyang" not in qwen


def test_with_voice_changes_only_the_voice_and_capabilities_follow_model_and_region() -> None:
    adapter = _adapter()
    changed = adapter.with_voice("longwan_v3")
    assert changed.voice == "longwan_v3" and changed.model == adapter.plugin_cfg.model
    assert adapter.plugin_cfg.voice == "longanyang"
    assert adapter.enrollment_capabilities().design and adapter.enrollment_capabilities().clone
    abroad = _adapter(url="wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference")
    assert abroad.enrollment_capabilities().clone and not abroad.enrollment_capabilities().design
    assert adapter.fingerprint() != _adapter("qwen-audio-3.1-tts-flash").fingerprint()


def test_the_host_prefix_is_what_bailian_accepts(monkeypatch) -> None:
    monkeypatch.setenv("EIDOLON_HOST_ID", "eidolon-pi5")
    prefix = host_prefix()
    assert len(prefix) == 10 and prefix.isalnum() and prefix == host_prefix()


@pytest.fixture
async def bailian(monkeypatch):
    """A customization endpoint that behaves like Bailian's (shapes from the 2026-10-30 live check)."""
    seen: list[dict] = []
    voices = {"qwen-audio-3.1-tts-flash-eprefix000-1": "DEPLOYING"}

    async def customization(request: web.Request) -> web.Response:
        body = await request.json()
        seen.append(body)
        assert request.headers["Authorization"] == "Bearer k"
        action = body["input"]["action"]
        if action == "create_voice" and "voice_prompt" in body["input"]:
            preview = base64.b64encode(wav_bytes(b"\x01\x00" * 2400, 24000)).decode()
            voices["m-vd-new"] = "OK"
            return web.json_response({"output": {"voice_id": "m-vd-new", "preview_audio": {"data": preview}}})
        if action == "create_voice":
            if body["input"]["audio"] == "data:audio/wav;base64,":
                return web.json_response({"code": "Audio.Invalid", "message": "bad audio"}, status=400)
            voices["m-clone"] = "DEPLOYING"
            return web.json_response({"output": {"voice_id": "m-clone"}})
        voice_id = body["input"].get("voice_id")
        if action in ("query_voice", "delete_voice") and voice_id not in voices:
            return web.json_response({"code": "BadRequest.ResourceNotExist",
                                      "message": "The Required resource not exist"}, status=400)
        if action == "query_voice":
            return web.json_response({"output": {"status": voices[voice_id], "resource_link": "http://secret"}})
        if action == "delete_voice":
            voices.pop(voice_id)
            return web.json_response({"output": {}})
        if action == "list_voice":
            ids = sorted(voices)
            page, size = body["input"]["page_index"], body["input"]["page_size"]
            return web.json_response({"output": {"voice_list": [{"voice_id": v} for v in ids[page * size:(page + 1) * size]]}})
        return web.json_response({"code": "?"}, status=500)

    app = web.Application()
    app.router.add_post("/api/v1/services/audio/tts/customization", customization)
    server = TestServer(app)
    await server.start_server()
    adapter = _adapter("qwen-audio-3.1-tts-flash")
    monkeypatch.setattr(type(adapter), "_customization_url",
                        property(lambda self: str(server.make_url("/api/v1/services/audio/tts/customization"))))
    yield adapter, seen, voices
    await server.close()


async def test_design_returns_a_ready_binding_and_its_preview(bailian) -> None:
    adapter, seen, _ = bailian
    made = await adapter.design(prompt="清亮的女声", preview_text="你好呀，我是你的新伙伴，今天想聊点什么？")
    assert made.binding.status == "ready" and made.binding.model == "qwen-audio-3.1-tts-flash"
    assert made.sample_rate == 24000 and len(made.preview_pcm) == 4800
    assert seen[0]["input"]["prefix"] == host_prefix() and seen[0]["parameters"]["response_format"] == "wav"


async def test_clone_status_delete_and_list_behave_like_the_provider(bailian) -> None:
    adapter, seen, voices = bailian
    binding = await adapter.clone(reference_wav=b"RIFF....WAVE")
    assert binding.status == "deploying"
    assert seen[-1]["input"]["audio"].startswith("data:audio/wav;base64,")
    voices["m-clone"] = "OK"
    assert (await adapter.status("m-clone")).status == "ready"
    voices["m-clone"] = "UNDEPLOYED"
    assert (await adapter.status("m-clone")).status == "rejected"
    assert (await adapter.status("never-existed")).status == "failed"
    await adapter.delete("m-clone")
    await adapter.delete("m-clone")  # gone already counts as deleted
    assert "m-clone" not in await adapter.list_own()
    with pytest.raises(EnrollmentRejected) as refused:
        await adapter.clone(reference_wav=b"")
    assert refused.value.code == "Audio.Invalid"
    assert all("resource_link" not in json.dumps(b.model_dump(mode="json"))
               for b in [await adapter.status("qwen-audio-3.1-tts-flash-eprefix000-1")])
