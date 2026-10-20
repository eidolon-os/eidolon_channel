import base64
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from eidolon.channel_provider.voiceprints import install_voiceprint_routes
from eidolon_sdk.biz.voiceprint import ExtractedSample


async def test_extraction_requires_auth_accepts_bounded_audio_without_echoing_it():
    seen = []

    class Extractor:
        async def extract(self, pcm):
            seen.append(len(pcm))
            return ExtractedSample(
                model_id="test",
                weights_hash="a" * 64,
                preprocessing_version="v1",
                vector=[1, 0],
                speech_ms=4000,
            )

    app = web.Application(client_max_size=256 * 1024)
    install_voiceprint_routes(
        app, lambda r: r.headers.get("Authorization") == "Bearer service", Extractor()
    )
    async with TestClient(TestServer(app)) as h:
        assert (await h.post("/v1/voiceprints/extract", json={})).status == 401
        headers = {"Authorization": "Bearer service"}
        encoded = base64.b64encode(b"\0" * 256000).decode()
        r = await h.post("/v1/voiceprints/extract", headers=headers, json={"pcm_base64": encoded})
        assert r.status == 200, await r.text()
        assert seen == [256000]
        r = await h.post(
            "/v1/voiceprints/extract", headers=headers, json={"pcm_base64": "secret-not-base64"}
        )
        assert r.status == 422
        assert "secret-not-base64" not in await r.text()
