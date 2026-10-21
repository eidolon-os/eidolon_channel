import asyncio
import base64

from aiohttp.test_utils import TestClient, TestServer

from eidolon.channel_provider.http import create_app
from eidolon_sdk.biz.voiceprint import ExtractedSample


class Service:
    async def start(self):
        pass

    async def shutdown(self):
        pass

    async def healthcheck(self):
        pass


class Engine:
    state = "not_started"

    def __init__(self):
        self.ready = asyncio.Event()
        self.calls = []

    async def start(self):
        self.state = "loading"
        await self.ready.wait()
        self.state = "ready"

    async def close(self):
        self.state = "closed"

    async def extract(self, pcm):
        self.calls.append(pcm)
        return ExtractedSample(
            model_id="fixture",
            weights_hash="a" * 64,
            preprocessing_version="reference-contiguous-pcm16-v1",
            vector=[1.0, 0.0],
            speech_ms=1500,
        )


async def test_warmup_does_not_block_provider_health_and_does_not_enable_admission():
    engine = Engine()
    client = TestClient(
        TestServer(create_app(service=Service(), bearer_token="test", speaker_engine=engine))
    )
    headers = {"Authorization": "Bearer test"}
    try:
        await client.start_server()
        assert (await client.get("/health")).status == 200
        assert (await client.get("/v1/voiceprints/readiness")).status == 401
        response = await client.get("/v1/voiceprints/readiness", headers=headers)
        assert (await response.json()) == dict(
            state="loading", extraction_ready=False, admission_ready=False
        )
        pcm = b"\x01\x00" * 480000  # Full 30-second request exceeds ordinary control API limit.
        body = dict(pcm_base64=base64.b64encode(pcm).decode())
        assert (
            await client.post("/v1/voiceprints/extract", headers=headers, json=body)
        ).status == 503
        engine.ready.set()
        await asyncio.sleep(0)
        response = await client.get("/v1/voiceprints/readiness", headers=headers)
        assert (await response.json())["admission_ready"] is False
        # A device identifier cannot turn this numerical API into device enrollment.
        assert (
            await client.post(
                "/v1/voiceprints/extract", headers=headers, json=body | {"device_id": "box"}
            )
        ).status == 422
        assert (
            await client.post("/v1/voiceprints/extract", headers=headers, json=body)
        ).status == 200
        assert engine.calls == [pcm]
    finally:
        await client.close()
    assert engine.state == "closed"


async def test_no_selected_engine_is_unavailable_without_affecting_core_health():
    client = TestClient(TestServer(create_app(service=Service(), bearer_token="test")))
    try:
        await client.start_server()
        response = await client.get(
            "/v1/voiceprints/readiness", headers={"Authorization": "Bearer test"}
        )
        assert (await response.json())["state"] == "disabled"
        assert (await client.get("/health")).status == 200
    finally:
        await client.close()


async def test_enrollment_evaluation_is_authenticated_and_never_certifies_identity():
    sample = dict(
        model_id="test",
        weights_hash="a" * 64,
        preprocessing_version="v1",
        vector=[1, 0],
        speech_ms=4000,
    )
    body = dict(samples={str(i): sample for i in range(3)}, trial=sample, recording_sha256="b" * 64)
    client = TestClient(TestServer(create_app(service=Service(), bearer_token="test")))
    try:
        await client.start_server()
        path = "/v1/voiceprints/evaluate-enrollment"
        assert (await client.post(path, json=body)).status == 401
        headers = {"Authorization": "Bearer test"}
        response = await client.post(path, headers=headers, json=body)
        assert response.status == 200
        result = await response.json()
        assert result["template"]["vector"] == [1, 0]
        assert result["validation_state"] == "unvalidated"
        assert (
            await client.post(path, headers=headers, json=body | {"device_id": "box"})
        ).status == 422
    finally:
        await client.close()
