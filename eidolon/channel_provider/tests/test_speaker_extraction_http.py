import asyncio
import base64

from aiohttp.test_utils import TestClient, TestServer

from eidolon.channel_provider.http import create_app
from eidolon_sdk.biz.voiceprint import CalibrationCrop, EnrollmentSample, ExtractedSample


class Service:
    async def start(self):
        pass

    async def shutdown(self):
        pass

    async def healthcheck(self):
        pass


class Engine:
    state = "not_started"
    model_identity = ("fixture", "a" * 64, "reference-contiguous-pcm16-v1")

    def __init__(self):
        self.ready = asyncio.Event()
        self.calls = []
        self.enrollments = []

    async def start(self):
        self.state = "loading"
        await self.ready.wait()
        self.state = "ready"

    async def close(self):
        self.state = "closed"

    async def extract(self, pcm, min_speech_ms=1500):
        self.calls.append(pcm)
        return ExtractedSample(
            model_id="fixture",
            weights_hash="a" * 64,
            preprocessing_version="reference-contiguous-pcm16-v1",
            vector=[1.0, 0.0],
            speech_ms=1500,
        )

    async def extract_enrollment(self, pcm, min_speech_ms=1500):
        self.enrollments.append(pcm)
        sample = await self.extract(pcm, min_speech_ms)
        window = CalibrationCrop(speech_ms=900, vector=[0.0, 1.0])
        return EnrollmentSample(**sample.model_dump(), crops=[window])


async def test_registration_samples_carry_windows_and_probes_do_not():
    engine = Engine()
    engine.ready.set()
    client = TestClient(
        TestServer(create_app(service=Service(), bearer_token="test", speaker_engine=engine))
    )
    headers = {"Authorization": "Bearer test"}
    pcm = base64.b64encode(b"\x01\x00" * 32000).decode()
    try:
        await client.start_server()
        while engine.state != "ready":
            await asyncio.sleep(0)
        enrolled = await client.post("/v1/voiceprints/extract", headers=headers,
                                     json=dict(pcm_base64=pcm))
        assert enrolled.status == 200
        assert EnrollmentSample.model_validate(await enrolled.json()).crops[0].speech_ms == 900
        probe = await client.post("/v1/voiceprints/extract", headers=headers,
                                  json=dict(pcm_base64=pcm, purpose="recognition",
                                            min_speech_ms=600))
        assert probe.status == 200
        assert "crops" not in await probe.json()
        assert len(engine.enrollments) == 1
    finally:
        await client.close()


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
            state="loading", extraction_ready=False
        )
        pcm = b"\x01\x00" * 480000  # Full 30-second request exceeds ordinary control API limit.
        body = dict(pcm_base64=base64.b64encode(pcm).decode())
        assert (
            await client.post("/v1/voiceprints/extract", headers=headers, json=body)
        ).status == 503
        engine.ready.set()
        await asyncio.sleep(0)
        response = await client.get("/v1/voiceprints/readiness", headers=headers)
        assert "admission_ready" not in (await response.json())
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


async def test_comparison_requires_auth_and_has_no_identity_or_audio_output():
    sample = dict(
        model_id="test",
        weights_hash="a" * 64,
        preprocessing_version="v1",
        vector=[1, 0],
        speech_ms=3000,
    )
    body = dict(probe=sample, templates=[dict(profile_id="p", revision=1, sample=sample)])
    client = TestClient(TestServer(create_app(service=Service(), bearer_token="test")))
    try:
        await client.start_server()
        path = "/v1/voiceprints/compare"
        assert (await client.post(path, json=body)).status == 401
        headers = {"Authorization": "Bearer test"}
        response = await client.post(path, headers=headers, json=body)
        assert response.status == 200
        result = await response.json()
        assert result["scores"][0]["similarity"] == 1
        assert result["validation_state"] == "unvalidated"
        assert "vector" not in str(result) and "matched" not in result
        assert response.headers["Cache-Control"] == "no-store"
        assert (
            await client.post(path, headers=headers, json=body | {"device_id": "box"})
        ).status == 422
    finally:
        await client.close()


async def test_retired_policy_endpoint_is_absent():
    client = TestClient(TestServer(create_app(service=Service(), bearer_token="test")))
    try:
        await client.start_server()
        assert (await client.post("/v1/voiceprints/approve-policy", json={})).status == 404
    finally:
        await client.close()


async def test_quality_rejection_roundtrips_from_provider_to_conversation(tmp_path):
    import httpx
    from eidolon.livekit.tests.agent.test_reference_speaker_extractor import engine as make_extractor
    from eidolon.livekit.plugins.speaker_verification.inference import SpeakerInferenceEngine
    from eidolon.livekit.agent.session.speaker_recognition import SpeakerRecognitionClient, SpeakerRecognizer
    from eidolon.livekit.tests.agent.test_conversation_speaker import client as candidates, POLICY
    from eidolon.livekit.common.audio_evidence import AudioEvidence
    extractor, model = make_extractor(tmp_path)
    engine = SpeakerInferenceEngine(extractor)
    await engine.start()
    server = TestServer(create_app(service=Service(), bearer_token='test', speaker_engine=engine))
    await server.start_server()
    try:
        async with httpx.AsyncClient(trust_env=False) as http:
            port = SpeakerRecognitionClient(http, data_url='http://unused', data_token='test',
                provider_url=str(server.make_url('')).rstrip('/'), provider_token='test')
            port.candidates = candidates().candidates
            pcm = b'\x01\x00' * 8000 + b'\x00\x00' * 24000
            evidence = AudioEvidence(source='mic', generation=1, candidate='c', sample_rate=16000,
                start=0, end=len(pcm)//2, complete=True, reason=None, pcm=pcm)
            value = await SpeakerRecognizer(port, POLICY.model_copy(update={'min_speech_ms':600})).recognize(
                evidence, owner_id='o', session_id='s', turn_id='t')
            assert value.status == 'uncertain' and value.person_id is None
            assert value.reason == 'speech_too_short'
            assert value.audio_quality.speech_ms == 500
            assert value.audio_quality.audio_ms == 2000
            assert value.audio_quality.required_speech_ms == 600
            # Warmup only: no rejected audio reaches the embedding model.
            assert len(model.received) == 1
    finally:
        await server.close()


def _recognition_policy(**overrides):
    import json
    return json.dumps({
        "version": "fixture-recognition-v1", "model_id": "fixture", "weights_hash": "a" * 64,
        "preprocessing_version": "reference-contiguous-pcm16-v1", "threshold": 0.55,
        "min_person_margin": 0.1, "min_speech_ms": 600, "min_unknown_speech_ms": 1500,
    } | overrides)


async def test_health_publishes_recognition_state_without_failing_the_provider(tmp_path):
    # 10-30: the policy was absent from model_dir, the worker turned recognition
    # off silently, and the Host still reported ready. Health now says which.
    engine = Engine()
    engine.model_dir = tmp_path
    client = TestClient(
        TestServer(create_app(service=Service(), bearer_token="test", speaker_engine=engine))
    )

    async def recognition():
        response = await client.get("/health")
        assert response.status == 200
        return (await response.json())["speaker_recognition"]

    try:
        await client.start_server()
        assert await recognition() == {"state": "model_loading"}
        engine.ready.set()
        await asyncio.sleep(0)
        assert await recognition() == {"state": "policy_missing"}
        (tmp_path / "recognition-controlled-trial.json").write_text("{")
        assert await recognition() == {"state": "policy_invalid"}
        (tmp_path / "recognition-controlled-trial.json").write_text(
            _recognition_policy(weights_hash="b" * 64)
        )
        assert await recognition() == {
            "state": "policy_model_mismatch", "policy_version": "fixture-recognition-v1",
        }
        (tmp_path / "recognition-controlled-trial.json").write_text(_recognition_policy())
        assert await recognition() == {
            "state": "ready", "model_id": "fixture", "policy_version": "fixture-recognition-v1",
        }
        # A calibrated policy decides with its cohort; the worker turns
        # recognition off without it, so health must not say ready.
        import hashlib
        import numpy as np
        calibrated = {"version": "fixture-recognition-v3", "calibration": {
            "version": "c", "max_false_accept": 0.005, "min_genuine_acceptance": 0.8,
            "min_genuine_scores": 4,
            "bands": [{"name": "short", "min_speech_ms": 600, "floor": 0.3, "margin": 0.1}],
            "cohort": {"path": "cohort.npz", "sha256": "c" * 64, "speakers": 1}}}
        (tmp_path / "recognition-controlled-trial.json").write_text(_recognition_policy(**calibrated))
        assert await recognition() == {"state": "cohort_missing", "policy_version": "fixture-recognition-v3"}
        with open(tmp_path / "cohort.npz", "wb") as f:
            np.savez(f, vectors=np.ones((1, 2), np.float32), speech_ms=[900], speakers=[0],
                     model_id=np.array("fixture"), weights_hash=np.array("a" * 64),
                     preprocessing_version=np.array("reference-contiguous-pcm16-v1"))
        assert await recognition() == {"state": "cohort_invalid", "policy_version": "fixture-recognition-v3"}
        calibrated["calibration"]["cohort"]["sha256"] = hashlib.sha256(
            (tmp_path / "cohort.npz").read_bytes()).hexdigest()
        (tmp_path / "recognition-controlled-trial.json").write_text(_recognition_policy(**calibrated))
        assert await recognition() == {
            "state": "ready", "model_id": "fixture", "policy_version": "fixture-recognition-v3",
        }
    finally:
        await client.close()


async def test_health_reports_recognition_disabled_when_capability_not_selected():
    client = TestClient(TestServer(create_app(service=Service(), bearer_token="test")))
    try:
        await client.start_server()
        body = await (await client.get("/health")).json()
        assert body["speaker_recognition"] == {"state": "disabled"}
    finally:
        await client.close()
