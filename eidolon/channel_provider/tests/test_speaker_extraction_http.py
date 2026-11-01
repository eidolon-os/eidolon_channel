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
    from eidolon.livekit.tests.agent.test_conversation_speaker import CHANNEL, client as candidates, POLICY
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
            authority = candidates()
            port.candidates = authority.candidates
            pcm = b'\x01\x00' * 8000 + b'\x00\x00' * 24000
            evidence = AudioEvidence(source='mic', generation=1, candidate='c', sample_rate=16000,
                start=0, end=len(pcm)//2, complete=True, reason=None, pcm=pcm)
            value = await SpeakerRecognizer(port, POLICY.model_copy(update={'min_speech_ms':600}),
                authority.backend, channel=CHANNEL).recognize(
                evidence, owner_id='o', session_id='s', turn_id='t')
            assert value.status == 'uncertain' and value.evidence == []
            assert value.decisions.persons() == set()
            assert value.reason == 'speech_too_short'
            assert value.audio_quality.speech_ms == 500
            assert value.audio_quality.audio_ms == 2000
            assert value.audio_quality.required_speech_ms == 600
            # Warmup only: no rejected audio reaches the embedding model.
            assert len(model.received) == 1
            assert authority.backend.trials == 0
    finally:
        await server.close()


def _identity_policy(**overrides):
    import json
    return json.dumps({
        "version": "fixture-identity-v1", "model_id": "fixture", "weights_hash": "a" * 64,
        "preprocessing_version": "reference-contiguous-pcm16-v1",
        "backend_version": "fixture-backend-v1", "min_speech_ms": 600, "min_unknown_speech_ms": 1500,
        "prior_unknown": 0.5, "household_margin_llr": 1.0,
        "conditions": {c: {"personal_memory": 0.99, "addressing": 0.9, "speaker_label": 0.9,
                           "candidate_evidence": 0.95, "discovery": 0.99}
                       for c in ("near_same", "far_same", "far_cross")},
    } | overrides)


def _write_backend(root, **overrides):
    """A tiny valid calibration and the arrays it names, as the Models artifact delivers them."""
    import hashlib
    import json
    import numpy as np
    from eidolon.livekit.plugins.speaker_verification import evidence
    with open(root / "identity-backend.npz", "wb") as f:
        np.savez(f, mean=np.zeros(2, np.float32), projection=np.eye(2, dtype=np.float32))
    spec = {
        "version": "fixture-backend-v1", "identity": list(Engine.model_identity),
        "max_speech_ms": 20000, "features": list(evidence.FEATURES),
        "conditions": {c: {"weights": [1.0] + [0.0] * (len(evidence.FEATURES) - 1), "bias": 0.0}
                       for c in ("near_same", "far_same", "far_cross")},
        "arrays": {"path": "identity-backend.npz",
                   "sha256": hashlib.sha256((root / "identity-backend.npz").read_bytes()).hexdigest()},
    } | overrides
    (root / evidence.BACKEND_FILE).write_text(json.dumps(spec))


async def test_health_publishes_recognition_state_without_failing_the_provider(tmp_path):
    # 10-30: the policy was absent from model_dir, the worker turned recognition
    # off silently, and the Host still reported ready. Health now says which.
    engine = Engine()
    engine.model_dir = tmp_path
    client = TestClient(
        TestServer(create_app(service=Service(), bearer_token="test", speaker_engine=engine))
    )
    policy_file = tmp_path / "identity-policy.json"

    async def recognition():
        response = await client.get("/health")
        assert response.status == 200
        return (await response.json())["speaker_recognition"]

    def off(state):
        return {"state": state, "policy_version": "fixture-identity-v1"}

    try:
        await client.start_server()
        assert await recognition() == {"state": "model_loading"}
        engine.ready.set()
        await asyncio.sleep(0)
        assert await recognition() == {"state": "policy_missing"}
        # A retired controlled-trial policy is not this release's identity policy.
        (tmp_path / "recognition-controlled-trial.json").write_text(_identity_policy())
        assert await recognition() == {"state": "policy_missing"}
        policy_file.write_text("{")
        assert await recognition() == {"state": "policy_invalid"}
        policy_file.write_text(_identity_policy(backend_version=None))
        assert await recognition() == {"state": "policy_invalid"}
        policy_file.write_text(_identity_policy(weights_hash="b" * 64))
        assert await recognition() == off("policy_model_mismatch")
        policy_file.write_text(_identity_policy())
        # The worker decides with the policy's calibration; without it, recognition is off.
        assert await recognition() == off("backend_missing")
        _write_backend(tmp_path, arrays={"path": "identity-backend.npz", "sha256": "c" * 64})
        assert await recognition() == off("backend_invalid")
        _write_backend(tmp_path, features=["score"])
        assert await recognition() == off("backend_invalid")
        _write_backend(tmp_path, version="fixture-backend-v2")
        assert await recognition() == off("backend_mismatch")
        _write_backend(tmp_path, identity=["fixture", "b" * 64, "reference-contiguous-pcm16-v1"])
        assert await recognition() == off("backend_mismatch")
        _write_backend(tmp_path)
        assert await recognition() == {
            "state": "ready", "model_id": "fixture", "policy_version": "fixture-identity-v1",
        }
        (tmp_path / "identity-backend.npz").unlink()
        assert await recognition() == off("backend_missing")
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
