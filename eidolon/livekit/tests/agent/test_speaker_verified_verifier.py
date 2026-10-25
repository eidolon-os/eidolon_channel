"""No permission from stale templates, poor audio or another model version."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from eidolon_sdk.biz.voiceprint import ExtractedSample, VoiceTemplate, VoicePolicySnapshot
from eidolon_sdk.core.http.errors import ServiceUpstreamError
from eidolon.livekit.agent.speaker_verified.verifier import CalibratedSpeakerVerifier
from eidolon_sdk.biz.speaker_calibration import SpeakerAdmissionCalibration, load_calibration


def fixture():
    sample = ExtractedSample(
        model_id="test",
        weights_hash="a" * 64,
        preprocessing_version="v1",
        vector=[1, 0],
        speech_ms=2000,
    )
    snapshot = VoicePolicySnapshot(
        owner_id="owner",
        companion_id="companion",
        mode="allowlist",
        revision=1,
        templates=[VoiceTemplate(profile_id="p", revision=2, sample=sample)],
    )
    calibration = SpeakerAdmissionCalibration(
        validation_state="calibrated",
        calibration_id="synthetic-test-only",
        model_id="test",
        weights_hash="a" * 64,
        preprocessing_version="v1",
        threshold=0.8,
        max_allowed_speakers=1,
        registration_min_similarity=0.7,
        trial_min_similarity=0.7,
        min_audio_ms=1500,
        max_audio_ms=8000,
        evidence_sha256="b" * 64,
    )
    snapshot.calibration_digest = calibration.digest
    data = SimpleNamespace(get_voiceprint_snapshot=AsyncMock(return_value=snapshot))
    extractor = SimpleNamespace(
        extract=AsyncMock(return_value=sample),
        readiness=AsyncMock(
            return_value=SimpleNamespace(
                admission_ready=True, calibration_digest=calibration.digest
            )
        ),
    )
    verifier = CalibratedSpeakerVerifier(
        data=data,
        extractor=extractor,
        calibration=calibration,
        owner_id="owner",
        companion_id="companion",
    )
    evidence = SimpleNamespace(complete=True, reason="", sample_rate=16000, pcm=b"\1\0" * 32000)
    return verifier, data, extractor, evidence, snapshot, sample


async def test_match_requires_fresh_scoped_template_and_matching_artifact():
    verifier, data, extractor, evidence, _, _ = fixture()
    result = await verifier(evidence)
    assert (result.status, result.profile_id, result.profile_revision) == ("match", "p", 2)
    assert result.calibration_version == verifier.calibration.digest
    extractor.extract.assert_awaited_once_with(evidence.pcm)
    data.get_voiceprint_snapshot.assert_awaited_once_with("companion")


@pytest.mark.parametrize(
    "change", [{"mode": "off"}, {"owner_id": "other"}, {"companion_id": "other"}, {"templates": []}]
)
async def test_changed_scope_never_extracts(change):
    verifier, data, extractor, evidence, snapshot, _ = fixture()
    data.get_voiceprint_snapshot.return_value = snapshot.model_copy(update=change)
    assert (await verifier(evidence)).status == "unavailable"
    extractor.extract.assert_not_awaited()


@pytest.mark.parametrize(
    "change", [{"weights_hash": "c" * 64}, {"preprocessing_version": "v2"}, {"vector": [0, 1]}]
)
async def test_incompatible_or_other_speaker_never_matches(change):
    verifier, _, extractor, evidence, _, sample = fixture()
    extractor.extract.return_value = sample.model_copy(update=change)
    assert (await verifier(evidence)).status == (
        "non_match" if "vector" in change else "unavailable"
    )


async def test_reenrollment_uses_current_template_revision_for_data_authorization():
    verifier, data, extractor, evidence, snapshot, _ = fixture()
    snapshot.templates[0].revision = 3
    result = await verifier(evidence)
    assert result.status == "match"
    assert result.profile_revision == 3
    extractor.extract.assert_awaited_once()


async def test_model_calibration_does_not_bind_an_owner_companion_or_device():
    verifier, data, extractor, evidence, snapshot, _ = fixture()
    data.get_voiceprint_snapshot.return_value = snapshot.model_copy(
        update={"owner_id": "another-owner", "companion_id": "another-companion"}
    )
    other = CalibratedSpeakerVerifier(
        data=data,
        extractor=extractor,
        calibration=verifier.calibration,
        owner_id="another-owner",
        companion_id="another-companion",
    )
    assert (await other(evidence)).status == "match"
    data.get_voiceprint_snapshot.assert_awaited_once_with("another-companion")


async def test_unevaluated_allowlist_size_is_not_permission():
    verifier, _, extractor, evidence, snapshot, _ = fixture()
    snapshot.templates.append(snapshot.templates[0].model_copy(update={"profile_id": "second"}))
    assert (await verifier(evidence)).status == "unavailable"
    extractor.extract.assert_not_awaited()


async def test_incomplete_or_short_audio_does_not_reuse_a_match():
    verifier, _, extractor, evidence, _, _ = fixture()
    assert (await verifier(evidence)).status == "match"
    evidence.complete = False
    assert (await verifier(evidence)).status == "inconclusive"
    assert extractor.extract.await_count == 1


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -0.1, 1.1, True, "0.8"])
def test_no_guessed_or_nonfinite_threshold(threshold):
    verifier, *_ = fixture()
    with pytest.raises(ValueError):
        SpeakerAdmissionCalibration.model_validate(
            verifier.calibration.model_dump() | {"threshold": threshold}
        )


@pytest.mark.parametrize(
    "status,expected", [(422, "inconclusive"), (401, "unavailable"), (503, "unavailable")]
)
async def test_quality_failure_is_distinct_from_identity_rejection(status, expected):
    verifier, _, extractor, evidence, _, _ = fixture()
    extractor.extract.side_effect = ServiceUpstreamError(status, "upstream private body")
    assert (await verifier(evidence)).status == expected


@pytest.mark.parametrize("contents", ["{private malformed artifact", "x" * 16385, "{}"])
def test_invalid_artifact_never_leaks_body_or_provides_default(tmp_path, contents):
    path = tmp_path / "calibration.json"
    path.write_text(contents)
    with pytest.raises(ValueError) as failure:
        load_calibration(str(path))
    assert str(failure.value) == "SPEAKER_VERIFIED_CALIBRATION_INVALID"


def test_missing_artifact_never_uses_legacy_threshold():
    with pytest.raises(ValueError, match="SPEAKER_VERIFIED_CALIBRATION_REQUIRED"):
        load_calibration("")


async def test_provider_release_changed_cannot_use_cached_worker_calibration():
    verifier, _, extractor, audio, _, _ = fixture()
    extractor.readiness.return_value.calibration_digest = 'f'*64
    assert (await verifier(audio)).status == 'unavailable'
    extractor.extract.assert_not_awaited()
