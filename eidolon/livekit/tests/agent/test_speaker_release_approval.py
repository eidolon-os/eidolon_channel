"""Release evaluation is numerical evidence, not a Companion permission write."""

import pytest
from eidolon_sdk.biz.voiceprint import (
    EnrollmentEvaluation,
    VoiceProfileEvidence,
    VoicePolicyEvidence,
)
from eidolon.livekit.plugins.speaker_verification.admission import approve_policy
from .test_speaker_verified_verifier import fixture


def evidence():
    verifier, _, _, _, _, sample = fixture()
    evaluation = EnrollmentEvaluation(
        evidence_digest="a" * 64,
        recording_sha256="b" * 64,
        template=sample,
        trial=sample,
        registration_min_similarity=0.9,
        trial_min_similarity=0.9,
    )
    value = VoicePolicyEvidence(
        owner_id="owner",
        companion_id="c",
        revision=2,
        profiles=[VoiceProfileEvidence(profile_id="p", revision=3, evaluation=evaluation)],
    )
    return value, verifier.calibration


def test_release_approval_binds_whole_current_evidence_and_content_digest():
    value, calibration = evidence()
    approval = approve_policy(value, calibration)
    assert approval.evidence_digest == value.digest()
    assert approval.calibration_digest == calibration.digest
    changed = calibration.model_copy(update={"threshold": 0.85})
    assert changed.calibration_id == calibration.calibration_id
    assert changed.digest != approval.calibration_digest


@pytest.mark.parametrize("change", ["weak_registration", "weak_trial", "model", "count"])
def test_unsupported_or_inconsistent_enrollment_does_not_get_approval(change):
    value, calibration = evidence()
    evaluation = value.profiles[0].evaluation
    if change == "weak_registration":
        evaluation.registration_min_similarity = 0.1
    elif change == "weak_trial":
        evaluation.trial_min_similarity = 0.1
    elif change == "model":
        evaluation.trial.weights_hash = "e" * 64
    else:
        value.profiles.append(value.profiles[0].model_copy(update={"profile_id": "other"}))
    with pytest.raises(ValueError):
        approve_policy(value, calibration)
