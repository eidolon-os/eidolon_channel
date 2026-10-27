import pytest
from eidolon_sdk.biz.voiceprint import EnrollmentEvaluationRequest, ExtractedSample
from eidolon.livekit.plugins.speaker_verification.enrollment import evaluate_enrollment


def sample(vector, version="v1"):
    return ExtractedSample(
        model_id="test",
        weights_hash="a" * 64,
        preprocessing_version=version,
        vector=vector,
        speech_ms=4000,
    )


def test_trial_never_changes_template_and_scores_are_not_identity_approval():
    request = EnrollmentEvaluationRequest(
        samples={str(i): sample([1, 0]) for i in range(3)},
        trial=sample([0, 1]),
        recording_sha256="b" * 64,
    )
    result = evaluate_enrollment(request)
    assert result.template.vector == [1, 0]
    assert result.registration_min_similarity == 1
    assert result.trial_min_similarity == 0
    assert result.validation_state == "unvalidated"


def test_incompatible_preprocessing_is_not_pooled():
    request = EnrollmentEvaluationRequest(
        samples={str(i): sample([1, 0]) for i in range(3)},
        trial=sample([1, 0], "different"),
        recording_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="incompatible"):
        evaluate_enrollment(request)


@pytest.mark.parametrize("part", ["trial", "registration"])
def test_short_conversation_evidence_cannot_be_used_for_enrollment(part):
    samples = {str(i): sample([1, 0]) for i in range(3)}
    trial = sample([1, 0])
    if part == "trial":
        trial.speech_ms = 400
    else:
        samples["0"].speech_ms = 400
    with pytest.raises(ValueError, match="insufficient_enrollment_speech"):
        evaluate_enrollment(EnrollmentEvaluationRequest(samples=samples, trial=trial,
                                                       recording_sha256="b" * 64))
