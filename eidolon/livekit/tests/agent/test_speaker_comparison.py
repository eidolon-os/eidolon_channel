import pytest
from eidolon_sdk.biz.voiceprint import ExtractedSample, VoiceTemplate, SpeakerComparisonRequest
from eidolon.livekit.plugins.speaker_verification.scoring import compare_speakers


def sample(values, **changes):
    return ExtractedSample(
        model_id="test",
        weights_hash="a" * 64,
        preprocessing_version="v1",
        vector=values,
        speech_ms=3000,
    ).model_copy(update=changes)


def test_each_template_is_scored_without_claiming_identity():
    result = compare_speakers(
        SpeakerComparisonRequest(
            probe=sample([1, 0]),
            templates=[
                VoiceTemplate(profile_id="a", revision=2, sample=sample([1, 0])),
                VoiceTemplate(profile_id="b", revision=1, sample=sample([0, 1])),
                VoiceTemplate(profile_id="c", revision=1, sample=sample([-1, 0])),
            ],
        )
    )
    assert [r.similarity for r in result.scores] == [1, 0, -1]
    assert result.validation_state == "unvalidated"
    assert not hasattr(result, "matched")
    assert "vector" not in result.model_dump_json()


@pytest.mark.parametrize(
    "change",
    [
        dict(weights_hash="b" * 64),
        dict(model_id="other"),
        dict(preprocessing_version="v2"),
        dict(vector=[1, 0, 0]),
    ],
)
def test_incompatible_profile_is_explicit_not_zero_or_cross_version_score(change):
    result = compare_speakers(
        SpeakerComparisonRequest(
            probe=sample([1, 0]),
            templates=[
                VoiceTemplate(profile_id="a", revision=1, sample=sample([1, 0], **change)),
            ],
        )
    )
    assert result.scores[0].status == "incompatible"
    assert result.scores[0].similarity is None
