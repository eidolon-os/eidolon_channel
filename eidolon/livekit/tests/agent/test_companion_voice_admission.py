import pytest
from eidolon_sdk.biz.voiceprint import ExtractedSample, VoicePolicySnapshot, VoiceTemplate
from eidolon.livekit.agent.speaker_verification.admission import CompanionVoiceAdmission


def sample(vector):
    return ExtractedSample(
        model_id="test",
        weights_hash="a" * 64,
        preprocessing_version="v1",
        speech_ms=2000,
        vector=vector,
    )


class Authority:
    def __init__(self):
        self.snapshot = VoicePolicySnapshot(
            companion_id="a",
            owner_id="owner",
            mode="allowlist",
            revision=1,
            templates=[VoiceTemplate(profile_id="alice", revision=1, sample=sample([1, 0]))],
        )

    async def get_voiceprint_snapshot(self, companion):
        return self.snapshot

    async def admit_voiceprint(self, companion, revision, profile):
        return revision == self.snapshot.revision and (
            self.snapshot.mode == "off" or profile == "alice"
        )


class Extractor:
    def __init__(self, vector=(1, 0)):
        self.vector = vector

    async def extract(self, audio, sample_rate=16000):
        return sample(list(self.vector))


async def test_matching_owner_is_bound_to_companion_revision():
    authority = Authority()
    gate = CompanionVoiceAdmission(authority, Extractor())
    result = await gate.verify("a", b"\0" * 64000)
    assert result.commit_allowed
    assert await gate.authorize("a", result)
    authority.snapshot.revision += 1
    assert not await gate.authorize("a", result)


async def test_unknown_speaker_and_missing_templates_fail_closed():
    authority = Authority()
    gate = CompanionVoiceAdmission(authority, Extractor((0, 1)))
    assert not (await gate.verify("a", b"\0" * 64000)).commit_allowed
    authority.snapshot.templates = []
    assert not (await gate.verify("a", b"\0" * 64000)).commit_allowed


async def test_missing_result_is_not_an_authorization():
    assert not await CompanionVoiceAdmission(Authority(), Extractor()).authorize("a", None)


async def test_extractor_failure_is_not_a_pass():
    class Broken:
        async def extract(self, *args, **kwargs):
            raise RuntimeError("missing weights")

    result = await CompanionVoiceAdmission(Authority(), Broken()).verify("a", b"x" * 64000)
    assert not result.commit_allowed
    assert result.commit_reason == "voiceprint_unavailable"


async def test_policy_off_still_requires_current_authority_and_companion_scope():
    authority = Authority()
    authority.snapshot.mode = "off"
    gate = CompanionVoiceAdmission(authority, Extractor())
    result = await gate.verify("a", b"")
    assert result.commit_allowed
    assert not await gate.authorize("other", result)
    authority.snapshot.mode = "allowlist"
    authority.snapshot.revision += 1
    assert not await gate.authorize("a", result)


async def test_model_version_mismatch_is_not_comparable():
    authority = Authority()
    authority.snapshot.templates[0].sample.weights_hash = "b" * 64
    result = await CompanionVoiceAdmission(authority, Extractor()).verify("a", b"x" * 64000)
    assert not result.commit_allowed
    assert result.commit_reason == "voiceprint_model_mismatch"


def test_temporal_speaker_switch_rejected_by_extraction_quality():
    import numpy as np
    from eidolon.livekit.plugins.speaker_verification.extractor import (
        validate_speaker_consistency,
        InsufficientSpeech,
    )

    validate_speaker_consistency([np.array([1.0, 0.0]), np.array([0.99, 0.1])])
    with pytest.raises(InsufficientSpeech):
        validate_speaker_consistency([np.array([1.0, 0.0]), np.array([0.0, 1.0])])


async def test_quality_diagnostic_has_code_and_duration_without_audio(caplog):
    import logging
    from eidolon.livekit.plugins.speaker_verification.extractor import InsufficientSpeech

    class Clipped:
        async def extract(self, *args):
            raise InsufficientSpeech("private detail", code="audio_clipped")

    with caplog.at_level(logging.INFO):
        result = await CompanionVoiceAdmission(Authority(), Clipped()).verify("a", b"x" * 64000)
    assert not result.commit_allowed
    assert "code=audio_clipped audio_ms=2000" in caplog.text
    assert "private detail" not in caplog.text
