import pytest

from eidolon.livekit.agent.session.user_turn_coordinator import UserTurnCoordinator
from eidolon.livekit.common.audio_timeline import AudioSource, AudioSpan
from eidolon.livekit.common.transcript_evidence import TranscriptEvidence


@pytest.fixture
def candidate():
    owner = UserTurnCoordinator(speech_merge_grace_sec=0.8)
    source = AudioSource("input", 1, 16000)
    evidence = TranscriptEvidence(
        stream_key="asr",
        revision_key="one",
        sequence=1,
        audio_span=AudioSpan("input", 1, 16000, 0, 32000),
    )
    owner.start_speech(timeline=None)
    owner.add_transcript("完整获批句子", is_final=True, evidence=evidence)
    owner.note_speech_stopped(eot_score=0.9)
    return owner, source, evidence


def test_verified_candidate_commits_exact_text_once_and_retires_provider_revision(candidate):
    owner, source, evidence = candidate
    reference = owner.capture_audio_reference(source=source)
    result = owner.commit_verified(reference, source=source)
    assert result.action == "commit" and result.transcript == "完整获批句子"
    assert owner.commit_verified(reference, source=source).action == "none"
    assert owner.is_retired_revision(evidence)
    assert not owner.absorb_committed_transcript_revision(
        "迟到新增内容", is_final=True, reason="late"
    )
    assert owner.selected_text == "完整获批句子"


@pytest.mark.parametrize("change", ["text", "speech", "source", "rejected"])
def test_changed_candidate_cannot_commit_with_old_reference(candidate, change):
    owner, source, evidence = candidate
    reference = owner.capture_audio_reference(source=source)
    if change == "text":
        owner.add_transcript("发生变化的转写", is_final=True, evidence=evidence)
    elif change == "speech":
        owner.start_speech(timeline=None, continue_pending=True)
    elif change == "source":
        source = AudioSource("replacement", 2, 16000)
    else:
        owner.reject_active("revoked")
    assert owner.commit_verified(reference, source=source).action == "none"
    assert owner.active.state != "committed"
