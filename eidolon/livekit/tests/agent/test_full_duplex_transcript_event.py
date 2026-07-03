from __future__ import annotations

from types import SimpleNamespace

from eidolon.livekit.agent.full_duplex.transcript_event import (
    FullDuplexTranscriptEvent,
)


def test_transcript_event_normalizes_livekit_event_shape() -> None:
    event = SimpleNamespace(
        transcript="停一下",
        speaker_id="user-1",
        is_final=True,
    )

    normalized = FullDuplexTranscriptEvent.from_event(event)

    assert normalized.transcript == "停一下"
    assert normalized.speaker_id == "user-1"
    assert normalized.is_final is True
    assert normalized.has_transcript is True
    assert normalized.timeline_mark == "transcript_final_at"


def test_transcript_event_handles_missing_or_empty_fields() -> None:
    normalized = FullDuplexTranscriptEvent.from_event(SimpleNamespace(transcript=None))

    assert normalized.transcript == ""
    assert normalized.speaker_id is None
    assert normalized.is_final is False
    assert normalized.has_transcript is False
    assert normalized.timeline_mark == "transcript_interim_first_at"


def test_transcript_event_from_event_is_idempotent() -> None:
    event = FullDuplexTranscriptEvent(
        transcript="你好",
        speaker_id="user-2",
        is_final=False,
    )

    assert FullDuplexTranscriptEvent.from_event(event) is event
