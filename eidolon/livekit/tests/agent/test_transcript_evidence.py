"""Transcript evidence gate tests."""

from __future__ import annotations

from eidolon.livekit.agent.turn_policy import TranscriptEvidenceGate


def test_short_latin_interim_is_not_cancel_evidence() -> None:
    evidence = TranscriptEvidenceGate().evaluate("If")

    assert evidence.allow_cancel is False
    assert evidence.reason == "short_latin_artifact"


def test_substantive_cjk_interim_is_cancel_evidence() -> None:
    evidence = TranscriptEvidenceGate().evaluate("我不相信")

    assert evidence.allow_cancel is True
    assert evidence.reason == "enough_cjk_interim"


def test_final_transcript_is_cancel_evidence() -> None:
    evidence = TranscriptEvidenceGate().evaluate("If", is_final=True)

    assert evidence.allow_cancel is True
    assert evidence.reason == "final_transcript"
