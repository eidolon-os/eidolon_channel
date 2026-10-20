"""Audio identity evidence must refer to exact samples from one source generation."""

import pytest

from eidolon.livekit.agent.session.audio_evidence import AudioEvidenceBuffer


def test_snapshot_slices_frames_without_reordering_or_including_neighbours():
    buf = AudioEvidenceBuffer(source="box", generation=1, sample_rate=16000, max_samples=12)
    buf.append(start=0, pcm=b"\x01\x00" * 4)
    buf.append(start=4, pcm=b"\x02\x00" * 4)
    evidence = buf.snapshot(candidate="turn", start=2, end=6, generation=1)
    assert evidence.complete
    assert evidence.pcm == b"\x01\x00" * 2 + b"\x02\x00" * 2
    assert (evidence.start, evidence.end, evidence.candidate) == (2, 6, "turn")


def test_gap_never_becomes_a_shortened_but_valid_utterance():
    buf = AudioEvidenceBuffer(source="box", generation=1, sample_rate=16000, max_samples=12)
    buf.append(start=0, pcm=b"\x01\x00" * 4)
    buf.append(start=6, pcm=b"\x02\x00" * 4)
    evidence = buf.snapshot(candidate="turn", start=0, end=10, generation=1)
    assert not evidence.complete
    assert evidence.reason == "missing_samples"
    assert evidence.pcm == b""


def test_evicted_audio_and_stale_source_are_not_valid_evidence():
    buf = AudioEvidenceBuffer(source="box", generation=3, sample_rate=16000, max_samples=4)
    buf.append(start=0, pcm=b"\x01\x00" * 8)
    assert buf.retained_samples == 4
    assert not buf.snapshot(candidate="old", start=0, end=4, generation=3).complete
    assert not buf.snapshot(candidate="wrong-source", start=4, end=8, generation=2).complete
    assert buf.snapshot(candidate="current", start=4, end=8, generation=3).complete


def test_duplicate_or_overlapping_frames_are_rejected():
    buf = AudioEvidenceBuffer(source="box", generation=1, sample_rate=16000, max_samples=12)
    buf.append(start=0, pcm=b"\x01\x00" * 4)
    with pytest.raises(ValueError, match="overlap"):
        buf.append(start=2, pcm=b"\x01\x00" * 4)


def test_clear_invalidates_retained_audio_and_snapshots_are_immutable():
    buf = AudioEvidenceBuffer(source="box", generation=1, sample_rate=16000, max_samples=12)
    incoming = bytearray(b"\x01\x00" * 4)
    buf.append(start=0, pcm=incoming)
    evidence = buf.snapshot(candidate="turn", start=0, end=4, generation=1)
    incoming[:] = b"\x02\x00" * 4
    assert evidence.pcm == b"\x01\x00" * 4
    buf.clear()
    assert not buf.snapshot(candidate="turn", start=0, end=4, generation=1).complete
