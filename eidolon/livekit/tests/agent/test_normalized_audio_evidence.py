"""Evidence owns one normalized stream; lifecycle changes never relabel old PCM."""

from eidolon.livekit.common.audio_evidence import AudioEvidenceCapture
from eidolon.livekit.common.audio_timeline import AudioSource, AudioSpan


def capture():
    return AudioEvidenceCapture(session_id="session", endpoint_id="endpoint", max_seconds=1)


def begin(value, name="stream", generation=1):
    value.select_stream(name)
    source = AudioSource(name, generation, 16000)
    value.begin_run(source)
    return source


def span(source, start=0, end=4):
    return AudioSpan(source.source, source.generation, source.sample_rate, start, end)


def test_exact_normalized_pcm_and_scope_without_exposing_pcm_in_repr():
    value = capture()
    source = begin(value)
    value.append(source, start=0, pcm=b"\x11\x22" * 8)
    evidence = value.snapshot(candidate="turn", span=span(source, 2, 6))
    assert evidence.complete and evidence.pcm == b"\x11\x22" * 4
    assert (evidence.session_id, evidence.endpoint_id) == ("session", "endpoint")
    assert "pcm=" not in repr(evidence)
    assert value.active_source == source


def test_provider_retry_clears_pcm_and_rejects_old_writer_and_old_cleanup():
    value = capture()
    old = begin(value)
    value.append(old, start=0, pcm=b"\x01\x00" * 4)
    new = AudioSource(old.source, 2, 16000)
    value.begin_run(new)
    value.append(old, start=0, pcm=b"\x02\x00" * 4)
    assert value.retained_samples == 0
    value.append(new, start=0, pcm=b"\x03\x00" * 4)
    value.end_run(old)
    assert not value.snapshot(candidate="old", span=span(old)).complete
    assert value.snapshot(candidate="new", span=span(new)).pcm == b"\x03\x00" * 4


def test_new_stream_cannot_be_replaced_by_previous_stream_retry():
    value = capture()
    old = begin(value, "old")
    new = begin(value, "new")
    value.begin_run(AudioSource("old", 100, 16000))
    value.append(old, start=0, pcm=b"\x01\x00" * 4)
    value.end_run(old)
    assert value.active_source == new
    assert value.retained_samples == 0


def test_input_invalidation_is_not_undone_by_reconnect_or_new_recognizer():
    value = capture()
    source = begin(value)
    value.append(source, start=0, pcm=b"\x01\x00" * 4)
    value.invalidate("input_detached")
    assert value.active_source is None and value.retained_samples == 0
    newer = begin(value, "new")
    value.append(newer, start=0, pcm=b"\x02\x00" * 4)
    evidence = value.snapshot(candidate="late", span=span(source))
    assert not evidence.complete and evidence.pcm == b""
    assert evidence.reason == "input_detached"
    assert value.active_source is None


def test_missing_or_expired_samples_never_return_retained_suffix():
    value = capture()
    source = begin(value)
    value.append(source, start=0, pcm=b"\x01\x00" * 16001)
    assert value.retained_samples == 16000
    assert not value.snapshot(candidate="expired", span=span(source, 0, 10)).complete
    assert not value.snapshot(candidate="long", span=span(source, 0, 16001)).complete
    value.append(source, start=16004, pcm=b"\x01\x00" * 4)
    assert not value.snapshot(candidate="gap", span=span(source, 16000, 16008)).complete


def test_end_of_run_removes_current_source_and_releases_audio():
    value = capture()
    source = begin(value)
    value.append(source, start=0, pcm=b"\x01\x00" * 4)
    value.end_run(source)
    assert value.active_source is None and value.retained_samples == 0
    assert not value.snapshot(candidate="late", span=span(source)).complete


def test_malformed_evidence_does_not_raise_into_media_or_allow_a_later_suffix():
    value = capture()
    source = begin(value)
    value.append(source, start=0, pcm=b"\x01\x00" * 4)
    value.append(source, start=2, pcm=b"\x02\x00" * 4)
    value.append(source, start=6, pcm=b"\x03\x00" * 4)
    evidence = value.snapshot(candidate="tail", span=span(source, 6, 10))
    assert not evidence.complete and evidence.reason == "invalid_audio_sequence"
    assert value.active_source is None and value.retained_samples == 0
