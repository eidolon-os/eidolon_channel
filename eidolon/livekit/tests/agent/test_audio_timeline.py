from dataclasses import replace

import pytest

from eidolon.livekit.common.audio_timeline import AudioSpan, ProviderAudioTimeline


def timeline():
    return ProviderAudioTimeline(
        source="stream", generation=1, sample_rate=16000, max_samples=16000
    )


def test_wire_timestamps_map_after_keepalive_without_shifting_input():
    mapping = timeline()
    mapping.record(samples=1600, input_start=None)
    mapping.record(samples=3200, input_start=0)
    assert mapping.resolve_ms(100, 250) == AudioSpan("stream", 1, 16000, 0, 2400)
    assert mapping.resolve_ms(50, 150) is None


def test_gate_gaps_replayed_preroll_and_unsent_tail_are_not_continuous_evidence():
    mapping = timeline()
    mapping.record(samples=1600, input_start=0)
    mapping.record(samples=1600, input_start=3200)
    assert mapping.resolve_ms(0, 200) is None
    assert mapping.resolve_ms(100, 200).start == 3200
    mapping.record(samples=1600, input_start=3200)
    assert mapping.resolve_ms(100, 300) is None
    assert mapping.resolve_ms(300, 400) is None


@pytest.mark.parametrize(
    "start,end", [(float("nan"), 2), (0, float("inf")), (-1, 1), (2, 1), (0, 0)]
)
def test_invalid_provider_time_has_no_audio_mapping(start, end):
    assert timeline().resolve_ms(start, end) is None


def test_bound_history_and_new_generation_do_not_alias_old_connection():
    mapping = timeline()
    for i in range(20):
        mapping.record(samples=1600, input_start=i * 1600)
    assert mapping.resolve_ms(0, 100) is None
    span = mapping.resolve_ms(1900, 2000)
    assert span.end == 32000
    assert span != replace(span, generation=2)


def test_audio_span_metadata_is_strict_and_scope_preserving():
    span = AudioSpan("stream", 2, 16000, 160, 320)
    assert AudioSpan.from_metadata(span.as_metadata()) == span
    for malformed in (
        {},
        {**span.as_metadata(), "end": True},
        {**span.as_metadata(), "start": 500},
        {**span.as_metadata(), "sample_rate": 0},
    ):
        assert AudioSpan.from_metadata(malformed) is None
