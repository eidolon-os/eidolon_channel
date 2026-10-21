from dataclasses import replace

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.session.user_turn_coordinator import UserTurnCoordinator
from eidolon.livekit.common.audio_timeline import AudioSource, AudioSpan
from eidolon.livekit.common.transcript_evidence import TranscriptEvidence


def evidence(revision, start, end, *, generation=1):
    return TranscriptEvidence(
        stream_key="task",
        revision_key=revision,
        audio_span=AudioSpan("input", generation, 16000, start, end),
    )


def owner():
    value = UserTurnCoordinator(speech_merge_grace_sec=0.8)
    value.start_speech(timeline=TurnTimeline("candidate"), now=0)
    return value


def test_paused_continuation_covers_both_sentences_and_invalidates_prefix_result():
    value = owner()
    value.add_transcript("第一句", is_final=True, evidence=evidence("a", 0, 1600), now=0.1)
    value.note_speech_stopped(eot_score=None, now=0.2)
    first = value.capture_audio_reference(source=AudioSource("input", 1, 16000))
    assert first.span == AudioSpan("input", 1, 16000, 0, 1600)
    assert value.is_audio_reference_current(first, source=AudioSource("input", 1, 16000))
    value.start_speech(timeline=TurnTimeline("same"), now=0.3)
    assert not value.is_audio_reference_current(first, source=AudioSource("input", 1, 16000))
    assert value.capture_audio_reference(source=AudioSource("input", 1, 16000)).span is None
    value.add_transcript("第二句", is_final=True, evidence=evidence("b", 3200, 4800), now=0.4)
    value.note_speech_stopped(eot_score=None, now=0.5)
    complete = value.capture_audio_reference(source=AudioSource("input", 1, 16000))
    assert complete.span == AudioSpan("input", 1, 16000, 0, 4800)
    assert value.selected_text == "第一句第二句"
    value.reject_active("test")
    assert not value.is_audio_reference_current(complete, source=AudioSource("input", 1, 16000))


def test_new_revision_missing_audio_cannot_borrow_previous_prefix_mapping():
    value = owner()
    value.add_transcript("前半句", is_final=False, evidence=evidence("a", 0, 1600))
    value.add_transcript(
        "前半句和后半句",
        is_final=True,
        evidence=TranscriptEvidence(stream_key="task", revision_key="a"),
    )
    value.note_speech_stopped(eot_score=None)
    assert (
        value.capture_audio_reference(source=AudioSource("input", 1, 16000)).reason
        == "audio_mapping_missing"
    )


def test_late_final_invalidates_frozen_reference_even_in_same_acoustic_generation():
    value = owner()
    value.add_transcript("一句话", is_final=True, evidence=evidence("a", 0, 1600))
    value.note_speech_stopped(eot_score=None)
    first = value.capture_audio_reference(source=AudioSource("input", 1, 16000))
    value.add_transcript("一句话的后半句", is_final=True, evidence=evidence("a", 0, 3200))
    assert not value.is_audio_reference_current(first, source=AudioSource("input", 1, 16000))
    assert value.capture_audio_reference(source=AudioSource("input", 1, 16000)).span.end == 3200


def test_reconnect_cannot_join_audio_from_different_provider_runs():
    value = owner()
    value.add_transcript("上一连接", is_final=True, evidence=evidence("a", 0, 1600))
    value.add_transcript(
        "下一连接",
        is_final=True,
        evidence=replace(evidence("b", 0, 1600, generation=2), stream_key="new"),
    )
    value.note_speech_stopped(eot_score=None)
    assert (
        value.capture_audio_reference(source=AudioSource("input", 1, 16000)).reason
        == "audio_source_changed"
    )


def test_pending_or_untranscribed_segment_never_uses_last_final_as_entire_turn():
    value = owner()
    value.add_transcript("先说一半", is_final=True, evidence=evidence("a", 0, 1600))
    value.add_transcript("还有", is_final=False, evidence=evidence("b", 1600, 3200))
    value.note_speech_stopped(eot_score=None)
    assert value.capture_audio_reference(source=AudioSource("input", 1, 16000)).span is None


def test_textual_coverage_does_not_prove_audio_coverage():
    value = owner()
    first = value.add_transcript("前半", is_final=False, evidence=evidence("a", 0, 1600))
    value.note_speech_stopped(eot_score=None, now=0.1)
    value.start_speech(timeline=TurnTimeline("same"), now=0.2)
    value.add_transcript("前半后半", is_final=False, evidence=evidence("b", 1600, 3200))
    last = value.add_transcript("前半后半", is_final=True, evidence=evidence("b", 1600, 3200))
    value.cover_pending_transcript_segments(
        candidate_id=first.candidate_id,
        segment_indexes=(0,),
        covered_by_generation_id=last.generation_id,
        covered_by_segment_index=last.segment_index,
        reason="provider_coverage",
    )
    value.note_speech_stopped(eot_score=None)
    assert (
        value.capture_audio_reference(source=AudioSource("input", 1, 16000)).reason
        == "audio_coverage_unproven"
    )


def test_old_connection_only_final_cannot_authorize_current_input():
    value = owner()
    value.add_transcript("旧连接结果", is_final=True, evidence=evidence("a", 0, 1600))
    value.note_speech_stopped(eot_score=None)
    assert (
        value.capture_audio_reference(source=AudioSource("input", 2, 16000)).reason
        == "audio_source_changed"
    )


def test_public_transcript_bridge_preserves_mapped_span_and_missing_revision_clears_it():
    from livekit.agents.stt import SpeechEvent, SpeechEventType, SpeechData
    from eidolon.livekit.agent.full_duplex.transcript_evidence_buffer import (
        TranscriptEvidenceBuffer,
    )

    expected = evidence("a", 0, 1600)
    buffer = TranscriptEvidenceBuffer()
    event = SpeechEvent(
        type=SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[SpeechData(language="zh", text="映射", metadata=expected.as_metadata())],
    )
    assert buffer.observe_speech_event(event)
    received = buffer.take("映射", is_final=True)
    value = owner()
    value.add_transcript("映射", is_final=True, evidence=received)
    value.note_speech_stopped(eot_score=None)
    assert (
        value.capture_audio_reference(source=AudioSource("input", 1, 16000)).span
        == expected.audio_span
    )
    assert TranscriptEvidence().merge(expected).audio_span is None
