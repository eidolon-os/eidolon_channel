from __future__ import annotations

from types import SimpleNamespace

import pytest
from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType

from eidolon.livekit.agent.full_duplex.transcript_evidence_buffer import (
    TranscriptEvidenceBuffer,
)
from eidolon.livekit.agent.full_duplex.transcript_settlement import (
    TranscriptSettlementLease,
)
from eidolon.livekit.agent.session.user_turn_coordinator import (
    FrameworkCompletionReadiness,
    UserTurnCoordinator,
)
from eidolon.livekit.common.transcript_evidence import (
    TranscriptBoundary,
    TranscriptEvidence,
    TranscriptEvidenceCapability,
    TranscriptSourceSpan,
    transcript_evidence_from_metadata,
    transcript_evidence_from_speech_event,
)


def _evidence(*, revision: str | None = None, span: tuple[float, float] | None = None):
    return TranscriptEvidence(
        stream_key="stream-a",
        revision_key=revision,
        source_span=(TranscriptSourceSpan(*span) if span is not None else None),
    )


@pytest.mark.parametrize(
    ("profile", "evidence", "expected"),
    [
        ("minimal", TranscriptEvidence(), frozenset()),
        (
            "identity_only",
            _evidence(revision="sentence-1"),
            frozenset({TranscriptEvidenceCapability.REVISION_IDENTITY}),
        ),
        (
            "timestamps_only",
            _evidence(span=(0.1, 0.8)),
            frozenset({TranscriptEvidenceCapability.SOURCE_TIMESTAMPS}),
        ),
        (
            "full",
            TranscriptEvidence(
                stream_key="stream-a",
                revision_key="sentence-1",
                sequence=1,
                source_span=TranscriptSourceSpan(0.1, 0.8),
                word_spans=(TranscriptSourceSpan(0.1, 0.3),),
                boundaries=frozenset({TranscriptBoundary.SENTENCE_END}),
                stable_prefix="你好",
            ),
            frozenset(
                {
                    TranscriptEvidenceCapability.REVISION_IDENTITY,
                    TranscriptEvidenceCapability.SOURCE_TIMESTAMPS,
                    TranscriptEvidenceCapability.WORD_TIMESTAMPS,
                    TranscriptEvidenceCapability.SENTENCE_BOUNDARY,
                    TranscriptEvidenceCapability.STABLE_PREFIX,
                    TranscriptEvidenceCapability.PROVIDER_SEQUENCE,
                }
            ),
        ),
    ],
)
def test_capability_profiles_are_provider_neutral(
    profile: str,
    evidence: TranscriptEvidence,
    expected: frozenset[TranscriptEvidenceCapability],
) -> None:
    assert profile
    assert evidence.capabilities == expected


def test_malformed_metadata_degrades_to_available_fields() -> None:
    evidence = transcript_evidence_from_metadata(
        {
            "eidolon.transcript_evidence": {
                "stream_key": "stream-a",
                "sequence": "not-a-number",
                "source_span": {"start_time": 10, "end_time": 2},
                "boundaries": ["sentence_end", "unknown-boundary"],
            }
        }
    )

    assert evidence.stream_key == "stream-a"
    assert evidence.sequence is None
    assert evidence.source_span is None
    assert evidence.boundaries == frozenset({TranscriptBoundary.SENTENCE_END})


def test_livekit_speech_event_is_the_evidence_transport() -> None:
    original = TranscriptEvidence(
        stream_key="task-a",
        revision_key="sentence-7",
        sequence=7,
        boundaries=frozenset({TranscriptBoundary.SENTENCE_END}),
    )
    event = SpeechEvent(
        type=SpeechEventType.FINAL_TRANSCRIPT,
        request_id="request-a",
        alternatives=[
            SpeechData(
                language="zh",
                text="你好。",
                start_time=0.2,
                end_time=0.9,
                metadata=original.as_metadata(),
            )
        ],
    )

    evidence = transcript_evidence_from_speech_event(event)

    assert evidence.stream_key == "task-a"
    assert evidence.revision_key == "sentence-7"
    assert evidence.sequence == 7
    assert evidence.source_span == TranscriptSourceSpan(0.2, 0.9)


def test_newer_revision_replaces_growing_optional_evidence() -> None:
    coordinator = UserTurnCoordinator(speech_merge_grace_sec=0.8)
    coordinator.start_speech(timeline=None)
    interim = TranscriptEvidence(
        stream_key="stream-a",
        revision_key="sentence-1",
        sequence=1,
        source_span=TranscriptSourceSpan(0.1, 0.5),
        stable_prefix="今天",
    )
    final = TranscriptEvidence(
        stream_key="stream-a",
        revision_key="sentence-1",
        sequence=2,
        source_span=TranscriptSourceSpan(0.1, 1.2),
        boundaries=frozenset({TranscriptBoundary.SENTENCE_END}),
        stable_prefix="今天天气很好",
    )

    coordinator.add_transcript("今天", is_final=False, evidence=interim)
    coordinator.add_transcript("今天天气很好", is_final=True, evidence=final)

    assert coordinator.active is not None
    merged = coordinator.active.segments[0].evidence
    assert merged is not None
    assert merged.sequence == 2
    assert merged.source_span == TranscriptSourceSpan(0.1, 1.2)
    assert merged.stable_prefix == "今天天气很好"
    assert merged.boundaries == frozenset({TranscriptBoundary.SENTENCE_END})


def test_buffer_pairs_original_speech_event_with_normalized_callback() -> None:
    evidence = _evidence(revision="sentence-3")
    event = SpeechEvent(
        type=SpeechEventType.INTERIM_TRANSCRIPT,
        alternatives=[
            SpeechData(
                language="zh",
                text="先记一下",
                metadata=evidence.as_metadata(),
            )
        ],
    )
    buffer = TranscriptEvidenceBuffer()

    buffer.observe_speech_event(event)

    assert buffer.take("先记一下", is_final=False) == evidence
    assert buffer.take("先记一下", is_final=False) is None


def test_identity_strategy_updates_one_segment_even_when_text_is_rewritten() -> None:
    coordinator = UserTurnCoordinator(speech_merge_grace_sec=0.8)
    coordinator.start_speech(timeline=None)
    evidence = _evidence(revision="sentence-1")

    coordinator.add_transcript("旧假设", is_final=False, evidence=evidence)
    coordinator.add_transcript("完全改写后的最终文本", is_final=True, evidence=evidence)

    assert coordinator.selected_text == "完全改写后的最终文本"
    assert coordinator.active is not None
    assert len(coordinator.active.segments) == 1


def test_timestamp_strategy_updates_one_segment_without_identity() -> None:
    coordinator = UserTurnCoordinator(speech_merge_grace_sec=0.8)
    coordinator.start_speech(timeline=None)

    coordinator.add_transcript(
        "天气",
        is_final=False,
        evidence=_evidence(span=(0.1, 0.7)),
    )
    coordinator.add_transcript(
        "今天天气很好",
        is_final=True,
        evidence=_evidence(span=(0.1, 0.8)),
    )

    assert coordinator.selected_text == "今天天气很好"
    assert coordinator.active is not None
    assert len(coordinator.active.segments) == 1


def test_conflicting_identity_outranks_overlapping_timestamp() -> None:
    coordinator = UserTurnCoordinator(speech_merge_grace_sec=0.8)
    coordinator.start_speech(timeline=None)

    coordinator.add_transcript(
        "第一句",
        is_final=False,
        evidence=_evidence(revision="sentence-1", span=(0.1, 0.8)),
    )
    coordinator.add_transcript(
        "第二句",
        is_final=False,
        evidence=_evidence(revision="sentence-2", span=(0.6, 1.2)),
    )

    assert coordinator.selected_text == "第一句第二句"
    assert coordinator.active is not None
    assert len(coordinator.active.segments) == 2


def test_minimal_profile_remains_supported() -> None:
    coordinator = UserTurnCoordinator(speech_merge_grace_sec=0.8)
    coordinator.start_speech(timeline=None)
    coordinator.add_transcript("只有标准字段", is_final=False)
    coordinator.add_transcript("只有标准字段。", is_final=True)

    assert coordinator.selected_text == "只有标准字段。"
    assert coordinator.snapshot()["transcript_evidence_capabilities"] == []


def test_normalized_livekit_item_id_is_optional_revision_identity() -> None:
    from eidolon.livekit.agent.full_duplex.transcript_event import FullDuplexTranscriptEvent

    event = FullDuplexTranscriptEvent.from_event(
        SimpleNamespace(
            transcript="实时模型文本",
            is_final=True,
            item_id="item-9",
        )
    )

    assert event.evidence.revision_key == "item-9"


@pytest.mark.asyncio
async def test_settlement_does_not_miss_evidence_arriving_during_readiness_check() -> None:
    class CoordinatorWithInterleavedEvidence:
        def __init__(self) -> None:
            self.active = SimpleNamespace(candidate_id="candidate-1")
            self.transcript_change_version = 0
            self._ready = False
            self.waited_after_versions: list[int] = []

        def framework_completion_readiness(self, _transcript: str):
            if not self._ready:
                # Deterministically model evidence arriving after the lease
                # snapshots the version but during readiness evaluation.
                self._ready = True
                self.transcript_change_version += 1
                return FrameworkCompletionReadiness(False, "pending")
            return FrameworkCompletionReadiness(True, "covered")

        async def wait_for_transcript_change(
            self,
            *,
            after_version: int,
            timeout_sec: float,
        ) -> bool:
            self.waited_after_versions.append(after_version)
            return self.transcript_change_version != after_version

    coordinator = CoordinatorWithInterleavedEvidence()
    result = await TranscriptSettlementLease(
        coordinator,  # type: ignore[arg-type]
        timeout_sec=0.1,
    ).settle(
        candidate_id="candidate-1",
        framework_transcript="旧 final",
    )

    assert result.outcome == "ready"
    assert result.evidence_updates == 1
    assert coordinator.waited_after_versions == [0]
