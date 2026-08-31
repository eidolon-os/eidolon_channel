from __future__ import annotations

from eidolon.livekit.agent.full_duplex.transcript_hypothesis_reconciler import (
    TranscriptHypothesisReconciler,
)
from eidolon.livekit.agent.session.user_turn_coordinator import TranscriptRevisionReceipt
from eidolon.livekit.common.transcript_evidence import TranscriptEvidence


def _receipt(
    candidate_id: str,
    generation_id: int,
    segment_index: int,
    evidence: TranscriptEvidence | None = None,
) -> TranscriptRevisionReceipt:
    return TranscriptRevisionReceipt(
        candidate_id=candidate_id,
        generation_id=generation_id,
        segment_index=segment_index,
        evidence=evidence,
    )


def test_final_covers_equivalent_interim_from_prior_generation() -> None:
    reconciler = TranscriptHypothesisReconciler(min_normalized_chars=4)

    assert reconciler.observe(
        _receipt("turn-a", 1, 0), text="铁锤三二五", is_final=False
    ) == ()
    assert reconciler.observe(
        _receipt("turn-a", 2, 1), text="铁锤三二五", is_final=False
    ) == ()

    assert reconciler.observe(
        _receipt("turn-a", 2, 1), text="铁锤三二五。", is_final=True
    ) == (0,)


def test_non_equivalent_hypothesis_is_not_covered() -> None:
    reconciler = TranscriptHypothesisReconciler(min_normalized_chars=4)
    reconciler.observe(_receipt("turn-a", 1, 0), text="第一句话", is_final=False)
    reconciler.observe(_receipt("turn-a", 2, 1), text="第二句话", is_final=False)

    assert reconciler.observe(
        _receipt("turn-a", 2, 1), text="第二句话。", is_final=True
    ) == ()


def test_repeated_phrase_without_target_interim_is_not_treated_as_alias() -> None:
    reconciler = TranscriptHypothesisReconciler(min_normalized_chars=4)
    reconciler.observe(_receipt("turn-a", 1, 0), text="再说一次", is_final=False)

    assert reconciler.observe(
        _receipt("turn-a", 2, 1), text="再说一次。", is_final=True
    ) == ()


def test_final_then_interim_in_one_generation_tracks_each_segment() -> None:
    reconciler = TranscriptHypothesisReconciler(min_normalized_chars=4)
    reconciler.observe(_receipt("turn-a", 1, 0), text="第一句。", is_final=True)
    reconciler.observe(_receipt("turn-a", 1, 1), text="第二句", is_final=False)
    reconciler.observe(_receipt("turn-a", 2, 2), text="第二句", is_final=False)

    assert reconciler.observe(
        _receipt("turn-a", 2, 2), text="第二句。", is_final=True
    ) == (1,)


def test_candidate_boundary_discards_prior_hypotheses() -> None:
    reconciler = TranscriptHypothesisReconciler(min_normalized_chars=4)
    reconciler.observe(_receipt("turn-a", 1, 0), text="不会跨轮", is_final=False)

    assert reconciler.observe(
        _receipt("turn-b", 2, 0), text="不会跨轮。", is_final=True
    ) == ()


def test_identity_covers_same_revision_without_text_heuristic() -> None:
    reconciler = TranscriptHypothesisReconciler(min_normalized_chars=20)
    evidence = TranscriptEvidence(stream_key="task-a", revision_key="sentence-1")
    reconciler.observe(
        _receipt("turn-a", 1, 0, evidence),
        text="旧假设",
        is_final=False,
    )
    reconciler.observe(
        _receipt("turn-a", 2, 1, evidence),
        text="完全不同的新假设",
        is_final=False,
    )

    assert reconciler.observe(
        _receipt("turn-a", 2, 1, evidence),
        text="最终改写",
        is_final=True,
    ) == (0,)


def test_conflicting_identity_prevents_text_based_aliasing() -> None:
    reconciler = TranscriptHypothesisReconciler(min_normalized_chars=4)
    first = TranscriptEvidence(stream_key="task-a", revision_key="sentence-1")
    second = TranscriptEvidence(stream_key="task-a", revision_key="sentence-2")
    reconciler.observe(
        _receipt("turn-a", 1, 0, first), text="再说一次", is_final=False
    )
    reconciler.observe(
        _receipt("turn-a", 2, 1, second), text="再说一次", is_final=False
    )

    assert reconciler.observe(
        _receipt("turn-a", 2, 1, second), text="再说一次。", is_final=True
    ) == ()
