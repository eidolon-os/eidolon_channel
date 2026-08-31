"""Resolve provider transcript hypotheses at the full-duplex ingress boundary."""

from __future__ import annotations

from dataclasses import dataclass

from eidolon.livekit.agent.session.transcript_revision import transcript_revision_matches
from eidolon.livekit.agent.session.user_turn_coordinator import TranscriptRevisionReceipt
from eidolon.livekit.common.transcript_evidence import TranscriptEvidence


@dataclass
class _HypothesisStream:
    latest_text: str = ""
    interim_seen: bool = False
    final_seen: bool = False
    covered_by_generation_id: int | None = None
    evidence: TranscriptEvidence | None = None


class TranscriptHypothesisReconciler:
    """Translate STT behavior into explicit transcript-segment generation links.

    LiveKit's normalized transcript callback has no provider utterance identifier.
    Some streaming providers repeat the last interim after a VAD boundary and
    emit only one final. This ingress owner performs the unavoidable text
    comparison while the generic turn coordinator receives only explicit
    generation coverage instructions.
    """

    def __init__(
        self,
        *,
        min_normalized_chars: int,
    ) -> None:
        self._min_normalized_chars = max(1, int(min_normalized_chars))
        self._candidate_id: str | None = None
        self._streams: dict[tuple[int, int], _HypothesisStream] = {}

    def observe(
        self,
        receipt: TranscriptRevisionReceipt,
        *,
        text: str,
        is_final: bool,
    ) -> tuple[int, ...]:
        if receipt.candidate_id != self._candidate_id:
            self._candidate_id = receipt.candidate_id
            self._streams.clear()

        stream_key = (receipt.generation_id, receipt.segment_index)
        stream = self._streams.setdefault(stream_key, _HypothesisStream())
        if receipt.evidence is not None:
            stream.evidence = (
                receipt.evidence.merge(stream.evidence)
                if stream.evidence is not None
                else receipt.evidence
            )
        if not is_final:
            stream.latest_text = text.strip()
            stream.interim_seen = True
            return ()

        stream.latest_text = text.strip()
        stream.final_seen = True
        # A repeated phrase spoken twice is not an alias. Require the exact
        # provider pattern observed in production: the final's generation must
        # first repeat an interim from another generation.
        if not stream.interim_seen:
            return ()

        covered_segment_indexes: list[int] = []
        for (generation_id, segment_index), prior in self._streams.items():
            if (
                (generation_id, segment_index) == stream_key
                or prior.final_seen
                or prior.covered_by_generation_id is not None
            ):
                continue
            if self._same_provider_revision(prior, stream) or (
                not self._has_conflicting_provider_identity(prior, stream)
                and transcript_revision_matches(
                    prior.latest_text,
                    stream.latest_text,
                    min_normalized_chars=self._min_normalized_chars,
                )
            ):
                prior.covered_by_generation_id = receipt.generation_id
                covered_segment_indexes.append(segment_index)
        return tuple(covered_segment_indexes)

    @staticmethod
    def _same_provider_revision(left: _HypothesisStream, right: _HypothesisStream) -> bool:
        return bool(
            left.evidence is not None
            and right.evidence is not None
            and left.evidence.same_revision(right.evidence)
        )

    @staticmethod
    def _has_conflicting_provider_identity(
        left: _HypothesisStream,
        right: _HypothesisStream,
    ) -> bool:
        if left.evidence is None or right.evidence is None:
            return False
        if not left.evidence.revision_key or not right.evidence.revision_key:
            return False
        return not left.evidence.same_revision(right.evidence)
