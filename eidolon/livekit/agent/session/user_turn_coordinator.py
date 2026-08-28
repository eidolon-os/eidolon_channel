"""Assemble transcripts and own Eidolon's product-level turn terminal state.

LiveKit owns transport and automatic endpointing. VAD start/stop events only
open speech segments and record acoustic boundaries. The framework-completed
hook is the sole normal path that commits or rejects a product user turn.
"""

from __future__ import annotations

import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Literal

from eidolon.livekit.agent.observability import TurnTimeline

CandidateState = Literal["open", "committed", "rejected"]
DecisionAction = Literal["none", "commit", "reject"]
OwnerKind = Literal[
    "provisional_user_turn",
    "accepted_user_turn",
    "rejected_user_turn",
    "merged_fragment",
]

DEFAULT_TRANSCRIPT_REVISION_MIN_NORMALIZED_CHARS = 4
DEFAULT_SPEECH_MERGE_GRACE_SEC = 0.8
TIMELINE_TEXT_PREVIEW_MAX_CHARS = 120
OWNER_LEDGER_MAX_TRANSITIONS = 16
COMMITTED_TURN_REVISION_REASON = "committed_turn_revision"


@dataclass(frozen=True)
class UserTurnDecision:
    action: DecisionAction
    candidate_id: str | None = None
    transcript: str = ""
    reason: str = ""


@dataclass(frozen=True)
class FrameworkCompletionReadiness:
    ready: bool
    reason: str
    pending_transcript: str = ""


@dataclass(frozen=True)
class OwnerTransition:
    owner: OwnerKind
    event: str
    reason: str
    at: float
    text_preview: str = ""
    segment_index: int | None = None

    def snapshot(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "owner": self.owner,
            "event": self.event,
            "reason": self.reason,
            "at": self.at,
            "text_preview": self.text_preview,
        }
        if self.segment_index is not None:
            payload["segment_index"] = self.segment_index
        return payload


@dataclass
class TranscriptRevision:
    text: str
    is_final: bool
    received_at: float
    segment_index: int
    generation_id: int


@dataclass
class SpeechSegment:
    started_at: float
    generation_id: int
    ended_at: float | None = None
    text: str = ""
    final_text: str = ""

    @property
    def selected_text(self) -> str:
        return (self.final_text or self.text).strip()


@dataclass
class UserTurnCandidate:
    candidate_id: str
    timeline: TurnTimeline | None
    state: CandidateState
    created_at: float
    updated_at: float
    segments: list[SpeechSegment] = field(default_factory=list)
    revisions: list[TranscriptRevision] = field(default_factory=list)
    eot_score: float | None = None
    voiceprint_reason: str = ""
    commit_reason: str = ""
    reject_reason: str = ""
    committed_at: float | None = None
    owner_transitions: list[OwnerTransition] = field(default_factory=list)
    latest_generation_id: int = 0

    @property
    def selected_text(self) -> str:
        text = ""
        for segment in self.segments:
            text = _merge_text(text, segment.selected_text)
        return text.strip()

    @property
    def last_speech_ended_at(self) -> float | None:
        for segment in reversed(self.segments):
            if segment.ended_at is not None:
                return segment.ended_at
        return None

    @property
    def current_segment(self) -> SpeechSegment | None:
        return self.segments[-1] if self.segments else None


class UserTurnCoordinator:
    """Side-effect-light transcript assembler and terminal-state authority."""

    def __init__(
        self,
        *,
        transcript_revision_min_normalized_chars: int = (
            DEFAULT_TRANSCRIPT_REVISION_MIN_NORMALIZED_CHARS
        ),
        speech_merge_grace_sec: float = DEFAULT_SPEECH_MERGE_GRACE_SEC,
        clock: Any | None = None,
    ) -> None:
        self._transcript_revision_min_normalized_chars = max(
            1, int(transcript_revision_min_normalized_chars)
        )
        self._clock = clock or time.monotonic
        self._speech_merge_grace_sec = max(0.0, float(speech_merge_grace_sec))
        self._active: UserTurnCandidate | None = None
        self._counter = 0
        self._acoustic_generation_counter = 0

    @property
    def active(self) -> UserTurnCandidate | None:
        return self._active

    @property
    def selected_text(self) -> str:
        return self._active.selected_text if self._active is not None else ""

    @property
    def current_generation_id(self) -> int | None:
        candidate = self._active
        segment = candidate.current_segment if candidate is not None else None
        return segment.generation_id if segment is not None else None

    def can_merge_new_speech(self, *, now: float | None = None) -> bool:
        """Merge only an acoustically continuous fragment into the open turn."""

        candidate = self._active
        if candidate is None or candidate.state != "open":
            return False
        segment = candidate.current_segment
        if segment is None:
            return False
        if segment.ended_at is None:
            return True
        return (self._now(now) - segment.ended_at) <= self._speech_merge_grace_sec

    def start_speech(
        self,
        *,
        timeline: TurnTimeline | None,
        now: float | None = None,
    ) -> UserTurnCandidate:
        current_time = self._now(now)
        if self.can_merge_new_speech(now=current_time):
            candidate = self._require_active()
            current_segment = candidate.current_segment
            if current_segment is not None and current_segment.ended_at is None:
                self._record_owner_transition(
                    candidate,
                    owner="provisional_user_turn",
                    event="duplicate_speech_started",
                    reason="candidate_already_open",
                    now=current_time,
                    segment_index=len(candidate.segments) - 1,
                )
                self._record_attrs(candidate, event="duplicate_speech_started")
                return candidate
            candidate.updated_at = current_time
            candidate.latest_generation_id = self._next_generation_id()
            candidate.segments.append(
                SpeechSegment(
                    started_at=current_time,
                    generation_id=candidate.latest_generation_id,
                )
            )
            self._record_owner_transition(
                candidate,
                owner="merged_fragment",
                event="speech_continued",
                reason="awaiting_framework_completed_turn",
                now=current_time,
                segment_index=len(candidate.segments) - 1,
            )
            self._record_attrs(candidate, event="speech_continued")
            return candidate

        previous = self._active
        if previous is not None and previous.state == "open":
            self._reject_candidate(
                previous,
                reason="superseded_by_new_speech",
                now=current_time,
            )
        candidate = self._new_candidate(
            timeline=timeline,
            state="open",
            now=current_time,
            with_initial_segment=True,
        )
        self._active = candidate
        self._record_owner_transition(
            candidate,
            owner="provisional_user_turn",
            event="speech_started",
            reason="new_candidate",
            now=current_time,
            segment_index=0,
        )
        self._record_attrs(candidate, event="speech_started")
        return candidate

    def add_transcript(
        self,
        text: str,
        *,
        is_final: bool,
        now: float | None = None,
    ) -> None:
        candidate = self._active
        if candidate is None or candidate.state != "open":
            return
        stripped = text.strip()
        if not stripped:
            return
        current_time = self._now(now)
        segment_index, segment = self._select_segment_for_revision(
            candidate,
            stripped,
            is_final=is_final,
            now=current_time,
        )
        if segment is None:
            if candidate.latest_generation_id == 0:
                candidate.latest_generation_id = self._next_generation_id()
            segment = SpeechSegment(
                started_at=current_time,
                generation_id=candidate.latest_generation_id,
            )
            candidate.segments.append(segment)
            segment_index = len(candidate.segments) - 1
        segment.text = stripped
        if is_final:
            segment.final_text = stripped
            self._finalize_equivalent_pending_segments(
                candidate,
                stripped,
                selected_index=segment_index,
            )
        candidate.revisions.append(
            TranscriptRevision(
                text=stripped,
                is_final=is_final,
                received_at=current_time,
                segment_index=segment_index,
                generation_id=segment.generation_id,
            )
        )
        candidate.updated_at = current_time
        self._record_attrs(
            candidate,
            event="transcript_final" if is_final else "transcript_interim",
        )

    def note_speech_stopped(
        self,
        *,
        eot_score: float | None,
        now: float | None = None,
    ) -> None:
        """Record an acoustic boundary without making a product decision."""

        candidate = self._active
        if candidate is None or candidate.state != "open":
            return
        current_time = self._now(now)
        segment = candidate.current_segment
        if segment is not None:
            segment.ended_at = current_time
        candidate.eot_score = eot_score
        candidate.updated_at = current_time
        self._record_owner_transition(
            candidate,
            owner="provisional_user_turn",
            event="speech_stopped",
            reason="awaiting_framework_completed_turn",
            now=current_time,
            transcript=candidate.selected_text,
        )
        self._record_attrs(candidate, event="speech_stopped")

    def absorb_committed_transcript_revision(
        self,
        text: str,
        *,
        is_final: bool,
        reason: str = COMMITTED_TURN_REVISION_REASON,
        require_framework_completed: bool = False,
        now: float | None = None,
    ) -> bool:
        if not is_final:
            return False
        candidate = self._active
        stripped = text.strip()
        if (
            candidate is None
            or candidate.state != "committed"
            or not stripped
            or not _normalized_text_equal(candidate.selected_text, stripped)
        ):
            return False
        if require_framework_completed and not self._has_framework_completed_owner(candidate):
            return False

        current_time = self._now(now)
        self._replace_committed_text(candidate, stripped, now=current_time)
        candidate.updated_at = current_time
        self._record_owner_transition(
            candidate,
            owner="accepted_user_turn",
            event="committed_revision",
            reason=reason,
            now=current_time,
            transcript=stripped,
        )
        self._record_attrs(candidate, event="committed_revision", transcript=stripped)
        return True

    def mark_framework_completed(
        self,
        *,
        transcript: str,
        reason: str,
        timeline: TurnTimeline | None = None,
        voiceprint_reason: str = "",
        now: float | None = None,
    ) -> UserTurnDecision:
        current_time = self._now(now)
        stripped = transcript.strip()
        candidate = self._active
        if self.absorb_committed_transcript_revision(
            stripped,
            is_final=True,
            reason=COMMITTED_TURN_REVISION_REASON,
            require_framework_completed=True,
            now=current_time,
        ):
            candidate = self._require_active()
            return UserTurnDecision(
                action="none",
                candidate_id=candidate.candidate_id,
                transcript=candidate.selected_text,
                reason=COMMITTED_TURN_REVISION_REASON,
            )

        if (
            candidate is not None
            and self._is_terminal(candidate)
            and timeline is not None
            and candidate.timeline is not None
            and candidate.timeline.turn_id == timeline.turn_id
        ):
            return UserTurnDecision(
                action="none",
                candidate_id=candidate.candidate_id,
                transcript=candidate.selected_text,
                reason=f"framework_completed_duplicate:{candidate.state}",
            )

        canonical = self.prepare_framework_completed(
            transcript=stripped,
            timeline=timeline,
            now=current_time,
        )
        candidate = self._require_active()
        if not canonical:
            return self._reject_candidate(
                candidate,
                reason="empty_transcript",
                now=current_time,
            )
        candidate.state = "committed"
        candidate.commit_reason = reason
        candidate.voiceprint_reason = voiceprint_reason
        candidate.committed_at = current_time
        candidate.updated_at = current_time
        self._record_owner_transition(
            candidate,
            owner="accepted_user_turn",
            event="framework_completed",
            reason=reason,
            now=current_time,
            transcript=canonical,
        )
        self._record_attrs(candidate, event="framework_completed", transcript=canonical)
        return UserTurnDecision(
            action="commit",
            candidate_id=candidate.candidate_id,
            transcript=canonical,
            reason=reason,
        )

    def prepare_framework_completed(
        self,
        *,
        transcript: str,
        timeline: TurnTimeline | None = None,
        now: float | None = None,
    ) -> str:
        """Assemble the canonical candidate without making it terminal.

        Framework completion may carry an older FINAL while later accepted STT
        revisions already belong to the same product turn.  Admission policy
        must inspect the assembled candidate before deciding whether the sole
        framework terminal boundary may commit it.
        """

        current_time = self._now(now)
        stripped = transcript.strip()
        candidate = self._active
        if candidate is not None and self._is_terminal(candidate):
            if (
                timeline is None
                or candidate.timeline is None
                or candidate.timeline.turn_id == timeline.turn_id
            ):
                return candidate.selected_text or stripped
            candidate = None

        if candidate is None:
            candidate = self._new_candidate(
                timeline=timeline,
                state="open",
                now=current_time,
                with_initial_segment=False,
            )
            self._active = candidate
        elif candidate.timeline is None and timeline is not None:
            candidate.timeline = timeline

        if stripped:
            self._merge_framework_transcript(candidate, stripped, now=current_time)
        canonical = candidate.selected_text or stripped
        candidate.updated_at = current_time
        self._record_attrs(
            candidate,
            event="framework_completed_prepared",
            transcript=canonical,
        )
        return canonical

    def framework_completion_readiness(
        self,
        transcript: str,
    ) -> FrameworkCompletionReadiness:
        """Require a framework boundary to cover every later STT revision.

        A provider may emit ``FINAL(A) -> INTERIM(B)`` before LiveKit delivers
        the completion for A.  Starting generation for A would split one
        physical utterance and let the later completion cancel that generation.
        The framework boundary is ready only when its own transcript covers the
        trailing interim.  An interim-only candidate remains admissible because
        there is no stale earlier FINAL to supersede.
        """

        candidate = self._active
        if candidate is None or candidate.state != "open":
            return FrameworkCompletionReadiness(True, "no_open_candidate")

        has_final = any(bool(segment.final_text.strip()) for segment in candidate.segments)
        if not has_final:
            return FrameworkCompletionReadiness(True, "interim_only_candidate")

        framework_text = transcript.strip()
        for segment in candidate.segments:
            pending = segment.text.strip() if not segment.final_text.strip() else ""
            if not pending:
                continue
            if framework_text and self._text_matches_revision(pending, framework_text):
                continue
            return FrameworkCompletionReadiness(
                False,
                "framework_completion_precedes_pending_interim",
                pending_transcript=pending,
            )
        return FrameworkCompletionReadiness(True, "candidate_segments_covered")

    def framework_completion_generation(self, transcript: str) -> int | None:
        """Resolve the acoustic generation that produced a framework completion.

        Provider callbacks can arrive after a later VAD segment has begun.  Match
        the completed transcript against recorded revisions instead of assuming
        the candidate's current segment owns it.
        """

        candidate = self._active
        if candidate is None:
            return None
        text = transcript.strip()
        if text:
            for final_only in (True, False):
                for revision in reversed(candidate.revisions):
                    if final_only and not revision.is_final:
                        continue
                    if self._text_matches_revision(revision.text, text):
                        return revision.generation_id
        segment = candidate.current_segment
        return segment.generation_id if segment is not None else None

    def reject_active(
        self,
        reason: str,
        *,
        now: float | None = None,
    ) -> UserTurnDecision:
        candidate = self._active
        if candidate is None:
            return UserTurnDecision(action="reject", reason=reason)
        if self._is_terminal(candidate):
            return UserTurnDecision(
                action="none",
                candidate_id=candidate.candidate_id,
                transcript=candidate.selected_text,
                reason=f"already_{candidate.state}",
            )
        return self._reject_candidate(candidate, reason=reason, now=self._now(now))

    def reset(self) -> None:
        self._active = None

    def snapshot(self) -> dict[str, Any]:
        return {"state": "idle"} if self._active is None else self._snapshot(self._active)

    def _require_active(self) -> UserTurnCandidate:
        if self._active is None:
            raise RuntimeError("user turn candidate missing")
        return self._active

    def _now(self, value: float | None) -> float:
        return self._clock() if value is None else value

    @staticmethod
    def _is_terminal(candidate: UserTurnCandidate) -> bool:
        return candidate.state in {"committed", "rejected"}

    def _new_candidate(
        self,
        *,
        timeline: TurnTimeline | None,
        state: CandidateState,
        now: float,
        with_initial_segment: bool,
    ) -> UserTurnCandidate:
        self._counter += 1
        initial_generation = self._next_generation_id() if with_initial_segment else 0
        return UserTurnCandidate(
            candidate_id=(
                timeline.turn_id if timeline is not None else f"user-turn-{self._counter}"
            ),
            timeline=timeline,
            state=state,
            created_at=now,
            updated_at=now,
            segments=(
                [SpeechSegment(started_at=now, generation_id=initial_generation)]
                if with_initial_segment
                else []
            ),
            latest_generation_id=initial_generation,
        )

    def _select_segment_for_revision(
        self,
        candidate: UserTurnCandidate,
        text: str,
        *,
        is_final: bool,
        now: float,
    ) -> tuple[int, SpeechSegment | None]:
        current = candidate.current_segment
        if current is None:
            return -1, None
        current_index = len(candidate.segments) - 1
        if not is_final:
            # FINAL closes one provider revision stream. A later INTERIM is a
            # new sentence even if VAD kept both in one acoustic interval.
            return (-1, None) if current.final_text else (current_index, current)

        current_text = current.selected_text
        if current_text and self._text_matches_revision(current_text, text):
            return current_index, current
        for index in range(len(candidate.segments) - 2, -1, -1):
            segment = candidate.segments[index]
            if self._text_matches_revision(segment.selected_text, text):
                return index, segment
        if not current_text:
            return current_index, current
        return (-1, None) if current.final_text else (current_index, current)

    def _merge_framework_transcript(
        self,
        candidate: UserTurnCandidate,
        transcript: str,
        *,
        now: float,
    ) -> None:
        selected = candidate.selected_text
        if selected and selected in transcript:
            candidate.segments = [
                SpeechSegment(
                    started_at=candidate.created_at,
                    generation_id=self._framework_segment_generation(candidate),
                    ended_at=now,
                    text=transcript,
                    final_text=transcript,
                )
            ]
            return
        if selected and transcript in selected:
            return
        if selected and self._text_matches_revision(selected, transcript):
            return
        candidate.segments.append(
            SpeechSegment(
                started_at=now,
                generation_id=self._framework_segment_generation(candidate),
                ended_at=now,
                text=transcript,
                final_text=transcript,
            )
        )

    def _replace_committed_text(
        self,
        candidate: UserTurnCandidate,
        transcript: str,
        *,
        now: float,
    ) -> None:
        candidate.segments = [
            SpeechSegment(
                started_at=candidate.created_at,
                generation_id=self._framework_segment_generation(candidate),
                ended_at=now,
                text=transcript,
                final_text=transcript,
            )
        ]

    def _finalize_equivalent_pending_segments(
        self,
        candidate: UserTurnCandidate,
        transcript: str,
        *,
        selected_index: int,
    ) -> None:
        """Close duplicate interim aliases created across a VAD boundary.

        Some STT providers repeat the last interim in the next acoustic segment
        before emitting its FINAL.  Leaving the alias interim-only makes the
        framework gate wait forever for a final that will never exist.  Marking
        equivalent aliases final preserves one canonical text via ``_merge_text``
        while closing every revision stream covered by the provider FINAL.
        """

        for index, segment in enumerate(candidate.segments):
            if index == selected_index or segment.final_text.strip():
                continue
            if self._text_matches_revision(segment.selected_text, transcript):
                segment.final_text = transcript

    def _framework_segment_generation(self, candidate: UserTurnCandidate) -> int:
        if candidate.latest_generation_id == 0:
            candidate.latest_generation_id = self._next_generation_id()
        return candidate.latest_generation_id

    def _next_generation_id(self) -> int:
        self._acoustic_generation_counter += 1
        return self._acoustic_generation_counter

    @staticmethod
    def _has_framework_completed_owner(candidate: UserTurnCandidate) -> bool:
        return candidate.commit_reason == "framework_completed_turn" or any(
            transition.event == "framework_completed" for transition in candidate.owner_transitions
        )

    def _reject_candidate(
        self,
        candidate: UserTurnCandidate,
        *,
        reason: str,
        now: float,
    ) -> UserTurnDecision:
        candidate.state = "rejected"
        candidate.reject_reason = reason
        candidate.updated_at = now
        self._record_owner_transition(
            candidate,
            owner="rejected_user_turn",
            event="rejected",
            reason=reason,
            now=now,
            transcript=candidate.selected_text,
        )
        self._record_attrs(candidate, event="rejected")
        return UserTurnDecision(
            action="reject",
            candidate_id=candidate.candidate_id,
            transcript=candidate.selected_text,
            reason=reason,
        )

    def _record_owner_transition(
        self,
        candidate: UserTurnCandidate,
        *,
        owner: OwnerKind,
        event: str,
        reason: str,
        now: float,
        transcript: str | None = None,
        segment_index: int | None = None,
    ) -> None:
        text = transcript if transcript is not None else candidate.selected_text
        transition = OwnerTransition(
            owner=owner,
            event=event,
            reason=reason,
            at=now,
            text_preview=text[:TIMELINE_TEXT_PREVIEW_MAX_CHARS],
            segment_index=segment_index,
        )
        candidate.owner_transitions.append(transition)
        if candidate.timeline is None:
            return
        candidate.timeline.set_attr(
            "user_turn_owner_ledger",
            {
                "candidate_id": candidate.candidate_id,
                "last": transition.snapshot(),
                "transitions": [
                    entry.snapshot()
                    for entry in candidate.owner_transitions[-OWNER_LEDGER_MAX_TRANSITIONS:]
                ],
            },
        )

    def _record_attrs(
        self,
        candidate: UserTurnCandidate,
        *,
        event: str,
        transcript: str | None = None,
    ) -> None:
        if candidate.timeline is None:
            return
        payload = self._snapshot(candidate)
        payload["event"] = event
        if transcript is not None:
            payload["committed_transcript_preview"] = transcript[:TIMELINE_TEXT_PREVIEW_MAX_CHARS]
        candidate.timeline.set_attr("user_turn_coordinator", payload)

    @staticmethod
    def _snapshot(candidate: UserTurnCandidate) -> dict[str, Any]:
        return {
            "candidate_id": candidate.candidate_id,
            "state": candidate.state,
            "segments": len(candidate.segments),
            "revisions": len(candidate.revisions),
            "acoustic_generation": (
                candidate.current_segment.generation_id
                if candidate.current_segment is not None
                else None
            ),
            "selected_text_preview": candidate.selected_text[:TIMELINE_TEXT_PREVIEW_MAX_CHARS],
            "eot_score": candidate.eot_score,
            "voiceprint_reason": candidate.voiceprint_reason,
            "commit_reason": candidate.commit_reason,
            "reject_reason": candidate.reject_reason,
            "owner_transition_count": len(candidate.owner_transitions),
            "last_owner_transition": (
                candidate.owner_transitions[-1].snapshot() if candidate.owner_transitions else None
            ),
        }

    def _text_matches_revision(self, existing: str, revision: str) -> bool:
        return _text_matches_revision(
            existing,
            revision,
            min_normalized_chars=self._transcript_revision_min_normalized_chars,
        )


def _merge_text(left: str, right: str) -> str:
    left = left.strip()
    right = right.strip()
    if not left:
        return right
    if not right:
        return left
    if right.startswith(left):
        return right
    if left.endswith(right):
        return left
    if _looks_cjk(left[-1]) or _looks_cjk(right[0]):
        return f"{left}{right}"
    return f"{left} {right}"


def _looks_cjk(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


def _text_matches_revision(
    existing: str,
    revision: str,
    *,
    min_normalized_chars: int = DEFAULT_TRANSCRIPT_REVISION_MIN_NORMALIZED_CHARS,
) -> bool:
    existing_norm = _normalize_revision_text(existing)
    revision_norm = _normalize_revision_text(revision)
    if not existing_norm or not revision_norm:
        return False
    if existing_norm == revision_norm:
        return True
    if existing_norm.startswith(revision_norm) or revision_norm.startswith(existing_norm):
        return True
    if min(len(existing_norm), len(revision_norm)) < max(1, min_normalized_chars):
        return False
    shorter, longer = (
        (existing_norm, revision_norm)
        if len(existing_norm) <= len(revision_norm)
        else (revision_norm, existing_norm)
    )
    return shorter in longer


def _normalized_text_equal(left: str, right: str) -> bool:
    left_norm = _normalize_revision_text(left)
    right_norm = _normalize_revision_text(right)
    return bool(left_norm and right_norm and left_norm == right_norm)


def _normalize_revision_text(text: str) -> str:
    return "".join(
        char
        for char in text.strip().lower()
        if not unicodedata.category(char).startswith(("P", "Z"))
    )
