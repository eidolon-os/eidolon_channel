"""Coordinate Eidolon's product-level user turn lifecycle.

This module deliberately does not call LiveKit APIs.  It accepts framework
events and returns decisions that ``StreamingPipeline`` can apply at the
framework boundary.  Keeping it side-effect-light makes the hardest realtime
turn rules unit-testable: transcript revision selection, short-pause merging,
voiceprint gating, duplicate prevention, and explicit reject reasons.
"""

from __future__ import annotations

import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Literal

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.session.voiceprint_reasons import (
    VOICEPRINT_INCONCLUSIVE_PREFIX,
    voiceprint_allowed_reason,
    voiceprint_blocked_reason,
)

CandidateState = Literal[
    "idle",
    "open",
    "waiting_merge",
    "waiting_voiceprint",
    "ready_to_commit",
    "committed",
    "rejected",
]
DecisionAction = Literal["none", "defer", "commit", "reject"]
OwnerKind = Literal[
    "provisional_user_turn",
    "accepted_user_turn",
    "rejected_user_turn",
    "merged_fragment",
    "dropped_fragment",
]

DEFAULT_MERGE_GRACE_SEC = 0.8
DEFAULT_STATEMENT_DEFERRED_MERGE_GRACE_SEC = 3.5
DEFAULT_VOICEPRINT_DEFERRED_MERGE_GRACE_SEC = 4.0
DEFAULT_LOW_EOT_DELAY_SEC = 0.8
DEFAULT_STATEMENT_SEQUENCE_MERGE_MAX_CJK_CHARS = 28
DEFAULT_STATEMENT_SEQUENCE_FRAGMENT_MAX_CJK_CHARS = 14
DEFAULT_TRANSCRIPT_REVISION_MIN_NORMALIZED_CHARS = 4
TIMELINE_TEXT_PREVIEW_MAX_CHARS = 120
OWNER_LEDGER_MAX_TRANSITIONS = 16
NON_ACTIONABLE_META_TURN_REASON = "non_actionable_meta_turn"
_NON_ACTIONABLE_META_TURN_PREFIX_SUFFIXES = {
    "那我再说": frozenset(("", "了", "一下", "一遍", "吧")),
    "我再说": frozenset(("", "了", "一下", "一遍", "吧")),
    "我重新说": frozenset(("", "一下", "一遍")),
    "我重说": frozenset(("", "一下", "一遍")),
    "等我再说": frozenset(("", "吧")),
    "等下我再说": frozenset(("", "吧")),
}


@dataclass(frozen=True)
class UserTurnDecision:
    """Decision emitted by :class:`UserTurnCoordinator`."""

    action: DecisionAction
    candidate_id: str | None = None
    transcript: str = ""
    reason: str = ""
    delay_sec: float = 0.0


@dataclass(frozen=True)
class OwnerTransition:
    """A side-effect-free record of which owner currently owns a candidate."""

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


@dataclass
class SpeechSegment:
    started_at: float
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
    merge_reason: str = ""
    voiceprint_reason: str = ""
    commit_reason: str = ""
    reject_reason: str = ""
    committed_at: float | None = None
    owner_transitions: list[OwnerTransition] = field(default_factory=list)

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
        if not self.segments:
            return None
        return self.segments[-1]


class UserTurnCoordinator:
    """Own the product-level user turn state machine.

    LiveKit still owns transport, endpointing, and ``commit_user_turn``.  This
    coordinator owns Eidolon's business decision: which transcript candidate is
    current, whether short pauses should merge, whether voiceprint permits the
    turn, and whether the candidate has already reached a terminal state.
    """

    def __init__(
        self,
        *,
        merge_grace_sec: float = DEFAULT_MERGE_GRACE_SEC,
        statement_deferred_merge_grace_sec: float = (
            DEFAULT_STATEMENT_DEFERRED_MERGE_GRACE_SEC
        ),
        voiceprint_deferred_merge_grace_sec: float = (
            DEFAULT_VOICEPRINT_DEFERRED_MERGE_GRACE_SEC
        ),
        low_eot_delay_sec: float = DEFAULT_LOW_EOT_DELAY_SEC,
        statement_sequence_merge_max_cjk_chars: int = (
            DEFAULT_STATEMENT_SEQUENCE_MERGE_MAX_CJK_CHARS
        ),
        statement_sequence_fragment_max_cjk_chars: int = (
            DEFAULT_STATEMENT_SEQUENCE_FRAGMENT_MAX_CJK_CHARS
        ),
        transcript_revision_min_normalized_chars: int = (
            DEFAULT_TRANSCRIPT_REVISION_MIN_NORMALIZED_CHARS
        ),
        clock: Any | None = None,
    ) -> None:
        self._merge_grace_sec = max(0.0, float(merge_grace_sec))
        self._statement_deferred_merge_grace_sec = max(
            self._merge_grace_sec,
            float(statement_deferred_merge_grace_sec),
        )
        self._voiceprint_deferred_merge_grace_sec = max(
            self._merge_grace_sec,
            float(voiceprint_deferred_merge_grace_sec),
        )
        self._low_eot_delay_sec = max(0.0, float(low_eot_delay_sec))
        self._statement_sequence_merge_max_cjk_chars = max(
            1,
            int(statement_sequence_merge_max_cjk_chars),
        )
        self._statement_sequence_fragment_max_cjk_chars = max(
            1,
            int(statement_sequence_fragment_max_cjk_chars),
        )
        self._transcript_revision_min_normalized_chars = max(
            1,
            int(transcript_revision_min_normalized_chars),
        )
        self._clock = clock or time.monotonic
        self._active: UserTurnCandidate | None = None
        self._counter = 0

    @property
    def active(self) -> UserTurnCandidate | None:
        return self._active

    @property
    def selected_text(self) -> str:
        candidate = self._active
        return candidate.selected_text if candidate is not None else ""

    def can_merge_new_speech(self, *, now: float | None = None) -> bool:
        candidate = self._active
        if candidate is None:
            return False
        if candidate.state not in {"waiting_merge", "waiting_voiceprint", "open"}:
            return False
        ended_at = candidate.last_speech_ended_at
        if ended_at is None:
            return False
        return (self._now(now) - ended_at) <= self._merge_window_sec(candidate)

    def merge_remaining_sec(self, *, now: float | None = None) -> float:
        candidate = self._active
        if candidate is None or candidate.state != "waiting_merge":
            return 0.0
        ended_at = candidate.last_speech_ended_at
        if ended_at is None:
            return 0.0
        elapsed = self._now(now) - ended_at
        return max(0.0, self._merge_window_sec(candidate) - elapsed)

    def should_wait_for_deferred_voiceprint_merge(
        self,
        *,
        now: float | None = None,
    ) -> bool:
        candidate = self._active
        if candidate is None:
            return False
        if not candidate.voiceprint_reason.startswith(
            f"{VOICEPRINT_INCONCLUSIVE_PREFIX}:"
        ):
            return False
        return self.merge_remaining_sec(now=now) > 0.0

    def should_wait_for_statement_sequence_merge(
        self,
        *,
        now: float | None = None,
    ) -> bool:
        candidate = self._active
        if candidate is None:
            return False
        if not self._uses_statement_sequence_merge_window(candidate):
            return False
        return self.merge_remaining_sec(now=now) > 0.0

    def start_speech(
        self,
        *,
        timeline: TurnTimeline | None,
        now: float | None = None,
    ) -> UserTurnCandidate:
        current_time = self._now(now)
        if self.can_merge_new_speech(now=current_time):
            candidate = self._require_active()
            candidate.state = "open"
            candidate.updated_at = current_time
            candidate.segments.append(SpeechSegment(started_at=current_time))
            self._record_owner_transition(
                candidate,
                owner="merged_fragment",
                event="speech_continued",
                reason="merge_window",
                now=current_time,
                segment_index=len(candidate.segments) - 1,
            )
            self._record_attrs(candidate, event="speech_continued")
            return candidate

        self._counter += 1
        candidate_id = timeline.turn_id if timeline is not None else f"user-turn-{self._counter}"
        candidate = UserTurnCandidate(
            candidate_id=candidate_id,
            timeline=timeline,
            state="open",
            created_at=current_time,
            updated_at=current_time,
            segments=[SpeechSegment(started_at=current_time)],
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
        if candidate is None or candidate.state in {"committed", "rejected"}:
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
            segment = SpeechSegment(started_at=current_time)
            candidate.segments.append(segment)
            segment_index = len(candidate.segments) - 1
        segment.text = stripped
        if is_final:
            segment.final_text = stripped
        candidate.revisions.append(
            TranscriptRevision(
                text=stripped,
                is_final=is_final,
                received_at=current_time,
                segment_index=segment_index,
            )
        )
        candidate.updated_at = current_time
        self._record_attrs(
            candidate, event="transcript_final" if is_final else "transcript_interim"
        )

    def finish_speech(
        self,
        *,
        eot_score: float | None,
        should_defer: bool,
        now: float | None = None,
    ) -> UserTurnDecision:
        candidate = self._active
        if candidate is None:
            return UserTurnDecision(action="reject", reason="no_active_candidate")
        current_time = self._now(now)
        segment = candidate.current_segment
        if segment is not None:
            segment.ended_at = current_time
        candidate.eot_score = eot_score
        candidate.updated_at = current_time
        self._drop_non_actionable_meta_tail_if_needed(
            candidate,
            now=current_time,
        )
        transcript = candidate.selected_text
        if not transcript:
            candidate.state = "rejected"
            candidate.reject_reason = "empty_transcript"
            self._record_owner_transition(
                candidate,
                owner="rejected_user_turn",
                event="rejected",
                reason="empty_transcript",
                now=current_time,
            )
            self._record_attrs(candidate, event="rejected")
            return UserTurnDecision(
                action="reject",
                candidate_id=candidate.candidate_id,
                reason="empty_transcript",
            )
        meta_turn_decision = self._reject_non_actionable_meta_turn_if_needed(
            candidate,
            transcript=transcript,
            now=current_time,
        )
        if meta_turn_decision is not None:
            return meta_turn_decision
        if should_defer:
            candidate.state = "waiting_merge"
            candidate.merge_reason = "low_eot_wait_for_continuation"
            self._record_owner_transition(
                candidate,
                owner="provisional_user_turn",
                event="deferred_low_eot",
                reason="low_eot_wait_for_continuation",
                now=current_time,
                transcript=transcript,
            )
            self._record_attrs(candidate, event="deferred_low_eot")
            return UserTurnDecision(
                action="defer",
                candidate_id=candidate.candidate_id,
                transcript=transcript,
                reason="low_eot_wait_for_continuation",
                delay_sec=self._low_eot_delay_sec,
            )
        candidate.state = "waiting_voiceprint"
        self._record_owner_transition(
            candidate,
            owner="provisional_user_turn",
            event="waiting_voiceprint",
            reason="speech_finished",
            now=current_time,
            transcript=transcript,
        )
        self._record_attrs(candidate, event="waiting_voiceprint")
        return UserTurnDecision(
            action="commit",
            candidate_id=candidate.candidate_id,
            transcript=transcript,
            reason="speech_finished",
        )

    def deferred_ready(self, *, now: float | None = None) -> UserTurnDecision:
        candidate = self._active
        if candidate is None:
            return UserTurnDecision(action="reject", reason="no_active_candidate")
        if candidate.state != "waiting_merge":
            return UserTurnDecision(
                action="none",
                candidate_id=candidate.candidate_id,
                transcript=candidate.selected_text,
                reason=f"not_waiting_merge:{candidate.state}",
            )
        candidate.state = "waiting_voiceprint"
        candidate.updated_at = self._now(now)
        self._record_owner_transition(
            candidate,
            owner="provisional_user_turn",
            event="deferred_ready",
            reason="low_eot_grace_elapsed",
            now=candidate.updated_at,
            transcript=candidate.selected_text,
        )
        self._record_attrs(candidate, event="deferred_ready")
        return UserTurnDecision(
            action="commit",
            candidate_id=candidate.candidate_id,
            transcript=candidate.selected_text,
            reason="low_eot_grace_elapsed",
        )

    def apply_voiceprint_result(
        self,
        result: Any,
        *,
        now: float | None = None,
    ) -> UserTurnDecision:
        candidate = self._active
        if candidate is None:
            return UserTurnDecision(action="reject", reason="no_active_candidate")
        allowed = bool(getattr(result, "commit_allowed", False))
        reason = str(getattr(result, "commit_reason", "") or "unknown")
        candidate.voiceprint_reason = reason
        candidate.updated_at = self._now(now)
        if not allowed:
            candidate.state = "rejected"
            candidate.reject_reason = voiceprint_blocked_reason(reason)
            self._record_owner_transition(
                candidate,
                owner="rejected_user_turn",
                event="voiceprint_rejected",
                reason=candidate.reject_reason,
                now=candidate.updated_at,
                transcript=candidate.selected_text,
            )
            self._record_attrs(candidate, event="voiceprint_rejected")
            return UserTurnDecision(
                action="reject",
                candidate_id=candidate.candidate_id,
                transcript=candidate.selected_text,
                reason=candidate.reject_reason,
            )
        candidate.state = "ready_to_commit"
        candidate.commit_reason = voiceprint_allowed_reason(reason)
        self._record_owner_transition(
            candidate,
            owner="accepted_user_turn",
            event="voiceprint_allowed",
            reason=candidate.commit_reason,
            now=candidate.updated_at,
            transcript=candidate.selected_text,
        )
        self._record_attrs(candidate, event="voiceprint_allowed")
        return UserTurnDecision(
            action="commit",
            candidate_id=candidate.candidate_id,
            transcript=candidate.selected_text,
            reason=candidate.commit_reason,
        )

    def mark_committed(
        self,
        *,
        transcript: str,
        reason: str,
        now: float | None = None,
    ) -> None:
        candidate = self._active
        if candidate is None:
            return
        candidate.state = "committed"
        candidate.commit_reason = reason
        candidate.committed_at = self._now(now)
        candidate.updated_at = candidate.committed_at
        self._record_owner_transition(
            candidate,
            owner="accepted_user_turn",
            event="committed",
            reason=reason,
            now=candidate.committed_at,
            transcript=transcript,
        )
        self._record_attrs(candidate, event="committed", transcript=transcript)

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
        if candidate is None or self._is_terminal(candidate):
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
        self._drop_non_actionable_meta_tail_if_needed(candidate, now=current_time)
        canonical = candidate.selected_text or stripped
        meta_turn_decision = self._reject_non_actionable_meta_turn_if_needed(
            candidate,
            transcript=canonical,
            now=current_time,
        )
        if meta_turn_decision is not None:
            return meta_turn_decision
        candidate.state = "committed"
        candidate.commit_reason = reason
        if voiceprint_reason:
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

    def defer_framework_completed(
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
        if candidate is None or self._is_terminal(candidate):
            candidate = self._new_candidate(
                timeline=timeline,
                state="open",
                now=current_time,
                with_initial_segment=False,
            )
            self._active = candidate
        elif candidate.timeline is None and timeline is not None:
            candidate.timeline = timeline

        preserve_statement_window = self._uses_statement_sequence_merge_window(candidate)
        delay_sec = self.merge_remaining_sec(now=current_time)
        if stripped:
            selected = candidate.selected_text
            if not (
                preserve_statement_window
                and selected
                and _normalize_revision_text(selected) == _normalize_revision_text(stripped)
            ):
                self._merge_framework_transcript(candidate, stripped, now=current_time)
        self._drop_non_actionable_meta_tail_if_needed(candidate, now=current_time)
        canonical = candidate.selected_text or stripped
        meta_turn_decision = self._reject_non_actionable_meta_turn_if_needed(
            candidate,
            transcript=canonical,
            now=current_time,
        )
        if meta_turn_decision is not None:
            return meta_turn_decision
        candidate.state = "waiting_merge"
        if not candidate.merge_reason:
            candidate.merge_reason = reason
        if voiceprint_reason and not candidate.voiceprint_reason.startswith(
            f"{VOICEPRINT_INCONCLUSIVE_PREFIX}:"
        ):
            candidate.voiceprint_reason = voiceprint_reason
        candidate.updated_at = current_time
        self._record_owner_transition(
            candidate,
            owner="provisional_user_turn",
            event="framework_completed_deferred",
            reason=reason,
            now=current_time,
            transcript=canonical,
        )
        if delay_sec <= 0.0:
            delay_sec = self.merge_remaining_sec(now=current_time)
        if delay_sec <= 0.0:
            delay_sec = self._low_eot_delay_sec
        self._record_attrs(
            candidate,
            event="framework_completed_deferred",
            transcript=canonical,
        )
        return UserTurnDecision(
            action="defer",
            candidate_id=candidate.candidate_id,
            transcript=canonical,
            reason=reason,
            delay_sec=delay_sec,
        )

    def defer_voiceprint_inconclusive(
        self,
        *,
        transcript: str,
        reason: str,
        timeline: TurnTimeline | None = None,
        now: float | None = None,
    ) -> UserTurnDecision:
        current_time = self._now(now)
        stripped = transcript.strip()
        candidate = self._active
        if candidate is None or self._is_terminal(candidate):
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
        candidate.state = "waiting_merge"
        candidate.merge_reason = reason
        candidate.voiceprint_reason = reason
        candidate.updated_at = current_time
        canonical = candidate.selected_text or stripped
        self._record_owner_transition(
            candidate,
            owner="provisional_user_turn",
            event="voiceprint_deferred",
            reason=reason,
            now=current_time,
            transcript=canonical,
        )
        self._record_attrs(
            candidate,
            event="voiceprint_deferred",
            transcript=canonical,
        )
        return UserTurnDecision(
            action="defer",
            candidate_id=candidate.candidate_id,
            transcript=canonical,
            reason=reason,
            delay_sec=self.merge_remaining_sec(now=current_time) or self._low_eot_delay_sec,
        )

    def reject_active(
        self,
        reason: str,
        *,
        now: float | None = None,
    ) -> UserTurnDecision:
        candidate = self._active
        if candidate is None:
            return UserTurnDecision(action="reject", reason=reason)
        candidate.state = "rejected"
        candidate.reject_reason = reason
        candidate.updated_at = self._now(now)
        self._record_owner_transition(
            candidate,
            owner="rejected_user_turn",
            event="rejected",
            reason=reason,
            now=candidate.updated_at,
            transcript=candidate.selected_text,
        )
        self._record_attrs(candidate, event="rejected")
        return UserTurnDecision(
            action="reject",
            candidate_id=candidate.candidate_id,
            transcript=candidate.selected_text,
            reason=reason,
        )

    def reset(self) -> None:
        self._active = None

    def snapshot(self) -> dict[str, Any]:
        candidate = self._active
        if candidate is None:
            return {"state": "idle"}
        return self._snapshot(candidate)

    def _require_active(self) -> UserTurnCandidate:
        if self._active is None:
            raise RuntimeError("user turn candidate missing")
        return self._active

    def _now(self, value: float | None) -> float:
        return self._clock() if value is None else value

    def _is_terminal(self, candidate: UserTurnCandidate) -> bool:
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
        return UserTurnCandidate(
            candidate_id=(
                timeline.turn_id if timeline is not None else f"user-turn-{self._counter}"
            ),
            timeline=timeline,
            state=state,
            created_at=now,
            updated_at=now,
            segments=[SpeechSegment(started_at=now)] if with_initial_segment else [],
        )

    def _merge_window_sec(self, candidate: UserTurnCandidate) -> float:
        if candidate.voiceprint_reason.startswith(
            "voiceprint_inconclusive:"
        ) or candidate.merge_reason.startswith("voiceprint_inconclusive:"):
            return self._voiceprint_deferred_merge_grace_sec
        if self._uses_statement_sequence_merge_window(candidate):
            return self._statement_deferred_merge_grace_sec
        return self._merge_grace_sec

    def _uses_statement_sequence_merge_window(
        self,
        candidate: UserTurnCandidate,
    ) -> bool:
        if candidate.state != "waiting_merge":
            return False
        if candidate.merge_reason not in {
            "low_eot_wait_for_continuation",
            "framework_completed_wait_for_continuation",
        }:
            return False
        selected = candidate.selected_text
        if not selected or any(mark in selected for mark in ("？", "?", "！", "!")):
            return False
        if selected.startswith(("帮我", "请", "麻烦", "换个话题", "换一个话题")):
            return False
        if _count_cjk_chars(selected) > self._statement_sequence_merge_max_cjk_chars:
            return False
        fragments = [
            segment.selected_text for segment in candidate.segments if segment.selected_text
        ]
        if len(fragments) < 2:
            return False
        return all(
            _looks_like_statement_fragment(
                text,
                max_cjk_chars=self._statement_sequence_fragment_max_cjk_chars,
            )
            for text in fragments[-2:]
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
            return current_index, current

        current_text = current.selected_text
        if current_text and self._text_matches_revision(current_text, text):
            return current_index, current

        for index in range(len(candidate.segments) - 2, -1, -1):
            segment = candidate.segments[index]
            if self._text_matches_revision(segment.selected_text, text):
                return index, segment

        if not current_text:
            return current_index, current
        if current.final_text:
            return -1, None
        return current_index, current

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
                ended_at=now,
                text=transcript,
                final_text=transcript,
            )
        )

    def _drop_non_actionable_meta_tail_if_needed(
        self,
        candidate: UserTurnCandidate,
        *,
        now: float,
    ) -> None:
        current = candidate.current_segment
        if current is None or not _looks_like_non_actionable_meta_turn(
            current.selected_text
        ):
            return
        prior_text = ""
        for segment in candidate.segments[:-1]:
            prior_text = _merge_text(prior_text, segment.selected_text)
        if not prior_text.strip():
            return
        current.text = ""
        current.final_text = ""
        current.ended_at = now
        candidate.updated_at = now
        self._record_owner_transition(
            candidate,
            owner="dropped_fragment",
            event="non_actionable_meta_tail_dropped",
            reason=NON_ACTIONABLE_META_TURN_REASON,
            now=now,
            transcript=prior_text.strip(),
            segment_index=len(candidate.segments) - 1,
        )
        self._record_attrs(
            candidate,
            event="non_actionable_meta_tail_dropped",
            transcript=prior_text.strip(),
        )

    def _reject_non_actionable_meta_turn_if_needed(
        self,
        candidate: UserTurnCandidate,
        *,
        transcript: str,
        now: float,
    ) -> UserTurnDecision | None:
        if not _looks_like_non_actionable_meta_turn(transcript):
            return None
        candidate.state = "rejected"
        candidate.reject_reason = NON_ACTIONABLE_META_TURN_REASON
        candidate.updated_at = now
        self._record_owner_transition(
            candidate,
            owner="rejected_user_turn",
            event="rejected",
            reason=NON_ACTIONABLE_META_TURN_REASON,
            now=now,
            transcript=transcript,
        )
        self._record_attrs(candidate, event="rejected", transcript=transcript)
        return UserTurnDecision(
            action="reject",
            candidate_id=candidate.candidate_id,
            transcript=transcript,
            reason=NON_ACTIONABLE_META_TURN_REASON,
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
        timeline = candidate.timeline
        if timeline is None:
            return
        timeline.set_attr(
            "user_turn_owner_ledger",
            {
                "candidate_id": candidate.candidate_id,
                "last": transition.snapshot(),
                "transitions": [
                    entry.snapshot()
                    for entry in candidate.owner_transitions[
                        -OWNER_LEDGER_MAX_TRANSITIONS:
                    ]
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
        timeline = candidate.timeline
        if timeline is None:
            return
        payload = self._snapshot(candidate)
        payload["event"] = event
        if transcript is not None:
            payload["committed_transcript_preview"] = transcript[
                :TIMELINE_TEXT_PREVIEW_MAX_CHARS
            ]
        timeline.set_attr("user_turn_coordinator", payload)

    def _snapshot(self, candidate: UserTurnCandidate) -> dict[str, Any]:
        return {
            "candidate_id": candidate.candidate_id,
            "state": candidate.state,
            "segments": len(candidate.segments),
            "revisions": len(candidate.revisions),
            "selected_text_preview": candidate.selected_text[
                :TIMELINE_TEXT_PREVIEW_MAX_CHARS
            ],
            "eot_score": candidate.eot_score,
            "merge_reason": candidate.merge_reason,
            "voiceprint_reason": candidate.voiceprint_reason,
            "commit_reason": candidate.commit_reason,
            "reject_reason": candidate.reject_reason,
            "owner_transition_count": len(candidate.owner_transitions),
            "last_owner_transition": (
                candidate.owner_transitions[-1].snapshot()
                if candidate.owner_transitions
                else None
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


def _count_cjk_chars(text: str) -> int:
    return sum(1 for char in text if _looks_cjk(char))


def _looks_like_statement_fragment(
    text: str,
    *,
    max_cjk_chars: int = DEFAULT_STATEMENT_SEQUENCE_FRAGMENT_MAX_CJK_CHARS,
) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if any(mark in stripped for mark in ("？", "?", "！", "!")):
        return False
    if _count_cjk_chars(stripped) > max(1, max_cjk_chars):
        return False
    if stripped.startswith(("帮我", "请", "麻烦", "换个话题", "换一个话题")):
        return False
    return stripped.endswith(("。", "，", ",", "、", "的", "了", "呢", "吧"))


def _looks_like_non_actionable_meta_turn(text: str) -> bool:
    normalized = _normalize_revision_text(text)
    if not normalized:
        return False
    for prefix, suffixes in _NON_ACTIONABLE_META_TURN_PREFIX_SUFFIXES.items():
        if not normalized.startswith(prefix):
            continue
        suffix = normalized[len(prefix) :]
        if suffix in suffixes:
            return True
    return False


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


def _normalize_revision_text(text: str) -> str:
    return "".join(
        char
        for char in text.strip().lower()
        if not unicodedata.category(char).startswith(("P", "Z"))
    )
