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


@dataclass(frozen=True)
class UserTurnDecision:
    """Decision emitted by :class:`UserTurnCoordinator`."""

    action: DecisionAction
    candidate_id: str | None = None
    transcript: str = ""
    reason: str = ""
    delay_sec: float = 0.0


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
        merge_grace_sec: float = 0.8,
        statement_deferred_merge_grace_sec: float = 3.5,
        voiceprint_deferred_merge_grace_sec: float = 4.0,
        low_eot_delay_sec: float = 0.8,
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
        if not candidate.voiceprint_reason.startswith("voiceprint_inconclusive:"):
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
            self._record_attrs(candidate, event="speech_continued")
            return candidate

        self._counter += 1
        candidate_id = (
            timeline.turn_id
            if timeline is not None
            else f"user-turn-{self._counter}"
        )
        candidate = UserTurnCandidate(
            candidate_id=candidate_id,
            timeline=timeline,
            state="open",
            created_at=current_time,
            updated_at=current_time,
            segments=[SpeechSegment(started_at=current_time)],
        )
        self._active = candidate
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
        self._record_attrs(candidate, event="transcript_final" if is_final else "transcript_interim")

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
        transcript = candidate.selected_text
        if not transcript:
            candidate.state = "rejected"
            candidate.reject_reason = "empty_transcript"
            self._record_attrs(candidate, event="rejected")
            return UserTurnDecision(
                action="reject",
                candidate_id=candidate.candidate_id,
                reason="empty_transcript",
            )
        if should_defer:
            candidate.state = "waiting_merge"
            candidate.merge_reason = "low_eot_wait_for_continuation"
            self._record_attrs(candidate, event="deferred_low_eot")
            return UserTurnDecision(
                action="defer",
                candidate_id=candidate.candidate_id,
                transcript=transcript,
                reason="low_eot_wait_for_continuation",
                delay_sec=self._low_eot_delay_sec,
            )
        candidate.state = "waiting_voiceprint"
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
            candidate.reject_reason = f"voiceprint_blocked:{reason}"
            self._record_attrs(candidate, event="voiceprint_rejected")
            return UserTurnDecision(
                action="reject",
                candidate_id=candidate.candidate_id,
                transcript=candidate.selected_text,
                reason=candidate.reject_reason,
            )
        candidate.state = "ready_to_commit"
        candidate.commit_reason = f"voiceprint_allowed:{reason}"
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
        if candidate is None:
            self._counter += 1
            candidate = UserTurnCandidate(
                candidate_id=(
                    timeline.turn_id
                    if timeline is not None
                    else f"user-turn-{self._counter}"
                ),
                timeline=timeline,
                state="open",
                created_at=current_time,
                updated_at=current_time,
                segments=[],
            )
            self._active = candidate
        elif candidate.timeline is None and timeline is not None:
            candidate.timeline = timeline

        if stripped:
            self._merge_framework_transcript(candidate, stripped, now=current_time)
        candidate.state = "committed"
        candidate.commit_reason = reason
        if voiceprint_reason:
            candidate.voiceprint_reason = voiceprint_reason
        candidate.committed_at = current_time
        candidate.updated_at = current_time
        canonical = candidate.selected_text or stripped
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
        if candidate is None:
            self._counter += 1
            candidate = UserTurnCandidate(
                candidate_id=(
                    timeline.turn_id
                    if timeline is not None
                    else f"user-turn-{self._counter}"
                ),
                timeline=timeline,
                state="open",
                created_at=current_time,
                updated_at=current_time,
                segments=[],
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
                and _normalize_revision_text(selected)
                == _normalize_revision_text(stripped)
            ):
                self._merge_framework_transcript(candidate, stripped, now=current_time)
        candidate.state = "waiting_merge"
        if not candidate.merge_reason:
            candidate.merge_reason = reason
        if (
            voiceprint_reason
            and not candidate.voiceprint_reason.startswith("voiceprint_inconclusive:")
        ):
            candidate.voiceprint_reason = voiceprint_reason
        candidate.updated_at = current_time
        canonical = candidate.selected_text or stripped
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
        if candidate is None:
            self._counter += 1
            candidate = UserTurnCandidate(
                candidate_id=(
                    timeline.turn_id
                    if timeline is not None
                    else f"user-turn-{self._counter}"
                ),
                timeline=timeline,
                state="open",
                created_at=current_time,
                updated_at=current_time,
                segments=[],
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
            delay_sec=self.merge_remaining_sec(now=current_time)
            or self._low_eot_delay_sec,
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
        if _count_cjk_chars(selected) > 28:
            return False
        fragments = [
            segment.selected_text
            for segment in candidate.segments
            if segment.selected_text
        ]
        if len(fragments) < 2:
            return False
        return all(_looks_like_statement_fragment(text) for text in fragments[-2:])

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
        if current_text and _text_matches_revision(current_text, text):
            return current_index, current

        for index in range(len(candidate.segments) - 2, -1, -1):
            segment = candidate.segments[index]
            if _text_matches_revision(segment.selected_text, text):
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
        if selected and _text_matches_revision(selected, transcript):
            return
        candidate.segments.append(
            SpeechSegment(
                started_at=now,
                ended_at=now,
                text=transcript,
                final_text=transcript,
            )
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
            payload["committed_transcript_preview"] = transcript[:120]
        timeline.set_attr("user_turn_coordinator", payload)

    def _snapshot(self, candidate: UserTurnCandidate) -> dict[str, Any]:
        return {
            "candidate_id": candidate.candidate_id,
            "state": candidate.state,
            "segments": len(candidate.segments),
            "revisions": len(candidate.revisions),
            "selected_text_preview": candidate.selected_text[:120],
            "eot_score": candidate.eot_score,
            "merge_reason": candidate.merge_reason,
            "voiceprint_reason": candidate.voiceprint_reason,
            "commit_reason": candidate.commit_reason,
            "reject_reason": candidate.reject_reason,
        }


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


def _looks_like_statement_fragment(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if any(mark in stripped for mark in ("？", "?", "！", "!")):
        return False
    if _count_cjk_chars(stripped) > 14:
        return False
    if stripped.startswith(("帮我", "请", "麻烦", "换个话题", "换一个话题")):
        return False
    return stripped.endswith(("。", "，", ",", "、", "的", "了", "呢", "吧"))


def _text_matches_revision(existing: str, revision: str) -> bool:
    existing_norm = _normalize_revision_text(existing)
    revision_norm = _normalize_revision_text(revision)
    if not existing_norm or not revision_norm:
        return False
    if existing_norm == revision_norm:
        return True
    if existing_norm.startswith(revision_norm) or revision_norm.startswith(existing_norm):
        return True
    if min(len(existing_norm), len(revision_norm)) < 4:
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
