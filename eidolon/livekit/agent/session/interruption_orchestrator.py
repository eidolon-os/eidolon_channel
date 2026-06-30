"""Product-level interruption owner for streaming voice sessions."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.turn_policy import Action, Decision

logger = logging.getLogger("agent.session.interruption_orchestrator")


class InterruptionState(str, Enum):
    """Lifecycle state for one possible barge-in candidate."""

    IDLE = "idle"
    CANDIDATE_STARTED = "candidate_started"
    SUSPENDED_WAITING_EVIDENCE = "suspended_waiting_evidence"
    SUSPENDED_POST_SPEECH_WAIT = "suspended_post_speech_wait"
    CONFIRMED_CANCELLED = "confirmed_cancelled"
    CONFIRMED_FALSE_RESUME = "confirmed_false_resume"
    REJECTED_NOISE_OR_ECHO = "rejected_noise_or_echo"


class InterruptionDecisionAction(str, Enum):
    """Typed decisions emitted by the interruption owner."""

    NO_OP = "no_op"
    SOFT_SUSPEND_OUTPUT = "soft_suspend_output"
    HOLD_FOR_EVIDENCE = "hold_for_evidence"
    CONFIRM_CANCEL = "confirm_cancel"
    RESUME_OUTPUT = "resume_output"
    DROP_STALE_BUFFER_AND_RESUME = "drop_stale_buffer_and_resume"
    REJECT_CANDIDATE = "reject_candidate"


@dataclass(frozen=True)
class InterruptionDecision:
    """Pure owner decision; side effects are applied by session adapters."""

    action: InterruptionDecisionAction
    reason: str
    candidate_id: str | None = None
    state: InterruptionState = InterruptionState.IDLE
    turn_policy_action: str = ""
    transcript_preview: str = ""
    drop_buffered: bool = False


@dataclass
class InterruptionCandidate:
    """A single possible barge-in while the agent is audible."""

    candidate_id: str | None
    started_at: float
    state: InterruptionState = InterruptionState.CANDIDATE_STARTED
    stopped_at: float | None = None
    awaiting_post_speech_evidence: bool = False
    resolved: bool = False
    transcript: str = ""
    final_transcript: str = ""
    last_policy_action: Action | None = None
    last_policy_reason: str = ""
    last_policy_source: str = ""

    @property
    def speech_duration_sec(self) -> float:
        end = self.stopped_at if self.stopped_at is not None else time.monotonic()
        return max(0.0, end - self.started_at)


class InterruptionOrchestrator:
    """Own the interruption candidate lifecycle.

    Attention admission decides whether an input is eligible, semantic policy
    decides cancel/rollback from evidence, and output controllers apply audio
    side effects. This orchestrator owns the missing product-level lifecycle in
    between: a VAD-start candidate remains alive after VAD-end long enough for
    delayed STT evidence to confirm cancel or false-resume.
    """

    def __init__(
        self,
        *,
        evidence_timeout_sec: float,
        min_speech_sec: float,
        clock: Any | None = None,
    ) -> None:
        self._evidence_timeout_sec = max(0.0, float(evidence_timeout_sec))
        self._min_speech_sec = max(0.0, float(min_speech_sec))
        self._clock = clock or time.monotonic
        self._candidate: InterruptionCandidate | None = None
        self._timeline: TurnTimeline | None = None

    @property
    def state(self) -> InterruptionState:
        candidate = self._candidate
        if candidate is None:
            return InterruptionState.IDLE
        return candidate.state

    @property
    def active(self) -> bool:
        candidate = self._candidate
        return candidate is not None and not candidate.resolved

    @property
    def awaiting_post_speech_evidence(self) -> bool:
        candidate = self._candidate
        return (
            candidate is not None
            and not candidate.resolved
            and candidate.awaiting_post_speech_evidence
        )

    def start_candidate(
        self,
        *,
        timeline: TurnTimeline | None,
        already_suspended: bool = True,
    ) -> InterruptionDecision:
        """Start owning a possible interruption after output has soft-ducked."""

        self._timeline = timeline
        if self.active:
            self._record_event("candidate_start_ignored_active")
            return self._decision(
                InterruptionDecisionAction.NO_OP,
                "candidate_already_active",
            )
        self._candidate = InterruptionCandidate(
            candidate_id=getattr(timeline, "turn_id", None),
            started_at=self._now(),
            state=(
                InterruptionState.SUSPENDED_WAITING_EVIDENCE
                if already_suspended
                else InterruptionState.CANDIDATE_STARTED
            ),
        )
        self._record_event("candidate_started")
        return self._decision(
            (
                InterruptionDecisionAction.SOFT_SUSPEND_OUTPUT
                if already_suspended
                else InterruptionDecisionAction.HOLD_FOR_EVIDENCE
            ),
            "vad_started",
        )

    def note_transcript(self, transcript: str, *, is_final: bool) -> None:
        """Record evidence arrival for observability.

        SemanticInterruptHandler still owns the policy decision. Recording here
        lets timelines explain why a candidate stayed suspended after VAD end.
        """

        candidate = self._candidate
        text = transcript.strip()
        if candidate is None or candidate.resolved or not text:
            return
        candidate.transcript = text
        if is_final:
            candidate.final_transcript = text
        self._record_event(
            "transcript_evidence",
            is_final=bool(is_final),
            text_preview=transcript[:80],
        )

    def note_turn_policy_decision(
        self,
        decision: Decision,
        *,
        source: str = "turn_policy",
        transcript: str = "",
        vad_active: bool | None = None,
        eot_score: float | None = None,
    ) -> InterruptionDecision:
        """Record a turn-policy decision and map it to owner semantics.

        The turn-policy layer remains the evidence authority. The owner uses
        that evidence to keep one terminal lifecycle for the candidate, so VAD
        end, STT final, and timeout cannot race into conflicting side effects.
        """

        candidate = self._candidate
        if candidate is None or candidate.resolved:
            return self._decision(
                InterruptionDecisionAction.NO_OP,
                "no_active_candidate",
                turn_policy_action=decision.action.value,
                transcript_preview=transcript[:80],
            )

        candidate.last_policy_action = decision.action
        candidate.last_policy_reason = decision.reason
        candidate.last_policy_source = source
        if transcript.strip():
            candidate.transcript = transcript.strip()

        self._record_event(
            "turn_policy_decision",
            source=source,
            action=decision.action.value,
            reason=decision.reason,
            vad_active=vad_active,
            eot_score=eot_score,
            transcript_preview=transcript[:80],
            drop_buffered=decision.rollback_drop_buffered,
        )

        if decision.action is Action.CANCEL:
            candidate.state = InterruptionState.CONFIRMED_CANCELLED
            return self._decision(
                InterruptionDecisionAction.CONFIRM_CANCEL,
                decision.reason,
                turn_policy_action=decision.action.value,
                transcript_preview=transcript[:80],
            )
        if decision.action is Action.ROLLBACK:
            candidate.state = InterruptionState.CONFIRMED_FALSE_RESUME
            return self._decision(
                (
                    InterruptionDecisionAction.DROP_STALE_BUFFER_AND_RESUME
                    if decision.rollback_drop_buffered
                    else InterruptionDecisionAction.RESUME_OUTPUT
                ),
                decision.reason,
                turn_policy_action=decision.action.value,
                transcript_preview=transcript[:80],
                drop_buffered=decision.rollback_drop_buffered,
            )
        if decision.action is Action.HOLD:
            if candidate.awaiting_post_speech_evidence:
                candidate.state = InterruptionState.SUSPENDED_POST_SPEECH_WAIT
            else:
                candidate.state = InterruptionState.SUSPENDED_WAITING_EVIDENCE
            return self._decision(
                InterruptionDecisionAction.HOLD_FOR_EVIDENCE,
                decision.reason,
                turn_policy_action=decision.action.value,
                transcript_preview=transcript[:80],
            )
        return self._decision(
            InterruptionDecisionAction.NO_OP,
            decision.reason,
            turn_policy_action=decision.action.value,
            transcript_preview=transcript[:80],
        )

    def defer_false_resume_after_speech_end(
        self,
        *,
        transcript: str,
        duck_suspended: bool,
    ) -> bool:
        """Return True when VAD-end should wait for delayed evidence.

        VAD-end is an input signal, not an interruption decision. When the agent
        is still suspended and no transcript has arrived, hold the candidate
        open if the near-end speech lasted long enough to plausibly be a real
        barge-in. Very short VAD blips continue down the existing fast rollback
        path.
        """

        candidate = self._candidate
        if candidate is None or candidate.resolved:
            return False
        candidate.stopped_at = self._now()
        if not duck_suspended:
            return False
        text = transcript.strip()
        if text:
            candidate.transcript = text
        if text and candidate.last_policy_action is Action.ROLLBACK:
            return False
        if text and candidate.last_policy_action is Action.CANCEL:
            return False
        should_wait_for_evidence = (
            not text
            or candidate.last_policy_action is None
            or candidate.last_policy_action is Action.HOLD
        )
        if not should_wait_for_evidence:
            return False
        duration_sec = candidate.speech_duration_sec
        if duration_sec < self._min_speech_sec:
            self._record_event(
                "candidate_too_short_for_evidence_wait",
                speech_ms=duration_sec * 1000.0,
                min_speech_ms=self._min_speech_sec * 1000.0,
            )
            return False
        candidate.awaiting_post_speech_evidence = True
        candidate.state = InterruptionState.SUSPENDED_POST_SPEECH_WAIT
        self._record_event(
            "post_speech_evidence_wait",
            speech_ms=duration_sec * 1000.0,
            timeout_ms=self._evidence_timeout_sec * 1000.0,
            transcript_preview=text[:80],
            last_policy_action=(
                candidate.last_policy_action.value
                if candidate.last_policy_action is not None
                else None
            ),
            last_policy_reason=candidate.last_policy_reason,
        )
        logger.info(
            "[InterruptionOrchestrator] holding post-speech evidence window "
            "speech_ms=%.0f timeout=%.2fs last_policy_action=%s",
            duration_sec * 1000.0,
            self._evidence_timeout_sec,
            (
                candidate.last_policy_action.value
                if candidate.last_policy_action is not None
                else "none"
            ),
        )
        return True

    def should_hold_deadline(self) -> bool:
        """Whether duck deadline should keep holding after VAD became idle."""

        return self.awaiting_post_speech_evidence

    def max_suspend_sec(self) -> float:
        """Max suspend window while waiting for post-speech evidence."""

        if not self.awaiting_post_speech_evidence:
            return 0.0
        return self._evidence_timeout_sec

    def resolve(self, *, action: str, reason: str) -> None:
        candidate = self._candidate
        if candidate is None:
            return
        candidate.resolved = True
        self._record_event("candidate_resolved", action=action, reason=reason)
        logger.info(
            "[InterruptionOrchestrator] resolved action=%s reason=%s",
            action,
            reason,
        )
        self._candidate = None

    def _decision(
        self,
        action: InterruptionDecisionAction,
        reason: str,
        *,
        turn_policy_action: str = "",
        transcript_preview: str = "",
        drop_buffered: bool = False,
    ) -> InterruptionDecision:
        candidate = self._candidate
        return InterruptionDecision(
            action=action,
            reason=reason,
            candidate_id=candidate.candidate_id if candidate is not None else None,
            state=candidate.state if candidate is not None else InterruptionState.IDLE,
            turn_policy_action=turn_policy_action,
            transcript_preview=transcript_preview,
            drop_buffered=drop_buffered,
        )

    def _now(self) -> float:
        return float(self._clock())

    def _record_event(self, event: str, **fields: object) -> None:
        timeline = self._timeline
        if timeline is None:
            return
        candidate = self._candidate
        payload = {
            "event": event,
            "state": candidate.state.value if candidate is not None else "idle",
            **fields,
        }
        events = list(timeline.attrs.get("interruption_orchestrator_events") or ())
        events.append(payload)
        timeline.set_attr("interruption_orchestrator_events", events)
        timeline.set_attr("interruption_orchestrator_last_event", payload)
