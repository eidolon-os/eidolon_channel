"""Product-level interruption owner for streaming voice sessions."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from collections.abc import Callable
from typing import Any

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.turn_policy import (
    Action,
    Decision,
    InterruptIntent,
    TurnPolicyRuntime,
)

logger = logging.getLogger("agent.session.interruption_orchestrator")


class InterruptionState(str, Enum):
    """Lifecycle state for one possible barge-in candidate."""

    IDLE = "idle"
    CANDIDATE_STARTED = "candidate_started"
    SUSPENDED_WAITING_EVIDENCE = "suspended_waiting_evidence"
    SUSPENDED_POST_SPEECH_WAIT = "suspended_post_speech_wait"
    CONFIRMED_CANCELLED = "confirmed_cancelled"
    CONFIRMED_CANCEL_COLLECTING_TURN = "confirmed_cancel_collecting_turn"
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


class InterruptionVerdictAction(str, Enum):
    """Terminal meaning of one resolved playback interruption."""

    CONFIRMED_CANCEL = "confirmed_cancel"
    REJECTED_RESUME = "rejected_resume"
    EXPIRED_RESUME = "expired_resume"
    REJECTED_CANDIDATE = "rejected_candidate"


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


@dataclass(frozen=True)
class InterruptionVerdict:
    """Immutable result consumed by the framework turn boundary.

    The terminal hook must not reconstruct interruption intent from transcript
    length, language, EOT, or timeline dictionaries.  It consumes this verdict
    produced by the interruption owner after output effects resolve.
    """

    action: InterruptionVerdictAction
    reason: str
    candidate_id: str | None
    generation_id: int
    continue_to_llm: bool
    turn_policy_action: str = ""
    intent: str = ""
    transcript: str = ""


@dataclass
class InterruptionCandidate:
    """A single possible barge-in while the agent is audible."""

    candidate_id: str | None
    started_at: float
    generation_id: int
    state: InterruptionState = InterruptionState.CANDIDATE_STARTED
    stopped_at: float | None = None
    awaiting_post_speech_evidence: bool = False
    resolved: bool = False
    transcript: str = ""
    final_transcript: str = ""
    last_policy_action: Action | None = None
    last_policy_reason: str = ""
    last_policy_source: str = ""
    last_intent: InterruptIntent | None = None
    last_eot_score: float | None = None
    last_vad_active: bool | None = None
    last_decision_signature: tuple[object, ...] | None = None
    last_event_at: float | None = None

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
        no_evidence_timeout_sec: float | None = None,
        on_terminal_verdict: Callable[[InterruptionVerdict], None] | None = None,
        clock: Any | None = None,
    ) -> None:
        self._evidence_timeout_sec = max(0.0, float(evidence_timeout_sec))
        self._min_speech_sec = max(0.0, float(min_speech_sec))
        # Shorter cap for the post-speech wait when NO transcript has arrived
        # (false trigger). Defaults to the full evidence window so existing
        # callers keep prior behavior.
        self._no_evidence_timeout_sec = (
            self._evidence_timeout_sec
            if no_evidence_timeout_sec is None
            else max(0.0, float(no_evidence_timeout_sec))
        )
        self._clock = clock or time.monotonic
        self._on_terminal_verdict = on_terminal_verdict
        self._candidate: InterruptionCandidate | None = None
        self._timeline: TurnTimeline | None = None
        self._resolved_verdicts: dict[tuple[str | None, int], InterruptionVerdict] = {}

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

    @property
    def current_transcript(self) -> str:
        candidate = self._candidate
        if candidate is None:
            return ""
        return candidate.final_transcript or candidate.transcript

    def verdict_for(
        self,
        candidate_id: str | None,
        *,
        generation_id: int | None,
    ) -> InterruptionVerdict | None:
        """Return a verdict only for its exact product/acoustic generation."""

        if generation_id is None:
            return None
        return self._resolved_verdicts.get((candidate_id, generation_id))

    def start_candidate(
        self,
        *,
        timeline: TurnTimeline | None,
        generation_id: int = 1,
        already_suspended: bool = True,
    ) -> InterruptionDecision:
        """Start owning a possible interruption after output has soft-ducked."""

        if self.active:
            current = self._candidate
            candidate_id = getattr(timeline, "turn_id", None)
            if (
                current is not None
                and current.candidate_id == candidate_id
                and current.generation_id == generation_id
            ):
                self._record_event("candidate_start_ignored_active")
                return self._decision(
                    InterruptionDecisionAction.NO_OP,
                    "candidate_already_active",
                )
            # A later acoustic generation has become the interruption owner.
            # Terminalize the stale owner before switching timelines; output is
            # already suspended and remains so for the replacement candidate.
            self.resolve(
                action="supersede",
                reason="superseded_by_new_acoustic_generation",
            )
        self._timeline = timeline
        self._candidate = InterruptionCandidate(
            candidate_id=getattr(timeline, "turn_id", None),
            started_at=self._now(),
            generation_id=generation_id,
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

    def decide_from_transcript(
        self,
        turn_runtime: TurnPolicyRuntime,
        text: str,
        eot_score: float,
        *,
        vad_active: bool,
        agent_speaking: bool,
        is_final: bool = False,
        event_time_ms: float | None = None,
    ) -> Decision:
        """Route transcript evidence through the owner-owned policy path."""

        self.note_transcript(text, is_final=is_final)
        decision = turn_runtime.decide_from_transcript(
            text,
            eot_score,
            vad_active=vad_active,
            agent_speaking=agent_speaking,
            is_final=is_final,
            event_time_ms=event_time_ms,
        )
        return decision

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
        candidate.last_intent = decision.intent
        candidate.last_eot_score = eot_score
        candidate.last_vad_active = vad_active
        if transcript.strip():
            candidate.transcript = transcript.strip()

        signature = (
            source,
            decision.action.value,
            decision.reason,
            decision.intent.value if decision.intent is not None else None,
            transcript.strip(),
            vad_active,
            round(float(eot_score), 4) if eot_score is not None else None,
            decision.rollback_drop_buffered,
        )
        if signature != candidate.last_decision_signature:
            candidate.last_decision_signature = signature
            self._record_event(
                "turn_policy_decision",
                source=source,
                action=decision.action.value,
                reason=decision.reason,
                intent=decision.intent.value if decision.intent is not None else None,
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
        if text and candidate.last_intent in (
            InterruptIntent.BACKCHANNEL,
            InterruptIntent.NOISE,
        ):
            self._record_event(
                "short_false_interruption_fast_resume",
                transcript_preview=text[:80],
                last_policy_action=(
                    candidate.last_policy_action.value
                    if candidate.last_policy_action is not None
                    else None
                ),
                last_policy_reason=candidate.last_policy_reason,
            )
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
        """Whether the duck deadline is allowed to terminate the candidate.

        While speech is still active, weak evidence can only HOLD: a short
        backchannel-looking prefix is not terminal until the acoustic segment
        closes.  After speech ends, keep holding only for the explicit delayed
        evidence window; a no-transcript trigger expires on its shorter grace.
        """

        remaining_sec = self.hold_remaining_sec()
        return remaining_sec is not None and remaining_sec > 0

    def hold_remaining_sec(self) -> float | None:
        """Return the owner-clock budget remaining for an evidence HOLD.

        Active-speech evidence is bounded from VAD start. Once speech stops,
        the post-speech evidence window is measured from VAD end; a long valid
        utterance must not consume that grace before STT can finalize it. A
        short Latin interim is provisional until FINAL and uses the shorter
        no-evidence grace.
        """

        candidate = self._candidate
        if candidate is None or candidate.resolved:
            return None
        if (
            candidate.stopped_at is None
            and candidate.last_policy_action in (None, Action.HOLD)
        ):
            return max(
                0.0,
                self._evidence_timeout_sec - (self._now() - candidate.started_at),
            )
        if not candidate.awaiting_post_speech_evidence or candidate.stopped_at is None:
            return None
        transcript = (candidate.final_transcript or candidate.transcript).strip()
        timeout_sec = (
            self._no_evidence_timeout_sec
            if not transcript or self._has_provisional_short_latin_artifact(candidate)
            else self._evidence_timeout_sec
        )
        return max(0.0, timeout_sec - (self._now() - candidate.stopped_at))

    @staticmethod
    def _has_provisional_short_latin_artifact(
        candidate: InterruptionCandidate,
    ) -> bool:
        return bool(
            not candidate.final_transcript.strip()
            and candidate.last_policy_action is Action.HOLD
            and "short_latin_artifact" in candidate.last_policy_reason
        )

    def blocks_framework_completed_turn(
        self,
        candidate_id: str | None = None,
        *,
        generation_id: int | None = None,
    ) -> bool:
        """True while LiveKit must not commit a not-yet-owned interrupt turn."""

        candidate = self._candidate
        if candidate is None or candidate.resolved:
            return False
        if candidate_id is None and generation_id is None:
            candidate_id = candidate.candidate_id
            generation_id = candidate.generation_id
        if (
            generation_id is None
            or candidate.candidate_id != candidate_id
            or candidate.generation_id != generation_id
        ):
            return False
        if candidate.awaiting_post_speech_evidence:
            return True
        if candidate.state is InterruptionState.CONFIRMED_CANCEL_COLLECTING_TURN:
            return True
        return (
            candidate.state
            in {
                InterruptionState.SUSPENDED_WAITING_EVIDENCE,
                InterruptionState.SUSPENDED_POST_SPEECH_WAIT,
            }
            and candidate.last_policy_action in (None, Action.HOLD)
        )

    def should_commit_after_confirmed_cancel(self) -> bool:
        """Whether a confirmed cancel should become the next user turn.

        Hard-stop/backchannel/noise are control or false-interruption signals.
        Normal/topic/correction interrupts are user utterances and must be
        committed after the agent is cancelled, subject to voiceprint gating.
        """

        candidate = self._candidate
        if not self._is_committable_cancel_candidate(candidate):
            return False
        if (
            not candidate.awaiting_post_speech_evidence
            and candidate.stopped_at is None
            and candidate.last_vad_active is True
        ):
            return False
        return bool((candidate.final_transcript or candidate.transcript).strip())

    def should_collect_after_confirmed_cancel(self) -> bool:
        """True when output is cancelled but the user speech segment is ongoing."""

        candidate = self._candidate
        if not self._is_committable_cancel_candidate(candidate):
            return False
        return (
            not candidate.awaiting_post_speech_evidence
            and candidate.stopped_at is None
            and candidate.last_vad_active is True
        )

    def mark_confirmed_cancel_collecting_turn(self) -> None:
        candidate = self._candidate
        if candidate is None or candidate.resolved:
            return
        candidate.state = InterruptionState.CONFIRMED_CANCEL_COLLECTING_TURN
        self._record_event(
            "confirmed_cancel_collecting_turn",
            transcript_preview=self.current_transcript[:80],
            last_policy_reason=candidate.last_policy_reason,
        )

    def finish_confirmed_cancel_speech(self, transcript: str) -> bool:
        """Terminalize a semantic cancel once its speech segment closes.

        Cancelling playback is irreversible, but a normal interruption can
        still be speaking at that instant.  In that case the owner stays in
        ``CONFIRMED_CANCEL_COLLECTING_TURN`` until VAD-end, then publishes the
        typed verdict consumed by the framework-completed boundary.  The
        already-confirmed cancel must never be reclassified from a later EOT
        score computed for the completed transcript.
        """

        candidate = self._candidate
        if not self._is_committable_cancel_candidate(candidate):
            return False
        candidate.stopped_at = self._now()
        text = transcript.strip()
        if text:
            candidate.transcript = text
        has_transcript = bool(self.current_transcript.strip())
        self._record_event(
            "confirmed_cancel_speech_stopped",
            transcript_preview=self.current_transcript[:80],
        )
        self.resolve(
            action="cancel",
            reason="confirmed_cancel_speech_completed",
        )
        return has_transcript

    def resolve(self, *, action: str, reason: str) -> None:
        candidate = self._candidate
        if candidate is None:
            return
        verdict = self._build_verdict(candidate, action=action, reason=reason)
        verdict_key = (verdict.candidate_id, verdict.generation_id)
        self._resolved_verdicts[verdict_key] = verdict
        while len(self._resolved_verdicts) > 16:
            self._resolved_verdicts.pop(next(iter(self._resolved_verdicts)))
        candidate.resolved = True
        self._record_event(
            "candidate_resolved",
            action=action,
            reason=reason,
            verdict=verdict.action.value,
            continue_to_llm=verdict.continue_to_llm,
            intent=verdict.intent,
        )
        if self._timeline is not None:
            verdict_payload = {
                "action": verdict.action.value,
                "reason": verdict.reason,
                "candidate_id": verdict.candidate_id,
                "generation_id": verdict.generation_id,
                "continue_to_llm": verdict.continue_to_llm,
                "turn_policy_action": verdict.turn_policy_action,
                "intent": verdict.intent,
                "transcript_preview": verdict.transcript[:120],
            }
            history = list(self._timeline.attrs.get("interruption_verdicts") or ())
            history.append(verdict_payload)
            self._timeline.set_attr("interruption_verdicts", history[-16:])
            self._timeline.set_attr("interruption_verdict", verdict_payload)
        logger.info(
            "[InterruptionOrchestrator] resolved action=%s verdict=%s "
            "continue_to_llm=%s reason=%s",
            action,
            verdict.action.value,
            verdict.continue_to_llm,
            reason,
        )
        self._candidate = None
        if self._on_terminal_verdict is not None:
            self._on_terminal_verdict(verdict)

    @staticmethod
    def _build_verdict(
        candidate: InterruptionCandidate,
        *,
        action: str,
        reason: str,
    ) -> InterruptionVerdict:
        normalized_action = action.strip().lower()
        intent = candidate.last_intent
        if normalized_action == "cancel":
            verdict_action = InterruptionVerdictAction.CONFIRMED_CANCEL
            continue_to_llm = (
                candidate.last_policy_action is Action.CANCEL
                and intent is InterruptIntent.NORMAL_INTERRUPT
            )
        elif normalized_action in {"rollback", "resume", "unduck"}:
            verdict_action = (
                InterruptionVerdictAction.EXPIRED_RESUME
                if "timeout" in reason or "deadline" in reason
                else InterruptionVerdictAction.REJECTED_RESUME
            )
            continue_to_llm = False
        else:
            verdict_action = InterruptionVerdictAction.REJECTED_CANDIDATE
            continue_to_llm = False
        return InterruptionVerdict(
            action=verdict_action,
            reason=reason,
            candidate_id=candidate.candidate_id,
            generation_id=candidate.generation_id,
            continue_to_llm=continue_to_llm,
            turn_policy_action=(
                candidate.last_policy_action.value
                if candidate.last_policy_action is not None
                else ""
            ),
            intent=intent.value if intent is not None else "",
            transcript=(candidate.final_transcript or candidate.transcript).strip(),
        )

    @staticmethod
    def _is_committable_cancel_candidate(
        candidate: InterruptionCandidate | None,
    ) -> bool:
        if candidate is None or candidate.resolved:
            return False
        if candidate.last_policy_action is not Action.CANCEL:
            return False
        return candidate.last_intent not in (
            InterruptIntent.HARD_STOP,
            InterruptIntent.BACKCHANNEL,
            InterruptIntent.NOISE,
        )

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
        now = self._now()
        elapsed_ms: float | None = None
        since_last_event_ms: float | None = None
        if candidate is not None:
            elapsed_ms = max(0.0, (now - candidate.started_at) * 1000.0)
            if candidate.last_event_at is not None:
                since_last_event_ms = max(0.0, (now - candidate.last_event_at) * 1000.0)
            candidate.last_event_at = now
        payload = {
            "event": event,
            "state": candidate.state.value if candidate is not None else "idle",
            "generation_id": candidate.generation_id if candidate is not None else None,
            **fields,
        }
        if elapsed_ms is not None:
            payload["elapsed_ms"] = elapsed_ms
        if since_last_event_ms is not None:
            payload["since_last_event_ms"] = since_last_event_ms
        events = list(timeline.attrs.get("interruption_orchestrator_events") or ())
        events.append(payload)
        timeline.set_attr("interruption_orchestrator_events", events)
        timeline.set_attr("interruption_orchestrator_last_event", payload)
