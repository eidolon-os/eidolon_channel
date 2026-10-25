"""One candidate's permission ordering; no input, model or playback implementation.

The existing coordinator owns terminal state. All async results are scoped to
its frozen audio reference; no speaker result is reusable across candidates.
"""

import asyncio
import logging
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from ...common.audio_evidence import AudioEvidence, AudioEvidenceCapture
from ..session.user_turn_coordinator import UserTurnCoordinator, UserTurnDecision

logger = logging.getLogger(__name__)
VERIFICATION_TIMEOUT_SEC = 8


@dataclass(frozen=True)
class SpeakerDecision:
    status: Literal["match", "non_match", "inconclusive", "unavailable"]
    profile_id: str = ""
    profile_revision: int = 0
    calibration_version: str = ""
    policy_revision: int = -1


class SpeakerVerifiedTurns:
    def __init__(
        self,
        *,
        coordinator: UserTurnCoordinator,
        capture: Callable[[], AudioEvidenceCapture | None],
        session: Any,
        verify: Callable[[AudioEvidence], Awaitable[SpeakerDecision]],
        is_complete: Callable[[str], Awaitable[bool]],
        classify: Callable[[str, Any], Awaitable[str]],
        authorize: Callable[[SpeakerDecision], Awaitable[bool]],
        committed: Callable[[UserTurnDecision], None],
        cancel_output: Callable[[], Awaitable[None]] | None = None,
        feedback: Callable[[str], None] | None = None,
        timeout_sec: float = VERIFICATION_TIMEOUT_SEC,
        min_audio_ms: int = 1500,
        max_audio_ms: int = 30000,
    ):
        if not math.isfinite(timeout_sec) or not 0 < timeout_sec <= 8:
            raise ValueError("bounded candidate verification deadline required")
        self._owner, self._capture, self._session = coordinator, capture, session
        self._verify, self._complete, self._classify = verify, is_complete, classify
        self._authorize, self._committed = authorize, committed
        self._cancel_output = cancel_output
        self._feedback = feedback or (lambda state: None)
        self._timeout = timeout_sec
        if not 1500 <= min_audio_ms <= max_audio_ms <= 30000:
            raise ValueError("bounded admitted audio range required")
        self._min_audio_ms, self._max_audio_ms = min_audio_ms, max_audio_ms
        self._pending: asyncio.Task | None = None
        self._closed = False
        self._epoch = 0

    @property
    def pending(self) -> asyncio.Task | None:
        return self._pending

    def invalidate(self, reason: str) -> None:
        self._epoch += 1
        self._owner.reject_active(reason)
        if self._pending is not None:
            self._pending.cancel()
        if not self._closed:
            self._feedback("inconclusive" if reason == "candidate_timeout" else "ready")

    def consider(self) -> asyncio.Task | None:
        """Attempt a complete candidate; never queues PCM or a second inference."""
        if self._closed or self._pending is not None:
            return None
        candidate = self._owner.active
        if candidate is None or candidate.state != "open":
            return None
        capture = self._capture()
        source = capture.active_source if capture else None
        if source is None:
            self._owner.reject_active("audio_source_unavailable")
            self._feedback("unavailable")
            return None
        reference = self._owner.capture_audio_reference(source=source)
        if reference.span is None:
            # Interim/renewed speech is still collecting. Caller owns its
            # overall candidate deadline, not a second verification queue.
            return None
        evidence = capture.snapshot(candidate=candidate.candidate_id, span=reference.span)
        reason = evidence.reason
        if evidence.sample_rate != 16000:
            reason = "unsupported_sample_rate"
        elif evidence.end - evidence.start < self._min_audio_ms * 16:
            reason = "audio_too_short"
        elif evidence.end - evidence.start > self._max_audio_ms * 16:
            reason = "audio_too_long"
        if not evidence.complete or reason:
            logger.info("Verified candidate rejected candidate=%s reason=%s audio_ms=%.1f",
                        candidate.candidate_id, reason or "audio_incomplete",
                        (evidence.end - evidence.start) / 16)
            self._owner.reject_active(reason or "audio_incomplete")
            self._feedback("audio_too_long" if reason == "audio_too_long" else "inconclusive")
            return None
        epoch, text, output = self._epoch, candidate.selected_text, self._session.current_speech

        def current():
            return (
                not self._closed
                and epoch == self._epoch
                and self._capture() is capture
                and capture.active_source == source
                and self._owner.is_audio_reference_current(reference, source=source)
            )

        def reject(reason, state="unavailable"):
            if current():
                logger.info("Verified candidate rejected candidate=%s reason=%s state=%s",
                            candidate.candidate_id, reason, state)
                self._owner.reject_active(reason)
                self._feedback(state)

        def output_current():
            actual = self._session.current_speech
            # Natural completion is fine; a different response is not the
            # output against which the candidate's intent was assessed.
            return actual is None or actual is output

        async def evaluate():
            committed = False
            try:
                async with asyncio.timeout(self._timeout):
                    if not await self._complete(text):
                        return  # EOT owns completeness; do not invent a terminal utterance.
                    if not current():
                        return
                    self._feedback("checking")
                    decision = await self._verify(evidence)
                    if not current():
                        return
                    if (
                        decision.status != "match"
                        or not decision.profile_id
                        or decision.profile_revision < 1
                        or not decision.calibration_version
                        or decision.policy_revision < 0
                    ):
                        reject(
                            "identity_" + decision.status,
                            decision.status
                            if decision.status in {"non_match", "inconclusive", "unavailable"}
                            else "unavailable",
                        )
                        return
                    action = await self._classify(text, output)
                    if not current():
                        return
                    if action not in {"reply", "stop_output"}:
                        reject("no_authorized_action", "no_action")
                        return
                    allowed = await self._authorize(decision)
                    if not current():
                        return
                    if not allowed or not output_current():
                        reject("authorization_or_output_changed")
                        return
                    # No await between scope check and issuing the cancellation.
                    actual = self._session.current_speech
                    if actual is not None and not actual.done():
                        if self._cancel_output is None:
                            await self._session.interrupt(force=True)
                        else:
                            await self._cancel_output()
                        if not current():
                            return
                    if action == "stop_output":
                        reject("authorized_stop", "accepted")
                        return
                    # Cancellation may drain tools/playout asynchronously. A
                    # revocation or newly scheduled output must still veto reply.
                    allowed = await self._authorize(decision)
                    if not current():
                        return
                    if not allowed or not output_current():
                        reject("authorization_or_output_changed")
                        return
                    result = self._owner.commit_verified(reference, source=source)
                    if result.action == "commit":
                        committed = True
                        self._committed(result)
                        self._session.generate_reply(
                            user_input=result.transcript,
                            input_modality="audio",
                            allow_interruptions=False,
                        )
                        self._feedback("accepted")
            except TimeoutError:
                reject("verification_timeout")
            except asyncio.CancelledError:
                raise
            except Exception:
                if committed:
                    # A dispatch failure cannot undo a completed authorization
                    # or be labelled a speaker rejection. The caller may await
                    # the task; asynchronous users still get a safe diagnostic.
                    logger.error("Verified turn dispatch failed after commit")
                    raise
                # Upstream bodies and candidate text are not diagnostic logs.
                reject("verification_unavailable")
            finally:
                self._pending = None

        self._pending = asyncio.create_task(evaluate())

        def completed(task):
            # A task cancelled before its coroutine starts does not run finally.
            if self._pending is task:
                self._pending = None
            if not task.cancelled():
                task.exception()  # observed even when scheduled from a VAD callback

        self._pending.add_done_callback(completed)
        return self._pending

    async def aclose(self) -> None:
        self._closed = True
        self.invalidate("session_closed")
        pending = self._pending
        if pending is not None:
            await asyncio.gather(pending, return_exceptions=True)
