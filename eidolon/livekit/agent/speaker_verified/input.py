"""Route existing VAD and mapped ASR events to one candidate owner.

No capture, recognizer or EOT model lives here. Timers only bound collection
and apply the configured speech-settlement grace before calling the owner.
"""

import asyncio
import math

from livekit.agents import stt

from ...common.transcript_evidence import transcript_evidence_from_speech_event
from .orchestration import SpeakerVerifiedTurns


class SpeakerVerifiedInput:
    def __init__(
        self,
        turns: SpeakerVerifiedTurns,
        *,
        coordinator,
        settle_sec: float,
        candidate_timeout_sec: float = 30,
    ):
        if (
            not math.isfinite(settle_sec)
            or not 0 <= settle_sec <= 5
            or not math.isfinite(candidate_timeout_sec)
            or not settle_sec < candidate_timeout_sec <= 60
        ):
            raise ValueError("bounded collection and settlement required")
        self._turns, self._owner = turns, coordinator
        self._settle_sec, self._timeout = settle_sec, candidate_timeout_sec
        self._speaking = False
        self._closed = False
        self._settle = None
        self._deadline = None
        self._deadline_candidate = None
        self._version = 0

    def _arm_deadline(self):
        candidate = self._owner.active
        if candidate is None or candidate.candidate_id == self._deadline_candidate:
            return
        if self._deadline:
            self._deadline.cancel()
        self._deadline_candidate = candidate.candidate_id
        self._deadline = asyncio.get_running_loop().call_later(
            self._timeout, self.invalidate, "candidate_timeout"
        )

    def user_state_changed(self, event):
        if self._closed:
            return
        if event.new_state == "speaking":
            self._speaking = True
            self._cancel_settlement()
            self._owner.start_speech(timeline=None, continue_pending=True)
            self._arm_deadline()
            self._version += 1
        elif event.old_state == "speaking":
            self._speaking = False
            self._owner.note_speech_stopped(eot_score=None)
            self._version += 1
            self._schedule()

    def observe_speech(self, event: stt.SpeechEvent):
        if (
            self._closed
            or event.type
            not in {
                stt.SpeechEventType.INTERIM_TRANSCRIPT,
                stt.SpeechEventType.FINAL_TRANSCRIPT,
            }
            or not event.alternatives
        ):
            return
        evidence = transcript_evidence_from_speech_event(event)
        if self._owner.is_retired_revision(evidence):
            return
        candidate = self._owner.active
        if candidate is None or candidate.state != "open":
            # Wording alone cannot reopen a completed turn. A new mapped
            # provider revision or a new VAD segment must establish ownership.
            if not self._owner.transcript_starts_new_turn(evidence):
                return
            self._owner.start_speech(timeline=None)
            self._arm_deadline()
        final = event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
        text = event.alternatives[0].text
        if not text.strip() and final:
            self._owner.withdraw_transcript_revision(evidence)
        else:
            self._owner.add_transcript(text, is_final=final, evidence=evidence)
        self._version += 1
        if not self._speaking:
            self._owner.note_speech_stopped(eot_score=None)
            self._schedule()

    def _cancel_settlement(self):
        if self._settle:
            self._settle.cancel()
            self._settle = None

    def _schedule(self):
        self._cancel_settlement()
        if not self._closed and not self._speaking:
            self._settle = asyncio.get_running_loop().call_later(self._settle_sec, self._consider)

    def _consider(self):
        self._settle = None
        if self._closed or self._speaking:
            return
        version = self._version
        pending = self._turns.pending
        task = self._turns.consider() if pending is None else pending
        if task is not None and pending is None:

            def settled(_):
                candidate = self._owner.active
                if candidate is not None and candidate.state != "open":
                    if self._deadline:
                        self._deadline.cancel()
                elif self._version != version:
                    # Only changed evidence gets another attempt, after the
                    # single in-flight task releases its capacity.
                    self._schedule()

            task.add_done_callback(settled)

    def invalidate(self, reason):
        self._cancel_settlement()
        if self._deadline:
            self._deadline.cancel()
        self._deadline_candidate = None
        self._version += 1
        self._turns.invalidate(reason)

    async def aclose(self):
        self._closed = True
        self.invalidate("session_closed")
        await self._turns.aclose()
