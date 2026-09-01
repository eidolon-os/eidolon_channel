"""Transcript admission gate for full-duplex realtime sessions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from eidolon.livekit.agent.session.transcript_echo import (
    EchoEvidence,
    TranscriptEchoGate,
)

from .transcript_event import FullDuplexTranscriptEvent


@dataclass(frozen=True)
class TranscriptAdmissionDecision:
    accepted: bool
    reason: str = "accepted"
    transcript: str = ""
    speaker_id: str | None = None
    is_final: bool | None = None


class TranscriptAdmissionGate:
    """Decide whether an STT transcript should enter turn/evidence logic."""

    def __init__(
        self,
        *,
        suppress_until_next_speech: Callable[[], bool],
        agent_output_active: Callable[[str | None], bool],
        echo_gate: Callable[[], TranscriptEchoGate],
        absorb_committed_turn_revision: Callable[[str, bool], bool] | None = None,
    ) -> None:
        self._suppress_until_next_speech = suppress_until_next_speech
        self._agent_output_active = agent_output_active
        self._echo_gate = echo_gate
        self._absorb_committed_turn_revision = (
            absorb_committed_turn_revision or (lambda _transcript, _is_final: False)
        )

    def evaluate(self, event: Any) -> TranscriptAdmissionDecision:
        transcript_event = FullDuplexTranscriptEvent.from_event(event)
        transcript = transcript_event.transcript
        speaker_id = transcript_event.speaker_id
        is_final = transcript_event.is_final
        if self._suppress_until_next_speech() and transcript:
            return TranscriptAdmissionDecision(
                accepted=False,
                reason="suppressed_until_next_speech",
                transcript=transcript,
                speaker_id=speaker_id,
                is_final=is_final,
            )
        if transcript and self._absorb_committed_turn_revision(transcript, bool(is_final)):
            return TranscriptAdmissionDecision(
                accepted=False,
                reason="committed_turn_revision",
                transcript=transcript,
                speaker_id=speaker_id,
                is_final=is_final,
            )
        if transcript and self._agent_output_active(speaker_id):
            echo_evidence = self._echo_gate().classify(transcript)
            if echo_evidence is EchoEvidence.CONFIRMED:
                return TranscriptAdmissionDecision(
                    accepted=False,
                    reason="agent_echo",
                    transcript=transcript,
                    speaker_id=speaker_id,
                    is_final=is_final,
                )
            if echo_evidence is EchoEvidence.POSSIBLE:
                return TranscriptAdmissionDecision(
                    accepted=False,
                    reason="possible_agent_echo_hold",
                    transcript=transcript,
                    speaker_id=speaker_id,
                    is_final=is_final,
                )
        return TranscriptAdmissionDecision(
            accepted=True,
            transcript=transcript,
            speaker_id=speaker_id,
            is_final=is_final,
        )
