"""Transcript admission gate for full-duplex realtime sessions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from eidolon.livekit.agent.session.transcript_echo import TranscriptEchoGate


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
    ) -> None:
        self._suppress_until_next_speech = suppress_until_next_speech
        self._agent_output_active = agent_output_active
        self._echo_gate = echo_gate

    def evaluate(self, event: Any) -> TranscriptAdmissionDecision:
        transcript = getattr(event, "transcript", "") or ""
        speaker_id = getattr(event, "speaker_id", None)
        is_final = getattr(event, "is_final", None)
        if self._suppress_until_next_speech() and transcript:
            return TranscriptAdmissionDecision(
                accepted=False,
                reason="suppressed_until_next_speech",
                transcript=transcript,
                speaker_id=speaker_id,
                is_final=is_final,
            )
        if (
            transcript
            and self._agent_output_active(speaker_id)
            and self._echo_gate().is_echo(transcript)
        ):
            return TranscriptAdmissionDecision(
                accepted=False,
                reason="agent_echo",
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
