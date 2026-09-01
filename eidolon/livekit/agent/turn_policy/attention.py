"""Pure attention-admission policy before duck/interruption side effects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from eidolon_sdk.biz.contracts import PLAYBACK_STATE_AGENT_SPEAKING

from eidolon.livekit.agent.integration.client_audio_state import ClientAudioState
from eidolon.livekit.common.config import TurnPolicyConfig

from .constants import TRANSCRIPT_PREVIEW_MAX_CHARS
from .evidence import TranscriptEvidenceGate


class AdmissionAction(str, Enum):
    IGNORE = "ignore"
    OBSERVE = "observe"
    DUCK_AND_DECIDE = "duck_and_decide"
    HARD_INTERRUPT = "hard_interrupt"


@dataclass(frozen=True)
class AttentionDecision:
    action: AdmissionAction
    reason: str
    transcript_preview: str = ""
    client_state_used: bool = False
    tier: str = ""
    tier_reason: str = ""
    evidence_reason: str = ""


@dataclass(frozen=True)
class AttentionInput:
    agent_speaking: bool
    client_state: ClientAudioState | None = None
    transcript: str = ""
    eot_score: float = 0.0
    speech_started: bool = False


class AttentionAdmission:
    """Gate raw VAD/STT activity before output ducking or cancellation.

    Channel owns interruption orchestration. Client audio state is treated as a
    useful hint, not as permission to start the server-side soft-duck path.
    """

    def __init__(self, config: TurnPolicyConfig) -> None:
        self._config = config.attention
        self._evidence_gate = TranscriptEvidenceGate(config.interrupt)

    def decide(self, signal: AttentionInput) -> AttentionDecision:
        text = signal.transcript.strip()
        preview = _preview(text)
        eot_score = _numeric_score(signal.eot_score)
        if not self._config.enabled:
            return AttentionDecision(
                AdmissionAction.DUCK_AND_DECIDE,
                "attention_disabled",
                transcript_preview=preview,
                client_state_used=signal.client_state is not None,
            )
        client = signal.client_state
        client_playback_active = (
            client is not None and client.playback_state == PLAYBACK_STATE_AGENT_SPEAKING
        )
        if not (signal.agent_speaking or client_playback_active):
            return AttentionDecision(
                AdmissionAction.DUCK_AND_DECIDE,
                "agent_not_speaking",
                transcript_preview=preview,
                client_state_used=signal.client_state is not None,
            )

        if client is None:
            return AttentionDecision(
                AdmissionAction.DUCK_AND_DECIDE,
                "no_client_state",
                transcript_preview=preview,
            )

        if self._config.ignore_when_mic_muted and client.mic_muted:
            return AttentionDecision(
                AdmissionAction.IGNORE,
                "client_mic_muted",
                transcript_preview=preview,
                client_state_used=True,
            )
        if client.ptt:
            # PTT is a deliberate button press — trust it and hard-cut.
            return AttentionDecision(
                AdmissionAction.HARD_INTERRUPT,
                "explicit_client_ptt",
                transcript_preview=preview,
                client_state_used=True,
            )
        # manual_interrupt is the device energy-gate barge-in guess, which residual
        # playback echo can falsely trip. Do NOT hard-cut on the signal alone; fall
        # through to the transcript-evidence gate so only real near-end content cuts.

        if signal.speech_started and not text:
            if self._config.soft_duck_on_playback_speech_start:
                return AttentionDecision(
                    AdmissionAction.DUCK_AND_DECIDE,
                    "playback_speech_start_soft_duck",
                    transcript_preview=preview,
                    client_state_used=True,
                )
            if self._config.require_direct_signal_during_playback:
                return AttentionDecision(
                    AdmissionAction.OBSERVE,
                    "playback_speech_start_soft_duck_disabled",
                    transcript_preview=preview,
                    client_state_used=True,
                )

        if not text:
            return AttentionDecision(
                AdmissionAction.OBSERVE,
                "playback_empty_transcript",
                transcript_preview=preview,
                client_state_used=True,
            )

        if text:
            evidence = self._evidence_gate.evaluate_attention(
                text,
                eot_score=eot_score,
            )
            if evidence.allow_decision and evidence.reason == "high_eot_transcript":
                return AttentionDecision(
                    AdmissionAction.DUCK_AND_DECIDE,
                    f"transcript_evidence:{evidence.reason}",
                    transcript_preview=preview,
                    client_state_used=True,
                    evidence_reason=evidence.reason,
                )
            if self._config.require_direct_signal_during_playback:
                return AttentionDecision(
                    AdmissionAction.OBSERVE,
                    f"playback_low_evidence_transcript:{evidence.reason}",
                    transcript_preview=preview,
                    client_state_used=True,
                    evidence_reason=evidence.reason,
                )

        return AttentionDecision(
            AdmissionAction.DUCK_AND_DECIDE,
            "client_playback_idle",
            transcript_preview=preview,
            client_state_used=True,
        )


def _preview(text: str) -> str:
    return text[:TRANSCRIPT_PREVIEW_MAX_CHARS]


def _numeric_score(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0
