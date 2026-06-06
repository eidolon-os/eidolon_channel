"""Pure attention-admission policy before duck/interruption side effects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from eidolon.livekit.agent.client_audio_state import (
    PLAYBACK_STATE_AGENT_SPEAKING,
    ClientAudioState,
)
from eidolon.livekit.common.config import TurnPolicyConfig

from .constants import TRANSCRIPT_PREVIEW_MAX_CHARS
from .intent_classifier import InterruptIntent, LexiconInterruptClassifier


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


@dataclass(frozen=True)
class AttentionInput:
    agent_speaking: bool
    client_state: ClientAudioState | None = None
    transcript: str = ""


class AttentionAdmission:
    """Gate raw VAD/STT activity before output ducking or cancellation.

    The default is intentionally conservative for compatibility: without a
    fresh client audio-state signal it preserves the existing duck/decide path.
    """

    def __init__(self, config: TurnPolicyConfig) -> None:
        self._config = config.attention
        self._classifier = LexiconInterruptClassifier(config.interrupt)

    def decide(self, signal: AttentionInput) -> AttentionDecision:
        text = signal.transcript.strip()
        preview = _preview(text)
        if not self._config.enabled:
            return AttentionDecision(
                AdmissionAction.DUCK_AND_DECIDE,
                "attention_disabled",
                transcript_preview=preview,
                client_state_used=signal.client_state is not None,
            )
        if not signal.agent_speaking:
            return AttentionDecision(
                AdmissionAction.DUCK_AND_DECIDE,
                "agent_not_speaking",
                transcript_preview=preview,
                client_state_used=signal.client_state is not None,
            )

        client = signal.client_state
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
        if client.manual_interrupt or client.ptt:
            return AttentionDecision(
                AdmissionAction.HARD_INTERRUPT,
                "explicit_client_interrupt",
                transcript_preview=preview,
                client_state_used=True,
            )

        if text:
            intent = self._classifier.classify(
                text,
                vad_active=True,
                agent_speaking=signal.agent_speaking,
                eot_score=0.0,
            ).intent
            if intent is InterruptIntent.HARD_STOP:
                return AttentionDecision(
                    AdmissionAction.HARD_INTERRUPT,
                    "transcript_hard_stop",
                    transcript_preview=preview,
                    client_state_used=True,
                )
            if intent in (InterruptIntent.TOPIC_SWITCH, InterruptIntent.CORRECTION):
                return AttentionDecision(
                    AdmissionAction.DUCK_AND_DECIDE,
                    f"transcript_{intent.value}",
                    transcript_preview=preview,
                    client_state_used=True,
                )

        if (
            self._config.require_direct_signal_during_playback
            and client.playback_state == PLAYBACK_STATE_AGENT_SPEAKING
        ):
            return AttentionDecision(
                AdmissionAction.OBSERVE,
                "client_playback_active_without_direct_signal",
                transcript_preview=preview,
                client_state_used=True,
            )

        return AttentionDecision(
            AdmissionAction.DUCK_AND_DECIDE,
            "client_playback_idle",
            transcript_preview=preview,
            client_state_used=True,
        )


def _preview(text: str) -> str:
    return text[:TRANSCRIPT_PREVIEW_MAX_CHARS]
