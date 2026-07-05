"""Apply attention-admission decisions around interruption hot path."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from eidolon.livekit.agent.integration.client_audio_state import ClientAudioState
from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.turn_policy import (
    AdmissionAction,
    AttentionDecision,
    AttentionInput,
    TurnPolicyRuntime,
)
from eidolon.livekit.common.config import TurnPolicyConfig

logger = logging.getLogger("agent.session.attention_effects")


class AttentionEffectHandler:
    """Gate hot-path EOT checks and execute attention side effects."""

    def __init__(
        self,
        *,
        turn_policy: TurnPolicyConfig,
        turn_runtime: TurnPolicyRuntime,
        get_agent_speaking: Callable[[], bool],
        get_duck_active: Callable[[], bool],
        latest_client_audio_state: Callable[[str | None], ClientAudioState | None],
        get_timeline: Callable[[], TurnTimeline | None],
        on_duck: Callable[[], bool | None],
        on_interrupt: Callable[[], None],
        get_eot_score: Callable[[], float] | None = None,
    ) -> None:
        self._turn_policy = turn_policy
        self._turn_runtime = turn_runtime
        self._get_agent_speaking = get_agent_speaking
        self._get_duck_active = get_duck_active
        self._latest_client_audio_state = latest_client_audio_state
        self._get_eot_score = get_eot_score or (lambda: 0.0)
        self._get_timeline = get_timeline
        self._on_duck = on_duck
        self._on_interrupt = on_interrupt

    def handle_speaking_started(self) -> bool:
        decision, state = self._decide_with_state("", speech_started=True)
        self.record_admission(decision, state=state)
        if decision.reason == "agent_not_speaking":
            logger.info(
                "[AttentionEffectHandler] attention admission: %s reason=%s; no duck",
                decision.action.value,
                decision.reason,
            )
            return False
        if not self._turn_policy.attention.enforce:
            return bool(self._on_duck())
        if decision.action is AdmissionAction.HARD_INTERRUPT:
            timeline = self._get_timeline()
            if timeline is not None:
                timeline.mark("interrupt_started_at")
            self._on_interrupt()
            return False
        if decision.action is AdmissionAction.DUCK_AND_DECIDE:
            return bool(self._on_duck())
        logger.info(
            "[AttentionEffectHandler] attention admission: %s reason=%s; no duck",
            decision.action.value,
            decision.reason,
        )
        return False

    def allows_eot_check(
        self,
        transcript: str,
        *,
        speaker_id: str | None = None,
    ) -> bool:
        decision, state = self._decide_with_state(
            transcript,
            participant_identity=speaker_id,
        )
        self.record_admission(decision, state=state)
        self._mark_direct_intent_admission(decision)
        if not self._turn_policy.attention.enforce:
            return True
        if decision.action is AdmissionAction.HARD_INTERRUPT:
            return True
        if decision.action is AdmissionAction.DUCK_AND_DECIDE:
            if self._get_agent_speaking() and not self._get_duck_active():
                self._on_duck()
            return True
        if decision.action is AdmissionAction.OBSERVE and self._get_duck_active():
            logger.info(
                "[AttentionEffectHandler] attention admission: %s reason=%s; "
                "duck active, route transcript as interruption evidence",
                decision.action.value,
                decision.reason,
            )
            return True
        logger.info(
            "[AttentionEffectHandler] attention admission: %s reason=%s; skip EOT",
            decision.action.value,
            decision.reason,
        )
        return False

    def decide(
        self,
        transcript: str,
        *,
        participant_identity: str | None = None,
        speech_started: bool = False,
    ) -> AttentionDecision:
        signal = self._attention_input(
            transcript,
            participant_identity=participant_identity,
            speech_started=speech_started,
        )
        return self._turn_runtime.admit_attention(signal)

    def record_admission(
        self,
        decision: AttentionDecision,
        *,
        state: dict[str, object] | None = None,
    ) -> None:
        timeline = self._get_timeline()
        if timeline is None:
            return
        payload = {
            "action": decision.action.value,
            "reason": decision.reason,
            "transcript_preview": decision.transcript_preview,
            "client_state_used": decision.client_state_used,
            "tier": decision.tier or None,
            "tier_reason": decision.tier_reason or None,
            "enforced": self._turn_policy.attention.enforce,
        }
        if state:
            payload["state"] = dict(state)
        timeline.set_attr("attention_admission", payload)
        events = list(timeline.attrs.get("attention_admission_events") or ())
        events.append(payload)
        timeline.set_attr("attention_admission_events", events)

    def _mark_direct_intent_admission(self, decision: AttentionDecision) -> None:
        if not _is_direct_intent_admission(decision):
            return
        timeline = self._get_timeline()
        if timeline is not None:
            timeline.mark("interrupt_intent_admitted_at")

    def _decide_with_state(
        self,
        transcript: str,
        *,
        participant_identity: str | None = None,
        speech_started: bool = False,
    ) -> tuple[AttentionDecision, dict[str, object]]:
        signal = self._attention_input(
            transcript,
            participant_identity=participant_identity,
            speech_started=speech_started,
        )
        decision = self._turn_runtime.admit_attention(signal)
        return decision, self._attention_state(signal)

    def _attention_input(
        self,
        transcript: str,
        *,
        participant_identity: str | None,
        speech_started: bool,
    ) -> AttentionInput:
        return AttentionInput(
            agent_speaking=self._get_agent_speaking(),
            client_state=self._latest_client_audio_state(participant_identity),
            transcript=transcript,
            eot_score=self._get_eot_score(),
            speech_started=speech_started,
        )

    def _attention_state(self, signal: AttentionInput) -> dict[str, object]:
        client = signal.client_state
        state: dict[str, object] = {
            "agent_speaking": signal.agent_speaking,
            "duck_active": self._get_duck_active(),
            "eot_score": signal.eot_score,
            "speech_started": signal.speech_started,
            "client_state_present": client is not None,
        }
        if client is None:
            return state

        now = time.monotonic()
        max_age_ms = self._turn_policy.attention.client_state_max_age_ms
        max_age_sec = max_age_ms / 1000.0
        client_age_ms = max(0.0, (now - client.received_at) * 1000.0)
        state.update(
            {
                "participant_identity": client.participant_identity,
                "client_input_mode": client.input_mode,
                "client_playback_state": client.playback_state,
                "client_ptt": client.ptt,
                "client_manual_interrupt": client.manual_interrupt,
                "client_mic_muted": client.mic_muted,
                "client_state_age_ms": round(client_age_ms),
                "client_state_fresh": client.is_fresh(
                    now=now,
                    max_age_sec=max_age_sec,
                ),
                "client_state_max_age_ms": max_age_ms,
                "client_rms": _optional_number(client.rms),
                "client_snr_hint": _optional_number(client.snr_hint),
            }
        )
        return state


def _is_direct_intent_admission(decision: AttentionDecision) -> bool:
    if decision.action is AdmissionAction.HARD_INTERRUPT:
        return True
    return decision.reason in {
        "explicit_client_interrupt",
        "transcript_hard_stop",
        "transcript_topic_switch",
        "transcript_correction",
    }


def _optional_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
