"""Apply attention-admission decisions around interruption hot path."""

from __future__ import annotations

import logging
from collections.abc import Callable

from eidolon.livekit.agent.client_audio_state import ClientAudioState
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
        on_duck: Callable[[], None],
        on_interrupt: Callable[[], None],
    ) -> None:
        self._turn_policy = turn_policy
        self._turn_runtime = turn_runtime
        self._get_agent_speaking = get_agent_speaking
        self._get_duck_active = get_duck_active
        self._latest_client_audio_state = latest_client_audio_state
        self._get_timeline = get_timeline
        self._on_duck = on_duck
        self._on_interrupt = on_interrupt

    def handle_speaking_started(self) -> None:
        decision = self.decide("")
        self.record_admission(decision)
        if not self._turn_policy.attention.enforce:
            self._on_duck()
            return
        if decision.action is AdmissionAction.HARD_INTERRUPT:
            timeline = self._get_timeline()
            if timeline is not None:
                timeline.mark("interrupt_started_at")
            self._on_interrupt()
            return
        if decision.action is AdmissionAction.DUCK_AND_DECIDE:
            self._on_duck()
            return
        logger.info(
            "[AttentionEffectHandler] attention admission: %s reason=%s; no duck",
            decision.action.value,
            decision.reason,
        )

    def allows_eot_check(
        self,
        transcript: str,
        *,
        speaker_id: str | None = None,
    ) -> bool:
        decision = self.decide(transcript, participant_identity=speaker_id)
        self.record_admission(decision)
        self._mark_direct_intent_admission(decision)
        if not self._turn_policy.attention.enforce:
            return True
        if decision.action is AdmissionAction.HARD_INTERRUPT:
            return True
        if decision.action is AdmissionAction.DUCK_AND_DECIDE:
            if self._get_agent_speaking() and not self._get_duck_active():
                self._on_duck()
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
    ) -> AttentionDecision:
        return self._turn_runtime.admit_attention(
            AttentionInput(
                agent_speaking=self._get_agent_speaking(),
                client_state=self._latest_client_audio_state(participant_identity),
                transcript=transcript,
            )
        )

    def record_admission(self, decision: AttentionDecision) -> None:
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


def _is_direct_intent_admission(decision: AttentionDecision) -> bool:
    if decision.action is AdmissionAction.HARD_INTERRUPT:
        return True
    return decision.reason in {
        "explicit_client_interrupt",
        "transcript_hard_stop",
        "transcript_topic_switch",
        "transcript_correction",
    }
