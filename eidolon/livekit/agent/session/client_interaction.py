"""Client interaction input handling for voice sessions.

This module owns product-level client controls that arrive over LiveKit data
packets.  It deliberately does not own natural open-mic interruption policy:
full-duplex barge-in/backchannel decisions stay in ``TurnPolicyRuntime`` and
``InterruptionOrchestrator``.  The client path here is limited to explicit
controls such as PTT/tap-to-stop. Half-duplex turn ownership lives in
``agent.half_duplex``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from eidolon_sdk.biz.contracts import CLIENT_AUDIO_STATE_TOPIC

from eidolon.livekit.agent.integration.client_audio_state import ClientAudioState
from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.turn_policy import (
    Action,
    Decision,
    InterruptIntent,
    TurnPolicyRuntime,
)

if TYPE_CHECKING:
    from eidolon.livekit.agent.session.decision_effects import DecisionEffectApplier

logger = logging.getLogger("agent")


class ExplicitClientInterruptLedger:
    """Record explicit client interrupt owner decisions on the active timeline."""

    def __init__(
        self,
        *,
        get_timeline: Callable[[], TurnTimeline | None],
        get_turn_runtime: Callable[[], TurnPolicyRuntime],
        get_decision_effects: Callable[[], "DecisionEffectApplier"],
        ensure_decision_effects: Callable[[], None],
    ) -> None:
        self.pending: dict[str, Any] | None = None
        self._get_timeline = get_timeline
        self._get_turn_runtime = get_turn_runtime
        self._get_decision_effects = get_decision_effects
        self._ensure_decision_effects = ensure_decision_effects

    def explicit_interrupt_decision(self) -> Decision:
        return Decision(
            action=Action.CANCEL,
            reason="explicit_client_ptt",
            intent=InterruptIntent.HARD_STOP,
            intent_source="client_ptt",
            intent_confidence=1.0,
        )

    def record(
        self,
        *,
        state_attr: dict[str, Any],
        received_at: float,
        resolved_at: float | None = None,
        timeline: TurnTimeline | None = None,
    ) -> None:
        timeline = timeline or self._get_timeline()
        if timeline is None:
            self.pending = {
                "state_attr": dict(state_attr),
                "received_at": received_at,
                "resolved_at": resolved_at,
            }
            return

        self._ensure_decision_effects()
        timeline.set_attr("explicit_client_interrupt", dict(state_attr))
        timeline.mark_at("interrupt_started_at", received_at)
        if resolved_at is not None:
            timeline.mark_at("interrupt_resolved_at", resolved_at)
            timeline.set_attr("cancel_reason", "explicit_client_ptt")
        decision = self.explicit_interrupt_decision()
        self._get_decision_effects().record_decision_attrs(
            decision,
            source="client_ptt",
            resolved_reason="explicit_client_ptt",
            vad_active=None,
        )
        timeline.set_attr(
            "turn_control",
            self._get_turn_runtime().control_signal_from_decision(decision).as_metadata(),
        )

    def apply_pending(self, timeline: TurnTimeline | None = None) -> None:
        pending = self.pending
        if not pending:
            return
        timeline = timeline or self._get_timeline()
        if timeline is None:
            return
        self.pending = None
        self.record(
            state_attr=pending.get("state_attr") or {},
            received_at=float(pending.get("received_at") or time.monotonic()),
            resolved_at=(
                float(pending["resolved_at"])
                if isinstance(pending.get("resolved_at"), (int, float))
                else None
            ),
            timeline=timeline,
        )

    def mark_resolved(self, received_at: float, resolved_at: float) -> None:
        pending = self.pending
        if (
            pending is not None
            and isinstance(pending.get("received_at"), (int, float))
            and float(pending["received_at"]) == received_at
        ):
            pending["resolved_at"] = resolved_at


class ClientInteractionHandler:
    """Handle explicit client controls that can preempt agent output."""

    def __init__(
        self,
        *,
        latest_client_audio_state: Callable[[str | None], ClientAudioState | None],
        agent_output_active_for_interrupts: Callable[[str | None], bool],
        ensure_ducking_controller: Callable[[], None],
        is_output_cancelled: Callable[[], bool],
        record_explicit_client_interrupt: Callable[[dict[str, Any], float], None],
        mark_explicit_client_interrupt_resolved: Callable[[float, float], None],
        cancel_agent_output: Callable[[bool], None],
        agent_turn_active_for_explicit_preempt: Callable[[str | None], bool] | None = None,
        preempt_agent_turn_for_explicit_control: Callable[[], None] | None = None,
    ) -> None:
        self._latest_client_audio_state = latest_client_audio_state
        self._agent_output_active_for_interrupts = agent_output_active_for_interrupts
        self._agent_turn_active_for_explicit_preempt = (
            agent_turn_active_for_explicit_preempt or agent_output_active_for_interrupts
        )
        self._ensure_ducking_controller = ensure_ducking_controller
        self._is_output_cancelled = is_output_cancelled
        self._record_explicit_client_interrupt = record_explicit_client_interrupt
        self._mark_explicit_client_interrupt_resolved = mark_explicit_client_interrupt_resolved
        self._cancel_agent_output = cancel_agent_output
        self._preempt_agent_turn_for_explicit_control = (
            preempt_agent_turn_for_explicit_control or (lambda: cancel_agent_output(True))
        )

    def on_client_room_packet(self, packet: Any) -> None:
        """Run packet side effects after ``RoomDataHandler`` stores state."""
        self.handle_explicit_client_interrupt(packet)

    def handle_explicit_client_interrupt(self, packet: Any) -> None:
        """Preempt the active agent turn for deliberate client controls."""
        if getattr(packet, "topic", None) != CLIENT_AUDIO_STATE_TOPIC:
            return
        participant = getattr(packet, "participant", None)
        identity = getattr(participant, "identity", "") or None
        state = self._latest_client_audio_state(identity)
        # PTT is the only explicit client interrupt.  Open-mic barge-in is a
        # server-side owner decision from transcript/attention evidence.
        if state is None or not state.ptt:
            return
        if not self._agent_turn_active_for_explicit_preempt(identity):
            return
        self._ensure_ducking_controller()
        if self._is_output_cancelled():
            return

        logger.info(
            "[ClientInteractionHandler] explicit client preempt received "
            "identity=%s playback=%s",
            state.participant_identity,
            state.playback_state,
        )
        received_at = time.monotonic()
        self._record_explicit_client_interrupt(
            state.as_timeline_attr(),
            received_at,
        )
        self._preempt_agent_turn_for_explicit_control()
        self._mark_explicit_client_interrupt_resolved(
            received_at,
            time.monotonic(),
        )
