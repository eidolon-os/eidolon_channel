"""Client interaction input handling for voice sessions.

This module owns product-level client controls that arrive over LiveKit data
packets.  It deliberately does not own natural open-mic interruption policy:
full-duplex barge-in/backchannel decisions stay in ``TurnPolicyRuntime`` and
``InterruptionOrchestrator``.  The client path here is limited to explicit
controls such as PTT/tap-to-stop and half-duplex PTT turn boundaries.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from eidolon_sdk.biz.contracts import CLIENT_AUDIO_STATE_TOPIC

from eidolon.livekit.agent.client_audio_state import ClientAudioState
from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.turn_policy import (
    Action,
    Decision,
    InterruptIntent,
    TurnPolicyRuntime,
)
from eidolon.livekit.agent.session.decision_effects import DecisionEffectApplier

logger = logging.getLogger("agent")


class ExplicitClientInterruptLedger:
    """Record explicit client interrupt owner decisions on the active timeline."""

    def __init__(
        self,
        *,
        get_timeline: Callable[[], TurnTimeline | None],
        get_turn_runtime: Callable[[], TurnPolicyRuntime],
        get_decision_effects: Callable[[], DecisionEffectApplier],
        ensure_decision_effects: Callable[[], None],
    ) -> None:
        self.pending: dict[str, Any] | None = None
        self._get_timeline = get_timeline
        self._get_turn_runtime = get_turn_runtime
        self._get_decision_effects = get_decision_effects
        self._ensure_decision_effects = ensure_decision_effects

    def ptt_decision(self) -> Decision:
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
        decision = self.ptt_decision()
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
    """Handle explicit client controls and half-duplex PTT turn boundaries."""

    def __init__(
        self,
        *,
        is_half_duplex: Callable[[], bool],
        latest_client_audio_state: Callable[[str | None], ClientAudioState | None],
        agent_output_active_for_interrupts: Callable[[str | None], bool],
        ensure_ducking_controller: Callable[[], None],
        is_output_cancelled: Callable[[], bool],
        record_explicit_client_interrupt: Callable[[dict[str, Any], float], None],
        mark_explicit_client_interrupt_resolved: Callable[[float, float], None],
        cancel_agent_output: Callable[[bool], None],
        commit_ptt_release_turn: Callable[[], None],
        sync_room_data: Callable[[], None],
        get_last_ptt_held: Callable[[], bool],
        set_last_ptt_held: Callable[[bool], None],
        set_ptt_turn_had_speech: Callable[[bool], None],
    ) -> None:
        self._is_half_duplex = is_half_duplex
        self._latest_client_audio_state = latest_client_audio_state
        self._agent_output_active_for_interrupts = agent_output_active_for_interrupts
        self._ensure_ducking_controller = ensure_ducking_controller
        self._is_output_cancelled = is_output_cancelled
        self._record_explicit_client_interrupt = record_explicit_client_interrupt
        self._mark_explicit_client_interrupt_resolved = mark_explicit_client_interrupt_resolved
        self._cancel_agent_output = cancel_agent_output
        self._commit_ptt_release_turn = commit_ptt_release_turn
        self._sync_room_data = sync_room_data
        self._get_last_ptt_held = get_last_ptt_held
        self._set_last_ptt_held = set_last_ptt_held
        self._set_ptt_turn_had_speech = set_ptt_turn_had_speech

    def on_client_room_packet(self, packet: Any) -> None:
        """Run packet side effects after ``RoomDataHandler`` stores state."""
        self._sync_room_data()
        self.handle_explicit_client_interrupt(packet)
        self.handle_ptt_turn_edges(packet)

    def mark_recognized_speech(self, text: str) -> None:
        """Arm the half-duplex empty-press guard when real transcript arrives."""
        if self._is_half_duplex() and text.strip():
            self._set_ptt_turn_had_speech(True)

    def handle_ptt_turn_edges(self, packet: Any) -> None:
        """Commit exactly one half-duplex user turn on the PTT release edge."""
        if not self._is_half_duplex():
            return
        if getattr(packet, "topic", None) != CLIENT_AUDIO_STATE_TOPIC:
            return
        participant = getattr(packet, "participant", None)
        identity = getattr(participant, "identity", "") or None
        state = self._latest_client_audio_state(identity)
        # Only react to the participant that actually published a PTT-bearing
        # audio state.  A stray packet must not fake a release while the device
        # is still held.
        if state is None:
            return
        held = bool(state.ptt)
        was_held = self._get_last_ptt_held()
        self._set_last_ptt_held(held)
        if held and not was_held:
            self._set_ptt_turn_had_speech(False)
            return
        if was_held and not held:
            self._commit_ptt_release_turn()

    def handle_explicit_client_interrupt(self, packet: Any) -> None:
        """Hard-cut agent output for deliberate PTT/tap-to-stop controls."""
        if getattr(packet, "topic", None) != CLIENT_AUDIO_STATE_TOPIC:
            return
        participant = getattr(packet, "participant", None)
        identity = getattr(participant, "identity", "") or None
        state = self._latest_client_audio_state(identity)
        # PTT is the only explicit client interrupt.  Open-mic barge-in is a
        # server-side owner decision from transcript/attention evidence.
        if state is None or not state.ptt:
            return
        if not self._agent_output_active_for_interrupts(identity):
            return
        self._ensure_ducking_controller()
        if self._is_output_cancelled():
            return

        logger.info(
            "[ClientInteractionHandler] explicit client PTT interrupt received "
            "identity=%s playback=%s",
            state.participant_identity,
            state.playback_state,
        )
        received_at = time.monotonic()
        self._record_explicit_client_interrupt(
            state.as_timeline_attr(),
            received_at,
        )
        self._cancel_agent_output(True)
        self._mark_explicit_client_interrupt_resolved(
            received_at,
            time.monotonic(),
        )
