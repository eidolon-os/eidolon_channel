"""Interaction-mode behavior for a StreamingPipeline session.

The session metadata decides whether the user experience is push-to-talk
half-duplex or natural full-duplex. Keep those product-mode differences in one
place so StreamingPipeline remains mostly a LiveKit AgentSession wrapper.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from eidolon_sdk.biz.contracts import (
    INTERACTION_MODE_FULL_DUPLEX,
    INTERACTION_MODE_HALF_DUPLEX,
)

from eidolon.livekit.common.config import TurnPolicyConfig


class InteractionModeBehavior:
    """Base behavior contract for mode-specific session decisions."""

    name = INTERACTION_MODE_FULL_DUPLEX
    is_half_duplex = False

    def turn_detection(self, get_eot_model: Callable[[], Any]) -> Any:
        """Return the Agent turn_detection option for this mode."""
        return get_eot_model()

    def interruption_options(
        self,
        *,
        allow_interruptions: bool,
        false_interruption_timeout: float | None,
        turn_policy: TurnPolicyConfig,
    ) -> dict[str, Any]:
        """Return AgentSession turn_handling.interruption options."""
        interruption: dict[str, Any] = {
            "enabled": allow_interruptions,
            "discard_audio_if_uninterruptible": True,
            "false_interruption_timeout": false_interruption_timeout,
        }
        if self.uses_livekit_native_adaptive_interruption(
            turn_policy=turn_policy,
            allow_interruptions=allow_interruptions,
        ):
            interruption["mode"] = "adaptive"
            interruption["resume_false_interruption"] = True
        return interruption

    def uses_livekit_native_adaptive_interruption(
        self,
        *,
        turn_policy: TurnPolicyConfig | None,
        allow_interruptions: bool,
    ) -> bool:
        return getattr(
            turn_policy, "interruption_owner", "channel"
        ) == "livekit_native_adaptive" and bool(allow_interruptions)

    def treats_vad_silence_as_turn_boundary(self) -> bool:
        return True

    def starts_natural_interruption_candidate(self) -> bool:
        return True

    def applies_agent_echo_gate(self) -> bool:
        return True

    def allows_low_eot_defer(self) -> bool:
        return True

    def allows_playback_low_evidence_reject(self) -> bool:
        return True


class FullDuplexInteractionMode(InteractionModeBehavior):
    """Open mic with server-owned barge-in, backchannel, and echo evidence."""

    name = INTERACTION_MODE_FULL_DUPLEX
    is_half_duplex = False


class HalfDuplexInteractionMode(InteractionModeBehavior):
    """Push-to-talk mode with explicit button turn boundaries."""

    name = INTERACTION_MODE_HALF_DUPLEX
    is_half_duplex = True

    def turn_detection(self, get_eot_model: Callable[[], Any]) -> str:
        return "manual"

    def interruption_options(
        self,
        *,
        allow_interruptions: bool,
        false_interruption_timeout: float | None,
        turn_policy: TurnPolicyConfig,
    ) -> dict[str, Any]:
        return {
            "enabled": allow_interruptions,
            "discard_audio_if_uninterruptible": False,
            "false_interruption_timeout": false_interruption_timeout,
        }

    def uses_livekit_native_adaptive_interruption(
        self,
        *,
        turn_policy: TurnPolicyConfig | None,
        allow_interruptions: bool,
    ) -> bool:
        return False

    def treats_vad_silence_as_turn_boundary(self) -> bool:
        return False

    def starts_natural_interruption_candidate(self) -> bool:
        return False

    def applies_agent_echo_gate(self) -> bool:
        return False

    def allows_low_eot_defer(self) -> bool:
        return False

    def allows_playback_low_evidence_reject(self) -> bool:
        return False


def build_interaction_mode_behavior(mode: str) -> InteractionModeBehavior:
    """Build a behavior object for a session interaction mode."""
    if mode == INTERACTION_MODE_HALF_DUPLEX:
        return HalfDuplexInteractionMode()
    return FullDuplexInteractionMode()
