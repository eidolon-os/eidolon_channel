"""AgentSession turn-handling config for the full-duplex runtime."""

from __future__ import annotations

from typing import Any

from eidolon.livekit.common.config import TurnPolicyConfig


def uses_livekit_native_adaptive_interruption(
    *,
    turn_policy: TurnPolicyConfig | None,
    allow_interruptions: bool,
) -> bool:
    return (
        getattr(turn_policy, "interruption_owner", "channel")
        == "livekit_native_adaptive"
        and bool(allow_interruptions)
    )


def build_full_duplex_turn_handling(
    *,
    turn_policy: TurnPolicyConfig,
    allow_interruptions: bool,
    false_interruption_timeout: float | None,
) -> dict[str, Any]:
    """Build ``AgentSession(turn_handling=...)`` for full-duplex mode."""

    interruption: dict[str, Any] = {
        "enabled": allow_interruptions,
        "discard_audio_if_uninterruptible": True,
        "false_interruption_timeout": false_interruption_timeout,
    }
    if uses_livekit_native_adaptive_interruption(
        turn_policy=turn_policy,
        allow_interruptions=allow_interruptions,
    ):
        interruption["mode"] = "adaptive"
        interruption["resume_false_interruption"] = True

    return {
        "interruption": interruption,
        "preemptive_generation": {
            "enabled": turn_policy.preemptive.enabled,
            "preemptive_tts": turn_policy.preemptive.preemptive_tts,
        },
    }
