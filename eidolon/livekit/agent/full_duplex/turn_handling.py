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
    avatar_mode: bool = False,
) -> dict[str, Any]:
    """Build ``AgentSession(turn_handling=...)`` for full-duplex mode.

    ``avatar_mode``: when the session's audio output is a
    ``DataStreamAudioOutput`` (video avatar worker), the framework's
    pause-based ``resume_false_interruption`` cannot run — that sink does not
    support ``pause`` (it logs a warning and no-ops). We disable it explicitly;
    the channel's own duck/soft-resume still applies (it propagates through the
    worker, which re-renders whatever PCM it receives).
    """

    interruption: dict[str, Any] = {
        "enabled": allow_interruptions,
        "discard_audio_if_uninterruptible": True,
        "false_interruption_timeout": false_interruption_timeout,
    }
    if avatar_mode:
        interruption["resume_false_interruption"] = False
    elif uses_livekit_native_adaptive_interruption(
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
