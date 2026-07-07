"""Turn-policy to EOT model configuration mapping."""

from __future__ import annotations

from typing import Any

from eidolon.livekit.common.config.schema import TurnPolicyConfig


def eot_kwargs_from_turn_policy(
    turn_policy: TurnPolicyConfig | None,
) -> dict[str, Any]:
    """Return ``ChineseModel`` kwargs derived from the effective turn policy."""

    if turn_policy is None:
        return {}
    return {
        "eot_unlikely_threshold": turn_policy.eot.eot_unlikely_threshold,
        "tail_hang_silence_sec": turn_policy.eot.tail_hang_silence_ms / 1000.0,
        "min_speech_duration_sec": turn_policy.vad.min_speech_duration_ms / 1000.0,
        "duck_enabled": turn_policy.ducking.enabled,
        "duck_fade_ms": turn_policy.ducking.fade_out_ms,
        "duck_fade_in_ms": turn_policy.ducking.fade_in_ms,
        "duck_suspend_volume": turn_policy.ducking.suspend_volume,
        "duck_suspended_passthrough_enabled": (
            turn_policy.ducking.suspended_passthrough_enabled
        ),
        "duck_suspended_passthrough_volume": (
            turn_policy.ducking.suspended_passthrough_volume
        ),
        "duck_buffer_max_sec": turn_policy.ducking.buffer_max_ms / 1000.0,
        "duck_suspend_timeout_sec": turn_policy.interrupt.decision_timeout_ms
        / 1000.0,
        "interrupt_min_interim_chars": turn_policy.interrupt.min_interim_chars,
        "duck_early_cancel_score_threshold": (
            turn_policy.interrupt.early_cancel_score_threshold
        ),
        "duck_early_resume_score_threshold": (
            turn_policy.interrupt.early_resume_score_threshold
        ),
        "duck_cooldown_sec": turn_policy.ducking.cooldown_ms / 1000.0,
        "filler_enabled": turn_policy.filler.enabled,
        "filler_phrases": turn_policy.filler.phrases,
        "filler_fade_in_ms": turn_policy.filler.fade_in_ms,
        "filler_fade_out_ms": turn_policy.filler.fade_out_ms,
        "filler_silence_lead_in_ms": turn_policy.filler.silence_lead_in_ms,
    }
