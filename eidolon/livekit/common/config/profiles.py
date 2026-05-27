"""Turn-policy profile defaults."""

from __future__ import annotations

from dataclasses import replace

from .schema import (
    DuckingPolicyConfig,
    EotPolicyConfig,
    InterruptPolicyConfig,
    TurnPolicyConfig,
    VadPolicyConfig,
)


def profile_defaults(name: str) -> TurnPolicyConfig:
    """Return the default turn-policy config for a named profile."""
    if name == "fast_e2e_like":
        return TurnPolicyConfig(
            profile="fast_e2e_like",
            vad=VadPolicyConfig(
                activation_threshold=0.50,
                prefix_padding_ms=300,
                min_speech_duration_ms=80,
                min_silence_duration_ms=350,
            ),
            eot=EotPolicyConfig(
                eot_unlikely_threshold=0.50,
                tail_hang_silence_ms=1_200,
            ),
            interrupt=InterruptPolicyConfig(
                decision_timeout_ms=350,
                min_interim_chars=2,
                early_cancel_score_threshold=0.60,
                early_resume_score_threshold=0.15,
            ),
            ducking=DuckingPolicyConfig(
                enabled=True,
                fade_out_ms=30,
                fade_in_ms=30,
            ),
        )
    if name == "patient_companion":
        return TurnPolicyConfig(
            profile="patient_companion",
            vad=VadPolicyConfig(
                activation_threshold=0.50,
                prefix_padding_ms=300,
                min_speech_duration_ms=100,
                min_silence_duration_ms=800,
            ),
            eot=EotPolicyConfig(
                eot_unlikely_threshold=0.50,
                tail_hang_silence_ms=3_500,
            ),
            interrupt=InterruptPolicyConfig(
                decision_timeout_ms=700,
                min_interim_chars=2,
                early_cancel_score_threshold=0.80,
                early_resume_score_threshold=0.25,
            ),
            ducking=DuckingPolicyConfig(
                enabled=True,
                fade_out_ms=30,
                fade_in_ms=30,
            ),
        )
    if name in ("balanced_semantic", "custom", ""):
        # "custom" starts from the balanced baseline and then applies user
        # overrides from settings.yaml.
        return replace(TurnPolicyConfig(), profile=name or "balanced_semantic")
    raise ValueError(
        f"Unknown turn_policy.profile {name!r}; expected fast_e2e_like, "
        "balanced_semantic, patient_companion, or custom"
    )
