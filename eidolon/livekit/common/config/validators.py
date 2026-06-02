"""Validation for effective channel configuration."""

from __future__ import annotations

from .schema import EffectiveAgentConfig


def validate_effective_config(cfg: EffectiveAgentConfig) -> None:
    errors: list[str] = []

    if cfg.behavior.agent_mode not in ("streaming", "batch"):
        errors.append("behavior.agent_mode must be 'streaming' or 'batch'")
    if cfg.providers.stt_provider not in ("bailian", "sensetime"):
        errors.append("providers.stt_provider must be 'bailian' or 'sensetime'")
    if cfg.providers.tts_provider not in ("bailian", "sensetime"):
        errors.append("providers.tts_provider must be 'bailian' or 'sensetime'")
    if cfg.providers.vad_provider not in (
        "firered",
        "firered_pvad",
        "silero",
        "none",
        "disabled",
    ):
        errors.append("providers.vad_provider is unsupported")
    if cfg.providers.brain_provider not in ("direct_llm", "eidolon_agent"):
        errors.append("providers.brain_provider must be 'direct_llm' or 'eidolon_agent'")

    if not (1 <= cfg.core.port <= 65535):
        errors.append("core.port must be in [1, 65535]")

    if cfg.providers.brain_provider == "direct_llm":
        if not cfg.llm.model:
            errors.append("llm.model is required for direct_llm")
        if not cfg.llm.base_url:
            errors.append("llm.base_url is required for direct_llm")
    else:
        if not cfg.remote_agent_rpc.target:
            errors.append("remote_agent_rpc.target is required for eidolon_agent")
        # Phase 32.D: device_token check moved out — token signing
        # belongs to runtime_admin now. server.run() does the
        # PAIRING_JWT_SECRET-or-file check at startup; here we only
        # require the gRPC target so this validator stays infrastructure-
        # focused (vs runtime-secret-focused).

    vad = cfg.turn_policy.vad
    if not 0.0 < vad.activation_threshold < 1.0:
        errors.append("turn_policy.vad.activation_threshold must be in (0, 1)")
    if not 0 <= vad.prefix_padding_ms <= 2_000:
        errors.append("turn_policy.vad.prefix_padding_ms must be in [0, 2000]")
    if not 10 <= vad.min_speech_duration_ms <= 2_000:
        errors.append("turn_policy.vad.min_speech_duration_ms must be in [10, 2000]")
    if not 100 <= vad.min_silence_duration_ms <= 5_000:
        errors.append("turn_policy.vad.min_silence_duration_ms must be in [100, 5000]")

    intr = cfg.turn_policy.interrupt
    if not 200 <= intr.decision_timeout_ms <= 1_000:
        errors.append("turn_policy.interrupt.decision_timeout_ms must be in [200, 1000]")
    if not 1 <= intr.min_interim_chars <= 12:
        errors.append("turn_policy.interrupt.min_interim_chars must be in [1, 12]")
    if not 0.0 <= intr.early_resume_score_threshold <= intr.early_cancel_score_threshold <= 1.0:
        errors.append(
            "turn_policy interrupt score thresholds must satisfy "
            "0 <= resume <= cancel <= 1"
        )

    duck = cfg.turn_policy.ducking
    if not 1 <= duck.fade_out_ms <= 500:
        errors.append("turn_policy.ducking.fade_out_ms must be in [1, 500]")
    if not 1 <= duck.fade_in_ms <= 500:
        errors.append("turn_policy.ducking.fade_in_ms must be in [1, 500]")
    if not 0.0 <= duck.suspend_volume <= 1.0:
        errors.append("turn_policy.ducking.suspend_volume must be in [0, 1]")

    if errors:
        raise ValueError("EffectiveAgentConfig validation failed:\n  - " + "\n  - ".join(errors))
