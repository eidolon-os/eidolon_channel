"""Validation for effective channel configuration."""

from __future__ import annotations

from .schema import EffectiveAgentConfig


def validate_effective_config(cfg: EffectiveAgentConfig) -> None:
    errors: list[str] = []

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

    worker = cfg.worker
    if worker.num_idle_processes is not None and not (0 <= worker.num_idle_processes <= 64):
        errors.append("worker.num_idle_processes must be in [0, 64]")
    if not (1.0 <= worker.setup_timeout_sec <= 600.0):
        errors.append("worker.setup_timeout_sec must be in [1.0, 600.0]")

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
        # belongs to runtime_authority now. server.run() does the
        # PAIRING_JWT_SECRET-or-file check at startup; here we only
        # require the gRPC target so this validator stays infrastructure-
        # focused (vs runtime-secret-focused).
    if (
        not 0.1
        <= cfg.runtime_authority.http_connect_timeout_sec
        <= cfg.runtime_authority.http_timeout_sec
        <= 60.0
    ):
        errors.append(
            "runtime_authority HTTP timeouts must satisfy "
            "0.1 <= http_connect_timeout_sec <= http_timeout_sec <= 60.0"
        )
    if cfg.runtime_authority.device_token_ttl_seconds <= 0:
        errors.append("runtime_authority.device_token_ttl_seconds must be positive")

    vad = cfg.turn_policy.vad
    if not 0.0 < vad.activation_threshold < 1.0:
        errors.append("turn_policy.vad.activation_threshold must be in (0, 1)")
    if not 0 <= vad.prefix_padding_ms <= 2_000:
        errors.append("turn_policy.vad.prefix_padding_ms must be in [0, 2000]")
    if not 10 <= vad.min_speech_duration_ms <= 2_000:
        errors.append("turn_policy.vad.min_speech_duration_ms must be in [10, 2000]")
    if not 100 <= vad.min_silence_duration_ms <= 5_000:
        errors.append("turn_policy.vad.min_silence_duration_ms must be in [100, 5000]")

    eot = cfg.turn_policy.eot
    if not 1 <= eot.transcript_revision_min_normalized_chars <= 20:
        errors.append("turn_policy.eot.transcript_revision_min_normalized_chars must be in [1, 20]")
    if not 0 <= eot.speech_merge_grace_ms <= 5_000:
        errors.append("turn_policy.eot.speech_merge_grace_ms must be in [0, 5000]")

    intr = cfg.turn_policy.interrupt
    if not 200 <= intr.decision_timeout_ms <= 1_000:
        errors.append("turn_policy.interrupt.decision_timeout_ms must be in [200, 1000]")
    if not 0 <= intr.framework_false_interruption_timeout_ms <= 30_000:
        errors.append(
            "turn_policy.interrupt.framework_false_interruption_timeout_ms must be in [0, 30000]"
        )
    if not 0 <= intr.stt_commit_transcript_timeout_ms <= 30_000:
        errors.append(
            "turn_policy.interrupt.stt_commit_transcript_timeout_ms must be in [0, 30000]"
        )
    if not 0 <= intr.post_speech_no_evidence_timeout_ms <= intr.post_speech_evidence_timeout_ms:
        errors.append(
            "turn_policy.interrupt.post_speech_no_evidence_timeout_ms must be in "
            "[0, post_speech_evidence_timeout_ms]"
        )
    if intr.aec_warmup_ms is not None and not 0 <= intr.aec_warmup_ms <= 30_000:
        errors.append("turn_policy.interrupt.aec_warmup_ms must be null or in [0, 30000]")
    if not 1 <= intr.min_interim_chars <= 12:
        errors.append("turn_policy.interrupt.min_interim_chars must be in [1, 12]")
    if not 1 <= intr.min_normal_interim_cjk_chars <= 12:
        errors.append("turn_policy.interrupt.min_normal_interim_cjk_chars must be in [1, 12]")
    if not 0 <= intr.latin_artifact_hold_max_chars <= 12:
        errors.append("turn_policy.interrupt.latin_artifact_hold_max_chars must be in [0, 12]")
    if not 1 <= intr.hard_stop_prefix_min_cjk_chars <= 8:
        errors.append("turn_policy.interrupt.hard_stop_prefix_min_cjk_chars must be in [1, 8]")
    if not 1 <= intr.redirect_prefix_min_cjk_chars <= 8:
        errors.append("turn_policy.interrupt.redirect_prefix_min_cjk_chars must be in [1, 8]")
    if not 1 <= intr.repeated_noise_min_chars <= intr.repeated_noise_max_chars <= 20:
        errors.append(
            "turn_policy.interrupt repeated noise chars must satisfy 1 <= min <= max <= 20"
        )
    if not 0 <= intr.weak_signal_followup_hold_ms <= 5_000:
        errors.append("turn_policy.interrupt.weak_signal_followup_hold_ms must be in [0, 5000]")
    if not 0 <= intr.correction_topic_stability_window_ms <= 1_000:
        errors.append(
            "turn_policy.interrupt.correction_topic_stability_window_ms must be in [0, 1000]"
        )
    if not 0 <= intr.normal_interrupt_stability_window_ms <= 2_000:
        errors.append(
            "turn_policy.interrupt.normal_interrupt_stability_window_ms must be in [0, 2000]"
        )
    if not 0.0 <= intr.early_resume_score_threshold <= intr.early_cancel_score_threshold <= 1.0:
        errors.append(
            "turn_policy interrupt score thresholds must satisfy 0 <= resume <= cancel <= 1"
        )
    if not 0 <= intr.cancel_residual_commit_suppress_ms <= 10_000:
        errors.append(
            "turn_policy.interrupt.cancel_residual_commit_suppress_ms must be in [0, 10000]"
        )

    ptt = cfg.turn_policy.ptt
    if ptt.segment_stt_strategy not in ("auto", "offline", "streaming"):
        errors.append(
            "turn_policy.ptt.segment_stt_strategy must be 'auto', 'offline', or 'streaming'"
        )
    if not 0 <= ptt.segment_min_audio_ms <= 5_000:
        errors.append("turn_policy.ptt.segment_min_audio_ms must be in [0, 5000]")
    if not 1_000 <= ptt.segment_max_audio_ms <= 120_000:
        errors.append("turn_policy.ptt.segment_max_audio_ms must be in [1000, 120000]")
    if ptt.segment_min_audio_ms > ptt.segment_max_audio_ms:
        errors.append("turn_policy.ptt.segment_min_audio_ms must be <= segment_max_audio_ms")
    if not 0 <= ptt.segment_min_rms_ppm <= 1_000_000:
        errors.append("turn_policy.ptt.segment_min_rms_ppm must be in [0, 1000000]")
    if not 0 <= ptt.segment_tap_to_stop_max_audio_ms <= 5_000:
        errors.append("turn_policy.ptt.segment_tap_to_stop_max_audio_ms must be in [0, 5000]")

    duck = cfg.turn_policy.ducking
    if not 1 <= duck.fade_out_ms <= 500:
        errors.append("turn_policy.ducking.fade_out_ms must be in [1, 500]")
    if not 1 <= duck.fade_in_ms <= 500:
        errors.append("turn_policy.ducking.fade_in_ms must be in [1, 500]")
    if not 0.0 <= duck.suspend_volume <= 1.0:
        errors.append("turn_policy.ducking.suspend_volume must be in [0, 1]")
    if not 0.0 <= duck.suspended_passthrough_volume <= 1.0:
        errors.append("turn_policy.ducking.suspended_passthrough_volume must be in [0, 1]")

    filler = cfg.turn_policy.filler
    if not 0 <= filler.fade_in_ms <= 500:
        errors.append("turn_policy.filler.fade_in_ms must be in [0, 500]")
    if not 0 <= filler.fade_out_ms <= 500:
        errors.append("turn_policy.filler.fade_out_ms must be in [0, 500]")
    if not 0 <= filler.silence_lead_in_ms <= 1_000:
        errors.append("turn_policy.filler.silence_lead_in_ms must be in [0, 1000]")
    if filler.enabled and not filler.phrases:
        errors.append("turn_policy.filler.phrases must not be empty when enabled")

    idle = cfg.turn_policy.idle
    if not 0 <= idle.disconnect_after_idle_ms <= 3_600_000:
        errors.append("turn_policy.idle.disconnect_after_idle_ms must be in [0, 3600000]")
    if not 0 <= idle.presence_disconnect_after_idle_ms <= 3_600_000:
        errors.append("turn_policy.idle.presence_disconnect_after_idle_ms must be in [0, 3600000]")
    if not 0 <= idle.proactive_disconnect_after_idle_ms <= 3_600_000:
        errors.append("turn_policy.idle.proactive_disconnect_after_idle_ms must be in [0, 3600000]")
    if not 0 <= idle.disconnect_grace_ms <= 10_000:
        errors.append("turn_policy.idle.disconnect_grace_ms must be in [0, 10000]")

    attention = cfg.turn_policy.attention
    if not 100 <= attention.client_state_max_age_ms <= 10_000:
        errors.append("turn_policy.attention.client_state_max_age_ms must be in [100, 10000]")
    if not 1 <= attention.echo_min_normalized_chars <= 32:
        errors.append("turn_policy.attention.echo_min_normalized_chars must be in [1, 32]")
    if not 0 <= attention.assistant_speech_recent_max_age_ms <= 30_000:
        errors.append(
            "turn_policy.attention.assistant_speech_recent_max_age_ms must be in [0, 30000]"
        )

    obs = cfg.observability
    if not 0 <= obs.llm_first_delta_timeout_ms <= 60_000:
        errors.append("observability.llm_first_delta_timeout_ms must be in [0, 60000]")
    if not 0 <= obs.stt_pending_provider_event_window_ms <= 10_000:
        errors.append("observability.stt_pending_provider_event_window_ms must be in [0, 10000]")
    if not 0 <= obs.stt_pending_provider_event_preroll_ms <= 5_000:
        errors.append("observability.stt_pending_provider_event_preroll_ms must be in [0, 5000]")
    if not 1 <= obs.stt_pending_provider_event_max_count <= 512:
        errors.append("observability.stt_pending_provider_event_max_count must be in [1, 512]")

    vp = cfg.voiceprint
    if vp.enabled:
        if vp.provider != "3d_speaker":
            errors.append("voiceprint.provider must be '3d_speaker'")
        if vp.model != "campplus_zh_16k_common":
            errors.append("voiceprint.model must be 'campplus_zh_16k_common'")
        if not 0.0 < vp.threshold < 1.0:
            errors.append("voiceprint.threshold must be in (0, 1)")
        if not 500 <= vp.min_audio_ms <= 30_000:
            errors.append("voiceprint.min_audio_ms must be in [500, 30000]")
        if not 500 <= vp.turn_max_audio_ms <= 60_000:
            errors.append("voiceprint.turn_max_audio_ms must be in [500, 60000]")
        if not 0 <= vp.accept_cache_ttl_ms <= 3_600_000:
            errors.append("voiceprint.accept_cache_ttl_ms must be in [0, 3600000]")
        if not 0 <= vp.accept_cache_short_audio_max_ms <= 30_000:
            errors.append("voiceprint.accept_cache_short_audio_max_ms must be in [0, 30000]")
        if not 0.0 < vp.owner_commit_threshold < 1.0:
            errors.append("voiceprint.owner_commit_threshold must be in (0, 1)")
        if not 0 <= vp.owner_short_audio_bypass_ms <= vp.min_audio_ms:
            errors.append("voiceprint.owner_short_audio_bypass_ms must be in [0, min_audio_ms]")

    if errors:
        raise ValueError("EffectiveAgentConfig validation failed:\n  - " + "\n  - ".join(errors))
