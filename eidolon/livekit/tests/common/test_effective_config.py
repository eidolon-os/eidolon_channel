"""Effective config schema/profile tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from eidolon.livekit.common.config import load_effective_config
from eidolon.livekit.common.config.profiles import profile_defaults


def _write_settings(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "settings.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_balanced_profile_defaults() -> None:
    cfg = profile_defaults("balanced_semantic")
    assert cfg.vad.activation_threshold == 0.50
    assert cfg.vad.prefix_padding_ms == 300
    assert cfg.vad.min_silence_duration_ms == 500
    assert cfg.eot.speech_merge_grace_ms == 800
    assert cfg.interrupt.decision_timeout_ms == 450
    assert cfg.interrupt.early_cancel_score_threshold == 0.70
    assert cfg.ducking.fade_out_ms == 30
    assert cfg.ducking.suspended_passthrough_enabled is False
    assert cfg.ducking.suspended_passthrough_volume == 0.25


def test_fast_profile_is_more_eager() -> None:
    fast = profile_defaults("fast_e2e_like")
    balanced = profile_defaults("balanced_semantic")
    assert fast.vad.min_silence_duration_ms < balanced.vad.min_silence_duration_ms
    assert fast.interrupt.decision_timeout_ms < balanced.interrupt.decision_timeout_ms
    assert (
        fast.interrupt.early_cancel_score_threshold
        < balanced.interrupt.early_cancel_score_threshold
    )


def test_load_effective_config_from_new_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _write_settings(
        tmp_path,
        """
core:
  livekit_url: ws://127.0.0.1:7880
  api_key: LIVEKIT_API_KEY
  api_secret: LIVEKIT_API_SECRET
providers:
  stt_provider: sensetime
  tts_provider: bailian
  vad_provider: firered
  brain_provider: direct_llm
behavior:
  instructions: test instructions
worker:
  num_idle_processes: 1
runtime_authority:
  kernel_api_url: http://kernel.local/api/kernel/v1
  data_api_url: http://data.local
  data_service_token_env: TEST_DATA_TOKEN
  http_timeout_sec: 8.5
  http_connect_timeout_sec: 1.5
llm:
  base_url: https://api.openai.com/v1
  model: gpt-4o-mini
  api_key: OPENAI_LLM_API_KEY
turn_policy:
  profile: balanced_semantic
  eot:
    transcript_revision_min_normalized_chars: 5
    speech_merge_grace_ms: 650
  interrupt:
    decision_timeout_ms: 450
    framework_false_interruption_timeout_ms: 5500
    stt_commit_transcript_timeout_ms: 4200
    aec_warmup_ms: 750
    cancel_residual_commit_suppress_ms: 1800
    hard_stop_prefix_min_cjk_chars: 3
    repeated_noise_min_chars: 3
    repeated_noise_max_chars: 8
  ptt:
    segment_stt_strategy: streaming
    segment_min_audio_ms: 120
    segment_max_audio_ms: 18000
    segment_min_rms_ppm: 25
    segment_tap_to_stop_max_audio_ms: 420
  ducking:
    suspended_passthrough_enabled: true
    suspended_passthrough_volume: 0.2
  filler:
    enabled: true
    phrases:
      - 嗯...
      - 收到...
    fade_in_ms: 40
    fade_out_ms: 90
    silence_lead_in_ms: 160
  idle:
    disconnect_after_idle_ms: 45000
    presence_disconnect_after_idle_ms: 11000
    proactive_disconnect_after_idle_ms: 9000
    disconnect_grace_ms: 450
observability:
  llm_first_delta_timeout_ms: 2500
  stt_pending_provider_event_window_ms: 1600
  stt_pending_provider_event_preroll_ms: 350
  stt_pending_provider_event_max_count: 48
voiceprint:
  enabled: true
  threshold: 0.42
  min_audio_ms: 2000
  turn_max_audio_ms: 9000
  accept_cache_ttl_ms: 120000
  accept_cache_short_audio_max_ms: 2400
  owner_commit_threshold: 0.64
  owner_short_audio_bypass_ms: 1200
""",
    )
    monkeypatch.setenv("EIDOLON_CHANNEL_SETTINGS_YAML", str(settings))
    monkeypatch.setenv("LIVEKIT_API_KEY", "devkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "devsecret")
    monkeypatch.setenv("OPENAI_LLM_API_KEY", "test")
    cfg = load_effective_config()
    assert cfg.providers.brain_provider == "direct_llm"
    assert cfg.turn_policy.interrupt.decision_timeout_ms == 450
    assert cfg.turn_policy.interrupt.framework_false_interruption_timeout_ms == 5500
    assert cfg.turn_policy.interrupt.stt_commit_transcript_timeout_ms == 4200
    assert cfg.turn_policy.ptt.segment_stt_strategy == "streaming"
    assert cfg.turn_policy.ptt.segment_min_audio_ms == 120
    assert cfg.turn_policy.ptt.segment_max_audio_ms == 18000
    assert cfg.turn_policy.ptt.segment_min_rms_ppm == 25
    assert cfg.turn_policy.ptt.segment_tap_to_stop_max_audio_ms == 420
    assert cfg.turn_policy.ducking.suspended_passthrough_enabled is True
    assert cfg.turn_policy.ducking.suspended_passthrough_volume == 0.2
    assert cfg.turn_policy.filler.enabled is True
    assert cfg.turn_policy.filler.phrases == ("嗯...", "收到...")
    assert cfg.turn_policy.filler.fade_in_ms == 40
    assert cfg.turn_policy.filler.fade_out_ms == 90
    assert cfg.turn_policy.filler.silence_lead_in_ms == 160
    assert cfg.turn_policy.idle.disconnect_after_idle_ms == 45000
    assert cfg.turn_policy.idle.presence_disconnect_after_idle_ms == 11000
    assert cfg.turn_policy.idle.proactive_disconnect_after_idle_ms == 9000
    assert cfg.turn_policy.idle.disconnect_grace_ms == 450
    assert cfg.turn_policy.interrupt.aec_warmup_ms == 750
    assert cfg.turn_policy.interrupt.cancel_residual_commit_suppress_ms == 1800
    assert cfg.turn_policy.interrupt.hard_stop_prefix_min_cjk_chars == 3
    assert cfg.turn_policy.interrupt.repeated_noise_min_chars == 3
    assert cfg.turn_policy.interrupt.repeated_noise_max_chars == 8
    assert cfg.turn_policy.eot.transcript_revision_min_normalized_chars == 5
    assert cfg.turn_policy.eot.speech_merge_grace_ms == 650
    assert cfg.observability.llm_first_delta_timeout_ms == 2500
    assert cfg.observability.stt_pending_provider_event_window_ms == 1600
    assert cfg.observability.stt_pending_provider_event_preroll_ms == 350
    assert cfg.observability.stt_pending_provider_event_max_count == 48
    assert cfg.llm.api_key == "test"
    assert cfg.voiceprint.enabled is True
    assert cfg.voiceprint.threshold == 0.42
    assert cfg.voiceprint.min_audio_ms == 2000
    assert cfg.voiceprint.turn_max_audio_ms == 9000
    assert cfg.voiceprint.accept_cache_ttl_ms == 120000
    assert cfg.voiceprint.accept_cache_short_audio_max_ms == 2400
    assert cfg.voiceprint.owner_commit_threshold == 0.64
    assert cfg.voiceprint.owner_short_audio_bypass_ms == 1200
    assert cfg.behavior.instructions == "test instructions"
    assert cfg.worker.num_idle_processes == 1
    assert cfg.runtime_authority.kernel_api_url == "http://kernel.local/api/kernel/v1"
    assert cfg.runtime_authority.data_api_url == "http://data.local"
    assert cfg.runtime_authority.data_service_token_env == "TEST_DATA_TOKEN"
    assert cfg.runtime_authority.http_timeout_sec == 8.5
    assert cfg.runtime_authority.http_connect_timeout_sec == 1.5


def test_load_effective_config_applies_settings_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _write_settings(
        tmp_path,
        """
core:
  livekit_url: ws://127.0.0.1:7880
  api_key: LIVEKIT_API_KEY
  api_secret: LIVEKIT_API_SECRET
providers:
  stt_provider: sensetime
  tts_provider: bailian
  vad_provider: firered
  brain_provider: eidolon_agent
runtime_authority:
  data_api_url: http://data.local
llm:
  base_url: http://llm.local/v1
  model: test-model
  api_key: OPENAI_LLM_API_KEY
remote_agent_rpc:
  target: 127.0.0.1:45051
turn_policy:
  attention:
    enforce: false
    soft_duck_on_playback_speech_start: false
voiceprint:
  enabled: true
""",
    )
    overlay = tmp_path / "box3_full_duplex.yaml"
    overlay.write_text(
        """
providers:
  brain_provider: direct_llm
turn_policy:
  attention:
    enforce: true
    soft_duck_on_playback_speech_start: true
voiceprint:
  enabled: false
""",
        encoding="utf-8",
    )

    monkeypatch.setenv("EIDOLON_CHANNEL_SETTINGS_YAML", str(settings))
    monkeypatch.setenv("EIDOLON_CHANNEL_SETTINGS_OVERLAY_YAML", str(overlay))
    monkeypatch.setenv("LIVEKIT_API_KEY", "devkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "devsecret")
    monkeypatch.setenv("OPENAI_LLM_API_KEY", "test")

    cfg = load_effective_config()

    assert cfg.providers.brain_provider == "direct_llm"
    assert cfg.turn_policy.attention.enforce is True
    assert cfg.turn_policy.attention.soft_duck_on_playback_speech_start is True
    assert cfg.voiceprint.enabled is False
    assert cfg.turn_policy.attention.require_direct_signal_during_playback is True
    assert cfg.turn_policy.interrupt.redirect_prefix_min_cjk_chars == 3
    assert cfg.turn_policy.attention.echo_min_normalized_chars == 3
    assert cfg.turn_policy.attention.assistant_speech_recent_max_age_ms == 3000


def test_invalid_voiceprint_threshold_fails_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _write_settings(
        tmp_path,
        """
core:
  livekit_url: ws://127.0.0.1:7880
  api_key: LIVEKIT_API_KEY
  api_secret: LIVEKIT_API_SECRET
providers:
  stt_provider: sensetime
  tts_provider: bailian
  vad_provider: firered
  brain_provider: direct_llm
llm:
  base_url: https://api.openai.com/v1
  model: gpt-4o-mini
  api_key: OPENAI_LLM_API_KEY
voiceprint:
  threshold: 1.2
""",
    )
    monkeypatch.setenv("EIDOLON_CHANNEL_SETTINGS_YAML", str(settings))
    monkeypatch.setenv("LIVEKIT_API_KEY", "devkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "devsecret")
    monkeypatch.setenv("OPENAI_LLM_API_KEY", "test")

    with pytest.raises(ValueError, match="voiceprint.threshold"):
        load_effective_config()


def test_non_positive_runtime_token_ttl_fails_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _write_settings(
        tmp_path,
        """
core:
  api_key: LIVEKIT_API_KEY
  api_secret: LIVEKIT_API_SECRET
providers:
  stt_provider: sensetime
  tts_provider: bailian
  vad_provider: firered
  brain_provider: direct_llm
runtime_authority:
  device_token_ttl_seconds: 0
llm:
  base_url: https://llm.example/v1
  model: test-model
  api_key: OPENAI_LLM_API_KEY
""",
    )
    monkeypatch.setenv("EIDOLON_CHANNEL_SETTINGS_YAML", str(settings))
    monkeypatch.setenv("LIVEKIT_API_KEY", "devkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "devsecret")
    monkeypatch.setenv("OPENAI_LLM_API_KEY", "test")

    with pytest.raises(ValueError, match="device_token_ttl_seconds must be positive"):
        load_effective_config()


def test_invalid_worker_num_idle_processes_fails_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _write_settings(
        tmp_path,
        """
core:
  livekit_url: ws://127.0.0.1:7880
  api_key: LIVEKIT_API_KEY
  api_secret: LIVEKIT_API_SECRET
providers:
  stt_provider: sensetime
  tts_provider: bailian
  vad_provider: firered
  brain_provider: direct_llm
llm:
  base_url: https://api.openai.com/v1
  model: gpt-4o-mini
  api_key: OPENAI_LLM_API_KEY
worker:
  num_idle_processes: -1
""",
    )
    monkeypatch.setenv("EIDOLON_CHANNEL_SETTINGS_YAML", str(settings))
    monkeypatch.setenv("LIVEKIT_API_KEY", "devkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "devsecret")
    monkeypatch.setenv("OPENAI_LLM_API_KEY", "test")

    with pytest.raises(ValueError, match="worker.num_idle_processes"):
        load_effective_config()


def test_invalid_ducking_passthrough_volume_fails_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _write_settings(
        tmp_path,
        """
core:
  livekit_url: ws://127.0.0.1:7880
  api_key: LIVEKIT_API_KEY
  api_secret: LIVEKIT_API_SECRET
providers:
  stt_provider: sensetime
  tts_provider: bailian
  vad_provider: firered
  brain_provider: direct_llm
llm:
  base_url: https://api.openai.com/v1
  model: gpt-4o-mini
  api_key: OPENAI_LLM_API_KEY
turn_policy:
  ducking:
    suspended_passthrough_volume: 1.5
""",
    )
    monkeypatch.setenv("EIDOLON_CHANNEL_SETTINGS_YAML", str(settings))
    monkeypatch.setenv("LIVEKIT_API_KEY", "devkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "devsecret")
    monkeypatch.setenv("OPENAI_LLM_API_KEY", "test")

    with pytest.raises(ValueError, match="suspended_passthrough_volume"):
        load_effective_config()


def test_invalid_idle_disconnect_grace_fails_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _write_settings(
        tmp_path,
        """
core:
  livekit_url: ws://127.0.0.1:7880
  api_key: LIVEKIT_API_KEY
  api_secret: LIVEKIT_API_SECRET
providers:
  stt_provider: sensetime
  tts_provider: bailian
  vad_provider: firered
  brain_provider: direct_llm
llm:
  base_url: https://api.openai.com/v1
  model: gpt-4o-mini
  api_key: OPENAI_LLM_API_KEY
turn_policy:
  idle:
    disconnect_grace_ms: 30000
""",
    )
    monkeypatch.setenv("EIDOLON_CHANNEL_SETTINGS_YAML", str(settings))
    monkeypatch.setenv("LIVEKIT_API_KEY", "devkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "devsecret")
    monkeypatch.setenv("OPENAI_LLM_API_KEY", "test")

    with pytest.raises(ValueError, match="turn_policy.idle.disconnect_grace_ms"):
        load_effective_config()


def test_settings_example_keeps_interrupt_lexicons_in_python_defaults() -> None:
    root = Path(__file__).resolve().parents[4]
    settings = yaml.safe_load(
        (root / "config" / "settings.example.yaml").read_text(encoding="utf-8")
    )
    interrupt = settings["turn_policy"]["interrupt"]

    assert "hard_stop_lexicon" not in interrupt
    assert "topic_switch_lexicon" not in interrupt
    assert "correction_lexicon" not in interrupt


def test_settings_example_loads_as_effective_config(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[4]
    settings = root / "config" / "settings.example.yaml"

    monkeypatch.setenv("EIDOLON_CHANNEL_SETTINGS_YAML", str(settings))
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
    monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)
    monkeypatch.delenv("OPENAI_LLM_API_KEY", raising=False)
    monkeypatch.delenv("PAIRING_JWT_SECRET", raising=False)

    cfg = load_effective_config()

    assert cfg.providers.brain_provider == "eidolon_agent"
    assert cfg.turn_policy.profile == "balanced_semantic"
    assert cfg.turn_policy.interrupt.fast_lexical_intents is True
    assert cfg.turn_policy.interrupt.correction_topic_stability_window_ms == 0
    assert cfg.turn_policy.ducking.suspended_passthrough_enabled is False


def test_remote_agent_rejects_legacy_device_token_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _write_settings(
        tmp_path,
        """
core:
  api_key: LIVEKIT_API_KEY
  api_secret: LIVEKIT_API_SECRET
providers:
  stt_provider: sensetime
  tts_provider: bailian
  vad_provider: firered
  brain_provider: eidolon_agent
remote_agent_rpc:
  target: 127.0.0.1:45051
  device_token: REMOTE_AGENT_RPC_DEVICE_TOKEN
llm:
  base_url: ""
  model: ignored-for-remote
  api_key: OPENAI_LLM_API_KEY
""",
    )
    monkeypatch.setenv("EIDOLON_CHANNEL_SETTINGS_YAML", str(settings))
    monkeypatch.setenv("LIVEKIT_API_KEY", "devkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "devsecret")
    monkeypatch.delenv("OPENAI_LLM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="unknown config field remote_agent_rpc.device_token"):
        load_effective_config()


def test_manual_config_sections_reject_unknown_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _write_settings(
        tmp_path,
        """
core:
  api_key: LIVEKIT_API_KEY
  api_secret: LIVEKIT_API_SECRET
behavior:
  typo_mode: streaming
providers:
  stt_provider: sensetime
  tts_provider: bailian
  vad_provider: firered
  brain_provider: direct_llm
llm:
  base_url: https://api.openai.com/v1
  model: gpt-4o-mini
  api_key: OPENAI_LLM_API_KEY
""",
    )
    monkeypatch.setenv("EIDOLON_CHANNEL_SETTINGS_YAML", str(settings))
    monkeypatch.setenv("LIVEKIT_API_KEY", "devkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "devsecret")
    monkeypatch.setenv("OPENAI_LLM_API_KEY", "test")

    with pytest.raises(ValueError, match="unknown config field behavior.typo_mode"):
        load_effective_config()


def test_behavior_pipeline_mode_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _write_settings(
        tmp_path,
        """
core:
  api_key: LIVEKIT_API_KEY
  api_secret: LIVEKIT_API_SECRET
behavior:
  pipeline_mode: batch
providers:
  stt_provider: sensetime
  tts_provider: bailian
  vad_provider: firered
  brain_provider: direct_llm
llm:
  base_url: https://api.openai.com/v1
  model: gpt-4o-mini
  api_key: OPENAI_LLM_API_KEY
""",
    )
    monkeypatch.setenv("EIDOLON_CHANNEL_SETTINGS_YAML", str(settings))
    monkeypatch.setenv("LIVEKIT_API_KEY", "devkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "devsecret")
    monkeypatch.setenv("OPENAI_LLM_API_KEY", "test")

    with pytest.raises(ValueError, match="unknown config field behavior.pipeline_mode"):
        load_effective_config()


def test_behavior_agent_mode_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _write_settings(
        tmp_path,
        """
core:
  api_key: LIVEKIT_API_KEY
  api_secret: LIVEKIT_API_SECRET
behavior:
  agent_mode: batch
providers:
  stt_provider: sensetime
  tts_provider: bailian
  vad_provider: firered
  brain_provider: direct_llm
llm:
  base_url: https://api.openai.com/v1
  model: gpt-4o-mini
  api_key: OPENAI_LLM_API_KEY
""",
    )
    monkeypatch.setenv("EIDOLON_CHANNEL_SETTINGS_YAML", str(settings))
    monkeypatch.setenv("LIVEKIT_API_KEY", "devkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "devsecret")
    monkeypatch.setenv("OPENAI_LLM_API_KEY", "test")

    with pytest.raises(ValueError, match="unknown config field behavior.agent_mode"):
        load_effective_config()


def test_brain_provider_must_be_explicit_for_remote_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _write_settings(
        tmp_path,
        """
core:
  api_key: LIVEKIT_API_KEY
  api_secret: LIVEKIT_API_SECRET
providers:
  brain_provider: direct_llm
remote_agent_rpc:
  target: 127.0.0.1:45051
llm:
  base_url: ""
  api_key: OPENAI_LLM_API_KEY
""",
    )
    monkeypatch.setenv("EIDOLON_CHANNEL_SETTINGS_YAML", str(settings))
    monkeypatch.setenv("LIVEKIT_API_KEY", "devkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "devsecret")
    monkeypatch.delenv("OPENAI_LLM_API_KEY", raising=False)

    with pytest.raises(ValueError, match="llm.base_url"):
        load_effective_config()
