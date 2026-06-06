"""Effective config schema/profile tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from eidolon.livekit.common.config import load_effective_config
from eidolon.livekit.common.config.defaults import (
    DEFAULT_CORRECTION_LEXICON,
    DEFAULT_HARD_STOP_LEXICON,
    DEFAULT_TOPIC_SWITCH_LEXICON,
)
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
    assert cfg.interrupt.decision_timeout_ms == 500
    assert cfg.interrupt.early_cancel_score_threshold == 0.70
    assert cfg.ducking.fade_out_ms == 30


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
  tts_provider: sensetime
  vad_provider: firered
  brain_provider: direct_llm
llm:
  base_url: https://api.openai.com/v1
  model: gpt-4o-mini
  api_key: OPENAI_LLM_API_KEY
turn_policy:
  profile: balanced_semantic
  interrupt:
    decision_timeout_ms: 450
""",
    )
    monkeypatch.setenv("EIDOLON_CHANNEL_SETTINGS_YAML", str(settings))
    monkeypatch.setenv("LIVEKIT_API_KEY", "devkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "devsecret")
    monkeypatch.setenv("OPENAI_LLM_API_KEY", "test")
    cfg = load_effective_config()
    assert cfg.providers.brain_provider == "direct_llm"
    assert cfg.turn_policy.interrupt.decision_timeout_ms == 450
    assert cfg.turn_policy.interrupt.hard_stop_lexicon == DEFAULT_HARD_STOP_LEXICON
    assert cfg.turn_policy.interrupt.topic_switch_lexicon == DEFAULT_TOPIC_SWITCH_LEXICON
    assert cfg.turn_policy.interrupt.correction_lexicon == DEFAULT_CORRECTION_LEXICON
    assert cfg.llm.api_key == "test"


def test_settings_example_keeps_interrupt_lexicons_in_python_defaults() -> None:
    root = Path(__file__).resolve().parents[4]
    settings = yaml.safe_load(
        (root / "config" / "settings.example.yaml").read_text(encoding="utf-8")
    )
    interrupt = settings["turn_policy"]["interrupt"]

    assert "hard_stop_lexicon" not in interrupt
    assert "topic_switch_lexicon" not in interrupt
    assert "correction_lexicon" not in interrupt


def test_remote_agent_validates_its_own_required_fields(
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
  tts_provider: sensetime
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
    monkeypatch.setenv("REMOTE_AGENT_RPC_DEVICE_TOKEN", "token")
    monkeypatch.delenv("OPENAI_LLM_API_KEY", raising=False)
    cfg = load_effective_config()
    assert cfg.providers.brain_provider == "eidolon_agent"
    assert cfg.remote_agent_rpc.target == "127.0.0.1:45051"


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
