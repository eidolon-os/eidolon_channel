"""Unit tests for G4: LLMConfig env parsing for temperature/timeout/max_tokens.

Verifies the "empty env → None → plugin default" semantics that decouples
us from the OpenAI plugin's NOT_GIVEN sentinel.
"""

from __future__ import annotations

import pytest

from eidolon.livekit.common.config import LLMConfig, _optional_float, _optional_int


class TestOptionalParsers:
    def test_empty_string_is_none(self) -> None:
        assert _optional_float("") is None
        assert _optional_int("") is None

    def test_whitespace_only_is_none(self) -> None:
        assert _optional_float("   ") is None
        assert _optional_int("\t\n") is None

    def test_valid_values_parse(self) -> None:
        assert _optional_float("0.7") == 0.7
        assert _optional_float(" 120 ") == 120.0
        assert _optional_int("100000") == 100000

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            _optional_float("not-a-number")
        with pytest.raises(ValueError):
            _optional_int("0.5")  # int parser rejects float strings


class TestLLMConfigDefaults:
    def test_all_optional_fields_default_to_none(self) -> None:
        """Default LLMConfig() leaves new G4 fields at None so the OpenAI
        plugin's NOT_GIVEN defaults apply unchanged."""
        cfg = LLMConfig()
        assert cfg.temperature is None
        assert cfg.timeout is None
        assert cfg.max_completion_tokens is None

    def test_explicit_values_round_trip(self) -> None:
        cfg = LLMConfig(
            base_url="https://test/v1",
            model="openai/MiniMax-M2.7",
            api_key="sk-test",
            temperature=0.7,
            timeout=120.0,
            max_completion_tokens=100_000,
        )
        assert cfg.temperature == 0.7
        assert cfg.timeout == 120.0
        assert cfg.max_completion_tokens == 100_000
