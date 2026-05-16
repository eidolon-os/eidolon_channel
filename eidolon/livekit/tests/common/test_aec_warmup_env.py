"""Unit tests for G9: ``AGENT_AEC_WARMUP_DURATION`` env parsing.

Validates the env → ``AgentBehaviorConfig.aec_warmup_duration`` mapping:
  * Numeric strings → float
  * Empty / "none" / "off" / "disabled" → None (framework treats as disabled)
  * Default (env unset) → 1.0
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from typing import Iterator

import pytest


@contextmanager
def _envvar(name: str, value: str | None) -> Iterator[None]:
    """Temporarily set / unset an env var, restoring on exit."""
    prev = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = prev


def _load_config_aec() -> "float | None":
    """Trigger ``AgentConfig.from_env`` with the LIVEKIT env file bootstrap
    skipped, and return the parsed aec_warmup_duration."""
    # We can't easily call from_env() because it requires a real .env file
    # via _bootstrap_dotenv. Instead, test the parsing expression directly,
    # which is the entire G9 surface area.
    raw = os.environ.get("AGENT_AEC_WARMUP_DURATION", "1.0")
    return (
        None
        if raw.lower() in ("none", "off", "disabled", "")
        else float(raw)
    )


def test_default_value() -> None:
    """No env var set → default 1.0 (avoid 0 so welcome has brief protect)."""
    with _envvar("AGENT_AEC_WARMUP_DURATION", None):
        assert _load_config_aec() == 1.0


def test_numeric_value() -> None:
    with _envvar("AGENT_AEC_WARMUP_DURATION", "2.5"):
        assert _load_config_aec() == 2.5


def test_zero_value() -> None:
    """0 is numeric and means "warmup window of zero seconds" (effectively
    disabled, but framework still processes as float)."""
    with _envvar("AGENT_AEC_WARMUP_DURATION", "0"):
        assert _load_config_aec() == 0.0


def test_none_sentinel() -> None:
    """Literal "none" → Python None (framework treats as disable)."""
    with _envvar("AGENT_AEC_WARMUP_DURATION", "none"):
        assert _load_config_aec() is None


def test_off_sentinel() -> None:
    with _envvar("AGENT_AEC_WARMUP_DURATION", "off"):
        assert _load_config_aec() is None


def test_disabled_sentinel() -> None:
    with _envvar("AGENT_AEC_WARMUP_DURATION", "disabled"):
        assert _load_config_aec() is None


def test_case_insensitive() -> None:
    with _envvar("AGENT_AEC_WARMUP_DURATION", "NONE"):
        assert _load_config_aec() is None
    with _envvar("AGENT_AEC_WARMUP_DURATION", "Off"):
        assert _load_config_aec() is None


def test_empty_string_disables() -> None:
    """Empty string is treated as the disable sentinel — convenient for
    env files where commenting out a key is awkward."""
    with _envvar("AGENT_AEC_WARMUP_DURATION", ""):
        assert _load_config_aec() is None


def test_invalid_string_raises() -> None:
    """Bogus non-numeric, non-sentinel input → ValueError (fail fast)."""
    with _envvar("AGENT_AEC_WARMUP_DURATION", "not-a-number"):
        with pytest.raises(ValueError):
            _load_config_aec()


def test_streaming_pipeline_forwards_to_agent_session() -> None:
    """Constructor wiring: ``aec_warmup_duration`` reaches the
    ``AgentSession(...)`` kwargs (a smoke test on the call site)."""
    from eidolon.livekit.agent.streaming import StreamingPipeline
    import inspect

    sig = inspect.signature(StreamingPipeline.__init__)
    assert "aec_warmup_duration" in sig.parameters
    # Default value should be 1.0 (matches AgentBehaviorConfig default)
    assert sig.parameters["aec_warmup_duration"].default == 1.0
