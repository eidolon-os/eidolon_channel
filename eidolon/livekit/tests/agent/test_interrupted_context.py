"""Unit tests for G2: `_snapshot_interrupted_context` API correctness.

The framework's `ChatContext.messages` is a METHOD, not a property. Pre-G2
code accessed `.messages` without calling it, then tried to `reversed()` a
method object → TypeError at runtime. The bug was latent because the only
caller is the EOT-cancel path, which never fired in the prior 0.55 VAD-gate
regime.

Regression assertions:
  * `_snapshot_interrupted_context` survives normal use (no TypeError).
  * It picks the last assistant message with non-empty text.
  * It tolerates a session.history that contains only user messages.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _make_pipeline_with_history(messages: list) -> "StreamingPipeline":
    """Build a minimally-initialised StreamingPipeline whose
    ``_session.history.messages()`` returns the supplied list."""
    from eidolon.livekit.agent.streaming import StreamingPipeline

    # Bypass full __init__ — _snapshot_interrupted_context only reads
    # ``self._session`` and ``self._get_eot_model()._config``.
    pipeline = StreamingPipeline.__new__(StreamingPipeline)
    pipeline._session = MagicMock()
    pipeline._session.history.messages = MagicMock(return_value=messages)
    pipeline._last_interrupted_context = None

    # Stub _get_eot_model() → cfg with interrupted_context_enabled=True
    cfg = SimpleNamespace(interrupted_context_enabled=True)
    eot = SimpleNamespace(_config=cfg)
    pipeline._get_eot_model = MagicMock(return_value=eot)

    return pipeline


def _msg(role: str, text: str) -> SimpleNamespace:
    return SimpleNamespace(role=role, text_content=text)


def test_picks_last_assistant_message() -> None:
    """G2 happy path: walking history in reverse finds the most-recent
    assistant turn with non-empty text."""
    pipeline = _make_pipeline_with_history([
        _msg("user", "hi"),
        _msg("assistant", "old reply"),
        _msg("user", "follow-up"),
        _msg("assistant", "newer reply"),
    ])

    pipeline._snapshot_interrupted_context()

    assert pipeline._last_interrupted_context is not None
    assert pipeline._last_interrupted_context["text"] == "newer reply"


def test_method_call_not_attribute_access() -> None:
    """G2 regression: ``messages`` must be called as a method, not a property.

    Asserts that .messages() is the call site (any future regression that
    reverts to attribute access will fail this — MagicMock's method tracker
    would not record a call)."""
    pipeline = _make_pipeline_with_history([
        _msg("assistant", "hello"),
    ])

    pipeline._snapshot_interrupted_context()

    pipeline._session.history.messages.assert_called_once_with()


def test_skips_empty_text_messages() -> None:
    """Empty text_content should be skipped — we want the last assistant
    turn that actually said something."""
    pipeline = _make_pipeline_with_history([
        _msg("assistant", "non-empty"),
        _msg("assistant", ""),
    ])

    pipeline._snapshot_interrupted_context()

    assert pipeline._last_interrupted_context is not None
    assert pipeline._last_interrupted_context["text"] == "non-empty"


def test_no_assistant_messages_leaves_context_none() -> None:
    """Pure-user history → nothing to snapshot, _last_interrupted_context
    stays None."""
    pipeline = _make_pipeline_with_history([
        _msg("user", "first"),
        _msg("user", "second"),
    ])

    pipeline._snapshot_interrupted_context()

    assert pipeline._last_interrupted_context is None


def test_disabled_config_is_noop() -> None:
    """If ``interrupted_context_enabled=False`` the function returns
    immediately without touching session.history."""
    from eidolon.livekit.agent.streaming import StreamingPipeline

    pipeline = StreamingPipeline.__new__(StreamingPipeline)
    pipeline._session = MagicMock()
    pipeline._last_interrupted_context = None
    cfg = SimpleNamespace(interrupted_context_enabled=False)
    pipeline._get_eot_model = MagicMock(return_value=SimpleNamespace(_config=cfg))

    pipeline._snapshot_interrupted_context()

    pipeline._session.history.messages.assert_not_called()
    assert pipeline._last_interrupted_context is None
