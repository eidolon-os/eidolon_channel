"""task #9: an unresolved-context turn drop must not be a silent dead-end.

When the session context can't be resolved (e.g. the user/device is bound to a
deleted agent → AdminResolveNotFound), every turn is cleared with a
``context_error`` reason. Before the fix the user just connected, heard the
welcome, and then got silence forever. Now the pipeline:
  - logs the drop at ERROR (operator-actionable, not a routine voiceprint reject),
  - speaks a one-time fallback so the user knows something is wrong,
without firing on ordinary turn clears (interrupts etc.).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import Mock

from eidolon.livekit.agent.full_duplex import StreamingPipeline

_CONTEXT_REASON = "voiceprint_blocked:context_error:AdminResolveNotFound"
_ROUTINE_REASON = "interrupt_cancel"


def _pipeline_with_session() -> StreamingPipeline:
    p = StreamingPipeline.__new__(StreamingPipeline)
    p._session = SimpleNamespace(clear_user_turn=Mock(), say=Mock())
    return p


def test_context_error_clear_announces_once(caplog):
    p = _pipeline_with_session()
    with caplog.at_level(logging.ERROR, logger="agent"):
        p._ensure_turn_completion().clear_session_user_turn(_CONTEXT_REASON)
    assert p._session.say.call_count == 1
    assert p._context_error_notified is True
    assert any("context unresolved" in r.message for r in caplog.records)


def test_context_error_announcement_is_once_only():
    p = _pipeline_with_session()
    p._ensure_turn_completion().clear_session_user_turn(_CONTEXT_REASON)
    p._ensure_turn_completion().clear_session_user_turn(_CONTEXT_REASON)  # second drop, same session
    assert p._session.say.call_count == 1


def test_routine_clear_does_not_announce():
    p = _pipeline_with_session()
    p._ensure_turn_completion().clear_session_user_turn(_ROUTINE_REASON)
    p._session.say.assert_not_called()
    assert getattr(p, "_context_error_notified", False) is False


def test_context_error_without_session_does_not_crash():
    p = StreamingPipeline.__new__(StreamingPipeline)
    p._session = None
    p._ensure_turn_completion().clear_session_user_turn(_CONTEXT_REASON)  # must not raise


def test_say_failure_is_swallowed():
    p = StreamingPipeline.__new__(StreamingPipeline)
    p._session = SimpleNamespace(
        clear_user_turn=Mock(), say=Mock(side_effect=RuntimeError("boom"))
    )
    p._ensure_turn_completion().clear_session_user_turn(_CONTEXT_REASON)  # must not raise
    assert p._context_error_notified is True
