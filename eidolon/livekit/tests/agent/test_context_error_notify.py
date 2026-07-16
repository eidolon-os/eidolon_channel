"""An unresolved voiceprint context must not be a silent dead-end.

When the session context can't be resolved (for example, the user/device is
bound to a deleted agent), the terminal voiceprint gate rejects with a
``context_error`` reason. The session boundary:
  - logs the drop at ERROR (operator-actionable, not a routine voiceprint reject),
  - speaks a one-time fallback so the user knows something is wrong.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import Mock

from eidolon.livekit.agent.full_duplex import StreamingPipeline

_CONTEXT_REASON = "voiceprint_blocked:context_error:AdminResolveNotFound"


def _pipeline_with_session() -> StreamingPipeline:
    p = StreamingPipeline.__new__(StreamingPipeline)
    p._session = SimpleNamespace(say=Mock())
    return p


def test_context_error_gate_announces_once(caplog):
    p = _pipeline_with_session()
    with caplog.at_level(logging.ERROR, logger="agent"):
        p._ensure_turn_completion()._session_turns.notify_context_error_once(_CONTEXT_REASON)
    assert p._session.say.call_count == 1
    assert p._context_error_notified is True
    assert any("context unresolved" in r.message for r in caplog.records)


def test_context_error_announcement_is_once_only():
    p = _pipeline_with_session()
    boundary = p._ensure_turn_completion()._session_turns
    boundary.notify_context_error_once(_CONTEXT_REASON)
    boundary.notify_context_error_once(_CONTEXT_REASON)
    assert p._session.say.call_count == 1


def test_context_error_without_session_does_not_crash():
    p = StreamingPipeline.__new__(StreamingPipeline)
    p._session = None
    p._ensure_turn_completion()._session_turns.notify_context_error_once(_CONTEXT_REASON)


def test_say_failure_is_swallowed():
    p = StreamingPipeline.__new__(StreamingPipeline)
    p._session = SimpleNamespace(say=Mock(side_effect=RuntimeError("boom")))
    p._ensure_turn_completion()._session_turns.notify_context_error_once(_CONTEXT_REASON)
    assert p._context_error_notified is True
