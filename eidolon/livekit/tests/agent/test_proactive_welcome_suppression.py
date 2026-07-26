"""Opening behavior derived from the exact session intent.

A proactive_initiated session is woken to deliver a report; the report (spoken by
the proactive consumer) is the opening, so the welcome must be suppressed —
otherwise the device says "你好…" and then the report. Explicit user and verified
presence sessions keep their welcome.
"""

from __future__ import annotations

from eidolon_sdk.biz.contracts import (
    SESSION_INTENT_PRESENCE,
    SESSION_INTENT_PROACTIVE,
    SESSION_INTENT_USER_INITIATED,
)

from eidolon.livekit.agent.full_duplex import StreamingPipeline


def _pipeline(*, intent: str, welcome: str) -> StreamingPipeline:
    p = StreamingPipeline.__new__(StreamingPipeline)
    p._session_intent = intent
    p._welcome_message = welcome
    return p


def test_proactive_suppresses_welcome():
    p = _pipeline(intent=SESSION_INTENT_PROACTIVE, welcome="你好！我是你的 AI 助手")
    assert p._welcome_on_enter_text() is None


def test_user_initiated_keeps_welcome():
    p = _pipeline(
        intent=SESSION_INTENT_USER_INITIATED,
        welcome="你好！我是你的 AI 助手",
    )
    assert p._welcome_on_enter_text() == "你好！我是你的 AI 助手"


def test_presence_initiated_keeps_welcome():
    p = _pipeline(
        intent=SESSION_INTENT_PRESENCE,
        welcome="你好！我是你的 AI 助手",
    )
    assert p._welcome_on_enter_text() == "你好！我是你的 AI 助手"


def test_user_initiated_empty_welcome_is_silent():
    p = _pipeline(intent=SESSION_INTENT_USER_INITIATED, welcome="")
    assert p._welcome_on_enter_text() is None
