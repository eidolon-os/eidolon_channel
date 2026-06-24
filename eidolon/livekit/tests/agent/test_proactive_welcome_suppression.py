"""Proactive session suppresses the canned welcome (plan §4.3.1 / §5.1).

A proactive_initiated session is woken to deliver a report; the report (spoken by
the proactive consumer) is the opening, so the welcome must be suppressed —
otherwise the device says "你好…" and then the report. A user_initiated session
keeps its welcome.
"""

from __future__ import annotations

from eidolon.livekit.agent.streaming import StreamingPipeline


def _pipeline(*, proactive: bool, welcome: str) -> StreamingPipeline:
    p = StreamingPipeline.__new__(StreamingPipeline)
    p._is_proactive = proactive
    p._welcome_message = welcome
    return p


def test_proactive_suppresses_welcome():
    p = _pipeline(proactive=True, welcome="你好！我是你的 AI 助手")
    assert p._welcome_on_enter_text() is None


def test_user_initiated_keeps_welcome():
    p = _pipeline(proactive=False, welcome="你好！我是你的 AI 助手")
    assert p._welcome_on_enter_text() == "你好！我是你的 AI 助手"


def test_user_initiated_empty_welcome_is_silent():
    p = _pipeline(proactive=False, welcome="")
    assert p._welcome_on_enter_text() is None
