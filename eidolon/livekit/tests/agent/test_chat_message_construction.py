"""G20 (2026-05-18): regression — ChatMessage construction.

The framework's ChatMessage is a pydantic model (livekit-agents 1.5.x). It
has no `.create()` classmethod; the older API was removed. Our
`InterruptedContextManager.inject()` previously called
`ChatMessage.create(...)` and crashed with AttributeError at the worst
possible moment (right after a real interrupt, when injecting the system
hint to the LLM).

Regression: directly constructing ChatMessage works and the resulting object
has the fields we care about.
"""

from __future__ import annotations


def test_chat_message_construct_directly() -> None:
    """G20: ChatMessage must be constructable via __init__ with role+content."""
    from livekit.agents.llm import ChatMessage

    msg = ChatMessage(role="system", content=["hello"])
    assert msg.role == "system"
    assert msg.content == ["hello"]
    # `.id` is auto-assigned by the pydantic default_factory
    assert msg.id


def test_chat_message_no_legacy_create_classmethod() -> None:
    """G20 documentation: `.create` does not exist in the current SDK.

    If a future SDK reintroduces it we want to KNOW (this test will start
    failing) so we can re-evaluate whether to migrate back.
    """
    from livekit.agents.llm import ChatMessage

    assert not hasattr(ChatMessage, "create"), (
        "ChatMessage gained a `.create` classmethod in livekit-agents; "
        "revisit eidolon/livekit/agent/context/interrupted.py "
        "and decide whether to migrate."
    )


def test_chat_message_accepts_long_chinese_content() -> None:
    """G20: smoke test on the exact construction path used in
    `InterruptedContextManager.inject()` — long Chinese content with mixed
    punctuation. Just exercising the constructor; no behavioural assertion
    beyond not raising."""
    from livekit.agents.llm import ChatMessage

    text = (
        "[系统提示] 上一轮助手回复被用户打断。"
        "请优先回答用户最新输入；除非用户明确要求继续上一轮，"
        "不要复述或主动续写被打断的内容，也不要提及这条系统提示。"
        "用户大约听到了前 2.3 秒。仅在判断用户是在追问上一轮时，"
        "把以下内容当作背景，不要直接复述："
        "「你好，今天天气不错，适合出去散步，记得带伞」"
    )
    msg = ChatMessage(role="system", content=[text])
    assert msg.role == "system"
    assert len(msg.content) == 1
    assert "[系统提示]" in msg.content[0]
    assert "不要复述" in msg.content[0]
