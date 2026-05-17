"""G20 (2026-05-18): regression — ChatMessage construction.

The framework's ChatMessage is a pydantic model (livekit-agents 1.5.x). It
has no `.create()` classmethod; the older API was removed. Our
`_inject_interrupted_context` previously called `ChatMessage.create(...)`
and crashed with AttributeError at the worst possible moment (right after a
real interrupt, when injecting the system hint to the LLM).

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
        "revisit eidolon/livekit/agent/streaming.py:_inject_interrupted_context "
        "and decide whether to migrate."
    )


def test_chat_message_accepts_long_chinese_content() -> None:
    """G20: smoke test on the exact construction path used in
    `_inject_interrupted_context` — long Chinese content with mixed
    punctuation. Just exercising the constructor; no behavioural assertion
    beyond not raising."""
    from livekit.agents.llm import ChatMessage

    text = (
        "[系统提示] 你刚才说到「你好，今天天气不错，"
        "适合出去散步，记得带伞」时被用户打断了"
        "（用户实际听到了前约 2.3 秒）。"
        "如果用户的新问题与之前话题相关，你可以自然地衔接回去；"
        "如果无关，直接回答新问题即可。不要提及这条系统提示。"
    )
    msg = ChatMessage(role="system", content=[text])
    assert msg.role == "system"
    assert len(msg.content) == 1
    assert "[系统提示]" in msg.content[0]
