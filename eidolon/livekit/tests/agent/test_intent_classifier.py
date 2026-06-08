"""Interrupt intent classifier tests."""

from __future__ import annotations

import pytest

from eidolon.livekit.agent.turn_policy import (
    InterruptIntent,
    LexiconInterruptClassifier,
)
from eidolon.livekit.agent.turn_policy.intent_classifier import (
    is_semantic_interrupt_prefix,
)
from eidolon.livekit.common.config.defaults import (
    DEFAULT_CORRECTION_EXCLUSION_LEXICON,
    DEFAULT_CORRECTION_LEXICON,
    DEFAULT_HARD_STOP_LEXICON,
    DEFAULT_TOPIC_SWITCH_LEXICON,
)


@pytest.fixture
def classifier() -> LexiconInterruptClassifier:
    return LexiconInterruptClassifier()


@pytest.mark.parametrize(
    "text,intent",
    [
        ("停一下", InterruptIntent.HARD_STOP),
        ("亭", InterruptIntent.HARD_STOP),
        ("你先停一下", InterruptIntent.HARD_STOP),
        ("暂停一下", InterruptIntent.HARD_STOP),
        ("可以了先到这里", InterruptIntent.HARD_STOP),
        ("别说了我想换个", InterruptIntent.HARD_STOP),
        ("please stop", InterruptIntent.HARD_STOP),
        ("that's enough", InterruptIntent.HARD_STOP),
        ("换个话题吧", InterruptIntent.TOPIC_SWITCH),
        ("半个话题吧", InterruptIntent.TOPIC_SWITCH),
        ("我们聊点", InterruptIntent.TOPIC_SWITCH),
        ("这个先不聊", InterruptIntent.TOPIC_SWITCH),
        ("别说这个", InterruptIntent.TOPIC_SWITCH),
        ("跳过这个", InterruptIntent.TOPIC_SWITCH),
        ("let's talk about something else", InterruptIntent.TOPIC_SWITCH),
        ("next topic", InterruptIntent.TOPIC_SWITCH),
        ("不是，我的意思是", InterruptIntent.CORRECTION),
        ("我刚才说", InterruptIntent.CORRECTION),
        ("我刚才", InterruptIntent.CORRECTION),
        ("是我刚", InterruptIntent.CORRECTION),
        ("不对不对你理解错了", InterruptIntent.CORRECTION),
        ("我纠正一下", InterruptIntent.CORRECTION),
        ("hold on a second", InterruptIntent.CORRECTION),
        ("let me rephrase", InterruptIntent.CORRECTION),
        ("that's not what i meant", InterruptIntent.CORRECTION),
        ("是我", InterruptIntent.UNCERTAIN),
        ("是不是应该这样", InterruptIntent.UNCERTAIN),
        ("对不对呢", InterruptIntent.UNCERTAIN),
        ("not enough detail", InterruptIntent.UNCERTAIN),
        ("嗯", InterruptIntent.BACKCHANNEL),
        ("咳咳", InterruptIntent.NOISE),
        ("帮我查一下天气", InterruptIntent.UNCERTAIN),
        ("亭子旁边有什么", InterruptIntent.UNCERTAIN),
    ],
)
def test_lexicon_classifier_intents(
    classifier: LexiconInterruptClassifier, text: str, intent: InterruptIntent
) -> None:
    result = classifier.classify(
        text,
        vad_active=True,
        agent_speaking=True,
        eot_score=0.0,
    )
    assert result.intent is intent
    assert result.source == "lexicon"


def test_interrupt_lexicons_are_non_empty_and_unique() -> None:
    for lexicon in (
        DEFAULT_HARD_STOP_LEXICON,
        DEFAULT_TOPIC_SWITCH_LEXICON,
        DEFAULT_CORRECTION_LEXICON,
        DEFAULT_CORRECTION_EXCLUSION_LEXICON,
    ):
        assert all(item.strip() for item in lexicon)
        assert len(lexicon) == len(set(lexicon))


@pytest.mark.parametrize(
    "text",
    [
        "换个",
        "我们换",
        "我刚",
        "let me",
    ],
)
def test_semantic_interrupt_prefix_detects_redirect_candidates(text: str) -> None:
    assert is_semantic_interrupt_prefix(text)


@pytest.mark.parametrize("text", ["换", "好", "那它", "是不是"])
def test_semantic_interrupt_prefix_rejects_weak_or_ambient_text(text: str) -> None:
    assert not is_semantic_interrupt_prefix(text)


def test_semantic_interrupt_prefix_accepts_attention_early_duck_only() -> None:
    assert is_semantic_interrupt_prefix(
        "换",
        min_chars=1,
        include_attention_early_duck=True,
    )
