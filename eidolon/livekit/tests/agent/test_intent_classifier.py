"""Interrupt intent classifier tests."""

from __future__ import annotations

import pytest

from eidolon_sdk.biz.dialogue_control import (
    InterruptIntent,
    LexiconInterruptClassifier,
    hard_stop_intent,
    hard_stop_prefix_intent,
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
        ("不要讲了", InterruptIntent.HARD_STOP),
        ("先别继续说了", InterruptIntent.HARD_STOP),
        ("please stop", InterruptIntent.HARD_STOP),
        ("that's enough", InterruptIntent.HARD_STOP),
        ("换个话题吧", InterruptIntent.UNCERTAIN),
        ("我们聊点", InterruptIntent.UNCERTAIN),
        ("这个先不聊", InterruptIntent.UNCERTAIN),
        ("跳过这个", InterruptIntent.UNCERTAIN),
        ("let's talk about something else", InterruptIntent.UNCERTAIN),
        ("next topic", InterruptIntent.UNCERTAIN),
        ("不是，我的意思是", InterruptIntent.UNCERTAIN),
        ("我刚才说", InterruptIntent.UNCERTAIN),
        ("我刚才", InterruptIntent.UNCERTAIN),
        ("不对不对你理解错了", InterruptIntent.UNCERTAIN),
        ("我纠正一下", InterruptIntent.UNCERTAIN),
        ("hold on a second", InterruptIntent.UNCERTAIN),
        ("let me rephrase", InterruptIntent.UNCERTAIN),
        ("that's not what i meant", InterruptIntent.UNCERTAIN),
        ("是我", InterruptIntent.UNCERTAIN),
        ("是不是应该这样", InterruptIntent.UNCERTAIN),
        ("对不对呢", InterruptIntent.UNCERTAIN),
        ("not enough detail", InterruptIntent.UNCERTAIN),
        ("嗯", InterruptIntent.BACKCHANNEL),
        ("咳咳", InterruptIntent.NOISE),
        ("帮我查一下天气", InterruptIntent.UNCERTAIN),
        ("亭子旁边有什么", InterruptIntent.UNCERTAIN),
        ("不要讲英文怎么说", InterruptIntent.UNCERTAIN),
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
    assert result.source.startswith("lexicon")


@pytest.mark.parametrize(
    "text,intent",
    [
        ("换个话题吧", InterruptIntent.TOPIC_SWITCH),
        ("我们聊点", InterruptIntent.TOPIC_SWITCH),
        ("不是，我的意思是", InterruptIntent.CORRECTION),
        ("我刚才说", InterruptIntent.CORRECTION),
        ("我纠正一下", InterruptIntent.CORRECTION),
        ("hold on a second", InterruptIntent.CORRECTION),
        ("let me rephrase", InterruptIntent.CORRECTION),
        ("嗯", InterruptIntent.BACKCHANNEL),
    ],
)
def test_fast_intents_mode_keeps_legacy_semantic_lexicon(
    text: str,
    intent: InterruptIntent,
) -> None:
    result = LexiconInterruptClassifier(fast_intents=True).classify(
        text,
        vad_active=True,
        agent_speaking=True,
        eot_score=0.0,
    )
    assert result.intent is intent


def test_repeated_noise_shape_thresholds_are_configurable() -> None:
    classifier = LexiconInterruptClassifier(
        repeated_noise_min_chars=3,
        repeated_noise_max_chars=4,
    )

    too_short = classifier.classify(
        "哈哈",
        vad_active=True,
        agent_speaking=True,
        eot_score=0.0,
    )
    in_range = classifier.classify(
        "哈哈哈",
        vad_active=True,
        agent_speaking=True,
        eot_score=0.0,
    )
    too_long = classifier.classify(
        "哈哈哈哈哈",
        vad_active=True,
        agent_speaking=True,
        eot_score=0.0,
    )

    assert too_short.intent is InterruptIntent.UNCERTAIN
    assert in_range.intent is InterruptIntent.NOISE
    assert too_long.intent is InterruptIntent.UNCERTAIN


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
        "别说",
        "先别说",
        "不要讲",
        "不要讲了",
        "先不要继续讲了",
        "停下",
        "你先停",
    ],
)
def test_hard_stop_prefix_detects_tier0_candidates(text: str) -> None:
    assert hard_stop_prefix_intent(text) is InterruptIntent.HARD_STOP


@pytest.mark.parametrize(
    "text",
    ["换", "好", "那它", "是不是", "我刚", "不要讲英文怎么说"],
)
def test_hard_stop_prefix_rejects_weak_or_ambient_text(text: str) -> None:
    assert hard_stop_prefix_intent(text) is None


def test_hard_stop_prefix_keeps_single_char_hotword_out_of_prefix_path() -> None:
    assert hard_stop_prefix_intent("停", min_chars=1, min_cjk_chars=1) is None


@pytest.mark.parametrize(
    "text",
    ["不要讲了", "不要讲啊", "你先不要说了", "别再继续说了"],
)
def test_hard_stop_intent_detects_speech_control_patterns(text: str) -> None:
    assert hard_stop_intent(text) is InterruptIntent.HARD_STOP


@pytest.mark.parametrize(
    "text",
    ["不要讲英文怎么说", "你不要说这是错的", "不要解释这个词是什么意思"],
)
def test_hard_stop_intent_rejects_non_control_negated_speech(text: str) -> None:
    assert hard_stop_intent(text) is None
