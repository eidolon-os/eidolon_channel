"""Interrupt intent classifier tests."""

from __future__ import annotations

import pytest

from eidolon.livekit.agent.turn_policy import (
    InterruptIntent,
    LexiconInterruptClassifier,
)
from eidolon.livekit.common.config import InterruptPolicyConfig


@pytest.fixture
def classifier() -> LexiconInterruptClassifier:
    return LexiconInterruptClassifier(InterruptPolicyConfig())


@pytest.mark.parametrize(
    "text,intent",
    [
        ("停一下", InterruptIntent.HARD_STOP),
        ("别说了我想换个", InterruptIntent.HARD_STOP),
        ("换个话题吧", InterruptIntent.TOPIC_SWITCH),
        ("半个话题吧", InterruptIntent.TOPIC_SWITCH),
        ("我们聊点", InterruptIntent.TOPIC_SWITCH),
        ("不是，我的意思是", InterruptIntent.CORRECTION),
        ("我刚才说", InterruptIntent.CORRECTION),
        ("我刚才", InterruptIntent.CORRECTION),
        ("是我刚", InterruptIntent.CORRECTION),
        ("是我", InterruptIntent.UNCERTAIN),
        ("嗯", InterruptIntent.BACKCHANNEL),
        ("咳咳", InterruptIntent.NOISE),
        ("帮我查一下天气", InterruptIntent.UNCERTAIN),
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
