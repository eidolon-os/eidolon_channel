from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from eidolon_sdk.biz.dialogue_control import InterruptIntent
from eidolon.livekit.agent.speaker_verified.semantics import CandidateSemantics
from eidolon.livekit.agent.turn_policy.intent_classifier import IntentClassification
from eidolon.livekit.plugins.eot import ChineseModel


@pytest.mark.asyncio
async def test_candidate_eot_reuses_score_without_text_log_or_history(monkeypatch, caplog):
    model = ChineseModel()
    scorer = Mock(return_value=0.9)
    monkeypatch.setattr(model._context_eot._base_eot, "p_complete_score", scorer)
    caplog.set_level("DEBUG")
    adapter = CandidateSemantics(model, classifier=AsyncMock(), assistant_text=lambda _: "")
    assert await adapter.is_complete("尚未获准的发言")
    scorer.assert_called_once_with("尚未获准的发言")
    assert "尚未获准的发言" not in caplog.text
    assert list(model._context_eot._dialogue_history) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "intent,action",
    [
        (InterruptIntent.NORMAL_INTERRUPT, "reply"),
        (InterruptIntent.CORRECTION, "reply"),
        (InterruptIntent.TOPIC_SWITCH, "reply"),
        (InterruptIntent.HARD_STOP, "stop_output"),
        (InterruptIntent.BACKCHANNEL, "no_action"),
        (InterruptIntent.UNCERTAIN, "no_action"),
    ],
)
async def test_existing_intent_results_map_to_actions_without_playback_effect(intent, action):
    classifier = SimpleNamespace(
        classify=AsyncMock(return_value=IntentClassification(intent, 0.9, "test", "test"))
    )
    model = SimpleNamespace(score_candidate=Mock(), unlikely_threshold=AsyncMock())
    adapter = CandidateSemantics(model, classifier=classifier, assistant_text=lambda _: "当前播报")
    output = SimpleNamespace(done=lambda: False, interrupt=Mock())
    assert await adapter.classify("测试", output) == action
    classifier.classify.assert_awaited_once_with("测试", assistant_text="当前播报")
    output.interrupt.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "intent,action",
    [
        (InterruptIntent.NORMAL_INTERRUPT, "reply"),
        (InterruptIntent.HARD_STOP, "stop_output"),
    ],
)
async def test_idle_request_still_distinguishes_reply_from_stop(intent, action):
    classifier = SimpleNamespace(
        classify=AsyncMock(return_value=IntentClassification(intent, 0.9, "test", "test"))
    )
    adapter = CandidateSemantics(None, classifier=classifier, assistant_text=lambda _: "")
    assert await adapter.classify("用户意图", None) == action
    classifier.classify.assert_awaited_once_with("用户意图", assistant_text="")


@pytest.mark.asyncio
async def test_missing_output_context_never_guesses_takeover():
    classifier = SimpleNamespace(classify=AsyncMock())
    adapter = CandidateSemantics(None, classifier=classifier, assistant_text=lambda _: "")
    assert await adapter.classify("继续", SimpleNamespace(done=lambda: False)) == "no_action"
    classifier.classify.assert_not_called()
