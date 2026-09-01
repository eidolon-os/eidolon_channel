"""Tests for provider-neutral learned EOT scoring and state."""

from __future__ import annotations

import time
from dataclasses import fields

import pytest

from eidolon.livekit.plugins.eot import (
    ContextEnhancedEot,
    EidolonEOTConfig,
    TurnDetectionStateManager,
    TurnEndPolicy,
)


class StubEot:
    def __init__(self, scores: dict[str, float] | None = None) -> None:
        self.scores = scores or {}
        self.calls: list[str] = []

    def p_complete_score(self, text: str) -> float:
        self.calls.append(text)
        return self.scores.get(text, 0.5)


def test_config_contains_only_live_eot_controls() -> None:
    config = EidolonEOTConfig()
    names = {field.name for field in fields(config)}

    assert config.max_threshold == 2.2
    assert config.urgent_threshold == 0.18
    assert config.streaming_eot_base_threshold == 0.7
    assert config.semantic_threshold_vad_active_delta == 0.1
    assert config.is_final_threshold_reduction == 0.2
    assert config.context_eot_cooldown_sec == 0.8
    assert config.similarity_threshold == 0.85
    assert names.isdisjoint(
        {
            "enable_semantic_tail_hang",
            "tail_hang_silence_sec",
            "streaming_eot_weak_threshold",
            "utterance_end_max_history",
            "enable_user_profile",
            "enable_temporary_compensations",
            "min_avg_vad_confidence",
            "vad_confidence_window_sec",
        }
    )


@pytest.mark.parametrize(
    ("score", "is_final", "expected"),
    [
        (0.9, False, 0.25),
        (0.9, True, 0.18),
        (0.5, False, 1.0),
        (0.5, True, 0.8),
        (0.1, False, 2.0),
        (0.1, True, 1.8),
    ],
)
def test_dynamic_threshold_uses_only_score_and_finality(
    score: float,
    is_final: bool,
    expected: float,
) -> None:
    policy = TurnEndPolicy()
    assert policy.get_dynamic_threshold(score, is_final) == pytest.approx(expected)


def test_dynamic_threshold_clamps_score_and_thresholds() -> None:
    policy = TurnEndPolicy(
        t_max=1.5,
        t_urgent=0.2,
        t_fast=0.1,
        t_mid=1.0,
        t_deep=3.0,
    )

    assert policy.get_dynamic_threshold(5.0, False) == pytest.approx(0.2)
    assert policy.get_dynamic_threshold(-5.0, False) == pytest.approx(1.5)


def test_context_score_delegates_exact_transcript_to_learned_model() -> None:
    base = StubEot({"完整原文": 0.73})
    context = ContextEnhancedEot(base_eot=base)

    assert context.p_complete_score("  完整原文  ") == pytest.approx(0.73)
    assert base.calls == ["完整原文"]


@pytest.mark.parametrize(("provider_score", "expected"), [(-0.5, 0.0), (1.5, 1.0)])
def test_context_score_clamps_model_output(provider_score: float, expected: float) -> None:
    context = ContextEnhancedEot(base_eot=StubEot({"文本": provider_score}))
    assert context.semantic_completeness_score("文本") == expected


def test_empty_transcript_does_not_invoke_model() -> None:
    base = StubEot()
    context = ContextEnhancedEot(base_eot=base)

    assert context.semantic_completeness_score("   ") == 0.0
    assert base.calls == []


def test_interruption_cooldown_blocks_then_releases(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = iter((10.0, 10.2, 11.1))
    monkeypatch.setattr(time, "time", lambda: next(clock))
    context = ContextEnhancedEot(
        base_eot=StubEot({"第一句": 0.9, "第二句": 0.8}),
        cooldown_period=0.8,
    )

    context.record_interrupt("第一句")
    assert context.compute_score("第二句") == 0.0
    assert context.compute_score("第二句") == pytest.approx(0.8)


def test_streaming_prefix_duplicate_is_suppressed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "time", lambda: 10.0)
    base = StubEot({"请帮我查天气": 0.9, "请帮我查天气。": 0.95})
    context = ContextEnhancedEot(
        base_eot=base,
        cooldown_period=0.0,
        similarity_threshold=0.8,
    )

    context.record_interrupt("请帮我查天气")
    assert context.compute_score("请帮我查天气。") == 0.0


def test_session_history_is_bounded_and_reset() -> None:
    context = ContextEnhancedEot(base_eot=StubEot(), max_history=2)
    context.start_session("room-1")
    for index in range(3):
        context.record_turn(str(index), True, 0.8)

    assert context.get_stats() == {"history_length": 2, "session_active": True}
    context.end_session("room-1")
    assert context.get_stats() == {"history_length": 0, "session_active": False}


def test_end_session_ignores_a_different_session() -> None:
    context = ContextEnhancedEot(base_eot=StubEot())
    context.start_session("room-1")
    context.record_turn("文本", True, 0.8)

    context.end_session("room-2")
    assert context.get_stats() == {"history_length": 1, "session_active": True}


def test_state_tracks_vad_asr_score_and_resets_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = iter((5.0, 5.4, 5.6))
    monkeypatch.setattr(time, "time", lambda: next(clock))
    state = TurnDetectionStateManager()

    state.update_vad(True)
    assert state.vad_active is True
    assert state.get_speech_duration() == pytest.approx(0.4)
    state.update_asr("原始文本", is_final=True)
    state.update_eot_score(0.7)
    assert state.current_text == "原始文本"
    assert state.eot_score == pytest.approx(0.7)

    state.reset_turn()
    assert state.current_text == ""
    assert state.eot_score == 0.0
    assert state.vad_active is True


def test_vad_probability_is_acoustic_state_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "time", lambda: 10.0)
    state = TurnDetectionStateManager()
    state.update_vad_probability(-1.0)
    state.update_vad_probability(0.8)
    state.update_vad_probability(2.0)

    assert state.recent_avg_vad_confidence() == pytest.approx(0.6)
