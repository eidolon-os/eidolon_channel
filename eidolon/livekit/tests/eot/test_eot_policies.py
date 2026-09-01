"""Tests for acoustic, timing, finality and model-score EOT policies."""

from __future__ import annotations

import time

import pytest

from eidolon.livekit.plugins.eot import (
    ASRFinalCutPolicy,
    ASRStabilityPolicy,
    DuplicateTextPolicy,
    EOTScorePolicy,
    EOTScoreSemanticPolicy,
    InterruptCooldownPolicy,
    MaxDurationPolicy,
    MinIntervalPolicy,
    MinSpeakingDurationPolicy,
    PolicyChain,
    TurnDetectionStateManager,
    TurnEndPolicy,
    VADStabilityPolicy,
    VADStalePolicy,
)


def state_with_text(text: str = "任意文本") -> TurnDetectionStateManager:
    state = TurnDetectionStateManager()
    state.update_asr(text, is_final=False)
    return state


def test_interrupt_cooldown_policy_blocks_recent_cut(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "time", lambda: 10.0)
    state = state_with_text()
    state._interrupt.last_interrupt_time = 9.8

    decision = InterruptCooldownPolicy().check(state, TurnEndPolicy())
    assert decision is not None and decision.should_cut is False


def test_min_interval_policy_blocks_early_recut(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "time", lambda: 10.0)
    state = state_with_text()
    state._sentence.last_cut_time = 9.8

    decision = MinIntervalPolicy().check(state, TurnEndPolicy())
    assert decision is not None and decision.should_cut is False


def test_duplicate_policy_blocks_streaming_revision() -> None:
    state = state_with_text("请查询天气。")
    state._sentence.last_cut_text = "请查询天气"

    decision = DuplicateTextPolicy(similarity_threshold=0.8).check(state, TurnEndPolicy())
    assert decision is not None and decision.should_cut is False


def test_min_speaking_duration_uses_audio_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "time", lambda: 10.0)
    state = state_with_text()
    state.vad_active = True
    state._vad.active_since = 9.95

    decision = MinSpeakingDurationPolicy(0.1).check(state, TurnEndPolicy())
    assert decision is not None and decision.should_cut is False


def test_min_speaking_duration_allows_long_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "time", lambda: 10.0)
    state = state_with_text()
    state.vad_active = True
    state._vad.active_since = 9.5

    assert MinSpeakingDurationPolicy(0.1).check(state, TurnEndPolicy()) is None


def test_vad_stability_blocks_rapid_transitions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "time", lambda: 10.0)
    state = state_with_text()
    state.vad_active = True
    state._vad.transition_times.extend((9.4, 9.6, 9.8))

    decision = VADStabilityPolicy(1.0, 3).check(state, TurnEndPolicy())
    assert decision is not None and decision.should_cut is False


def test_asr_stability_blocks_fresh_final(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "time", lambda: 10.0)
    state = state_with_text()
    state._asr.is_final = True
    state._asr.stable_since = 9.95

    decision = ASRStabilityPolicy(stability_short_sec=0.1).check(state, TurnEndPolicy())
    assert decision is not None and decision.should_cut is False


def test_max_duration_is_an_active_speech_safety_net(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "time", lambda: 20.0)
    state = TurnDetectionStateManager(max_sentence_duration=5.0)
    state.update_asr("长文本", is_final=False)
    state.vad_active = True
    state._sentence.start_time = 10.0

    decision = MaxDurationPolicy().check(state, TurnEndPolicy())
    assert decision is not None and decision.should_cut is True


def test_vad_stale_policy_uses_silence_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "time", lambda: 10.0)
    state = TurnDetectionStateManager(vad_stale_timeout=0.3)
    state.update_asr("文本", is_final=False)
    state.vad_active = True
    state._vad.last_active_time = 9.5

    decision = VADStalePolicy().check(state, TurnEndPolicy())
    assert decision is not None and decision.should_cut is True


def test_asr_final_cut_requires_silence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "time", lambda: 10.0)
    state = TurnDetectionStateManager()
    state.update_asr("文本", is_final=True)
    state._vad.last_active_time = 9.5

    decision = ASRFinalCutPolicy(0.2).check(state, TurnEndPolicy())
    assert decision is not None and decision.should_cut is True


def test_eot_score_policy_uses_numeric_dynamic_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "time", lambda: 10.0)
    state = state_with_text("同分文本不应有词法差异")
    state.update_eot_score(0.9)
    state._vad.last_active_time = 9.5

    decision = EOTScorePolicy().check(state, TurnEndPolicy())
    assert decision is not None and decision.should_cut is True


@pytest.mark.parametrize(("vad_active", "score", "cuts"), [(True, 0.6, True), (False, 0.6, False)])
def test_semantic_score_policy_uses_only_model_score_and_vad(
    vad_active: bool,
    score: float,
    cuts: bool,
) -> None:
    state = state_with_text()
    state.vad_active = vad_active
    state.update_eot_score(score)

    decision = EOTScoreSemanticPolicy(base_threshold=0.7, vad_active_delta=0.1).check(
        state,
        TurnEndPolicy(),
    )
    assert (decision is not None and decision.should_cut) is cuts


def test_semantic_chain_contains_only_live_signal_policies() -> None:
    names = {
        type(policy).__name__
        for policy in PolicyChain.for_semantic_interruption()._policies
    }
    assert names == {
        "InterruptCooldownPolicy",
        "MinSpeakingDurationPolicy",
        "VADStabilityPolicy",
        "MinIntervalPolicy",
        "DuplicateTextPolicy",
        "MaxDurationPolicy",
        "VADStalePolicy",
        "EOTScoreSemanticPolicy",
    }


def test_normal_chain_contains_finality_and_silence_policies() -> None:
    names = {type(policy).__name__ for policy in PolicyChain.for_normal_turn_end()._policies}
    assert names == {
        "InterruptCooldownPolicy",
        "MinSpeakingDurationPolicy",
        "VADStabilityPolicy",
        "MinIntervalPolicy",
        "DuplicateTextPolicy",
        "ASRStabilityPolicy",
        "MaxDurationPolicy",
        "VADStalePolicy",
        "ASRFinalCutPolicy",
        "EOTScorePolicy",
    }
