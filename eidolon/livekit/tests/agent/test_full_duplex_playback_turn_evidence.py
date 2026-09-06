import pytest

from eidolon.livekit.agent.full_duplex.playback_turn_evidence import (
    resolve_playback_turn_decision,
)
from eidolon.livekit.agent.turn_policy import Action, Decision, InterruptIntent


@pytest.mark.parametrize('intent', [InterruptIntent.NORMAL_INTERRUPT, InterruptIntent.TOPIC_SWITCH])
def test_topic_switch_completed_playback_turn_continues_to_llm(intent) -> None:
    decision = Decision(
        action=Action.CANCEL,
        reason="topic_switch",
        intent=intent,
        topic_switch_hint=True,
    )

    resolution = resolve_playback_turn_decision(decision)

    assert resolution.should_apply is True
    assert resolution.reason == "topic_switch"
    assert resolution.continue_to_llm is True


@pytest.mark.parametrize('intent', [InterruptIntent.NORMAL_INTERRUPT, InterruptIntent.CORRECTION])
def test_correction_completed_playback_turn_continues_to_llm(intent) -> None:
    decision = Decision(
        action=Action.CANCEL,
        reason="correction",
        intent=intent,
        correction_hint=True,
    )

    resolution = resolve_playback_turn_decision(decision)

    assert resolution.should_apply is True
    assert resolution.continue_to_llm is True


def test_hard_stop_completed_playback_turn_resolves_without_llm() -> None:
    decision = Decision(
        action=Action.CANCEL,
        reason="hard_stop",
        intent=InterruptIntent.HARD_STOP,
    )

    resolution = resolve_playback_turn_decision(decision)

    assert resolution.should_apply is True
    assert resolution.reason == "hard_stop"
    assert resolution.continue_to_llm is False


def test_rollback_completed_playback_turn_resolves_without_llm() -> None:
    decision = Decision(
        action=Action.ROLLBACK,
        reason="backchannel",
        intent=InterruptIntent.BACKCHANNEL,
    )

    resolution = resolve_playback_turn_decision(decision)

    assert resolution.should_apply is True
    assert resolution.reason == "backchannel"
    assert resolution.continue_to_llm is False


def test_normal_cancel_without_phrase_hint_resolves_as_user_turn() -> None:
    decision = Decision(
        action=Action.CANCEL,
        reason="final_eot_score_high",
        intent=InterruptIntent.NORMAL_INTERRUPT,
    )

    resolution = resolve_playback_turn_decision(decision)

    assert resolution.should_apply is True
    assert resolution.reason == "final_eot_score_high"
    assert resolution.continue_to_llm is True


def test_hold_decision_does_not_resolve_playback_turn() -> None:
    decision = Decision(
        action=Action.HOLD,
        reason="wait_for_better_evidence",
        intent=InterruptIntent.UNCERTAIN,
    )

    resolution = resolve_playback_turn_decision(decision)

    assert resolution.should_apply is False
    assert resolution.reason == "decision_not_resolvable:hold"
    assert resolution.continue_to_llm is False


def test_missing_decision_does_not_resolve_playback_turn() -> None:
    resolution = resolve_playback_turn_decision(None)

    assert resolution.should_apply is False
    assert resolution.reason == "no_decision"
    assert resolution.continue_to_llm is False
