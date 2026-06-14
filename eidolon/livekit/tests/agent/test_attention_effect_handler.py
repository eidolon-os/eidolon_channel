"""AttentionEffectHandler boundary tests."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import cast
from unittest.mock import MagicMock

from eidolon.livekit.agent.client_audio_state import ClientAudioState
from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.session import AttentionEffectHandler
from eidolon.livekit.agent.turn_policy import TurnPolicyRuntime
from eidolon.livekit.common.config import (
    AttentionPolicyConfig,
    InterruptMode,
    TurnPolicyConfig,
)


def _client_state(**kwargs) -> ClientAudioState:
    values = {
        "participant_identity": "alice",
        "input_mode": "auto",
        "playback_state": "agent_speaking",
        "received_at": time.monotonic(),
    }
    values.update(kwargs)
    return ClientAudioState(**values)


def _policy(
    *,
    enforce: bool = True,
    interrupt_mode: str = "balanced",
) -> TurnPolicyConfig:
    return replace(
        TurnPolicyConfig(),
        interrupt_mode=cast(InterruptMode, interrupt_mode),
        attention=replace(AttentionPolicyConfig(), enforce=enforce),
    )


def _handler(
    *,
    client_state=None,
    enforce: bool = True,
    agent_speaking: bool = True,
    eot_score: float = 0.0,
    interrupt_mode: str = "balanced",
):
    policy = _policy(enforce=enforce, interrupt_mode=interrupt_mode)
    timeline = TurnTimeline("turn-attention")
    on_duck = MagicMock()
    on_interrupt = MagicMock()
    handler = AttentionEffectHandler(
        turn_policy=policy,
        turn_runtime=TurnPolicyRuntime(policy),
        get_agent_speaking=lambda: agent_speaking,
        get_duck_active=lambda: False,
        latest_client_audio_state=lambda participant_identity: client_state,
        get_timeline=lambda: timeline,
        on_duck=on_duck,
        on_interrupt=on_interrupt,
        get_eot_score=lambda: eot_score,
    )
    return handler, timeline, on_duck, on_interrupt


def test_allows_eot_observes_substantive_playback_speech_without_eot() -> None:
    handler, timeline, on_duck, on_interrupt = _handler(client_state=_client_state())

    allowed = handler.allows_eot_check("那它有什么风险")

    assert allowed is False
    on_duck.assert_not_called()
    on_interrupt.assert_not_called()
    assert timeline.attrs["attention_admission"]["action"] == "observe"


def test_allows_eot_ducks_for_high_eot_playback_speech() -> None:
    handler, timeline, on_duck, on_interrupt = _handler(
        client_state=_client_state(),
        eot_score=0.82,
    )

    allowed = handler.allows_eot_check("那它有什么风险")

    assert allowed is True
    on_duck.assert_called_once_with()
    on_interrupt.assert_not_called()
    assert timeline.attrs["attention_admission"]["action"] == "duck_and_decide"
    assert timeline.attrs["attention_admission"]["reason"] == (
        "transcript_evidence:high_eot_transcript"
    )


def test_allows_eot_observes_short_low_score_playback_speech() -> None:
    handler, timeline, on_duck, on_interrupt = _handler(client_state=_client_state())

    allowed = handler.allows_eot_check("不是")

    assert allowed is False
    on_duck.assert_not_called()
    on_interrupt.assert_not_called()
    assert timeline.attrs["attention_admission"]["action"] == "observe"


def test_allows_eot_ducks_for_short_high_eot_playback_speech() -> None:
    handler, timeline, on_duck, on_interrupt = _handler(
        client_state=_client_state(),
        eot_score=0.82,
    )

    allowed = handler.allows_eot_check("不是")

    assert allowed is True
    on_duck.assert_called_once_with()
    on_interrupt.assert_not_called()
    assert timeline.attrs["attention_admission"]["action"] == "duck_and_decide"


def test_allows_eot_ducks_when_no_client_state() -> None:
    handler, timeline, on_duck, on_interrupt = _handler(client_state=None)

    allowed = handler.allows_eot_check("那它有什么风险")

    assert allowed is True
    on_duck.assert_called_once_with()
    on_interrupt.assert_not_called()
    assert timeline.attrs["attention_admission"]["action"] == "duck_and_decide"


def test_hard_stop_during_playback_allows_eot_without_ducking() -> None:
    handler, timeline, on_duck, on_interrupt = _handler(client_state=_client_state())

    allowed = handler.allows_eot_check("别说了")

    assert allowed is True
    on_duck.assert_not_called()
    on_interrupt.assert_not_called()
    assert timeline.attrs["attention_admission"]["action"] == "hard_interrupt"
    assert "interrupt_intent_admitted_at" in timeline.timestamps


def test_single_char_prefix_observes_without_direct_intent_mark() -> None:
    handler, timeline, on_duck, on_interrupt = _handler(client_state=_client_state())

    allowed = handler.allows_eot_check("换")

    assert allowed is False
    on_duck.assert_not_called()
    on_interrupt.assert_not_called()
    assert (
        timeline.attrs["attention_admission"]["reason"]
        == "client_playback_active_without_direct_signal"
    )
    assert "interrupt_intent_admitted_at" not in timeline.timestamps


def test_speaking_started_hard_client_interrupt_marks_and_interrupts() -> None:
    handler, timeline, on_duck, on_interrupt = _handler(
        client_state=_client_state(manual_interrupt=True),
    )

    handler.handle_speaking_started()

    on_duck.assert_not_called()
    on_interrupt.assert_called_once_with()
    assert "interrupt_started_at" in timeline.timestamps


def test_observe_only_rollout_records_but_allows_eot() -> None:
    handler, timeline, on_duck, on_interrupt = _handler(
        client_state=_client_state(),
        enforce=False,
    )

    allowed = handler.allows_eot_check("那它有什么风险")

    assert allowed is True
    on_duck.assert_not_called()
    on_interrupt.assert_not_called()
    assert timeline.attrs["attention_admission"]["enforced"] is False


def test_responsive_mode_does_not_disable_attention_enforcement() -> None:
    # Regression: responsive mode used to force attention_enforce=False, silently
    # overriding the operator's turn_policy.attention.enforce=true and disabling
    # the manual_interrupt / mic_muted gates (full-duplex barge-in bug). The mode
    # no longer overrides enforcement — it comes solely from config. With
    # enforce=True, a substantive playback overlap without EOT is observed (not
    # admitted as an EOT check), regardless of interrupt_mode.
    handler, timeline, on_duck, on_interrupt = _handler(
        client_state=_client_state(),
        enforce=True,
        interrupt_mode="responsive",
    )

    allowed = handler.allows_eot_check("不是")

    assert allowed is False
    on_duck.assert_not_called()
    on_interrupt.assert_not_called()
    assert timeline.attrs["attention_admission"]["enforced"] is True
    assert timeline.attrs["attention_admission"]["interrupt_mode"] == "responsive"
