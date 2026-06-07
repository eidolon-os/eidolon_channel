"""AttentionEffectHandler boundary tests."""

from __future__ import annotations

import time
from dataclasses import replace
from unittest.mock import MagicMock

from eidolon.livekit.agent.client_audio_state import ClientAudioState
from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.session import AttentionEffectHandler
from eidolon.livekit.agent.turn_policy import TurnPolicyRuntime
from eidolon.livekit.common.config import AttentionPolicyConfig, TurnPolicyConfig


def _client_state(**kwargs) -> ClientAudioState:
    values = {
        "participant_identity": "alice",
        "input_mode": "auto",
        "playback_state": "agent_speaking",
        "received_at": time.monotonic(),
    }
    values.update(kwargs)
    return ClientAudioState(**values)


def _policy(*, enforce: bool = True) -> TurnPolicyConfig:
    return replace(
        TurnPolicyConfig(),
        attention=replace(AttentionPolicyConfig(), enforce=enforce),
    )


def _handler(*, client_state=None, enforce: bool = True, agent_speaking: bool = True):
    policy = _policy(enforce=enforce)
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
    )
    return handler, timeline, on_duck, on_interrupt


def test_allows_eot_observes_ambient_playback_speech() -> None:
    handler, timeline, on_duck, on_interrupt = _handler(client_state=_client_state())

    allowed = handler.allows_eot_check("那它有什么风险")

    assert allowed is False
    on_duck.assert_not_called()
    on_interrupt.assert_not_called()
    assert timeline.attrs["attention_admission"]["action"] == "observe"
    assert timeline.attrs["attention_admission"]["tier"] == "tier4_attention"


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
