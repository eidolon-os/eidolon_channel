"""AttentionEffectHandler boundary tests."""

from __future__ import annotations

import time
from dataclasses import replace
from unittest.mock import MagicMock

from eidolon.livekit.agent.integration.client_audio_state import ClientAudioState
from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.session.attention_effects import AttentionEffectHandler
from eidolon.livekit.agent.turn_policy import TurnPolicyRuntime
from eidolon.livekit.common.config import (
    AttentionPolicyConfig,
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
    soft_duck_on_playback_speech_start: bool = False,
) -> TurnPolicyConfig:
    return replace(
        TurnPolicyConfig(),
        attention=replace(
            AttentionPolicyConfig(),
            enforce=enforce,
            soft_duck_on_playback_speech_start=soft_duck_on_playback_speech_start,
        ),
    )


def _handler(
    *,
    client_state=None,
    enforce: bool = True,
    agent_speaking: bool = True,
    duck_active: bool = False,
    eot_score: float = 0.0,
    soft_duck_on_playback_speech_start: bool = False,
):
    policy = _policy(
        enforce=enforce,
        soft_duck_on_playback_speech_start=soft_duck_on_playback_speech_start,
    )
    timeline = TurnTimeline("turn-attention")
    on_duck = MagicMock()
    on_interrupt = MagicMock()
    handler = AttentionEffectHandler(
        turn_policy=policy,
        turn_runtime=TurnPolicyRuntime(policy),
        get_agent_speaking=lambda: agent_speaking,
        get_duck_active=lambda: duck_active,
        latest_client_audio_state=lambda participant_identity: client_state,
        get_timeline=lambda: timeline,
        on_duck=on_duck,
        on_interrupt=on_interrupt,
        get_eot_score=lambda: eot_score,
    )
    return handler, timeline, on_duck, on_interrupt


def test_allows_eot_routes_substantive_fresh_playback_without_ducking() -> None:
    handler, timeline, on_duck, on_interrupt = _handler(client_state=_client_state())

    allowed = handler.allows_eot_check("那它有什么风险")

    assert allowed is True
    on_duck.assert_not_called()
    on_interrupt.assert_not_called()
    assert timeline.attrs["attention_admission"]["action"] == "observe"
    assert timeline.attrs["attention_admission"]["reason"] == (
        "playback_low_evidence_transcript:substantive_cjk_transcript"
    )


def test_allows_eot_blocks_substantive_stale_playback_evidence() -> None:
    received_at = (
        time.monotonic()
        - TurnPolicyConfig().attention.client_state_max_age_ms / 1000.0
        - 1.0
    )
    handler, timeline, on_duck, on_interrupt = _handler(
        client_state=_client_state(received_at=received_at),
    )

    allowed = handler.allows_eot_check("那它有什么风险")

    assert allowed is False
    on_duck.assert_not_called()
    on_interrupt.assert_not_called()
    assert timeline.attrs["attention_admission"]["action"] == "observe"
    assert timeline.attrs["attention_admission"]["state"]["client_state_fresh"] is False


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


def test_duck_active_routes_low_evidence_transcript_as_evidence() -> None:
    handler, timeline, on_duck, on_interrupt = _handler(
        client_state=_client_state(),
        duck_active=True,
    )

    allowed = handler.allows_eot_check("不是")

    assert allowed is True
    on_duck.assert_not_called()
    on_interrupt.assert_not_called()
    assert timeline.attrs["attention_admission"]["action"] == "observe"


def test_attention_admission_records_state_snapshot() -> None:
    received_at = time.monotonic() - 0.321
    handler, timeline, _, _ = _handler(
        client_state=_client_state(
            received_at=received_at,
            rms=42.0,
            snr_hint=18.5,
        ),
        duck_active=True,
        eot_score=0.42,
    )

    handler.allows_eot_check("换个话")

    state = timeline.attrs["attention_admission"]["state"]
    assert state["agent_speaking"] is True
    assert state["duck_active"] is True
    assert state["eot_score"] == 0.42
    assert state["client_state_present"] is True
    assert state["client_state_fresh"] is True
    assert state["client_playback_state"] == "agent_speaking"
    assert 250 <= state["client_state_age_ms"] <= 500
    assert state["client_rms"] == 42.0
    assert state["client_snr_hint"] == 18.5


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
        == "playback_low_evidence_transcript:insufficient_transcript_evidence"
    )
    assert "interrupt_intent_admitted_at" not in timeline.timestamps


def test_speaking_started_ptt_interrupt_marks_and_interrupts() -> None:
    # PTT (deliberate button) is the hard-cut signal. manual_interrupt no longer
    # hard-cuts (P1: it ducks + evidence-gates), so drive the hard path via ptt.
    handler, timeline, on_duck, on_interrupt = _handler(
        client_state=_client_state(ptt=True),
    )

    handler.handle_speaking_started()

    on_duck.assert_not_called()
    on_interrupt.assert_called_once_with()
    assert "interrupt_started_at" in timeline.timestamps


def test_speaking_started_soft_ducks_playback_when_configured() -> None:
    handler, timeline, on_duck, on_interrupt = _handler(
        client_state=_client_state(),
        soft_duck_on_playback_speech_start=True,
    )

    handler.handle_speaking_started()

    on_duck.assert_called_once_with()
    on_interrupt.assert_not_called()
    assert timeline.attrs["attention_admission"]["action"] == "duck_and_decide"
    assert timeline.attrs["attention_admission"]["reason"] == (
        "playback_speech_start_soft_duck"
    )


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


def test_attention_enforce_observes_short_overlap_without_direct_signal() -> None:
    # Regression: the retired responsive mode used to force attention_enforce=
    # False, silently overriding the operator's turn_policy.attention.enforce and
    # disabling the manual_interrupt / mic_muted gates (full-duplex barge-in bug).
    # The interrupt_mode axis is gone; enforcement comes solely from config. With
    # enforce=True, short low-evidence playback speech is observed without being
    # committed or escalated.
    handler, timeline, on_duck, on_interrupt = _handler(
        client_state=_client_state(),
        enforce=True,
    )

    allowed = handler.allows_eot_check("不是")

    assert allowed is False
    on_duck.assert_not_called()
    on_interrupt.assert_not_called()
    assert timeline.attrs["attention_admission"]["enforced"] is True
