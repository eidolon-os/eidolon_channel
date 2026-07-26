"""Interaction-mode × session-intent metadata-bus handshake.

The device⇄server contract has two halves that must agree:
  hub stamps ``participant_metadata`` into the voice token  →  channel resolves
  it into a per-session policy. This test drives a contract-correct
  ``SimulatedDevice`` (the same packet/metadata builder a real client uses)
  through channel's resolve + apply for every combination of
  (interaction_mode × session_intent), and round-trips the device's
  ``client.audio_state`` packet through the channel parser.

It is the regression net for exactly the drift class we keep hitting (a body
``type`` typo, an enum mismatch, a key the other side never reads): if hub and
channel ever disagree on the bus, one of these four cells fails.
"""

from __future__ import annotations

import pytest

from eidolon_sdk.biz.contracts import (
    INTERACTION_MODE_FULL_DUPLEX,
    INTERACTION_MODE_HALF_DUPLEX,
    INTERACTION_MODE_PTT,
    INPUT_MODE_AUTO,
    INPUT_MODE_PTT,
    SESSION_INTENT_PRESENCE,
    SESSION_INTENT_PROACTIVE,
    SESSION_INTENT_USER_INITIATED,
)

from eidolon.livekit.agent.integration.client_audio_state import parse_client_audio_state
from eidolon.livekit.agent.runtime import (
    apply_interaction_mode,
    resolve_interaction_mode,
    resolve_session_intent,
)
from eidolon.livekit.common.config.schema import TurnPolicyConfig
from eidolon.livekit.tests._harness.device_sim import SimulatedDevice

_MODES = [INTERACTION_MODE_HALF_DUPLEX, INTERACTION_MODE_FULL_DUPLEX]
_INTENTS = [
    SESSION_INTENT_USER_INITIATED,
    SESSION_INTENT_PRESENCE,
    SESSION_INTENT_PROACTIVE,
]


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("intent", _INTENTS)
def test_metadata_bus_roundtrip_per_cell(mode: str, intent: str) -> None:
    device = SimulatedDevice(
        device_id="device-abc", interaction_mode=mode, session_intent=intent
    )

    # Hub-stamped metadata → channel resolves both bus dimensions from ONE read.
    meta = device.token_metadata_json()
    assert resolve_interaction_mode(meta) == mode
    assert resolve_session_intent(meta) == intent

    # Derived per-session policy: half disables barge-in (framework interruption
    # off + attention guessing off); full is the unchanged status quo.
    base = TurnPolicyConfig()
    assert base.attention.enabled is True  # precondition
    policy, allow = apply_interaction_mode(
        turn_policy=base, allow_interruptions=True, interaction_mode=mode
    )
    if mode == INTERACTION_MODE_HALF_DUPLEX:
        assert allow is False
        assert policy.attention.enabled is False
    else:
        assert allow is True
        assert policy is base

    # Exact intent drives opening + idle behavior downstream; no cell may
    # silently degrade to another valid intent.
    assert resolve_session_intent(meta) == intent


@pytest.mark.parametrize("mode", _MODES)
def test_device_audio_state_packet_parses_clean(mode: str) -> None:
    # The device's own packet must satisfy the channel parser in STRICT mode —
    # proving the wire body (schema_v / type / enum values) matches the contract.
    device = SimulatedDevice(interaction_mode=mode)
    state = parse_client_audio_state(
        device.ptt_press() if mode == INTERACTION_MODE_PTT else device.audio_state_bytes(),
        participant_identity="device-abc",
        strict=True,
    )
    # Only ptt reports input_mode "ptt"; half_duplex and full_duplex auto-record.
    expected_input_mode = (
        INPUT_MODE_PTT if mode == INTERACTION_MODE_PTT else INPUT_MODE_AUTO
    )
    assert state.input_mode == expected_input_mode
    if mode == INTERACTION_MODE_PTT:
        assert state.ptt is True


def test_half_duplex_ptt_press_release_round_trip() -> None:
    device = SimulatedDevice(interaction_mode=INTERACTION_MODE_HALF_DUPLEX)
    press = parse_client_audio_state(
        device.ptt_press(), participant_identity="d", strict=True
    )
    release = parse_client_audio_state(
        device.ptt_release(), participant_identity="d", strict=True
    )
    assert press.ptt is True
    assert release.ptt is False
