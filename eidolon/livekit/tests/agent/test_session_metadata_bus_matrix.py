"""Interaction-mode × session-intent bus handshake — across TWO buses.

The contract has two halves that must agree, and they do not share a wire:

  * the joining body stamps ``participant_metadata`` into its voice token, and
    channel resolves the turn-taking mode out of it;
  * the Channel Provider stamps the agent-dispatch metadata, and channel
    resolves out of it why this session exists.

The split is a trust boundary, not a routing detail: ``interaction_mode`` is a
fact the body may state about its own hardware, while ``presence_initiated``
and ``proactive_initiated`` are grants the body would otherwise be handing
itself. This test drives a contract-correct ``SimulatedDevice`` (the same
packet/metadata builder a real client uses) plus a Provider-shaped dispatch
through channel's resolve + apply for every combination of the two, and
round-trips the device's ``client.audio_state`` packet through the parser.

It is the regression net for exactly the drift class we keep hitting (a body
``type`` typo, an enum mismatch, a key the other side never reads).

Two things this file deliberately does NOT cover, because each has a home where
it can be exercised against the real code rather than a mirror of it: that the
adapter writes the dispatch shape assumed here
(``channel_provider/tests/test_livekit_adapter.py``), and that a body claiming
an intent in its own metadata is not believed
(``test_server_session_metadata.py``).
"""

from __future__ import annotations

import json

import pytest

from eidolon_sdk.biz.contracts import (
    INTERACTION_MODE_FULL_DUPLEX,
    INTERACTION_MODE_HALF_DUPLEX,
    INTERACTION_MODE_PTT,
    INPUT_MODE_AUTO,
    INPUT_MODE_PTT,
    SESSION_CONVERSATION_ID_FIELD,
    SESSION_INTENT_FIELD,
    SESSION_INTENT_PRESENCE,
    SESSION_INTENT_PROACTIVE,
    SESSION_INTENT_USER_INITIATED,
    WIRE_SCHEMA_VERSION,
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


def _dispatch_metadata(intent: str) -> str:
    """What the Channel Provider puts on the agent dispatch for one session.

    Mirrors ``LiveKitChannelAdapter.open_session``; the adapter's own tests pin
    that this is the shape it actually writes.
    """
    return json.dumps(
        {
            "schema_v": WIRE_SCHEMA_VERSION,
            SESSION_CONVERSATION_ID_FIELD: "conversation-1",
            SESSION_INTENT_FIELD: intent,
        },
        separators=(",", ":"),
    )


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("intent", _INTENTS)
def test_metadata_bus_roundtrip_per_cell(mode: str, intent: str) -> None:
    device = SimulatedDevice(device_id="device-abc", interaction_mode=mode)

    # Body-stamped token metadata → the turn-taking half.
    meta = device.token_metadata_json()
    assert resolve_interaction_mode(meta) == mode
    # Provider-stamped dispatch metadata → the why half.
    assert resolve_session_intent(_dispatch_metadata(intent)) == intent
    # And the body's own metadata says nothing about why, in any cell.
    assert SESSION_INTENT_FIELD not in device.token_metadata()

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
    assert resolve_session_intent(_dispatch_metadata(intent)) == intent


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
