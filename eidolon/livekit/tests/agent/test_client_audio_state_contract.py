"""Contract tests for the inbound client.audio_state parser (plan Track A3/A5).

The parser must:
  - accept a well-formed packet and map enums faithfully,
  - be *loud* on field-level drift — an unknown key (the canonical ``ppt`` typo),
    an unrecognized enum value, or an unsupported ``schema_v`` — raising in
    strict mode and degrading best-effort (no raise) otherwise,
  - always reject malformed framing regardless of strictness,
  - tolerate the deprecated-but-known ``manual_interrupt`` field.
"""

from __future__ import annotations

import json

import pytest

from eidolon_sdk.biz.contracts import (
    CLIENT_AUDIO_STATE_KNOWN_KEYS,
    CLIENT_AUDIO_STATE_TYPE,
    INPUT_MODE_PTT,
    INPUT_MODE_UNKNOWN,
    PLAYBACK_STATE_AGENT_SPEAKING,
    PLAYBACK_STATE_UNKNOWN,
    WIRE_SCHEMA_VERSION,
)

from eidolon.livekit.agent.client_audio_state import parse_client_audio_state


def _packet(**fields) -> bytes:
    body = {"type": CLIENT_AUDIO_STATE_TYPE, **fields}
    return json.dumps(body).encode("utf-8")


def _parse(data: bytes, *, strict: bool):
    return parse_client_audio_state(
        data, participant_identity="alice", strict=strict
    )


# ── happy path ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("strict", [False, True])
def test_well_formed_packet_parses(strict: bool) -> None:
    state = _parse(
        _packet(
            schema_v=WIRE_SCHEMA_VERSION,
            input_mode=INPUT_MODE_PTT,
            ptt=True,
            playback_state=PLAYBACK_STATE_AGENT_SPEAKING,
            mic_muted=True,
        ),
        strict=strict,
    )
    assert state.input_mode == INPUT_MODE_PTT
    assert state.ptt is True
    assert state.playback_state == PLAYBACK_STATE_AGENT_SPEAKING
    assert state.mic_muted is True


def test_missing_optional_fields_default_unknown() -> None:
    state = _parse(_packet(), strict=True)
    assert state.input_mode == INPUT_MODE_UNKNOWN
    assert state.playback_state == PLAYBACK_STATE_UNKNOWN
    assert state.ptt is False


# ── the canonical typo: ``ppt`` instead of ``ptt`` ──────────────────────────


def test_unknown_key_is_loud_in_strict() -> None:
    with pytest.raises(ValueError, match="unknown field"):
        _parse(_packet(ppt=True), strict=True)


def test_unknown_key_degrades_in_non_strict(caplog) -> None:
    # Non-strict: warn + parse best-effort; the typo'd key is simply not applied.
    state = _parse(_packet(ppt=True), strict=False)
    assert state.ptt is False  # 'ppt' never reaches 'ptt'
    assert any("[contract]" in r.message for r in caplog.records)


# ── unrecognized enum values ────────────────────────────────────────────────


def test_bad_input_mode_is_loud_in_strict() -> None:
    with pytest.raises(ValueError, match="input_mode"):
        _parse(_packet(input_mode="duplex"), strict=True)


def test_bad_input_mode_degrades_to_unknown_in_non_strict() -> None:
    state = _parse(_packet(input_mode="duplex"), strict=False)
    assert state.input_mode == INPUT_MODE_UNKNOWN


def test_bad_playback_state_is_loud_in_strict() -> None:
    with pytest.raises(ValueError, match="playback_state"):
        _parse(_packet(playback_state="talking"), strict=True)


# ── schema_v ────────────────────────────────────────────────────────────────


def test_unsupported_schema_v_is_loud_in_strict() -> None:
    with pytest.raises(ValueError, match="schema_v"):
        _parse(_packet(schema_v=WIRE_SCHEMA_VERSION + 1), strict=True)


def test_missing_schema_v_is_accepted() -> None:
    # Firmware predating schema_v must keep working.
    state = _parse(_packet(input_mode=INPUT_MODE_PTT), strict=True)
    assert state.input_mode == INPUT_MODE_PTT


# ── deprecated-but-known field ──────────────────────────────────────────────


def test_manual_interrupt_is_tolerated_even_in_strict() -> None:
    # Deprecated but a KNOWN key: firmware still emitting it must not trip the
    # unknown-key check. It is parsed (for logging) but the policy layer ignores it.
    assert "manual_interrupt" in CLIENT_AUDIO_STATE_KNOWN_KEYS
    state = _parse(_packet(manual_interrupt=True), strict=True)
    assert state.manual_interrupt is True


# ── malformed framing: always rejected ──────────────────────────────────────


@pytest.mark.parametrize("strict", [False, True])
def test_non_json_always_rejected(strict: bool) -> None:
    with pytest.raises(ValueError):
        _parse(b"\xff\xfenot json", strict=strict)


@pytest.mark.parametrize("strict", [False, True])
def test_non_object_always_rejected(strict: bool) -> None:
    with pytest.raises(ValueError):
        _parse(b"[1, 2, 3]", strict=strict)


@pytest.mark.parametrize("strict", [False, True])
def test_wrong_type_always_rejected(strict: bool) -> None:
    with pytest.raises(ValueError):
        _parse(json.dumps({"type": "not.audio_state"}).encode("utf-8"), strict=strict)
