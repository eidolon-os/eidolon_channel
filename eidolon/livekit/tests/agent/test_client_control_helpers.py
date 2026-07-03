from __future__ import annotations

from eidolon_sdk.biz.contracts import CONTROL_OP_PLAYBACK_STOP, CONTROL_OP_PTT_TURN_STATUS

from eidolon.livekit.agent.session.client_control import (
    PTT_OUTCOME_COMMITTED,
    build_ptt_turn_status_payload,
    build_session_client_control_envelope,
    ptt_rejected_outcome,
    should_drop_pending_client_control_event,
)


def test_session_client_control_envelope_keeps_channel_wire_shape() -> None:
    envelope = build_session_client_control_envelope(
        op=CONTROL_OP_PLAYBACK_STOP,
        reason="interrupt_cancel",
        payload={"extra": "value"},
        turn_id="turn-1",
        now_ms=1234,
    )

    assert envelope == {
        "v": 1,
        "kind": "cmd",
        "id": "playback.stop:1234",
        "op": "playback.stop",
        "src": {"type": "channel", "id": "eidolon_channel"},
        "payload": {
            "reason": "interrupt_cancel",
            "turn_id": "turn-1",
            "extra": "value",
        },
        "ts": 1234,
        "ttl_ms": 5000,
    }


def test_ptt_turn_status_payload_and_reject_outcome() -> None:
    payload = build_ptt_turn_status_payload(
        PTT_OUTCOME_COMMITTED,
        "final_transcript",
        transcript="a" * 100,
    )

    assert payload == {
        "outcome": "committed",
        "reason": "final_transcript",
        "transcript_preview": "a" * 80,
    }
    assert ptt_rejected_outcome("empty_hold") == "rejected:empty_hold"


def test_no_turn_ptt_terminal_status_does_not_attach_to_next_timeline() -> None:
    assert should_drop_pending_client_control_event(
        op=CONTROL_OP_PTT_TURN_STATUS,
        reason="empty_hold",
        turn_id="",
    )
    assert not should_drop_pending_client_control_event(
        op=CONTROL_OP_PTT_TURN_STATUS,
        reason="empty_hold",
        turn_id="turn-active",
    )
    assert not should_drop_pending_client_control_event(
        op=CONTROL_OP_PLAYBACK_STOP,
        reason="interrupt_cancel",
        turn_id="",
    )
