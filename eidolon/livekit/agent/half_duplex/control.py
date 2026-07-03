"""Half-duplex PTT client-control helpers."""

from __future__ import annotations

from eidolon_sdk.biz.contracts import CONTROL_OP_PTT_TURN_STATUS

PTT_OUTCOME_RECORDING = "recording"
PTT_OUTCOME_FINALIZING = "finalizing"
PTT_OUTCOME_COMMITTED = "committed"
PTT_TRANSCRIPT_PREVIEW_CHARS = 80

PTT_NO_TURN_TERMINAL_REASONS = frozenset(
    {
        "empty_hold",
        "speech_without_transcript",
        "tap_to_stop",
    }
)


def ptt_rejected_outcome(reason: str) -> str:
    return f"rejected:{reason or 'ptt_rejected'}"


def build_ptt_turn_status_payload(
    outcome: str,
    reason: str,
    *,
    transcript: str = "",
) -> dict[str, object]:
    return {
        "outcome": outcome,
        "reason": reason,
        "transcript_preview": transcript[:PTT_TRANSCRIPT_PREVIEW_CHARS],
    }


def should_drop_pending_ptt_control_event(
    *,
    op: str,
    reason: str,
    turn_id: str = "",
) -> bool:
    return (
        not turn_id
        and op == CONTROL_OP_PTT_TURN_STATUS
        and reason in PTT_NO_TURN_TERMINAL_REASONS
    )
