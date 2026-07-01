from __future__ import annotations

from unittest.mock import MagicMock

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.session.ptt_turn import PttTurnFinalizer


def test_ptt_turn_finalizer_commits_speech_hold() -> None:
    session = MagicMock()
    reset_had_speech = MagicMock()
    reset_eot = MagicMock()
    timeline = TurnTimeline(turn_id="ptt-1")

    committed = PttTurnFinalizer().commit_release(
        session=session,
        had_speech=True,
        reset_had_speech=reset_had_speech,
        reset_eot=reset_eot,
        transcript_timeout=1.0,
        timeline=timeline,
    )

    assert committed is True
    reset_had_speech.assert_called_once_with()
    reset_eot.assert_called_once_with()
    session.commit_user_turn.assert_called_once_with(transcript_timeout=1.0)
    assert "turn_committed_at" in timeline.timestamps
    assert timeline.attrs["ptt_release_commit"] == {
        "transcript_timeout_ms": 1000,
        "text_source_policy": "final_or_interim_on_timeout",
    }


def test_ptt_turn_finalizer_guards_empty_hold() -> None:
    session = MagicMock()
    reset_had_speech = MagicMock()
    reset_eot = MagicMock()

    committed = PttTurnFinalizer().commit_release(
        session=session,
        had_speech=False,
        reset_had_speech=reset_had_speech,
        reset_eot=reset_eot,
        transcript_timeout=1.0,
        timeline=None,
    )

    assert committed is False
    reset_had_speech.assert_not_called()
    reset_eot.assert_not_called()
    session.commit_user_turn.assert_not_called()
