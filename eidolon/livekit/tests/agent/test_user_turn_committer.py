"""UserTurnCommitter boundary tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.session.turn_commit import UserTurnCommitter


def _session(audio=None):
    return SimpleNamespace(
        output=SimpleNamespace(audio=audio),
        commit_user_turn=MagicMock(),
    )


def _eot_model() -> MagicMock:
    eot = MagicMock()
    eot._current_eot_score = 0.42
    return eot


def test_commit_or_skip_commits_when_transcript_present() -> None:
    session = _session(audio=None)
    eot = _eot_model()
    timeline = TurnTimeline("turn-commit")
    inject = MagicMock()

    committed = UserTurnCommitter().commit_or_skip(
        session=session,
        eot_model=eot,
        transcript="你好世界",
        transcript_timeout=5.0,
        timeline=timeline,
        inject_interrupted_context=inject,
        filler=None,
    )

    assert committed is True
    eot.record_turn.assert_called_once_with(
        "你好世界",
        is_complete=True,
        eot_score=0.42,
    )
    eot.reset.assert_called_once_with()
    inject.assert_called_once_with()
    assert "turn_committed_at" in timeline.timestamps
    session.commit_user_turn.assert_called_once_with(transcript_timeout=5.0)


def test_commit_or_skip_resets_without_empty_transcript() -> None:
    session = _session()
    eot = _eot_model()
    inject = MagicMock()

    committed = UserTurnCommitter().commit_or_skip(
        session=session,
        eot_model=eot,
        transcript="",
        transcript_timeout=5.0,
        timeline=None,
        inject_interrupted_context=inject,
        filler=None,
    )

    assert committed is False
    eot.record_turn.assert_not_called()
    eot.reset.assert_called_once_with()
    inject.assert_not_called()
    session.commit_user_turn.assert_not_called()


def test_commit_or_skip_injects_filler_when_available() -> None:
    audio = MagicMock()
    session = _session(audio=audio)
    eot = _eot_model()
    filler = SimpleNamespace(inject=MagicMock())

    UserTurnCommitter().commit_or_skip(
        session=session,
        eot_model=eot,
        transcript="你好",
        transcript_timeout=5.0,
        timeline=None,
        inject_interrupted_context=lambda: None,
        filler=filler,
    )

    filler.inject.assert_called_once_with(audio)
