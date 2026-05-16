"""Unit tests for G8: no-ASR no-commit guard.

When VAD detects speech but STT produces no text (AEC warmup window,
brief noise, or STT hiccup), we must NOT call ``session.commit_user_turn``.
The framework would otherwise wait ``transcript_timeout`` for a FINAL
that never arrives, then promote whatever INTERIM is currently in its
global ``_audio_interim_transcript`` — often from the NEXT user
utterance — producing a ghost LLM call with cross-segment-contaminated
text.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_pipeline_with_session(*, latest_asr_text: str) -> "StreamingPipeline":
    """Build a minimally-initialised StreamingPipeline ready to receive a
    ``user_state: speaking → listening`` transition.

    We bypass full __init__ because ``_on_user_state_changed`` only consults
    the small slice of state set up below.
    """
    from eidolon.livekit.agent.streaming import StreamingPipeline

    pipeline = StreamingPipeline.__new__(StreamingPipeline)
    pipeline._session = MagicMock()
    pipeline._session.output = MagicMock()
    pipeline._session.output.audio = MagicMock()
    pipeline._callbacks = MagicMock()
    pipeline._stt_commit_transcript_timeout = 5.0
    pipeline._latest_asr_text = latest_asr_text
    pipeline._filler = None
    pipeline._duck_mixer = None
    pipeline._duck_timeout_task = None
    pipeline._duck_suspend_start = 0.0
    pipeline._soft_interrupt_active = False
    pipeline._soft_interrupt_timer = None
    pipeline._user_speaking_start_time = None
    pipeline._last_unduck_time = 0.0

    # Stub EOT model — its methods are called regardless of guard branch.
    eot = MagicMock()
    eot.reset = MagicMock()
    eot.record_turn = MagicMock()
    eot._current_eot_score = 0.0
    pipeline._get_eot_model = MagicMock(return_value=eot)

    # Stub interrupt-context capture path.
    pipeline._inject_interrupted_context = MagicMock()
    pipeline._cancel_soft_interrupt = MagicMock()
    pipeline._duck_unduck_if_suspended = MagicMock()

    return pipeline


def _user_state_event(old: str, new: str) -> SimpleNamespace:
    return SimpleNamespace(old_state=old, new_state=new)


def test_commit_skipped_when_asr_text_empty() -> None:
    """G8: VAD-end with empty ASR text → no commit_user_turn call."""
    pipeline = _make_pipeline_with_session(latest_asr_text="")

    pipeline._on_user_state_changed(_user_state_event("speaking", "listening"))

    pipeline._session.commit_user_turn.assert_not_called()
    # record_turn also gated on text — should not fire either
    eot = pipeline._get_eot_model.return_value
    eot.record_turn.assert_not_called()
    # But reset SHOULD fire — clean per-turn state regardless of branch
    eot.reset.assert_called_once()



def test_commit_called_when_asr_text_present() -> None:
    """Sanity: normal path (non-empty ASR text) still commits."""
    pipeline = _make_pipeline_with_session(latest_asr_text="你好世界")

    pipeline._on_user_state_changed(_user_state_event("speaking", "listening"))

    pipeline._session.commit_user_turn.assert_called_once_with(transcript_timeout=5.0)
    eot = pipeline._get_eot_model.return_value
    eot.record_turn.assert_called_once()



def test_asr_text_cleared_after_either_branch() -> None:
    """``_latest_asr_text`` is wiped after the VAD-end handler regardless of
    which branch fired, so the next turn starts clean."""
    pipeline = _make_pipeline_with_session(latest_asr_text="你好")
    pipeline._on_user_state_changed(_user_state_event("speaking", "listening"))
    assert pipeline._latest_asr_text == ""

    pipeline2 = _make_pipeline_with_session(latest_asr_text="")
    pipeline2._on_user_state_changed(_user_state_event("speaking", "listening"))
    assert pipeline2._latest_asr_text == ""
