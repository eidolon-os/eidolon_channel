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
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.session.voiceprint import VoiceprintTurnResult
from eidolon.livekit.agent.speaker_verification.signal import SpeakerSignal


def _make_pipeline_with_session(*, latest_asr_text: str) -> Any:
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
    pipeline._skip_commit_after_interrupt_cancel = False
    pipeline._suppress_commit_after_interrupt_until = 0.0

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


def _transcript_event(text: str, *, final: bool = True) -> SimpleNamespace:
    return SimpleNamespace(transcript=text, is_final=final, speaker_id="manson")


def test_commit_skipped_when_asr_text_empty() -> None:
    """G8: VAD-end with empty ASR text → no commit_user_turn call."""
    pipeline = _make_pipeline_with_session(latest_asr_text="")

    pipeline._on_user_state_changed(_user_state_event("speaking", "listening"))

    pipeline._session.commit_user_turn.assert_not_called()
    pipeline._session.clear_user_turn.assert_called_once()
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


def test_commit_skipped_after_interrupt_cancel() -> None:
    pipeline = _make_pipeline_with_session(latest_asr_text="换个话题")
    pipeline._skip_commit_after_interrupt_cancel = True
    pipeline._timeline = TurnTimeline("turn-after-cancel")
    pipeline._timeline_debug_flushed = False

    pipeline._on_user_state_changed(_user_state_event("speaking", "listening"))

    pipeline._session.commit_user_turn.assert_not_called()
    pipeline._session.clear_user_turn.assert_called_once()
    eot = pipeline._get_eot_model.return_value
    eot.record_turn.assert_not_called()
    eot.reset.assert_called_once()
    assert pipeline._skip_commit_after_interrupt_cancel is False
    assert pipeline._timeline is None
    assert pipeline._latest_asr_text == ""


def test_commit_skipped_during_post_interrupt_suppression_window() -> None:
    pipeline = _make_pipeline_with_session(latest_asr_text="这是一段迟到的识别")
    pipeline._suppress_commit_after_interrupt_until = time.monotonic() + 1.0
    pipeline._timeline = TurnTimeline("late-stt-after-cancel")
    pipeline._timeline_debug_flushed = False

    pipeline._on_user_state_changed(_user_state_event("speaking", "listening"))

    pipeline._session.commit_user_turn.assert_not_called()
    pipeline._session.clear_user_turn.assert_called_once()
    eot = pipeline._get_eot_model.return_value
    eot.record_turn.assert_not_called()
    eot.reset.assert_called_once()
    assert pipeline._skip_commit_after_interrupt_cancel is False
    assert pipeline._timeline is None
    assert pipeline._latest_asr_text == ""


def test_late_final_transcript_is_dropped_after_voiceprint_block() -> None:
    pipeline = _make_pipeline_with_session(latest_asr_text="")
    pipeline._suppress_transcripts_until_next_speech = True

    pipeline._on_user_transcribed(_transcript_event("迟到的噪音字幕", final=True))

    pipeline._callbacks.on_user_message.assert_not_called()
    pipeline._get_eot_model.return_value.update_asr.assert_not_called()
    assert pipeline._latest_asr_text == ""


def test_new_speech_reopens_transcript_admission() -> None:
    pipeline = _make_pipeline_with_session(latest_asr_text="")
    pipeline._suppress_transcripts_until_next_speech = True

    pipeline._on_user_state_changed(_user_state_event("listening", "speaking"))

    assert pipeline._suppress_transcripts_until_next_speech is False


@pytest.mark.asyncio
async def test_completed_turn_hook_allows_owner_voiceprint() -> None:
    pipeline = _make_pipeline_with_session(latest_asr_text="")
    pipeline._timeline = TurnTimeline("owner-hook-turn")
    signal = SpeakerSignal(
        provider="3d_speaker",
        model="campplus_zh_16k_common",
        known=True,
        score=0.66,
        audio_ms=2400,
        latency_ms=20.0,
        profile_id="vp_manson_default",
    )
    result = VoiceprintTurnResult(
        signal=signal,
        cached=False,
        commit_allowed=True,
        commit_reason="owner_high_confidence",
    )
    pipeline._completed_turn_voiceprint_task = asyncio.create_task(
        asyncio.sleep(0, result=result)
    )
    pipeline._completed_turn_voiceprint_timeline = pipeline._timeline

    allowed = await pipeline._voiceprint_allows_completed_turn(
        new_message=SimpleNamespace(text_content="主人正常说话")
    )

    assert allowed is True
    pipeline._session.clear_user_turn.assert_not_called()
    assert pipeline._timeline.attrs["voiceprint_commit_gate"]["allowed"] is True


@pytest.mark.asyncio
async def test_completed_turn_hook_blocks_late_noise_final() -> None:
    pipeline = _make_pipeline_with_session(latest_asr_text="")
    pipeline._timeline = TurnTimeline("noise-hook-turn")
    signal = SpeakerSignal(
        provider="3d_speaker",
        model="campplus_zh_16k_common",
        known=False,
        score=0.22,
        audio_ms=3200,
        latency_ms=20.0,
        profile_id="vp_manson_default",
    )
    result = VoiceprintTurnResult(
        signal=signal,
        cached=False,
        commit_allowed=False,
        commit_reason="speaker_not_owner",
    )
    pipeline._completed_turn_voiceprint_task = asyncio.create_task(
        asyncio.sleep(0, result=result)
    )
    pipeline._completed_turn_voiceprint_timeline = pipeline._timeline

    allowed = await pipeline._voiceprint_allows_completed_turn(
        new_message=SimpleNamespace(text_content="迟到的噪音字幕")
    )

    assert allowed is False
    pipeline._session.clear_user_turn.assert_called_once()
    assert pipeline._suppress_transcripts_until_next_speech is True
    assert pipeline._timeline is None


@pytest.mark.asyncio
async def test_voiceprint_gate_allows_high_confidence_owner_commit() -> None:
    pipeline = _make_pipeline_with_session(latest_asr_text="你好世界")
    pipeline._timeline = TurnTimeline("owner-turn")
    signal = SpeakerSignal(
        provider="3d_speaker",
        model="campplus_zh_16k_common",
        known=True,
        score=0.66,
        audio_ms=2400,
        latency_ms=20.0,
        profile_id="vp_manson_default",
    )
    result = VoiceprintTurnResult(
        signal=signal,
        cached=False,
        commit_allowed=True,
        commit_reason="owner_high_confidence",
    )
    verify_task = asyncio.create_task(asyncio.sleep(0, result=result))
    pipeline._voiceprint_turns = SimpleNamespace(finish_turn=MagicMock(return_value=verify_task))

    pipeline._on_user_state_changed(_user_state_event("speaking", "listening"))
    await asyncio.gather(*pipeline._pending_voiceprint_commit_tasks)

    pipeline._session.commit_user_turn.assert_called_once_with(transcript_timeout=5.0)
    pipeline._session.clear_user_turn.assert_not_called()
    assert pipeline._timeline.attrs["voiceprint_commit_gate"]["allowed"] is True


@pytest.mark.asyncio
async def test_voiceprint_gate_blocks_low_score_commit() -> None:
    pipeline = _make_pipeline_with_session(latest_asr_text="视频里的声音")
    pipeline._timeline = TurnTimeline("noise-turn")
    signal = SpeakerSignal(
        provider="3d_speaker",
        model="campplus_zh_16k_common",
        known=False,
        score=0.22,
        audio_ms=3200,
        latency_ms=20.0,
        profile_id="vp_manson_default",
    )
    result = VoiceprintTurnResult(
        signal=signal,
        cached=False,
        commit_allowed=False,
        commit_reason="speaker_not_owner",
    )
    verify_task = asyncio.create_task(asyncio.sleep(0, result=result))
    pipeline._voiceprint_turns = SimpleNamespace(finish_turn=MagicMock(return_value=verify_task))

    pipeline._on_user_state_changed(_user_state_event("speaking", "listening"))
    await asyncio.gather(*pipeline._pending_voiceprint_commit_tasks)

    pipeline._session.commit_user_turn.assert_not_called()
    pipeline._session.clear_user_turn.assert_called_once()
    eot = pipeline._get_eot_model.return_value
    eot.reset.assert_called_once()
    assert pipeline._timeline is None
