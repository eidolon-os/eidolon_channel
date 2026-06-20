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
from eidolon.livekit.agent.turn_policy import TurnPolicyRuntime
from eidolon.livekit.common.config import EotPolicyConfig, TurnPolicyConfig


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
    pipeline._room = None
    pipeline._allow_interruptions = False
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


def test_interrupt_cancel_no_longer_drops_owner_transcript() -> None:
    pipeline = _make_pipeline_with_session(latest_asr_text="换个话题")
    pipeline._skip_commit_after_interrupt_cancel = True
    pipeline._timeline = TurnTimeline("turn-after-cancel")
    pipeline._timeline_debug_flushed = False

    pipeline._on_user_state_changed(_user_state_event("speaking", "listening"))

    pipeline._session.commit_user_turn.assert_called_once_with(transcript_timeout=5.0)
    pipeline._session.clear_user_turn.assert_not_called()
    eot = pipeline._get_eot_model.return_value
    eot.record_turn.assert_called_once()
    assert pipeline._skip_commit_after_interrupt_cancel is False
    assert pipeline._latest_asr_text == ""


def test_post_interrupt_suppression_does_not_drop_owner_transcript() -> None:
    pipeline = _make_pipeline_with_session(latest_asr_text="这是一段迟到的识别")
    pipeline._suppress_commit_after_interrupt_until = time.monotonic() + 1.0
    pipeline._timeline = TurnTimeline("late-stt-after-cancel")
    pipeline._timeline_debug_flushed = False

    pipeline._on_user_state_changed(_user_state_event("speaking", "listening"))

    pipeline._session.commit_user_turn.assert_called_once_with(transcript_timeout=5.0)
    pipeline._session.clear_user_turn.assert_not_called()
    eot = pipeline._get_eot_model.return_value
    eot.record_turn.assert_called_once()
    assert pipeline._skip_commit_after_interrupt_cancel is False
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


def test_short_statement_fragment_defers_even_when_eot_is_high() -> None:
    pipeline = _make_pipeline_with_session(latest_asr_text="")
    pipeline._get_eot_model.return_value.current_eot_score = 0.99

    assert pipeline._should_defer_low_eot_commit(
        transcript="给医生做的系统。",
        eot_model=pipeline._get_eot_model.return_value,
    )
    assert not pipeline._should_defer_low_eot_commit(
        transcript="你觉得这个系统怎么定价？",
        eot_model=pipeline._get_eot_model.return_value,
    )
    assert not pipeline._should_defer_low_eot_commit(
        transcript="今天我想聊一下一个新的医疗项目。",
        eot_model=pipeline._get_eot_model.return_value,
    )
    assert not pipeline._should_defer_low_eot_commit(
        transcript="帮我详细介绍一下这个方案。",
        eot_model=pipeline._get_eot_model.return_value,
    )
    assert not pipeline._should_defer_low_eot_commit(
        transcript="换个话题，我们聊一下定价。",
        eot_model=pipeline._get_eot_model.return_value,
    )


def test_ptt_mode_never_defers_low_eot_commit() -> None:
    """Push-to-talk: release is an explicit end-of-turn, so a fragment that
    WOULD defer in open-mic mode commits immediately instead of being held for a
    continuation that will never come (the trailing-filler no-reply bug)."""
    pipeline = _make_pipeline_with_session(latest_asr_text="")
    pipeline._is_half_duplex = True
    pipeline._get_eot_model.return_value.current_eot_score = 0.01

    # Same short-statement fragment that defers in open-mic mode (see
    # test_short_statement_fragment_defers_even_when_eot_is_high).
    assert not pipeline._should_defer_low_eot_commit(
        transcript="给医生做的系统。",
        eot_model=pipeline._get_eot_model.return_value,
    )


def test_ptt_mode_never_defers_framework_completed_turn() -> None:
    """Push-to-talk: the framework-completed turn is never re-held for
    continuation in PTT mode."""
    pipeline = _make_pipeline_with_session(latest_asr_text="")
    pipeline._ensure_runtime_defaults()
    pipeline._is_half_duplex = True

    assert not pipeline._should_defer_framework_completed_turn("私立医院的。")


@pytest.mark.asyncio
async def test_low_eot_commit_is_deferred_until_grace_expires() -> None:
    pipeline = _make_pipeline_with_session(latest_asr_text="看你能不能")
    policy = TurnPolicyConfig(
        eot=EotPolicyConfig(eot_unlikely_threshold=0.5, tail_hang_silence_ms=10)
    )
    pipeline._turn_policy = policy
    pipeline._turn_runtime = TurnPolicyRuntime(policy)
    pipeline._get_eot_model.return_value.current_eot_score = 0.01

    pipeline._on_user_state_changed(_user_state_event("speaking", "listening"))

    pipeline._session.commit_user_turn.assert_not_called()
    await asyncio.sleep(0.05)
    pipeline._session.commit_user_turn.assert_called_once_with(transcript_timeout=5.0)


@pytest.mark.asyncio
async def test_low_eot_deferred_commit_merges_short_continuation() -> None:
    pipeline = _make_pipeline_with_session(latest_asr_text="看你能不能")
    policy = TurnPolicyConfig(
        eot=EotPolicyConfig(eot_unlikely_threshold=0.5, tail_hang_silence_ms=100)
    )
    pipeline._turn_policy = policy
    pipeline._turn_runtime = TurnPolicyRuntime(policy)
    pipeline._get_eot_model.return_value.current_eot_score = 0.01

    pipeline._on_user_state_changed(_user_state_event("speaking", "listening"))
    task = pipeline._deferred_low_eot_commit_task
    assert task is not None

    pipeline._on_user_state_changed(_user_state_event("listening", "speaking"))
    await asyncio.sleep(0)
    assert task.cancelled()
    pipeline._on_user_transcribed(_transcript_event("帮我", final=True))
    pipeline._get_eot_model.return_value.current_eot_score = 1.0
    await asyncio.sleep(0.12)
    pipeline._on_user_state_changed(_user_state_event("speaking", "listening"))
    pipeline._session.commit_user_turn.assert_called_once_with(transcript_timeout=5.0)
    eot = pipeline._get_eot_model.return_value
    eot.record_turn.assert_called_once()
    assert eot.record_turn.call_args.args[0] == "看你能不能帮我"


@pytest.mark.asyncio
async def test_low_eot_merged_turn_requires_all_voiceprint_segments_allowed() -> None:
    pipeline = _make_pipeline_with_session(latest_asr_text="看你能不能")
    policy = TurnPolicyConfig(
        eot=EotPolicyConfig(eot_unlikely_threshold=0.5, tail_hang_silence_ms=100)
    )
    pipeline._turn_policy = policy
    pipeline._turn_runtime = TurnPolicyRuntime(policy)
    pipeline._get_eot_model.return_value.current_eot_score = 0.01

    blocked = VoiceprintTurnResult(
        signal=SpeakerSignal(
            provider="3d_speaker",
            model="campplus_zh_16k_common",
            known=False,
            score=0.22,
            audio_ms=900,
            latency_ms=20.0,
            profile_id="vp_manson_default",
        ),
        cached=False,
        commit_allowed=False,
        commit_reason="speaker_not_owner",
    )
    allowed = VoiceprintTurnResult(
        signal=SpeakerSignal(
            provider="3d_speaker",
            model="campplus_zh_16k_common",
            known=True,
            score=0.66,
            audio_ms=900,
            latency_ms=20.0,
            profile_id="vp_manson_default",
        ),
        cached=False,
        commit_allowed=True,
        commit_reason="owner_high_confidence",
    )
    pipeline._voiceprint_turns = SimpleNamespace(
        start_turn=MagicMock(),
        finish_turn=MagicMock(
            side_effect=[
                asyncio.create_task(asyncio.sleep(0, result=blocked)),
                asyncio.create_task(asyncio.sleep(0, result=allowed)),
            ]
        ),
    )

    pipeline._on_user_state_changed(_user_state_event("speaking", "listening"))
    task = pipeline._deferred_low_eot_commit_task
    assert task is not None

    pipeline._on_user_state_changed(_user_state_event("listening", "speaking"))
    await asyncio.sleep(0)
    assert task.cancelled()
    pipeline._on_user_transcribed(_transcript_event("帮我", final=True))
    pipeline._get_eot_model.return_value.current_eot_score = 1.0
    await asyncio.sleep(0.12)
    pipeline._on_user_state_changed(_user_state_event("speaking", "listening"))
    await asyncio.gather(*pipeline._pending_voiceprint_commit_tasks)

    pipeline._session.commit_user_turn.assert_not_called()
    pipeline._session.clear_user_turn.assert_called_once()


@pytest.mark.asyncio
async def test_merged_turn_ignores_inconclusive_short_voiceprint_segment() -> None:
    pipeline = _make_pipeline_with_session(latest_asr_text="私立医院的")
    policy = TurnPolicyConfig(
        eot=EotPolicyConfig(eot_unlikely_threshold=0.5, tail_hang_silence_ms=100)
    )
    pipeline._turn_policy = policy
    pipeline._turn_runtime = TurnPolicyRuntime(policy)
    pipeline._get_eot_model.return_value.current_eot_score = 0.01

    short = VoiceprintTurnResult(
        signal=SpeakerSignal(
            provider="3d_speaker",
            model="campplus_zh_16k_common",
            known=False,
            score=0.0,
            audio_ms=400,
            latency_ms=10.0,
            profile_id="vp_manson_default",
        ),
        cached=False,
        commit_allowed=False,
        commit_reason="audio_too_short",
    )
    allowed = VoiceprintTurnResult(
        signal=SpeakerSignal(
            provider="3d_speaker",
            model="campplus_zh_16k_common",
            known=True,
            score=0.66,
            audio_ms=1800,
            latency_ms=20.0,
            profile_id="vp_manson_default",
        ),
        cached=False,
        commit_allowed=True,
        commit_reason="owner_high_confidence",
    )
    pipeline._voiceprint_turns = SimpleNamespace(
        start_turn=MagicMock(),
        finish_turn=MagicMock(
            side_effect=[
                asyncio.create_task(asyncio.sleep(0, result=short)),
                asyncio.create_task(asyncio.sleep(0, result=allowed)),
            ]
        ),
    )

    pipeline._on_user_state_changed(_user_state_event("speaking", "listening"))
    task = pipeline._deferred_low_eot_commit_task
    assert task is not None

    pipeline._on_user_state_changed(_user_state_event("listening", "speaking"))
    await asyncio.sleep(0)
    assert task.cancelled()
    pipeline._on_user_transcribed(_transcript_event("给医生做的系统", final=True))
    pipeline._get_eot_model.return_value.current_eot_score = 1.0
    await asyncio.sleep(0.12)
    pipeline._on_user_state_changed(_user_state_event("speaking", "listening"))
    await asyncio.gather(*pipeline._pending_voiceprint_commit_tasks)

    pipeline._session.commit_user_turn.assert_called_once_with(transcript_timeout=5.0)
    eot = pipeline._get_eot_model.return_value
    assert eot.record_turn.call_args.args[0] == "私立医院的给医生做的系统"


@pytest.mark.asyncio
async def test_short_voiceprint_result_waits_for_continuation_before_reject() -> None:
    pipeline = _make_pipeline_with_session(latest_asr_text="私立医院的。")
    pipeline._timeline = TurnTimeline("short-audio-waits")
    pipeline._ensure_runtime_defaults()
    pipeline._user_turns.start_speech(timeline=pipeline._timeline)
    pipeline._user_turns.add_transcript("私立医院的。", is_final=True)
    pipeline._user_turns.finish_speech(eot_score=1.0, should_defer=False)
    short = VoiceprintTurnResult(
        signal=SpeakerSignal(
            provider="3d_speaker",
            model="campplus_zh_16k_common",
            known=False,
            score=0.0,
            audio_ms=700,
            latency_ms=10.0,
            profile_id="vp_manson_default",
        ),
        cached=False,
        commit_allowed=False,
        commit_reason="audio_too_short",
    )
    verify_task = asyncio.create_task(asyncio.sleep(0, result=short))
    pipeline._schedule_voiceprint_gated_commit(
        verify_task=verify_task,
        eot_model=pipeline._get_eot_model.return_value,
        transcript="私立医院的。",
        timeline=pipeline._timeline,
    )

    await asyncio.gather(*pipeline._pending_voiceprint_commit_tasks)

    pipeline._session.commit_user_turn.assert_not_called()
    pipeline._session.clear_user_turn.assert_called_once()
    assert pipeline._user_turns.active is not None
    assert pipeline._user_turns.active.state == "waiting_merge"
    assert pipeline._timeline.attrs["voiceprint_deferred"]["state"] == "waiting_merge"


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
async def test_completed_turn_hook_aligns_waiting_candidate_and_cancels_deferred_commit() -> None:
    pipeline = _make_pipeline_with_session(latest_asr_text="")
    pipeline._timeline = TurnTimeline("owner-hook-waiting-turn")
    setter = MagicMock()
    pipeline._factory = SimpleNamespace(
        llm=SimpleNamespace(llm=SimpleNamespace(set_next_user_text=setter))
    )
    pipeline._ensure_runtime_defaults()
    pipeline._user_turns.start_speech(timeline=pipeline._timeline)
    pipeline._user_turns.add_transcript("换个话题。", is_final=True)
    pipeline._user_turns.add_transcript("我们聊一下定价。", is_final=True)
    pipeline._user_turns.finish_speech(eot_score=0.1, should_defer=True)
    deferred = asyncio.create_task(asyncio.sleep(10))
    pipeline._deferred_low_eot_commit_task = deferred
    result = VoiceprintTurnResult(
        signal=SpeakerSignal(
            provider="3d_speaker",
            model="campplus_zh_16k_common",
            known=True,
            score=0.66,
            audio_ms=2400,
            latency_ms=20.0,
            profile_id="vp_manson_default",
        ),
        cached=True,
        commit_allowed=True,
        commit_reason="cached_owner_context",
    )
    pipeline._completed_turn_voiceprint_task = asyncio.create_task(
        asyncio.sleep(0, result=result)
    )
    pipeline._completed_turn_voiceprint_timeline = pipeline._timeline

    allowed = await pipeline._voiceprint_allows_completed_turn(
        new_message=SimpleNamespace(text_content="我们聊一下定价。")
    )
    await asyncio.sleep(0)

    assert allowed is True
    assert deferred.cancelled()
    assert pipeline._user_turns.snapshot()["state"] == "committed"
    assert (
        pipeline._timeline.attrs["canonical_user_text"]["text_preview"]
        == "换个话题。我们聊一下定价。"
    )
    setter.assert_called_once_with(
        "换个话题。我们聊一下定价。",
        source="framework_completed_turn",
    )


@pytest.mark.asyncio
async def test_completed_turn_hook_keeps_waiting_merge_on_short_voiceprint() -> None:
    pipeline = _make_pipeline_with_session(latest_asr_text="")
    pipeline._turn_policy = TurnPolicyConfig()
    pipeline._turn_runtime = TurnPolicyRuntime(pipeline._turn_policy)
    timeline = TurnTimeline("waiting-short-voiceprint")
    pipeline._timeline = timeline
    pipeline._ensure_user_turn_coordinator()
    pipeline._user_turns.start_speech(timeline=timeline)
    pipeline._user_turns.add_transcript("私立医院的。", is_final=True)
    pipeline._user_turns.finish_speech(eot_score=0.01, should_defer=True)
    result = VoiceprintTurnResult(
        signal=SpeakerSignal(
            provider="3d_speaker",
            model="campplus_zh_16k_common",
            known=False,
            score=0.0,
            audio_ms=400,
            latency_ms=10.0,
            profile_id="vp_manson_default",
        ),
        cached=False,
        commit_allowed=False,
        commit_reason="audio_too_short",
    )
    pipeline._completed_turn_voiceprint_task = asyncio.create_task(
        asyncio.sleep(0, result=result)
    )
    pipeline._completed_turn_voiceprint_timeline = timeline

    allowed = await pipeline._voiceprint_allows_completed_turn(
        new_message=SimpleNamespace(text_content="私立医院的。")
    )

    assert allowed is False
    assert pipeline._user_turns.active is not None
    assert pipeline._user_turns.active.state == "waiting_merge"
    assert timeline.attrs["voiceprint_deferred"]["state"] == "waiting_merge"
    pipeline._session.clear_user_turn.assert_called_once()


@pytest.mark.asyncio
async def test_completed_turn_hook_defers_short_statement_for_continuation() -> None:
    pipeline = _make_pipeline_with_session(latest_asr_text="")
    policy = TurnPolicyConfig(
        eot=EotPolicyConfig(eot_unlikely_threshold=0.5, tail_hang_silence_ms=100)
    )
    pipeline._turn_policy = policy
    pipeline._turn_runtime = TurnPolicyRuntime(policy)
    timeline = TurnTimeline("statement-fragment-hook")
    pipeline._timeline = timeline
    pipeline._ensure_runtime_defaults()
    # EOT is unsure here (low score), so the short-statement hedge still applies
    # and the framework-completed turn defers for continuation.
    pipeline._get_eot_model.return_value.current_eot_score = 0.01
    pipeline._user_turns.start_speech(timeline=timeline)
    pipeline._user_turns.add_transcript("私立医院的。", is_final=True)
    pipeline._user_turns.finish_speech(eot_score=0.01, should_defer=True)
    deferred = asyncio.create_task(asyncio.sleep(10))
    pipeline._deferred_low_eot_commit_task = deferred
    result = VoiceprintTurnResult(
        signal=SpeakerSignal(
            provider="3d_speaker",
            model="campplus_zh_16k_common",
            known=True,
            score=0.66,
            audio_ms=1600,
            latency_ms=10.0,
            profile_id="vp_manson_default",
        ),
        cached=False,
        commit_allowed=True,
        commit_reason="owner_high_confidence",
    )
    voiceprint_task = asyncio.create_task(asyncio.sleep(0, result=result))
    pipeline._completed_turn_voiceprint_task = voiceprint_task
    pipeline._candidate_voiceprint_tasks = [voiceprint_task]
    pipeline._completed_turn_voiceprint_timeline = timeline

    allowed = await pipeline._voiceprint_allows_completed_turn(
        new_message=SimpleNamespace(text_content="私立医院的。 给医生做的系统。")
    )
    await asyncio.sleep(0)

    assert allowed is False
    assert deferred.cancelled()
    assert pipeline._deferred_low_eot_commit_task is not None
    assert pipeline._session.commit_user_turn.call_count == 0
    pipeline._session.clear_user_turn.assert_called_once_with()
    assert pipeline._user_turns.active is not None
    assert pipeline._user_turns.active.state == "waiting_merge"
    assert timeline.attrs["framework_completed_deferred"]["state"] == "waiting_merge"
    assert (
        timeline.attrs["user_turn_coordinator"]["selected_text_preview"]
        == "私立医院的。 给医生做的系统。"
    )


@pytest.mark.asyncio
async def test_completed_turn_hook_does_not_defer_when_eot_confident() -> None:
    # SOTA alignment: when the framework AND the EOT model both consider the turn
    # complete, do not re-hold it on the short-statement text heuristic (which a
    # dropped 「吗？」 would defeat). The turn takes the normal reply path instead of
    # the fragile return-False + re-commit defer path. Mirrors the production bug
    # where "你的笑话已经讲完了吗？" (EOT=1.0) was wrongly deferred and never answered.
    pipeline = _make_pipeline_with_session(latest_asr_text="")
    policy = TurnPolicyConfig(
        eot=EotPolicyConfig(eot_unlikely_threshold=0.5, tail_hang_silence_ms=100)
    )
    pipeline._turn_policy = policy
    pipeline._turn_runtime = TurnPolicyRuntime(policy)
    timeline = TurnTimeline("statement-fragment-confident")
    pipeline._timeline = timeline
    pipeline._ensure_runtime_defaults()
    # EOT is confident the turn is complete -> trust it, don't defer.
    pipeline._get_eot_model.return_value.current_eot_score = 1.0
    pipeline._user_turns.start_speech(timeline=timeline)
    pipeline._user_turns.add_transcript("私立医院的。", is_final=True)
    pipeline._user_turns.finish_speech(eot_score=1.0, should_defer=False)
    result = VoiceprintTurnResult(
        signal=SpeakerSignal(
            provider="3d_speaker",
            model="campplus_zh_16k_common",
            known=True,
            score=0.66,
            audio_ms=1600,
            latency_ms=10.0,
            profile_id="vp_manson_default",
        ),
        cached=False,
        commit_allowed=True,
        commit_reason="owner_high_confidence",
    )
    voiceprint_task = asyncio.create_task(asyncio.sleep(0, result=result))
    pipeline._completed_turn_voiceprint_task = voiceprint_task
    pipeline._candidate_voiceprint_tasks = [voiceprint_task]
    pipeline._completed_turn_voiceprint_timeline = timeline

    allowed = await pipeline._voiceprint_allows_completed_turn(
        new_message=SimpleNamespace(text_content="私立医院的。")
    )

    # Not deferred -> framework proceeds to reply (allowed True), no waiting_merge.
    assert allowed is True
    assert pipeline._user_turns.active is None or (
        pipeline._user_turns.active.state != "waiting_merge"
    )


@pytest.mark.asyncio
async def test_completed_turn_hook_respects_voiceprint_deferred_merge_window() -> None:
    pipeline = _make_pipeline_with_session(latest_asr_text="")
    policy = TurnPolicyConfig(
        eot=EotPolicyConfig(eot_unlikely_threshold=0.5, tail_hang_silence_ms=100)
    )
    pipeline._turn_policy = policy
    pipeline._turn_runtime = TurnPolicyRuntime(policy)
    timeline = TurnTimeline("voiceprint-merge-window-hook")
    pipeline._timeline = timeline
    pipeline._ensure_runtime_defaults()
    pipeline._user_turns.start_speech(timeline=timeline)
    pipeline._user_turns.add_transcript("私立医院的。", is_final=True)
    pipeline._user_turns.finish_speech(eot_score=1.0, should_defer=False)
    pipeline._user_turns.defer_voiceprint_inconclusive(
        transcript="私立医院的。",
        reason="voiceprint_inconclusive:audio_too_short",
        timeline=timeline,
    )
    pipeline._user_turns.start_speech(timeline=timeline)
    pipeline._user_turns.add_transcript("主要给医生做的系统。", is_final=True)
    pipeline._user_turns.finish_speech(eot_score=0.01, should_defer=True)
    deferred = asyncio.create_task(asyncio.sleep(10))
    pipeline._deferred_low_eot_commit_task = deferred
    result = VoiceprintTurnResult(
        signal=SpeakerSignal(
            provider="3d_speaker",
            model="campplus_zh_16k_common",
            known=True,
            score=0.78,
            audio_ms=1900,
            latency_ms=10.0,
            profile_id="vp_manson_default",
        ),
        cached=False,
        commit_allowed=True,
        commit_reason="owner_high_confidence",
    )
    pipeline._completed_turn_voiceprint_task = asyncio.create_task(
        asyncio.sleep(0, result=result)
    )
    pipeline._completed_turn_voiceprint_timeline = timeline

    allowed = await pipeline._voiceprint_allows_completed_turn(
        new_message=SimpleNamespace(
            text_content="私立医院的。主要给医生做的系统。"
        )
    )
    await asyncio.sleep(0)
    new_deferred = pipeline._deferred_low_eot_commit_task

    assert allowed is False
    assert deferred.cancelled()
    assert new_deferred is not None
    assert new_deferred is not deferred
    assert pipeline._session.commit_user_turn.call_count == 0
    assert pipeline._user_turns.active is not None
    assert pipeline._user_turns.active.state == "waiting_merge"
    assert pipeline._user_turns.active.voiceprint_reason == (
        "voiceprint_inconclusive:audio_too_short"
    )
    assert (
        timeline.attrs["framework_completed_deferred"]["text_preview"]
        == "私立医院的。主要给医生做的系统。"
    )

    new_deferred.cancel()
    try:
        await new_deferred
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_completed_turn_hook_respects_statement_sequence_merge_window() -> None:
    pipeline = _make_pipeline_with_session(latest_asr_text="")
    policy = TurnPolicyConfig(
        eot=EotPolicyConfig(eot_unlikely_threshold=0.5, tail_hang_silence_ms=100)
    )
    pipeline._turn_policy = policy
    pipeline._turn_runtime = TurnPolicyRuntime(policy)
    timeline = TurnTimeline("statement-sequence-window-hook")
    pipeline._timeline = timeline
    pipeline._ensure_runtime_defaults()
    pipeline._user_turns.start_speech(timeline=timeline)
    pipeline._user_turns.add_transcript("私立医院的。", is_final=True)
    pipeline._user_turns.finish_speech(eot_score=0.01, should_defer=True)
    pipeline._user_turns.start_speech(timeline=timeline)
    pipeline._user_turns.add_transcript("主要给医生做的系统。", is_final=True)
    pipeline._user_turns.finish_speech(eot_score=0.55, should_defer=True)
    deferred = asyncio.create_task(asyncio.sleep(10))
    pipeline._deferred_low_eot_commit_task = deferred
    result = VoiceprintTurnResult(
        signal=SpeakerSignal(
            provider="3d_speaker",
            model="campplus_zh_16k_common",
            known=True,
            score=0.78,
            audio_ms=1900,
            latency_ms=10.0,
            profile_id="vp_manson_default",
        ),
        cached=False,
        commit_allowed=True,
        commit_reason="owner_high_confidence",
    )
    pipeline._completed_turn_voiceprint_task = asyncio.create_task(
        asyncio.sleep(0, result=result)
    )
    pipeline._completed_turn_voiceprint_timeline = timeline

    allowed = await pipeline._voiceprint_allows_completed_turn(
        new_message=SimpleNamespace(
            text_content="私立医院的。主要给医生做的系统。"
        )
    )
    await asyncio.sleep(0)
    new_deferred = pipeline._deferred_low_eot_commit_task

    assert allowed is False
    assert deferred.cancelled()
    assert new_deferred is not None
    assert pipeline._session.commit_user_turn.call_count == 0
    assert pipeline._user_turns.active is not None
    assert pipeline._user_turns.active.state == "waiting_merge"
    assert pipeline._user_turns.active.merge_reason == "low_eot_wait_for_continuation"
    assert timeline.attrs["framework_completed_deferred"]["state"] == "waiting_merge"

    new_deferred.cancel()
    try:
        await new_deferred
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_deferred_framework_completed_commits_after_grace() -> None:
    pipeline = _make_pipeline_with_session(latest_asr_text="")
    policy = TurnPolicyConfig(
        eot=EotPolicyConfig(eot_unlikely_threshold=0.5, tail_hang_silence_ms=10)
    )
    pipeline._turn_policy = policy
    pipeline._turn_runtime = TurnPolicyRuntime(policy)
    timeline = TurnTimeline("statement-fragment-grace")
    pipeline._timeline = timeline
    pipeline._ensure_runtime_defaults()
    # EOT unsure (low score) -> short-statement hedge applies -> defers.
    pipeline._get_eot_model.return_value.current_eot_score = 0.01
    pipeline._user_turns.start_speech(timeline=timeline)
    pipeline._user_turns.add_transcript("私立医院的。", is_final=True)
    pipeline._user_turns.finish_speech(eot_score=0.01, should_defer=True)
    result = VoiceprintTurnResult(
        signal=SpeakerSignal(
            provider="3d_speaker",
            model="campplus_zh_16k_common",
            known=True,
            score=0.66,
            audio_ms=1600,
            latency_ms=10.0,
            profile_id="vp_manson_default",
        ),
        cached=False,
        commit_allowed=True,
        commit_reason="owner_high_confidence",
    )
    voiceprint_task = asyncio.create_task(asyncio.sleep(0, result=result))
    pipeline._completed_turn_voiceprint_task = voiceprint_task
    pipeline._candidate_voiceprint_tasks = [voiceprint_task]
    pipeline._completed_turn_voiceprint_timeline = timeline

    allowed = await pipeline._voiceprint_allows_completed_turn(
        new_message=SimpleNamespace(text_content="私立医院的。 给医生做的系统。")
    )
    assert allowed is False
    await asyncio.sleep(0.05)
    await asyncio.gather(*pipeline._pending_voiceprint_commit_tasks)

    pipeline._session.commit_user_turn.assert_called_once_with(transcript_timeout=5.0)
    eot = pipeline._get_eot_model.return_value
    assert eot.record_turn.call_args.args[0] == "私立医院的。 给医生做的系统。"


@pytest.mark.asyncio
async def test_completed_turn_hook_blocks_backchannel_response_even_when_voiceprint_allows() -> None:
    pipeline = _make_pipeline_with_session(latest_asr_text="")
    pipeline._timeline = TurnTimeline("owner-hook-backchannel")
    timeline = pipeline._timeline
    timeline.set_attr(
        "decision",
        {
            "action": "rollback",
            "intent": "backchannel",
            "reason": "intent:backchannel",
        },
    )
    pipeline._ensure_runtime_defaults()
    pipeline._user_turns.start_speech(timeline=pipeline._timeline)
    pipeline._user_turns.add_transcript("对呀。", is_final=True)
    pipeline._user_turns.finish_speech(eot_score=0.0, should_defer=True)
    deferred = asyncio.create_task(asyncio.sleep(10))
    pipeline._deferred_low_eot_commit_task = deferred
    result = VoiceprintTurnResult(
        signal=SpeakerSignal(
            provider="3d_speaker",
            model="campplus_zh_16k_common",
            known=True,
            score=0.66,
            audio_ms=900,
            latency_ms=0.0,
            profile_id="vp_manson_default",
        ),
        cached=True,
        commit_allowed=True,
        commit_reason="cached_owner_context",
    )
    pipeline._completed_turn_voiceprint_task = asyncio.create_task(
        asyncio.sleep(0, result=result)
    )
    pipeline._completed_turn_voiceprint_timeline = pipeline._timeline

    allowed = await pipeline._voiceprint_allows_completed_turn(
        new_message=SimpleNamespace(text_content="对呀。")
    )
    await asyncio.sleep(0)

    assert allowed is False
    assert deferred.cancelled()
    pipeline._session.clear_user_turn.assert_called_once()
    assert pipeline._user_turns.snapshot()["state"] == "rejected"
    assert "canonical_user_text" not in timeline.attrs
    assert timeline.attrs["voiceprint_commit_gate"]["allowed"] is True
    assert pipeline._timeline is None


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
