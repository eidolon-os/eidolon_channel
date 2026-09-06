"""Product-turn completion invariants for the full-duplex runtime.

LiveKit owns automatic acoustic endpointing.  VAD boundaries only record
evidence; the framework-completed hook is the single normal product boundary
that may admit or reject a turn before the LLM.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from livekit.agents.llm import ChatContext, ChatMessage

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.output.ducking import OutputDuckingController
from eidolon.livekit.agent.context import InterruptedContextManager
from eidolon.livekit.agent.full_duplex.context_ledger import FullDuplexContextLedger
from eidolon.livekit.agent.full_duplex.state_machine import FullDuplexPhase
from eidolon.livekit.agent.turn_policy import Action, Decision, InterruptIntent
from eidolon.livekit.agent.session.voiceprint import VoiceprintTurnResult
from eidolon.livekit.common.speaker_verification import SpeakerSignal


def _make_pipeline_with_session(*, latest_asr_text: str = "") -> Any:
    from eidolon.livekit.agent.full_duplex import StreamingPipeline

    pipeline = StreamingPipeline.__new__(StreamingPipeline)
    pipeline._session = MagicMock()
    pipeline._session.output = MagicMock()
    pipeline._session.output.audio = MagicMock()
    pipeline._callbacks = MagicMock()
    pipeline._room = None
    pipeline._allow_interruptions = False
    pipeline._latest_asr_text = latest_asr_text
    pipeline._ducking = OutputDuckingController()
    pipeline._ducking.mixer = None
    pipeline._ducking.timeout_task = None
    pipeline._ducking.suspend_start = 0.0
    pipeline._ducking.last_unduck_time = 0.0
    pipeline._user_speaking_start_time = None
    pipeline._skip_commit_after_interrupt_cancel = False
    pipeline._suppress_commit_after_interrupt_until = 0.0
    pipeline._stt_commit_transcript_timeout = 0.05

    eot = MagicMock()
    eot._current_eot_score = 1.0
    eot.current_eot_score = 1.0
    pipeline._get_eot_model = MagicMock(return_value=eot)

    effects = MagicMock()
    effects._ducking = pipeline._ducking
    effects.soft_interrupt_active.return_value = False
    pipeline._interruption_effects = effects
    pipeline._context_ledger = MagicMock()

    llm = SimpleNamespace(
        set_turn_decision_metadata=MagicMock(),
    )
    pipeline._factory = SimpleNamespace(llm=SimpleNamespace(llm=llm))
    return pipeline


def _user_state_event(old: str, new: str) -> SimpleNamespace:
    return SimpleNamespace(old_state=old, new_state=new)


def _transcript_event(text: str, *, is_final: bool) -> SimpleNamespace:
    return SimpleNamespace(transcript=text, is_final=is_final, speaker_id="owner")


def _voiceprint_result(*, allowed: bool, reason: str) -> VoiceprintTurnResult:
    return VoiceprintTurnResult(
        signal=SpeakerSignal(
            provider="test",
            model="test",
            known=allowed,
            score=0.9 if allowed else None,
            audio_ms=1600,
            latency_ms=1.0,
            profile_id="owner",
            error=(reason if reason in {"audio_too_short", "insufficient_audio"} else ""),
        ),
        cached=False,
        commit_allowed=allowed,
        commit_reason=reason,
    )


@pytest.mark.parametrize("text", ["", "你好世界"])
def test_vad_stop_never_commits_or_clears_product_turn(text: str) -> None:
    pipeline = _make_pipeline_with_session(latest_asr_text=text)

    pipeline._on_user_state_changed(_user_state_event("speaking", "listening"))

    pipeline._session.commit_user_turn.assert_not_called()
    pipeline._session.clear_user_turn.assert_not_called()
    assert pipeline._latest_asr_text == text
    pipeline._get_eot_model.return_value.update_vad.assert_called_once_with(False)


@pytest.mark.asyncio
async def test_framework_completed_is_the_single_normal_commit_boundary() -> None:
    pipeline = _make_pipeline_with_session()
    timeline = TurnTimeline("framework-owner")
    pipeline._timeline = timeline
    pipeline._ensure_runtime_defaults()
    pipeline._user_turns.start_speech(timeline=timeline)
    pipeline._user_turns.add_transcript("你好世界", is_final=True)

    message = ChatMessage(role="user", content=["你好世界"])
    allowed = await pipeline._ensure_turn_completion().voiceprint_allows_completed_turn(
        turn_ctx=ChatContext.empty(), new_message=message
    )

    assert allowed is True
    assert pipeline._user_turns.snapshot()["state"] == "committed"
    pipeline._session.commit_user_turn.assert_not_called()
    pipeline._session.clear_user_turn.assert_not_called()
    assert message.content == ["你好世界"]
    assert "turn_committed_at" in timeline.timestamps
    setter = pipeline._factory.llm.llm.set_turn_decision_metadata
    setter.assert_called_once()
    metadata = setter.call_args.args[0]
    assert metadata["decision"] == "commit"
    assert metadata["evidence"]["boundary"] == "framework_completed_turn"
    assert timeline.attrs["committed_turn_decision"] == metadata


def test_livekit_transcription_timeout_rejects_transcriptless_candidate() -> None:
    pipeline = _make_pipeline_with_session()
    pipeline._ensure_runtime_defaults()
    timeline = TurnTimeline("transcriptless-deadline")
    pipeline._timeline = timeline
    candidate = pipeline._user_turns.start_speech(timeline=timeline)
    pipeline._user_turns.note_speech_stopped(eot_score=0.0)

    rejected = pipeline._ensure_turn_completion().handle_transcription_timeout(
        SimpleNamespace(speech_duration=1.2, vad_speech_started_at=123.0)
    )

    assert rejected is True
    assert candidate.state == "rejected"
    assert candidate.reject_reason == "speech_stopped_without_transcript_deadline"
    expiry = timeline.attrs["transcriptless_candidate_expiry"]
    assert expiry["outcome"] == "rejected"
    assert expiry["source"] == "livekit_user_transcription_timeout"
    assert expiry["speech_duration"] == 1.2
    pipeline._session.say.assert_called_once_with(
        "抱歉，刚才没听清，请再说一遍好吗？",
        add_to_chat_ctx=False,
    )
    assert timeline.attrs["timeline_flush_reason"] == ("speech_stopped_without_transcript_deadline")


def test_livekit_transcription_timeout_ignores_candidate_with_text() -> None:
    pipeline = _make_pipeline_with_session()
    pipeline._ensure_runtime_defaults()
    timeline = TurnTimeline("transcript-before-deadline")
    pipeline._timeline = timeline
    candidate = pipeline._user_turns.start_speech(timeline=timeline)
    pipeline._user_turns.note_speech_stopped(eot_score=0.0)

    pipeline._user_turns.add_transcript("迟到但有效的转写", is_final=False)
    rejected = pipeline._ensure_turn_completion().handle_transcription_timeout(
        SimpleNamespace(speech_duration=1.2, vad_speech_started_at=123.0)
    )

    assert rejected is False
    assert candidate.state == "open"
    assert candidate.selected_text == "迟到但有效的转写"
    assert "transcriptless_candidate_expiry" not in timeline.attrs


@pytest.mark.asyncio
async def test_cross_vad_repeated_stt_hypothesis_commits_once() -> None:
    """A provider FINAL explicitly covers its prior-generation interim alias."""

    pipeline = _make_pipeline_with_session()
    pipeline._ensure_runtime_defaults()
    timeline = TurnTimeline("cross-vad-repeated-hypothesis")
    pipeline._timeline = timeline
    pipeline._user_turns.start_speech(timeline=timeline, now=0.0)
    pipeline._on_user_transcribed(_transcript_event("铁锤三二五", is_final=False))
    pipeline._user_turns.note_speech_stopped(eot_score=0.0, now=0.2)
    pipeline._user_turns.start_speech(timeline=timeline, now=0.5)
    pipeline._on_user_transcribed(_transcript_event("铁锤三二五", is_final=False))
    pipeline._on_user_transcribed(_transcript_event("铁锤三二五。", is_final=True))

    assert pipeline._user_turns.active is not None
    first_segment, final_segment = pipeline._user_turns.active.segments
    assert first_segment.final_text == ""
    assert first_segment.covered_by_generation_id == final_segment.generation_id

    message = ChatMessage(role="user", content=["铁锤三二五。"])
    first = await pipeline._ensure_turn_completion().voiceprint_allows_completed_turn(
        turn_ctx=ChatContext.empty(), new_message=message
    )
    duplicate = await pipeline._ensure_turn_completion().voiceprint_allows_completed_turn(
        turn_ctx=ChatContext.empty(), new_message=message
    )

    assert first is True
    assert duplicate is False
    assert message.text_content == "铁锤三二五。"
    assert "turn_committed_at" in timeline.timestamps


@pytest.mark.asyncio
async def test_completion_covering_latest_interim_commits_complete_candidate() -> None:
    pipeline = _make_pipeline_with_session()
    timeline = TurnTimeline("final-then-interim")
    pipeline._timeline = timeline
    pipeline._ensure_runtime_defaults()
    pipeline._user_turns.start_speech(timeline=timeline)
    pipeline._user_turns.add_transcript("好啊。", is_final=True)
    pipeline._user_turns.add_transcript(
        "那你记下来吧，这是我们约定。",
        is_final=False,
    )

    message = ChatMessage(
        role="user",
        content=["那你记下来吧，这是我们约定。"],
    )
    allowed = await pipeline._ensure_turn_completion().voiceprint_allows_completed_turn(
        turn_ctx=ChatContext.empty(), new_message=message
    )

    assert allowed is True
    canonical = "好啊。那你记下来吧，这是我们约定。"
    assert pipeline._user_turns.selected_text == canonical
    assert message.content == [canonical]
    pipeline._session.clear_user_turn.assert_not_called()


@pytest.mark.asyncio
async def test_framework_gate_waits_for_evidence_without_second_completion() -> None:
    """Replay the event order observed in the 13-turn Box-3 timeline.

    LiveKit can complete the earlier short FINAL after a new VAD segment has
    already contributed a substantive INTERIM.  Admission must evaluate the
    assembled product-turn candidate; evaluating only the stale framework text
    recreates the observed low-evidence rejection and drops the follow-up.
    """

    pipeline = _make_pipeline_with_session()
    pipeline._get_eot_model.return_value.current_eot_score = 0.0
    pipeline._get_eot_model.return_value._current_eot_score = 0.0
    pipeline._ensure_runtime_defaults()
    pipeline._on_user_state_changed(_user_state_event("listening", "speaking"))
    timeline = pipeline._timeline
    assert timeline is not None
    timeline.set_attr(
        "attention_admission_events",
        [
            {
                "action": "observe",
                "reason": "playback_low_evidence_transcript",
            }
        ],
    )
    pipeline._on_user_transcribed(_transcript_event("好啊。", is_final=True))
    pipeline._on_user_state_changed(_user_state_event("speaking", "listening"))
    pipeline._on_user_state_changed(_user_state_event("listening", "speaking"))
    pipeline._on_user_transcribed(_transcript_event("那你记下来吧，这是我们约定。", is_final=False))

    message = ChatMessage(role="user", content=["好啊。"])
    completion = asyncio.create_task(
        pipeline._ensure_turn_completion().voiceprint_allows_completed_turn(
            turn_ctx=ChatContext.empty(), new_message=message
        )
    )
    await asyncio.sleep(0)
    assert completion.done() is False
    assert pipeline._user_turns.snapshot()["state"] == "open"

    pipeline._on_user_transcribed(_transcript_event("那你记下来吧，这是我们约定。", is_final=True))
    allowed = await completion

    canonical = "好啊。那你记下来吧，这是我们约定。"
    assert allowed is True
    assert pipeline._user_turns.selected_text == canonical
    assert message.content == [canonical]
    assert timeline.attrs["framework_completed_settlement"]["outcome"] == "ready"
    assert timeline.attrs["framework_completed_settlement"]["evidence_updates"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("framework_final", "later_interim", "canonical"),
    [
        ("好的。", "那太好", "好的。那太好"),
        ("OK呀。", "到时候我还可以带几个朋友", "OK呀。到时候我还可以带几个朋友"),
    ],
)
async def test_stale_final_is_released_by_later_final_without_second_completion(
    framework_final: str,
    later_interim: str,
    canonical: str,
) -> None:
    """Replay the other stale-final patterns from Box-3 turns 10 and 11."""

    pipeline = _make_pipeline_with_session()
    timeline = TurnTimeline("box3-stale-final-variants")
    pipeline._timeline = timeline
    pipeline._ensure_runtime_defaults()
    pipeline._user_turns.start_speech(timeline=timeline)
    pipeline._on_user_transcribed(_transcript_event(framework_final, is_final=True))
    pipeline._on_user_transcribed(_transcript_event(later_interim, is_final=False))

    message = ChatMessage(role="user", content=[framework_final])
    completion = asyncio.create_task(
        pipeline._ensure_turn_completion().voiceprint_allows_completed_turn(
            turn_ctx=ChatContext.empty(), new_message=message
        )
    )
    await asyncio.sleep(0)
    assert completion.done() is False
    pipeline._on_user_transcribed(_transcript_event(later_interim, is_final=True))
    allowed = await completion

    assert allowed is True
    assert pipeline._user_turns.selected_text == canonical
    assert message.text_content == canonical


@pytest.mark.asyncio
async def test_correction_multi_final_starts_one_generation_with_complete_canonical_turn() -> None:
    """A late sentence FINAL must close the same physical speech before commit."""

    pipeline = _make_pipeline_with_session()
    timeline = TurnTimeline("correction-multi-final")
    pipeline._timeline = timeline
    pipeline._ensure_runtime_defaults()
    pipeline._user_turns.start_speech(timeline=timeline)
    pipeline._on_user_transcribed(_transcript_event("不是。", is_final=True))
    pipeline._on_user_transcribed(_transcript_event("我刚才说", is_final=False))

    message = ChatMessage(role="user", content=["不是。"])
    completion = asyncio.create_task(
        pipeline._ensure_turn_completion().voiceprint_allows_completed_turn(
            turn_ctx=ChatContext.empty(), new_message=message
        )
    )
    await asyncio.sleep(0)
    assert completion.done() is False

    pipeline._on_user_transcribed(_transcript_event("我刚才说错了。", is_final=True))
    assert await completion is True
    assert message.text_content == "不是。我刚才说错了。"


@pytest.mark.asyncio
async def test_orphan_interim_commits_best_known_text_at_hard_deadline() -> None:
    """No provider callback is required after LiveKit completes the turn."""

    pipeline = _make_pipeline_with_session()
    pipeline._stt_commit_transcript_timeout = 0.01
    timeline = TurnTimeline("orphan-interim-deadline")
    pipeline._timeline = timeline
    pipeline._ensure_runtime_defaults()
    pipeline._user_turns.start_speech(timeline=timeline)
    pipeline._on_user_transcribed(_transcript_event("好啊。", is_final=True))
    pipeline._on_user_transcribed(_transcript_event("那你记下来吧，这是我们约定。", is_final=False))

    message = ChatMessage(role="user", content=["好啊。"])
    allowed = await pipeline._ensure_turn_completion().voiceprint_allows_completed_turn(
        turn_ctx=ChatContext.empty(),
        new_message=message,
    )

    assert allowed is True
    assert message.text_content == "好啊。那你记下来吧，这是我们约定。"
    assert pipeline._user_turns.snapshot()["state"] == "committed"
    assert timeline.attrs["framework_completed_settlement"]["outcome"] == "deadline"
    assert pipeline._user_turns.snapshot()["commit_reason"] == (
        "framework_completed_turn:transcript_settlement_deadline"
    )


@pytest.mark.asyncio
async def test_new_candidate_supersedes_turn_waiting_for_provider_evidence() -> None:
    """A late completion cannot commit across a newer acoustic turn."""

    pipeline = _make_pipeline_with_session()
    pipeline._stt_commit_transcript_timeout = 1.0
    old_timeline = TurnTimeline("provider-evidence-waiting")
    pipeline._timeline = old_timeline
    pipeline._ensure_runtime_defaults()
    old_candidate = pipeline._user_turns.start_speech(
        timeline=old_timeline,
        now=0.0,
    )
    pipeline._user_turns.add_transcript("好啊。", is_final=True, now=0.1)
    pipeline._user_turns.add_transcript("旧轮次仍在修订", is_final=False, now=0.2)
    pipeline._user_turns.note_speech_stopped(eot_score=0.0, now=0.3)

    message = ChatMessage(role="user", content=["好啊。"])
    completion = asyncio.create_task(
        pipeline._ensure_turn_completion().voiceprint_allows_completed_turn(
            turn_ctx=ChatContext.empty(),
            new_message=message,
        )
    )
    await asyncio.sleep(0)
    assert completion.done() is False

    new_timeline = TurnTimeline("new-acoustic-turn")
    new_candidate = pipeline._user_turns.start_speech(
        timeline=new_timeline,
        now=2.0,
    )

    assert await completion is False
    assert old_candidate.state == "rejected"
    assert old_candidate.reject_reason == "superseded_by_new_speech"
    assert new_candidate.state == "open"
    assert pipeline._user_turns.active is new_candidate
    assert old_timeline.attrs["framework_completed_settlement"]["outcome"] == ("candidate_replaced")


@pytest.mark.asyncio
async def test_interim_only_candidate_replaces_empty_framework_message() -> None:
    """A VAD/EOT completion may arrive before the STT provider emits FINAL."""

    pipeline = _make_pipeline_with_session()
    timeline = TurnTimeline("box3-interim-only-framework-completion")
    pipeline._timeline = timeline
    pipeline._ensure_runtime_defaults()
    pipeline._user_turns.start_speech(timeline=timeline)
    pipeline._on_user_transcribed(_transcript_event("那你记下来吧，这是我们约定。", is_final=False))

    message = ChatMessage(role="user", content=[""])
    allowed = await pipeline._ensure_turn_completion().voiceprint_allows_completed_turn(
        turn_ctx=ChatContext.empty(), new_message=message
    )

    assert allowed is True
    assert message.text_content == "那你记下来吧，这是我们约定。"


@pytest.mark.asyncio
async def test_wrong_speaker_is_explicitly_rejected_without_clearing_next_audio() -> None:
    pipeline = _make_pipeline_with_session()
    timeline = TurnTimeline("wrong-speaker")
    pipeline._timeline = timeline
    pipeline._ensure_runtime_defaults()
    pipeline._user_turns.start_speech(timeline=timeline)
    pipeline._user_turns.add_transcript("视频里的声音", is_final=True)
    result = _voiceprint_result(allowed=False, reason="speaker_not_owner")
    pipeline._completed_turn_voiceprint_task = asyncio.create_task(asyncio.sleep(0, result=result))
    pipeline._completed_turn_voiceprint_timeline = timeline

    allowed = await pipeline._ensure_turn_completion().voiceprint_allows_completed_turn(
        turn_ctx=ChatContext.empty(), new_message=SimpleNamespace(text_content="视频里的声音")
    )

    assert allowed is False
    assert pipeline._user_turns.snapshot()["state"] == "rejected"
    assert pipeline._user_turns.snapshot()["reject_reason"] == (
        "voiceprint_blocked:speaker_not_owner"
    )
    pipeline._session.clear_user_turn.assert_not_called()
    assert pipeline._suppress_transcripts_until_next_speech is True


@pytest.mark.asyncio
async def test_framework_respects_observer_inconclusive_voiceprint_decision() -> None:
    pipeline = _make_pipeline_with_session()
    timeline = TurnTimeline("short-owner-sample")
    pipeline._timeline = timeline
    pipeline._ensure_runtime_defaults()
    pipeline._user_turns.start_speech(timeline=timeline)
    pipeline._user_turns.add_transcript("好的", is_final=True)
    result = _voiceprint_result(allowed=False, reason="audio_too_short")
    pipeline._completed_turn_voiceprint_task = asyncio.create_task(asyncio.sleep(0, result=result))
    pipeline._completed_turn_voiceprint_timeline = timeline

    allowed = await pipeline._ensure_turn_completion().voiceprint_allows_completed_turn(
        turn_ctx=ChatContext.empty(), new_message=SimpleNamespace(text_content="好的")
    )

    assert allowed is False
    assert pipeline._user_turns.snapshot()["state"] == "rejected"
    assert not hasattr(pipeline, "_deferred_low_eot_commit_task")
    pipeline._session.commit_user_turn.assert_not_called()
    pipeline._session.clear_user_turn.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["所", "OK", "我再说一下", "等下我再说吧"])
async def test_framework_terminal_boundary_is_text_agnostic(text: str) -> None:
    pipeline = _make_pipeline_with_session()
    timeline = TurnTimeline(f"text-agnostic-{text}")
    pipeline._timeline = timeline
    pipeline._ensure_runtime_defaults()
    pipeline._user_turns.start_speech(timeline=timeline)
    pipeline._user_turns.add_transcript(text, is_final=True)
    timeline.set_attr(
        "attention_admission_events",
        [{"action": "observe", "reason": "playback_low_evidence_transcript"}],
    )

    message = ChatMessage(role="user", content=[text])
    allowed = await pipeline._ensure_turn_completion().voiceprint_allows_completed_turn(
        turn_ctx=ChatContext.empty(), new_message=message
    )

    assert allowed is True
    assert message.text_content == text


@pytest.mark.asyncio
async def test_non_semantic_completed_turn_is_rejected_only_at_product_boundary() -> None:
    pipeline = _make_pipeline_with_session()
    timeline = TurnTimeline("cough")
    pipeline._timeline = timeline
    pipeline._ensure_runtime_defaults()
    pipeline._interruption_orchestrator.start_candidate(timeline=timeline)
    pipeline._interruption_orchestrator.note_turn_policy_decision(
        Decision(
            action=Action.ROLLBACK,
            intent=InterruptIntent.NOISE,
            reason="intent:noise",
        ),
        transcript="咳咳。",
        vad_active=False,
    )
    pipeline._interruption_orchestrator.resolve(
        action="rollback",
        reason="intent:noise",
    )
    pipeline._user_turns.start_speech(timeline=timeline)
    pipeline._user_turns.add_transcript("咳咳。", is_final=True)

    allowed = await pipeline._ensure_turn_completion().voiceprint_allows_completed_turn(
        turn_ctx=ChatContext.empty(), new_message=SimpleNamespace(text_content="咳咳。")
    )

    assert allowed is False
    assert pipeline._user_turns.snapshot()["state"] == "rejected"
    pipeline._session.clear_user_turn.assert_not_called()


@pytest.mark.asyncio
async def test_no_transcript_false_interruption_cannot_reject_later_real_speech() -> None:
    """Welcome playback blip expires, then an idle utterance commits exactly once."""

    pipeline = _make_pipeline_with_session()
    pipeline._ensure_runtime_defaults()
    blip_timeline = TurnTimeline("welcome-vad-blip")
    pipeline._timeline = blip_timeline
    blip = pipeline._user_turns.start_speech(timeline=blip_timeline, now=0.0)
    pipeline._user_turns.note_speech_stopped(eot_score=0.0, now=0.5)
    pipeline._interruption_orchestrator.start_candidate(
        timeline=blip_timeline,
        generation_id=1,
    )

    pipeline._interruption_orchestrator.resolve(
        action="rollback",
        reason="timeout",
    )

    assert blip.state == "rejected"
    assert blip_timeline.attrs["timeline_flush_reason"] == (
        "interruption_expired_resume_no_transcript"
    )

    real_timeline = TurnTimeline("real-idle-speech")
    pipeline._timeline = real_timeline
    real = pipeline._user_turns.start_speech(timeline=real_timeline, now=5.0)
    pipeline._user_turns.add_transcript("你，你好，你好。", is_final=True, now=5.8)
    message = ChatMessage(role="user", content=["你，你好，你好。"])

    first = await pipeline._ensure_turn_completion().voiceprint_allows_completed_turn(
        turn_ctx=ChatContext.empty(),
        new_message=message,
    )
    duplicate = await pipeline._ensure_turn_completion().voiceprint_allows_completed_turn(
        turn_ctx=ChatContext.empty(),
        new_message=message,
    )

    assert real is not blip
    assert real.candidate_id == "real-idle-speech"
    assert first is True
    assert duplicate is False
    assert pipeline._user_turns.snapshot()["state"] == "committed"
    assert real_timeline.attrs["full_duplex_state"]["phase"] == "user_turn_committed"


@pytest.mark.asyncio
async def test_recorded_verdict_cannot_cross_generation_on_shared_timeline() -> None:
    from eidolon.livekit.agent.session.interruption_orchestrator import (
        InterruptionOrchestrator,
    )

    pipeline = _make_pipeline_with_session()
    pipeline._ensure_runtime_defaults()
    timeline = TurnTimeline("shared-product-turn")
    pipeline._timeline = timeline
    pipeline._user_turns.start_speech(timeline=timeline, now=0.0)
    pipeline._user_turns.note_speech_stopped(eot_score=0.0, now=0.2)
    interruption = InterruptionOrchestrator(
        evidence_timeout_sec=6.0,
        min_speech_sec=0.25,
    )
    interruption.start_candidate(timeline=timeline, generation_id=1)
    interruption.resolve(action="rollback", reason="timeout")
    pipeline._interruption_orchestrator = interruption

    pipeline._user_turns.start_speech(timeline=timeline, now=0.5)
    pipeline._user_turns.add_transcript("新的真实语音", is_final=True, now=0.6)
    message = ChatMessage(role="user", content=["新的真实语音"])

    allowed = await pipeline._ensure_turn_completion().voiceprint_allows_completed_turn(
        turn_ctx=ChatContext.empty(),
        new_message=message,
    )

    assert pipeline._user_turns.framework_completion_generation("新的真实语音") == 2
    assert allowed is True
    assert pipeline._user_turns.snapshot()["state"] == "committed"


@pytest.mark.asyncio
async def test_confirmed_cjk_barge_in_stops_playback_and_reaches_output_once() -> None:
    """Replay the Box3 cancel-while-speaking sequence at the product boundary."""

    pipeline = _make_pipeline_with_session()
    pipeline._allow_interruptions = True
    pipeline._room = SimpleNamespace(local_participant=SimpleNamespace(publish_data=AsyncMock()))
    pipeline._ensure_runtime_defaults()
    pipeline._interruption_effects = pipeline._build_interruption_effects()
    timeline = TurnTimeline("confirmed-cjk-barge-in")
    pipeline._timeline = timeline

    # The same product turn already used generation 1 for an empty/noisy VAD
    # fragment.  The substantive interruption belongs to generation 2.
    pipeline._user_turns.start_speech(timeline=timeline, now=0.0)
    pipeline._user_turns.note_speech_stopped(eot_score=0.0, now=0.1)
    pipeline._user_turns.start_speech(timeline=timeline, now=0.2)
    assert pipeline._user_turns.current_generation_id == 2
    pipeline._record_full_duplex_transition(
        FullDuplexPhase.USER_SPEECH_OPEN,
        event="speech_started",
        reason="new_speech_started",
        timeline=timeline,
    )
    pipeline._record_full_duplex_transition(
        FullDuplexPhase.PROVISIONAL_DUCK,
        event="duck_started",
        reason="vad_started",
        timeline=timeline,
    )
    pipeline._user_turns.add_transcript("停一下，请只回答二", is_final=False, now=0.3)
    pipeline._interruption_orchestrator.start_candidate(
        timeline=timeline,
        generation_id=2,
    )
    pipeline._interruption_orchestrator.note_turn_policy_decision(
        Decision(
            action=Action.CANCEL,
            reason="stable_normal_interrupt score=0.48",
            intent=InterruptIntent.NORMAL_INTERRUPT,
        ),
        transcript="停一下，请只回答二",
        vad_active=True,
        eot_score=0.48,
    )

    pipeline._ensure_interruption_effects().cancel_and_interrupt()
    await asyncio.sleep(0)

    pipeline._room.local_participant.publish_data.assert_awaited_once()
    assert timeline.attrs["client_control_events"][-1]["op"] == "playback.stop"
    assert "interrupt_cancel_resolved_at" in timeline.timestamps
    assert pipeline._interruption_orchestrator.state.value == ("confirmed_cancel_collecting_turn")

    # VAD closes while only an interim is available.  The final framework text
    # arrives later with a low EOT score; that score must not undo the already
    # confirmed playback cancel.
    interim = "请只回答2等于几"
    final = "请只回答2等于几。"
    pipeline._user_turns.add_transcript(interim, is_final=False, now=1.7)
    pipeline._user_turns.note_speech_stopped(eot_score=0.237, now=1.8)
    assert pipeline._interruption_orchestrator.finish_confirmed_cancel_speech(interim)
    pipeline._user_turns.add_transcript(final, is_final=True, now=2.1)
    pipeline._get_eot_model.return_value.current_eot_score = 0.237
    pipeline._get_eot_model.return_value._current_eot_score = 0.237

    verdict = pipeline._interruption_orchestrator.verdict_for(
        timeline.turn_id,
        generation_id=2,
    )
    assert verdict is not None
    assert verdict.action.value == "confirmed_cancel"
    assert verdict.continue_to_llm is True

    downstream_brain_tts = MagicMock()
    for message in (
        ChatMessage(role="user", content=[final]),
        ChatMessage(role="user", content=[final]),
    ):
        allowed = await pipeline._ensure_turn_completion().voiceprint_allows_completed_turn(
            turn_ctx=ChatContext.empty(),
            new_message=message,
        )
        if allowed:
            downstream_brain_tts(message.text_content)

    downstream_brain_tts.assert_called_once_with(final)
    assert pipeline._user_turns.snapshot()["state"] == "committed"
    assert pipeline._user_turns.snapshot()["commit_reason"] == "framework_completed_turn"
    assert pipeline._active_agent_output_timeline() is timeline
    assert "turn_committed_at" in timeline.timestamps
    assert any(
        event.get("verdict") == "confirmed_cancel"
        for event in timeline.attrs["framework_completed_gate_events"]
    )


@pytest.mark.asyncio
async def test_duplicate_framework_completion_is_noop_without_audio_clear() -> None:
    pipeline = _make_pipeline_with_session()
    timeline = TurnTimeline("duplicate")
    pipeline._timeline = timeline
    pipeline._ensure_runtime_defaults()
    pipeline._user_turns.start_speech(timeline=timeline)
    pipeline._user_turns.add_transcript("记住了。", is_final=True)

    first = await pipeline._ensure_turn_completion().voiceprint_allows_completed_turn(
        turn_ctx=ChatContext.empty(), new_message=SimpleNamespace(text_content="记住了。")
    )
    second = await pipeline._ensure_turn_completion().voiceprint_allows_completed_turn(
        turn_ctx=ChatContext.empty(), new_message=SimpleNamespace(text_content="记住了。")
    )

    assert first is True
    assert second is False
    pipeline._session.clear_user_turn.assert_not_called()


@pytest.mark.asyncio
async def test_hard_stop_terminal_closes_before_next_turn_consumes_context() -> None:
    """Replay the Box-3 stop -> new question sequence across both owners."""

    pipeline = _make_pipeline_with_session()
    pipeline._ensure_runtime_defaults()
    manager = InterruptedContextManager()
    manager.last_context = {
        "text": "上一轮被打断的回答",
        "timestamp": time.monotonic(),
        "played_seconds": 1.2,
        "source": "tts_in_flight",
    }
    config = SimpleNamespace(interrupted_context_max_age_sec=60.0)
    pipeline._context_ledger = FullDuplexContextLedger(
        get_session=lambda: pipeline._session,
        get_factory=lambda: pipeline._factory,
        get_duck_mixer=lambda: None,
        get_config=lambda: config,
        get_timeline=lambda: pipeline._timeline,
        manager=manager,
    )

    stop_timeline = TurnTimeline("box3-hard-stop")
    pipeline._timeline = stop_timeline
    pipeline._user_turns.start_speech(timeline=stop_timeline)
    pipeline._user_turns.add_transcript("停，不要说了。", is_final=True)
    pipeline._record_full_duplex_transition(
        FullDuplexPhase.USER_SPEECH_OPEN,
        event="speech_started",
        reason="new_speech_started",
        timeline=stop_timeline,
    )
    pipeline._record_full_duplex_transition(
        FullDuplexPhase.PROVISIONAL_DUCK,
        event="duck_started",
        reason="vad_started",
        timeline=stop_timeline,
    )
    pipeline._record_full_duplex_transition(
        FullDuplexPhase.ACCEPTED_INTERRUPTION,
        event="turn_policy_cancel_accepted",
        reason="intent:hard_stop",
        timeline=stop_timeline,
    )
    pipeline._record_full_duplex_transition(
        FullDuplexPhase.USER_TURN_PENDING,
        event="speech_stopped_waiting_framework",
        reason="automatic_turn_lifecycle",
        timeline=stop_timeline,
    )
    pipeline._interruption_orchestrator.start_candidate(timeline=stop_timeline)
    pipeline._interruption_orchestrator.note_turn_policy_decision(
        Decision(
            action=Action.CANCEL,
            intent=InterruptIntent.HARD_STOP,
            reason="intent:hard_stop",
        ),
        transcript="停，不要说了。",
        vad_active=False,
    )
    pipeline._interruption_orchestrator.resolve(action="cancel", reason="eot_cancel")

    rejected = await pipeline._ensure_turn_completion().voiceprint_allows_completed_turn(
        turn_ctx=ChatContext.empty(),
        new_message=ChatMessage(role="user", content=["停，不要说了。"]),
    )

    assert rejected is False
    assert pipeline._user_turns.snapshot()["state"] == "rejected"
    assert stop_timeline.attrs["full_duplex_state"]["phase"] == "user_turn_rejected"
    assert stop_timeline.attrs["timeline_flush_reason"] == ("interruption_confirmed_cancel")
    assert manager.last_context is not None

    next_timeline = TurnTimeline("box3-next-question")
    pipeline._timeline = next_timeline
    pipeline._user_turns.start_speech(timeline=next_timeline)
    pipeline._user_turns.add_transcript("你还能听到我吗。", is_final=True)
    pipeline._record_full_duplex_transition(
        FullDuplexPhase.USER_SPEECH_OPEN,
        event="speech_started",
        reason="new_speech_started",
        timeline=next_timeline,
    )
    next_context = ChatContext.empty()

    accepted = await pipeline._ensure_turn_completion().voiceprint_allows_completed_turn(
        turn_ctx=next_context,
        new_message=ChatMessage(role="user", content=["你还能听到我吗。"]),
    )

    assert accepted is True
    assert pipeline._user_turns.snapshot()["state"] == "committed"
    assert next_timeline.attrs["full_duplex_state"]["phase"] == "user_turn_committed"
    assert next_timeline.attrs["interrupted_context_consumption"]["outcome"] == ("applied")
    assert len(next_context.messages()) == 1
    assert next_context.messages()[0].role == "system"
    assert manager.last_context is None
    assert pipeline._full_duplex_state.unexpected_transition_count == 0
