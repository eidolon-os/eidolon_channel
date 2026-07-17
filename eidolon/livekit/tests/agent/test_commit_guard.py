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
from unittest.mock import MagicMock

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
        set_turn_control_metadata=MagicMock(),
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
async def test_framework_gate_defers_stale_final_until_assembled_candidate_is_closed() -> None:
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
    first_allowed = await pipeline._ensure_turn_completion().voiceprint_allows_completed_turn(
        turn_ctx=ChatContext.empty(), new_message=message
    )

    assert first_allowed is False
    assert pipeline._user_turns.snapshot()["state"] == "open"

    pipeline._on_user_transcribed(_transcript_event("那你记下来吧，这是我们约定。", is_final=True))
    final_message = ChatMessage(
        role="user",
        content=["那你记下来吧，这是我们约定。"],
    )
    allowed = await pipeline._ensure_turn_completion().voiceprint_allows_completed_turn(
        turn_ctx=ChatContext.empty(), new_message=final_message
    )

    canonical = "好啊。那你记下来吧，这是我们约定。"
    assert allowed is True
    assert pipeline._user_turns.selected_text == canonical
    assert final_message.content == [canonical]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("framework_final", "later_interim", "canonical"),
    [
        ("好的。", "那太好", "好的。那太好"),
        ("OK呀。", "到时候我还可以带几个朋友", "OK呀。到时候我还可以带几个朋友"),
    ],
)
async def test_stale_final_waits_for_later_interim_to_become_final(
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
    first_allowed = await pipeline._ensure_turn_completion().voiceprint_allows_completed_turn(
        turn_ctx=ChatContext.empty(), new_message=message
    )

    assert first_allowed is False
    pipeline._on_user_transcribed(_transcript_event(later_interim, is_final=True))
    final_message = ChatMessage(role="user", content=[later_interim])
    allowed = await pipeline._ensure_turn_completion().voiceprint_allows_completed_turn(
        turn_ctx=ChatContext.empty(), new_message=final_message
    )

    assert allowed is True
    assert pipeline._user_turns.selected_text == canonical
    assert final_message.text_content == canonical


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

    stale = ChatMessage(role="user", content=["不是。"])
    assert (
        await pipeline._ensure_turn_completion().voiceprint_allows_completed_turn(
            turn_ctx=ChatContext.empty(), new_message=stale
        )
        is False
    )

    pipeline._on_user_transcribed(_transcript_event("我刚才说错了。", is_final=True))
    completed = ChatMessage(role="user", content=["我刚才说错了。"])
    assert (
        await pipeline._ensure_turn_completion().voiceprint_allows_completed_turn(
            turn_ctx=ChatContext.empty(), new_message=completed
        )
        is True
    )
    assert completed.text_content == "不是。我刚才说错了。"


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
