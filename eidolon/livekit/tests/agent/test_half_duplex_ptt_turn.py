"""Half-duplex (push-to-talk) turn-boundary tests.

In half_duplex the turn boundary is the explicit PTT signal, NOT VAD/EOT:
the rising edge (``ptt=True``) starts a turn and the falling edge
(``ptt=False``) ends it. The regression these tests pin down:

    User holds PTT and says one sentence with a brief mid-pause —
    "你帮我查一下今天的天气，北京的天气" — then releases. Bailian ASR splits at
    the comma into two finals. The framework's EOT used to fire in the gap
    AFTER release but BEFORE the trailing final landed, committing only
    "你帮我查一下今天的天气。" and dropping "北京的天气" — so the agent asked
    "你目前在哪个城市呢?" as if it never heard the city.

The fix: while a PTT turn is in flight, VAD/EOT must HOLD (defer) the turn;
the PTT release waits (bounded) for the trailing ASR to flush, AGGREGATES every
segment since press into ONE turn, and commits once. full_duplex (barge-in) is
unchanged: VAD/EOT still segment turns there (no PTT signal exists).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from eidolon.livekit.agent.client_audio_state import (
    CLIENT_AUDIO_STATE_TOPIC,
    ClientAudioState,
)
from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.common.config import TurnPolicyConfig


def _make_pipeline(*, half_duplex: bool) -> tuple[Any, dict]:
    """A minimally-initialised StreamingPipeline with a real UserTurnCoordinator.

    Voiceprint is disabled (no remembered gate tasks) so the commit takes the
    direct ``_commit_user_turn_now`` path. ``captured`` records the canonical
    user text published to the (gRPC) LLM via ``set_next_user_text``.
    """
    from eidolon.livekit.agent.streaming import StreamingPipeline

    pipeline = StreamingPipeline.__new__(StreamingPipeline)
    pipeline._session = MagicMock()
    pipeline._session.output = MagicMock()
    pipeline._session.output.audio = MagicMock()
    pipeline._callbacks = MagicMock()
    pipeline._room = None
    pipeline._allow_interruptions = not half_duplex
    pipeline._stt_commit_transcript_timeout = 5.0
    pipeline._latest_asr_text = ""
    pipeline._filler = None
    pipeline._turn_policy = TurnPolicyConfig()

    captured: dict = {"text": None, "source": None}

    def _set_next_user_text(text: str, *, source: str) -> None:
        captured["text"] = text
        captured["source"] = source

    pipeline._factory = SimpleNamespace(
        llm=SimpleNamespace(llm=SimpleNamespace(set_next_user_text=_set_next_user_text))
    )

    eot = MagicMock()
    eot.reset = MagicMock()
    eot.record_turn = MagicMock()
    eot.update_asr = MagicMock()
    eot.update_vad = MagicMock()
    eot._current_eot_score = 0.99
    eot.current_eot_score = 0.99
    pipeline._get_eot_model = MagicMock(return_value=eot)

    pipeline._inject_interrupted_context = MagicMock()
    pipeline._cancel_soft_interrupt = MagicMock()
    pipeline._duck_unduck_if_suspended = MagicMock()

    pipeline._ensure_runtime_defaults()
    pipeline._is_half_duplex = half_duplex
    # Make the release flush deterministic (no real waiting).
    pipeline._ptt_release_max_wait_sec = 0.3
    pipeline._ptt_release_settle_sec = 0.0
    pipeline._ptt_release_poll_sec = 0.01
    pipeline._timeline = TurnTimeline("ptt-turn")
    pipeline._timeline_debug_flushed = False
    return pipeline, captured


def _state(*, ptt: bool, identity: str = "dev1") -> ClientAudioState:
    return ClientAudioState(
        participant_identity=identity,
        playback_state="idle",
        ptt=ptt,
    )


def _packet(identity: str = "dev1") -> SimpleNamespace:
    return SimpleNamespace(
        topic=CLIENT_AUDIO_STATE_TOPIC,
        participant=SimpleNamespace(identity=identity),
    )


def _ptt(pipeline: Any, *, held: bool) -> None:
    """Drive a PTT edge through the real edge-tracking entry point."""
    pipeline._latest_client_audio_state = MagicMock(return_value=_state(ptt=held))
    pipeline._track_ptt_turn_edges(_packet())


def _user_state(old: str, new: str) -> SimpleNamespace:
    return SimpleNamespace(old_state=old, new_state=new)


def _tx(text: str, *, final: bool) -> SimpleNamespace:
    return SimpleNamespace(transcript=text, is_final=final, speaker_id="dev1")


@pytest.mark.asyncio
async def test_half_duplex_aggregates_split_sentences_and_commits_once_on_release() -> None:
    """THE bug: a mid-sentence pause splits ASR into two finals; an EOT firing
    after release but before the trailing final must NOT commit the first
    sentence alone. On release we wait for the trailing final and commit BOTH."""
    pipeline, captured = _make_pipeline(half_duplex=True)

    # 1. Press → turn boundary armed.
    _ptt(pipeline, held=True)
    assert pipeline._ptt_turn_active is True

    # 2. Speech starts; first sentence finalizes (split at the comma pause).
    pipeline._on_user_state_changed(_user_state("listening", "speaking"))
    pipeline._on_user_transcribed(_tx("你帮我查一下今天的天气。", final=True))

    # 3. Premature framework EOU fires in the pause — must be HELD, not committed.
    allowed = await pipeline._voiceprint_allows_completed_turn(
        new_message=SimpleNamespace(text_content="你帮我查一下今天的天气。")
    )
    assert allowed is False
    pipeline._session.commit_user_turn.assert_not_called()

    # 4. The trailing clause (still being held by PTT) finalizes.
    pipeline._on_user_transcribed(_tx("北", final=False))
    pipeline._on_user_transcribed(_tx("北京的天气。", final=True))

    # 5. Release → the single commit fires, aggregating every segment.
    _ptt(pipeline, held=False)
    assert pipeline._ptt_release_task is not None
    await pipeline._ptt_release_task
    if pipeline._pending_voiceprint_commit_tasks:
        await asyncio.gather(*pipeline._pending_voiceprint_commit_tasks)

    pipeline._session.commit_user_turn.assert_called_once_with(transcript_timeout=5.0)
    committed = pipeline._get_eot_model.return_value.record_turn.call_args.args[0]
    assert "今天的天气" in committed
    assert "北京" in committed  # the city is NOT dropped
    # Canonical text handed to the LLM carries the full utterance.
    assert "今天的天气" in captured["text"]
    assert "北京" in captured["text"]


@pytest.mark.asyncio
async def test_half_duplex_release_with_no_speech_does_not_commit() -> None:
    """Empty/silent hold → graceful no-turn (no ghost commit)."""
    pipeline, _ = _make_pipeline(half_duplex=True)

    _ptt(pipeline, held=True)
    pipeline._on_user_state_changed(_user_state("listening", "speaking"))
    # No ASR at all.
    _ptt(pipeline, held=False)
    await pipeline._ptt_release_task

    pipeline._session.commit_user_turn.assert_not_called()


@pytest.mark.asyncio
async def test_half_duplex_rapid_re_press_cancels_pending_release_commit() -> None:
    """A new press before the release commit lands cancels it (no double-commit)."""
    pipeline, _ = _make_pipeline(half_duplex=True)
    # Stall the flush so the release task is still pending when we re-press.
    pipeline._ptt_release_settle_sec = 5.0
    pipeline._ptt_release_max_wait_sec = 5.0

    _ptt(pipeline, held=True)
    pipeline._on_user_state_changed(_user_state("listening", "speaking"))
    pipeline._on_user_transcribed(_tx("第一句。", final=True))
    _ptt(pipeline, held=False)
    release_task = pipeline._ptt_release_task
    assert release_task is not None

    # Re-press before the (stalled) flush commits.
    _ptt(pipeline, held=True)
    await asyncio.sleep(0)
    assert release_task.cancelled()
    assert pipeline._ptt_turn_active is True
    assert pipeline._ptt_commit_authorized is False
    pipeline._session.commit_user_turn.assert_not_called()


def test_half_duplex_without_ptt_signal_still_commits_on_eot() -> None:
    """Defensive: a half_duplex session that never sees a PTT edge keeps the
    framework's VAD/EOT commit (the 9642a0c no-defer behaviour)."""
    pipeline, _ = _make_pipeline(half_duplex=True)
    assert pipeline._ptt_turn_active is False
    # No PTT observed → EOT/VAD is not held back.
    assert pipeline._ptt_release_drives_commit() is False
    assert pipeline._should_defer_framework_completed_turn("私立医院的。") is False
    assert (
        pipeline._should_defer_low_eot_commit(
            transcript="给医生做的系统。",
            eot_model=pipeline._get_eot_model.return_value,
        )
        is False
    )


@pytest.mark.asyncio
async def test_full_duplex_eot_still_commits_turn() -> None:
    """full_duplex (barge-in) is unchanged: a confident framework EOU commits the
    turn — there is no PTT signal to gate on."""
    pipeline, captured = _make_pipeline(half_duplex=False)
    assert pipeline._ptt_release_drives_commit() is False

    pipeline._user_turns.start_speech(timeline=pipeline._timeline)
    pipeline._user_turns.add_transcript("今天天气怎么样？", is_final=True)

    # A confident EOU (no voiceprint task) drives the commit straight through.
    allowed = await pipeline._voiceprint_allows_completed_turn(
        new_message=SimpleNamespace(text_content="今天天气怎么样？")
    )

    assert allowed is True
    assert captured["text"] == "今天天气怎么样？"
