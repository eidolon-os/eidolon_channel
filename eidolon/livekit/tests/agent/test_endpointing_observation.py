"""Characterize the SDK admission race and verify observation is non-invasive."""
import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pytest
from livekit.agents.llm import ChatContext
from livekit.agents.voice.agent_activity import AgentActivity

from eidolon.livekit.agent.full_duplex.pipeline import StreamingPipeline
from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.session.eot_model import InterruptAwareTurnDetector
from eidolon.livekit.agent.session.interruption_orchestrator import InterruptionState
from scripts.report_session_trace import _render_interruption_diagnostics


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_before_eot,allow", [(False, False), (True, False), (False, True)])
async def test_real_sdk_admission_order(cancel_before_eot, allow):
    """The real SDK gate runs before the application hook; late cancel cannot replay it.

    This is a characterization, not an assertion that losing the reply is right.
    Private SDK access is confined to this test; runtime tracing uses public APIs.
    """
    hook = AsyncMock()
    speech = NS(allow_interruptions=allow, interrupt=AsyncMock())
    activity = NS(
        _interrupt_background_speeches=lambda **_: [],
        _init_metrics_from_end_of_turn=lambda _: {}, llm=None,
        _current_speech=None if cancel_before_eot else speech,
        _agent=NS(chat_ctx=ChatContext.empty(), on_user_turn_completed=hook),
        _rt_session=None, _scheduling_paused=False, _new_turns_blocked=False,
        _cancel_speech_pause=AsyncMock(), _cancel_speech_pause_task=None,
    )
    info = NS(new_transcript="这个故事不好听。", transcript_confidence=1.0, skip_reply=False)
    await AgentActivity._user_turn_completed_task(activity, None, info)
    activity._current_speech = None  # A later accepted cancellation.
    await asyncio.sleep(0)
    assert hook.await_count == int(cancel_before_eot or allow)


def owner():
    return NS(
        _timeline=TurnTimeline("turn-original"),
        _session=NS(current_speech=NS(id="speech-old", allow_interruptions=False,
                                     interrupted=False, done=lambda: False)),
        _user_turns=NS(active=NS(candidate_id="original"), current_generation_id=1),
        _interruption_orchestrator=NS(state=InterruptionState.SUSPENDED_POST_SPEECH_WAIT, active=True),
        _session_trace=Mock(),
    )


@pytest.mark.asyncio
async def test_observation_keeps_original_candidate_across_await_and_keeps_score():
    pipeline = owner()
    original = pipeline._timeline

    async def predict(*a, **kw):
        pipeline._timeline = TurnTimeline("turn-next")
        pipeline._user_turns.active = NS(candidate_id="next")
        pipeline._user_turns.current_generation_id = 2
        return .687

    wrapper = InterruptAwareTurnDetector(
        NS(predict_end_of_turn=predict), AsyncMock(),
        begin_observation=lambda: StreamingPipeline._begin_endpointing_observation(pipeline),
    )
    assert await wrapper.predict_end_of_turn(object()) == .687
    row = original.attrs["endpointing_observations"][0]
    assert row["before"]["candidate_id"] == "original"
    assert row["after"]["candidate_id"] == "next"
    assert row["after"]["speech_allow_interruptions"] is False
    assert "endpointing_observations" not in pipeline._timeline.attrs
    assert pipeline._session_trace.event.call_args.args[1]["channel_turn_id"] == "turn-original"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [asyncio.CancelledError, ValueError])
async def test_observation_preserves_cancellation_and_errors(failure):
    finish = Mock()
    wrapper = InterruptAwareTurnDetector(
        NS(predict_end_of_turn=AsyncMock(side_effect=failure)), AsyncMock(),
        begin_observation=lambda: finish,
    )
    with pytest.raises(failure):
        await wrapper.predict_end_of_turn(object())
    finish.assert_called_once_with("cancelled" if failure is asyncio.CancelledError else "error", None)


@pytest.mark.asyncio
@pytest.mark.parametrize("break_at", ["begin", "finish"])
async def test_broken_observer_cannot_change_eot(break_at):
    def fail(*a):
        raise RuntimeError("observer unavailable")
    wrapper = InterruptAwareTurnDetector(
        NS(predict_end_of_turn=AsyncMock(return_value=.687)), AsyncMock(),
        begin_observation=fail if break_at == "begin" else lambda: fail,
    )
    assert await wrapper.predict_end_of_turn(object()) == .687


def test_history_is_bounded_and_absence_is_not_reported_as_proven_sdk_failure():
    pipeline = owner()
    for _ in range(30):
        StreamingPipeline._begin_endpointing_observation(pipeline)("returned", .687)
    assert len(pipeline._timeline.attrs["endpointing_observations"]) == 16
    assert pipeline._timeline.attrs["endpointing_observation_count"] == 30
    turn = pipeline._timeline.snapshot()
    turn["attrs"]["interruption_verdict"] = {"action": "confirmed_cancel", "continue_to_llm": True}
    report = "\n".join(_render_interruption_diagnostics(turn))
    assert "GAP:" in report and "no framework completion observed" in report
    assert "not proof" in report
    turn["timestamps"]["framework_completed_turn_at"] = 1.0
    assert "inspect product admission" in "\n".join(_render_interruption_diagnostics(turn))
    turn["timestamps"]["turn_committed_at"] = 2.0
    assert "GAP:" not in "\n".join(_render_interruption_diagnostics(turn))
    del turn["timestamps"]["turn_committed_at"]
    turn["attrs"]["interruption_verdict"]["continue_to_llm"] = False
    assert "GAP:" not in "\n".join(_render_interruption_diagnostics(turn))


def test_report_includes_endpointing_that_finished_after_terminal_snapshot_once():
    event = {"event_type": "channel.endpointing.observed", "started_at": 1.0,
             "finished_at": 2.0, "outcome": "returned", "score": .687,
             "before": {}, "after": {}}
    turn = {"attrs": {}, "timestamps": {}}
    assert "EOT returned: score=0.687" in "\n".join(_render_interruption_diagnostics(turn, [event]))
    turn["attrs"]["endpointing_observations"] = [event]
    assert "\n".join(_render_interruption_diagnostics(turn, [event])).count("EOT returned") == 1
