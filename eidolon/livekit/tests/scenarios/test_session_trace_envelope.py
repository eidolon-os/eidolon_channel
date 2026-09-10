"""The session trace, produced by the real pipeline rather than a stub.

``test_session_trace.py`` proves the writer's own contract. This proves the
wiring: that the events the pipeline already emitted and dropped — phase
transitions, milestones, a turn terminal — reach the file, and that one turn
lands as exactly one settled row.
"""
import json

import pytest

from eidolon.livekit.agent.observability import ChannelEventContext
from eidolon.livekit.agent.observability.session_trace import (
    RECORD_EVENT,
    RECORD_SESSION_CLOSE,
    RECORD_SESSION_MARK,
    RECORD_SESSION_OPEN,
    RECORD_TURN_FINAL,
)
from eidolon.livekit.common.config import ObservabilityConfig

from .._harness.audio import synth_voiced
from .._harness.mocks import MockLLM, MockSTT, MockTTS, MockVAD, MockVADEvent, ScriptedTranscript
from .._harness.production import production_session


def _rows(root):
    files = sorted(root.rglob("*.ndjson"))
    assert len(files) == 1, files
    return [json.loads(line) for line in files[0].open(encoding="utf-8") if line.strip()]


@pytest.mark.asyncio
async def test_a_rejected_turn_still_lands_as_one_settled_row(tmp_path) -> None:
    """A barge-in the policy rejects never reaches the brain — and only the
    Channel has a row for it. That absence elsewhere is why this trace exists."""

    observability = ObservabilityConfig(session_trace_path=str(tmp_path))

    async with production_session(
        welcome="我先说一句欢迎语。",
        llm=MockLLM.scripted([("帮我把灯打开", "好的，这就为你办。")]),
        stt=MockSTT.scripted([ScriptedTranscript(text="帮我把灯打开", trigger_after_ms=700)]),
        tts=MockTTS(char_seconds=0.02, chunk_delay_ms=10),
        vad=MockVAD.scripted([MockVADEvent("start", 350), MockVADEvent("end", 1050, 0.1)]),
        observability=observability,
    ) as (pipeline, handle):
        # ``production_session`` binds the session directly instead of running
        # ``lifecycle.run``, so open the trace the way the lifecycle would and
        # give the sink the identity it resolves from the room there.
        pipeline._ensure_turn_event_sink()._context = ChannelEventContext(
            owner_id="owner-1",
            companion_id="companion-1",
            device_id="device-1",
            room_name="room-1",
            session_flow_id=None,
        )
        pipeline.session_mark("room_joined")
        pipeline.open_session_trace(
            None, owner_id="owner-1", companion_id="companion-1", interaction_mode="full_duplex"
        )
        pipeline.session_mark("session_started")

        handle.audio_in.feed_pcm(synth_voiced(1.3))
        await handle.audio_out.wait_for_first_audio()

    # ``production_session`` closes the pipeline on exit, and BasePipeline.shutdown
    # closes the trace with it.
    rows = _rows(tmp_path)
    kinds = [row["record_kind"] for row in rows]

    assert kinds[0] == RECORD_SESSION_OPEN
    assert kinds[-1] == RECORD_SESSION_CLOSE
    assert kinds.count(RECORD_SESSION_CLOSE) == 1

    # ``first_turn`` is not set by this test: it comes from real speech reaching
    # the speech lifecycle, which is what makes "how long until this session was
    # used" measurable rather than declared.
    marks = [row["mark"] for row in rows if row["record_kind"] == RECORD_SESSION_MARK]
    assert marks == ["room_joined", "session_started", "first_turn"]

    # The events the sink used to drop.
    events = [row for row in rows if row["record_kind"] == RECORD_EVENT]
    types = {row["event_type"] for row in events}
    assert "channel.turn.phase_changed" in types
    assert "channel.session.ended" in types
    # Rejected, not completed: the policy declined this barge-in, so the turn
    # never became an Agent turn and this is the only place it is recorded.
    assert "channel.turn.rejected" in types
    assert all(row.get("channel_turn_id") for row in events if row["event_type"].startswith("channel.turn."))

    # One turn, one settled row — the whole point of the final/progress split.
    finals = [row for row in rows if row["record_kind"] == RECORD_TURN_FINAL]
    assert len(finals) == 1
    settled = finals[0]
    assert settled["final"] is True
    assert settled["turn_id"]

    # Self-describing: the row carries the pair that makes its monotonic marks
    # absolute, and a mark to apply it to.
    anchor = settled["clock_anchor"]
    assert anchor["unix_ns"] > 0
    assert settled["timestamps"].get("speech_started_at") is not None

    closing = rows[-1]
    assert closing["dropped_record_count"] == 0
    assert closing["truncated"] is False
    assert "first_turn" not in closing["missing_session_marks"]


@pytest.mark.asyncio
async def test_a_committed_turn_records_its_milestones(tmp_path) -> None:
    """The main path: speech after the welcome commits and answers.

    Milestones are what give a turn its stage breakdown, so a trace that only
    ever saw rejected turns would be missing the shape of a normal one.
    """

    observability = ObservabilityConfig(session_trace_path=str(tmp_path))

    async with production_session(
        welcome="",
        llm=MockLLM.scripted([("帮我把灯打开", "好的，这就为你办。")]),
        stt=MockSTT.scripted([ScriptedTranscript(text="帮我把灯打开", trigger_after_ms=700)]),
        tts=MockTTS(char_seconds=0.02, chunk_delay_ms=10),
        vad=MockVAD.scripted([MockVADEvent("start", 300), MockVADEvent("end", 900, 0.1)]),
        observability=observability,
    ) as (pipeline, handle):
        pipeline._ensure_turn_event_sink()._context = ChannelEventContext(
            owner_id="owner-1",
            companion_id="companion-1",
            device_id="device-1",
            room_name="room-1",
            session_flow_id=None,
        )
        pipeline.open_session_trace(
            None, owner_id="owner-1", companion_id="companion-1", interaction_mode="full_duplex"
        )
        handle.audio_in.feed_pcm(synth_voiced(1.2))
        await handle.audio_out.wait_for_first_audio()

    rows = _rows(tmp_path)
    events = [row for row in rows if row["record_kind"] == RECORD_EVENT]
    milestones = [row["milestone"] for row in events if row["event_type"] == "channel.turn.milestone"]
    assert milestones, [row["event_type"] for row in events]
    # Reaching audio means the turn was committed and answered.
    assert {"turn_committed", "first_audio"} & set(milestones)

    finals = [row for row in rows if row["record_kind"] == RECORD_TURN_FINAL]
    assert len(finals) == 1
    assert finals[0]["timestamps"].get("turn_committed_at") is not None
