"""attention.speaker in, attention.lock out: bind reports to adopted turns.

The harness mirrors production wiring: the pipeline's user-state handler opens
a new turn per non-merged speech onset and the speech lifecycle tells the
addressee which turn that speech belongs to (as does a transcript that opens a
turn no VAD edge opened); the real ``InteractionPublisher`` numbers facts
packets, with ``user_speaking`` true while either the session VAD
(``user_state``) or the speaker-evidence VAD state (``voice``) hears speech;
the real ``RoomDataHandler`` dispatches inbound packets; agent-state changes
are forwarded with the timeline that owns the current output (None for
welcome lines and proactive reports, and already None once an accepted
interruption has released it).
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from types import SimpleNamespace

import pytest

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.session.gaze_addressee import attach_gaze_addressee
from eidolon.livekit.agent.session.interaction_facts import attach_interaction_publisher
from eidolon.livekit.agent.session.room_data import RoomDataHandler
from eidolon_sdk.biz.presentation import OutputSelection, SessionOutputPlan
from eidolon_sdk.biz.presentation.attention import (
    LOCK_HOLD_LEASE_MS,
    LOCK_RELEASE_LEASE_MS,
    LOCK_TOPIC,
    SPEAKER_TOPIC,
    AttentionLock,
    AttentionSpeaker,
)
from eidolon_sdk.biz.presentation.interaction import TOPIC as FACTS_TOPIC

DEVICE = "reachy-mini-1"
SESSION = "session-1"
LOCK_FIELDS = {
    "version", "session_id", "policy_revision", "sequence", "phase", "turn_id",
    "track_id", "speaker_sequence", "lease_ms",
}


async def settle() -> None:
    for _ in range(8):
        await asyncio.sleep(0)


class FakeSession:
    def __init__(self) -> None:
        self.handlers: dict[str, list] = defaultdict(list)

    def on(self, name, handler) -> None:
        self.handlers[name].append(handler)

    def off(self, name, handler) -> None:
        self.handlers[name].remove(handler)

    def emit(self, name, **fields) -> None:
        for handler in list(self.handlers[name]):
            handler(SimpleNamespace(**fields))


class FakeRoom:
    def __init__(self) -> None:
        self.sent: list[tuple[str, dict, list[str] | None]] = []
        self.local_participant = SimpleNamespace(publish_data=self.publish_data)

    async def publish_data(self, data, *, topic, reliable=True, destination_identities=None):
        assert reliable
        self.sent.append((topic, json.loads(data), destination_identities))


class FakeVoice:
    """ConversationSpeaker.add_voice_listener: the speaker-evidence VAD state."""

    def __init__(self) -> None:
        self.listeners: list = []

    def add_voice_listener(self, listener):
        self.listeners.append(listener)
        return lambda: self.listeners.remove(listener)

    def set(self, speaking: bool) -> None:
        for listener in list(self.listeners):
            listener(speaking)


class Host:
    def __init__(self, *, motion: bool = True, facts_first: bool = False) -> None:
        self.session = FakeSession()
        self.room = FakeRoom()
        self.voice = FakeVoice()
        self.gaze = None
        self.timeline: TurnTimeline | None = None
        self.output: TurnTimeline | None = None
        self.merge_next = False
        self._turns = 0
        plan = SessionOutputPlan(session_id=SESSION, policy_revision=1,
                                 outputs=OutputSelection(speech=True, motion=motion))
        if facts_first:
            # Registration order must not matter: facts are built in their own
            # task, after every synchronous handler of the same event ran.
            self.facts = attach_interaction_publisher(
                self.session, self.room, plan, voice=self.voice)
            self.session.on("user_state_changed", self._open_turn)
        else:
            # Production: bind_session registers the pipeline handler first.
            self.session.on("user_state_changed", self._open_turn)
            self.facts = attach_interaction_publisher(
                self.session, self.room, plan, voice=self.voice)
        self.session.on("agent_state_changed", self._forward_agent_state)
        self.room_data = RoomDataHandler(get_timeline=lambda: self.timeline)
        self.gaze = attach_gaze_addressee(
            self.room, self.facts, room_data=self.room_data, participant_identity=DEVICE)
        self.agent_state = "listening"

    def _open_turn(self, event) -> None:
        if event.new_state != "speaking":
            return
        if self.merge_next and self.timeline is not None:
            self.merge_next = False
        else:
            self._new_turn()
        self._speech_started()

    def _new_turn(self) -> None:
        self._turns += 1
        self.timeline = TurnTimeline(f"{self._turns:016x}")

    def _speech_started(self) -> None:
        """speech_lifecycle._start_candidate."""
        if self.gaze is not None:
            self.gaze.on_speech_started(self.timeline)

    def _forward_agent_state(self, event) -> None:
        if self.gaze is not None:
            self.gaze.on_agent_state(event.old_state, event.new_state, self.output)

    async def user_speaks(self) -> tuple[TurnTimeline, int]:
        self.session.emit("user_state_changed", old_state="listening", new_state="speaking")
        await settle()
        return self.timeline, self.facts_sent()[-1]["sequence"]

    async def user_stops(self) -> None:
        self.session.emit("user_state_changed", old_state="speaking", new_state="listening")
        await settle()

    async def user_away(self) -> None:
        """LiveKit's user_away_timeout: 15 s with neither side speaking."""
        self.session.emit("user_state_changed", old_state="listening", new_state="away")
        await settle()

    async def evidence(self, speaking: bool) -> int:
        """The speaker-evidence VAD state starts or stops hearing speech."""
        self.voice.set(speaking)
        await settle()
        return self.facts_sent()[-1]["sequence"]

    async def transcript_opens_turn(self) -> TurnTimeline:
        """A transcript no session-VAD edge preceded opens a turn
        (speech_lifecycle.handle_transcript_started)."""
        self._new_turn()
        self._speech_started()
        await settle()
        return self.timeline

    def user_speaking(self) -> list[bool]:
        return [body["user_speaking"] for body in self.facts_sent()]

    def adopt(self, timeline: TurnTimeline) -> None:
        """framework_completed_turn claims the output for the committed turn."""
        self.output = timeline

    async def agent(self, state: str) -> None:
        old, self.agent_state = self.agent_state, state
        self.session.emit("agent_state_changed", old_state=old, new_state=state)
        if state in {"listening", "idle"}:
            self.output = None  # playback done flushes the output timeline
        await settle()

    async def interrupt(self) -> None:
        """Accepted interruption: the output is released before the framework
        moves the agent out of speaking (interruption_effects.cancel_and_interrupt)."""
        self.output = None
        await self.agent("listening")

    def report(self, sequence: int, facts_sequence: int, track: str | None, *,
               basis: str = "face", final: bool = True, sender: str = DEVICE,
               session_id: str = SESSION, raw: bytes | None = None) -> None:
        data = raw if raw is not None else AttentionSpeaker(
            session_id=session_id, policy_revision=1, sequence=sequence,
            facts_sequence=facts_sequence, final=final, track_id=track,
            basis=basis if track else "none", confidence=0.8, in_view=basis == "face",
        ).model_dump_json().encode()
        self.room_data.handle_packet(SimpleNamespace(
            topic=SPEAKER_TOPIC, data=data, participant=SimpleNamespace(identity=sender)))

    def facts_sent(self) -> list[dict]:
        return [body for topic, body, _ in self.room.sent if topic == FACTS_TOPIC]

    def locks(self) -> list[AttentionLock]:
        return [AttentionLock.model_validate(body)
                for topic, body, _ in self.room.sent if topic == LOCK_TOPIC]

    def new_topics(self) -> set[str]:
        return {topic for topic, _, _ in self.room.sent} - {FACTS_TOPIC}

    async def aclose(self) -> None:
        if self.gaze is not None:
            await self.gaze.aclose()
        await self.facts.aclose()


@pytest.fixture
async def host():
    host = Host()
    await settle()
    yield host
    await host.aclose()


async def ask(host: Host, track: str | None, sequence: int) -> tuple[TurnTimeline, int]:
    turn, opened = await host.user_speaks()
    host.report(sequence, opened, track, final=False)
    await host.user_stops()
    host.report(sequence + 1, opened, track, final=True)
    return turn, opened


@pytest.mark.asyncio
async def test_one_question_one_answer_holds_on_the_speaker_then_releases(host):
    turn, _ = await ask(host, "k3", 7)
    host.adopt(turn)
    await host.agent("thinking")
    await host.agent("speaking")
    await host.agent("listening")

    hold, release = host.locks()
    assert (hold.phase, hold.turn_id, hold.track_id, hold.speaker_sequence, hold.lease_ms) == (
        "hold", turn.turn_id, "k3", 8, LOCK_HOLD_LEASE_MS)
    assert (release.phase, release.turn_id, release.track_id, release.lease_ms) == (
        "release", turn.turn_id, "k3", LOCK_RELEASE_LEASE_MS)
    assert hold.sequence < release.sequence
    assert all(dest == [DEVICE] for topic, _, dest in host.room.sent if topic == LOCK_TOPIC)
    assert turn.attrs["gaze_addressee"] == {
        "track_id": "k3", "basis": "face", "speaker_sequence": 8, "facts_sequence": 2,
        "final": True}


@pytest.mark.asyncio
async def test_b_speaking_after_a_does_not_take_the_lock_when_a_was_adopted(host):
    a_turn, a_opened = await host.user_speaks()
    host.report(1, a_opened, "kA", final=False)
    await host.user_stops()
    host.adopt(a_turn)
    await host.agent("thinking")
    # B coughs before the reply: a separate candidate turn that is never adopted.
    b_turn, b_opened = await host.user_speaks()
    assert b_turn is not a_turn and b_opened > a_opened
    host.report(2, b_opened, "kB", basis="voice", final=False)
    await host.user_stops()
    # A's final arrives late (array converged slowly) but still before the reply.
    host.report(3, a_opened, "kA", final=True)
    await host.agent("speaking")

    (hold,) = host.locks()
    assert (hold.turn_id, hold.track_id, hold.speaker_sequence) == (a_turn.turn_id, "kA", 3)


@pytest.mark.asyncio
async def test_a_merged_continuation_belongs_to_the_same_turn(host):
    turn, first = await host.user_speaks()
    host.report(1, first, "kA")
    await host.user_stops()
    host.merge_next = True
    merged, second = await host.user_speaks()
    assert merged is turn and second > first
    host.report(2, second, "kB", basis="voice")
    await host.user_stops()
    host.adopt(turn)
    await host.agent("speaking")
    assert [(lock.turn_id, lock.track_id) for lock in host.locks()] == [(turn.turn_id, "kB")]


@pytest.mark.asyncio
async def test_a_rejected_turn_produces_no_lock(host):
    turn, _ = await ask(host, "k1", 1)
    # Rejected: never claims the output. Whatever speaks next is not its reply.
    await host.agent("speaking")
    await host.agent("listening")
    assert host.locks() == []
    assert host.gaze.enabled  # reports were accepted; still nothing to lock
    assert "gaze_addressee" not in turn.attrs


@pytest.mark.asyncio
async def test_welcome_and_proactive_speech_produce_no_lock(host):
    await host.agent("speaking")  # welcome: before any report
    await host.agent("listening")
    turn, _ = await ask(host, "k1", 1)
    host.adopt(turn)
    await host.agent("speaking")
    await host.agent("listening")
    before = len(host.locks())
    await host.agent("speaking")  # proactive report: no adopted turn owns it
    await host.agent("listening")
    assert before == 2 and len(host.locks()) == 2


@pytest.mark.asyncio
async def test_one_reply_spoken_in_two_parts_holds_twice_and_releases_once(host):
    turn, _ = await ask(host, "k2", 1)
    host.adopt(turn)
    await host.agent("thinking")
    await host.agent("speaking")
    await host.agent("thinking")  # tool call between the two spoken parts
    await host.agent("speaking")
    await host.agent("listening")
    assert [(lock.phase, lock.turn_id, lock.track_id) for lock in host.locks()] == [
        ("hold", turn.turn_id, "k2"), ("hold", turn.turn_id, "k2"),
        ("release", turn.turn_id, "k2")]
    assert [lock.sequence for lock in host.locks()] == [1, 2, 3]


@pytest.mark.asyncio
async def test_a_reply_that_ends_in_a_tool_call_still_releases(host):
    turn, _ = await ask(host, "k2", 1)
    host.adopt(turn)
    await host.agent("speaking")
    await host.agent("thinking")
    await host.agent("listening")
    assert [lock.phase for lock in host.locks()] == ["hold", "release"]


@pytest.mark.asyncio
async def test_an_accepted_interruption_releases_at_once(host):
    turn, _ = await ask(host, "k1", 1)
    host.adopt(turn)
    await host.agent("speaking")
    # B barges in; while ducked the agent stays "speaking", so nothing is sent.
    b_turn, b_opened = await host.user_speaks()
    host.report(3, b_opened, "kB", basis="voice")
    assert [lock.phase for lock in host.locks()] == ["hold"]
    await host.interrupt()
    assert [(lock.phase, lock.turn_id, lock.track_id) for lock in host.locks()] == [
        ("hold", turn.turn_id, "k1"), ("release", turn.turn_id, "k1")]
    # The interrupter's own turn then gets its reply, and the lock follows it.
    await host.user_stops()
    host.adopt(b_turn)
    await host.agent("thinking")
    await host.agent("speaking")
    assert (host.locks()[-1].phase, host.locks()[-1].turn_id, host.locks()[-1].track_id) == (
        "hold", b_turn.turn_id, "kB")


@pytest.mark.asyncio
async def test_late_reports_are_dropped_by_facts_sequence(host):
    first, first_opened = await ask(host, "k1", 1)
    host.adopt(first)
    await host.agent("speaking")
    await host.agent("listening")
    second, second_opened = await host.user_speaks()
    host.report(3, second_opened, "k2", final=False)
    # A report about the first segment, after its reply already started.
    host.report(4, first_opened, "k9")
    # A report older than any segment this session has seen.
    host.report(5, 1, "k8")
    await host.user_stops()
    host.adopt(second)
    await host.agent("speaking")
    hold = host.locks()[-1]
    assert (hold.turn_id, hold.track_id, hold.speaker_sequence) == (second.turn_id, "k2", 3)
    assert first.attrs["gaze_addressee"]["track_id"] == "k1"


@pytest.mark.asyncio
async def test_a_report_after_the_reply_started_does_not_move_the_lock(host):
    turn, opened = await host.user_speaks()
    host.report(1, opened, "k1", final=False)
    await host.user_stops()
    host.adopt(turn)
    await host.agent("speaking")
    host.report(2, opened, "k7")  # final, but the binding window has closed
    await host.agent("thinking")
    await host.agent("speaking")
    assert {lock.track_id for lock in host.locks()} == {"k1"}


@pytest.mark.asyncio
async def test_an_adopted_turn_without_a_report_holds_with_no_track(host):
    turn, _ = await ask(host, "k1", 1)
    host.adopt(turn)
    await host.agent("speaking")
    await host.agent("listening")
    silent, _ = await host.user_speaks()  # this segment never gets a report
    await host.user_stops()
    host.adopt(silent)
    await host.agent("speaking")
    hold = host.locks()[-1]
    assert (hold.phase, hold.turn_id, hold.track_id, hold.speaker_sequence) == (
        "hold", silent.turn_id, None, None)


@pytest.mark.asyncio
async def test_a_session_without_speaker_reports_sends_nothing_new(host):
    turn, _ = await host.user_speaks()
    await host.user_stops()
    host.adopt(turn)
    await host.agent("thinking")
    await host.agent("speaking")
    await host.interrupt()
    await host.agent("speaking")
    await host.agent("listening")
    assert host.new_topics() == set()
    assert not host.gaze.enabled


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [
    {"sender": "someone-else"},
    {"sender": ""},
    {"session_id": "other-session"},
    {"raw": b"{not json"},
    {"raw": json.dumps({"version": 1, "session_id": SESSION, "policy_revision": 1,
                        "sequence": 1, "facts_sequence": 2, "final": True, "track_id": "k1",
                        "basis": "face", "confidence": 0.9, "in_view": True,
                        "yaw_rad": 0.5}).encode()},
])
async def test_reports_from_elsewhere_or_malformed_do_not_enable_the_lock(host, bad):
    turn, opened = await host.user_speaks()
    host.report(1, opened, "k1", **bad)
    await host.user_stops()
    host.adopt(turn)
    await host.agent("speaking")
    assert host.locks() == [] and not host.gaze.enabled


@pytest.mark.asyncio
async def test_a_report_sequence_that_does_not_advance_is_ignored(host):
    turn, opened = await host.user_speaks()
    host.report(5, opened, "k1")
    host.report(5, opened, "k2")
    host.report(4, opened, "k3")
    await host.user_stops()
    host.adopt(turn)
    await host.agent("speaking")
    assert host.locks()[-1].track_id == "k1"


@pytest.mark.asyncio
async def test_lock_json_carries_exactly_the_field_table_and_no_geometry(host):
    turn, _ = await ask(host, "k3", 1)
    host.adopt(turn)
    await host.agent("speaking")
    await host.interrupt()
    bodies = [body for topic, body, _ in host.room.sent if topic == LOCK_TOPIC]
    assert len(bodies) == 2
    for body in bodies:
        assert set(body) == LOCK_FIELDS
        assert body["session_id"] == SESSION and body["policy_revision"] == 1


@pytest.mark.asyncio
async def test_turn_onset_is_mapped_regardless_of_handler_registration_order():
    host = Host(facts_first=True)
    await settle()
    try:
        turn, opened = await ask(host, "k5", 1)
        host.adopt(turn)
        await host.agent("speaking")
        assert [(lock.turn_id, lock.track_id) for lock in host.locks()] == [(turn.turn_id, "k5")]
        onset = next(body for body in host.facts_sent() if body["sequence"] == opened)
        assert onset["user_speaking"] and not host.facts_sent()[0]["user_speaking"]
    finally:
        await host.aclose()


@pytest.mark.asyncio
async def test_a_question_only_the_evidence_vad_heard_still_gets_its_speaker_locked(host):
    """2026-11-12 11:45:51-11:46:14: 15 s of silence put the user away; the
    session VAD then missed a quiet question that the speaker-evidence VAD and
    STT heard, and the first transcript opened the turn. user_speaking used to
    follow the session VAD only, so the device opened no segment, sent no
    report, and the hold went out with track=None."""
    first, _ = await ask(host, "k1", 1)
    host.adopt(first)
    await host.agent("speaking")
    await host.agent("listening")
    await host.user_away()
    opened = await host.evidence(True)
    assert host.user_speaking()[-2:] == [False, True]
    turn = await host.transcript_opens_turn()  # the first interim, ~0.1 s later
    host.report(3, opened, "k1", basis="voice", final=False)
    await host.evidence(False)
    assert host.user_speaking()[-1] is False
    host.report(4, opened, "k1", final=True)
    host.adopt(turn)
    await host.agent("thinking")
    await host.agent("speaking")
    hold = host.locks()[-1]
    assert (hold.phase, hold.turn_id, hold.track_id, hold.speaker_sequence) == (
        "hold", turn.turn_id, "k1", 4)


@pytest.mark.asyncio
async def test_the_evidence_vad_starting_first_still_binds_to_the_turn_the_session_vad_opens(host):
    """Today's log: the evidence VAD state started up to 1.4 s before the
    session VAD six times out of 22. Its rising edge then precedes the new
    turn, so the onset can no longer stand for "the turn current right now"
    (that was the previous, already answered one)."""
    first, _ = await ask(host, "k1", 1)
    host.adopt(first)
    await host.agent("speaking")
    await host.agent("listening")
    before = len(host.facts_sent())
    opened = await host.evidence(True)
    host.report(3, opened, "k2", basis="voice", final=False)  # before any turn exists
    turn, _ = await host.user_speaks()
    assert turn is not first
    await host.user_stops()  # the session VAD ends first, the evidence one later
    await host.evidence(False)
    # One segment for the device: the session VAD's edges inside it change nothing.
    assert host.user_speaking()[before:] == [True, True, True, False]
    host.report(4, opened, "k2", final=True)
    host.adopt(turn)
    await host.agent("speaking")
    hold = host.locks()[-1]
    assert (hold.turn_id, hold.track_id, hold.speaker_sequence) == (turn.turn_id, "k2", 4)
    assert first.attrs["gaze_addressee"]["track_id"] == "k1"


@pytest.mark.asyncio
async def test_a_sound_no_turn_was_made_of_does_not_take_the_lock(host):
    turn, opened = await host.user_speaks()
    host.report(1, opened, "kA", final=False)
    await host.user_stops()
    host.report(2, opened, "kA", final=True)
    host.adopt(turn)
    await host.agent("thinking")
    # B coughs before the reply: only the evidence VAD hears it, STT makes
    # nothing of it, so no turn claims the segment.
    cough = await host.evidence(True)
    host.report(3, cough, "kB", basis="voice", final=False)
    await host.evidence(False)
    host.report(4, cough, "kB", basis="voice", final=True)
    await host.agent("speaking")
    (hold,) = host.locks()
    assert (hold.turn_id, hold.track_id, hold.speaker_sequence) == (turn.turn_id, "kA", 2)


@pytest.mark.asyncio
async def test_a_turn_no_vad_heard_lends_its_claim_to_no_later_segment(host):
    first, _ = await ask(host, "k1", 1)
    host.adopt(first)
    await host.agent("speaking")
    await host.agent("listening")
    turn = await host.transcript_opens_turn()  # neither VAD heard it: no segment
    await host.agent("thinking")  # the next packet opens no segment; the claim lapses
    cough = await host.evidence(True)
    host.report(3, cough, "kB", basis="voice")
    await host.evidence(False)
    host.adopt(turn)
    await host.agent("speaking")
    hold = host.locks()[-1]
    assert (hold.turn_id, hold.track_id, hold.speaker_sequence) == (turn.turn_id, None, None)


@pytest.mark.asyncio
async def test_the_next_turn_of_one_long_segment_takes_the_segments_report(host):
    opened = await host.evidence(True)
    first, _ = await host.user_speaks()
    host.report(1, opened, "k1", basis="voice", final=False)
    await host.user_stops()
    host.adopt(first)
    await host.agent("speaking")  # the reply starts; the evidence VAD still hears speech
    second, _ = await host.user_speaks()
    assert second is not first
    await host.interrupt()
    await host.user_stops()
    host.report(2, opened, "k2", basis="voice", final=False)
    await host.evidence(False)
    host.adopt(second)
    await host.agent("speaking")
    assert [(lock.phase, lock.turn_id, lock.track_id) for lock in host.locks()] == [
        ("hold", first.turn_id, "k1"), ("release", first.turn_id, "k1"),
        ("hold", second.turn_id, "k2")]


@pytest.mark.asyncio
async def test_failed_delivery_is_not_retried_and_later_locks_still_go_out(host):
    calls = []

    async def flaky(data, *, topic, reliable=True, destination_identities=None):
        calls.append(topic)
        if topic == LOCK_TOPIC and calls.count(LOCK_TOPIC) == 1:
            raise ConnectionError("room gone")
        host.room.sent.append((topic, json.loads(data), destination_identities))

    host.room.local_participant.publish_data = flaky
    turn, _ = await ask(host, "k1", 1)
    host.adopt(turn)
    await host.agent("speaking")
    await host.agent("listening")
    assert [lock.phase for lock in host.locks()] == ["release"]
    assert calls.count(LOCK_TOPIC) == 2


@pytest.mark.asyncio
async def test_no_motion_plan_attaches_nothing():
    host = Host(motion=False)
    assert host.facts is None and host.gaze is None
    assert attach_gaze_addressee(host.room, None, room_data=host.room_data) is None


@pytest.mark.asyncio
async def test_close_detaches_from_the_publisher_and_the_room_data(host):
    gaze, facts = host.gaze, host.facts
    assert gaze.on_facts in facts.observers
    await gaze.aclose()
    assert gaze.on_facts not in facts.observers
    assert gaze._task.done()
    turn, opened = await host.user_speaks()
    host.report(1, opened, "k1")
    assert not gaze.enabled
    host.gaze = None


def test_room_data_subscribers_see_their_topic_only_and_cannot_break_it():
    handler = RoomDataHandler(get_timeline=lambda: None)
    seen = []

    def boom(packet):
        raise RuntimeError("subscriber bug")

    unsubscribe = handler.subscribe("eidolon.test.v1", seen.append)
    handler.subscribe("eidolon.test.v1", boom)
    packet = SimpleNamespace(topic="eidolon.test.v1", data=b"{}",
                             participant=SimpleNamespace(identity=DEVICE))
    handler.handle_packet(packet)
    handler.handle_packet(SimpleNamespace(topic="eidolon.other.v1", data=b"{}", participant=None))
    unsubscribe()
    handler.handle_packet(packet)
    assert seen == [packet]
    assert handler.room_data_packet_count == 3


def _bare_pipeline():
    from eidolon.livekit.agent.full_duplex import StreamingPipeline
    from eidolon.livekit.agent.output.ducking import OutputDuckingController
    from eidolon.livekit.agent.shared.types import PipelineState

    pipeline = StreamingPipeline.__new__(StreamingPipeline)
    pipeline._ducking = OutputDuckingController()
    pipeline._filler = None
    pipeline._state = PipelineState.IDLE
    return pipeline


def test_pipeline_forwards_the_output_timeline_and_an_interruption_has_already_released_it():
    pipeline = _bare_pipeline()
    calls = []
    pipeline._gaze_addressee = SimpleNamespace(
        on_agent_state=lambda old, new, timeline: calls.append((old, new, timeline)))
    turn = TurnTimeline("00000000000000aa")
    pipeline._claim_agent_output_timeline(turn)  # framework_completed_turn commit
    pipeline._on_agent_state_changed(SimpleNamespace(old_state="thinking", new_state="speaking"))
    # interruption_effects.cancel_and_interrupt: output released before the
    # framework ends the speech handle and leaves "speaking".
    pipeline._finish_agent_output("interrupted_by_user")
    pipeline._on_agent_state_changed(SimpleNamespace(old_state="speaking", new_state="listening"))
    pipeline._on_agent_state_changed(SimpleNamespace(old_state="listening", new_state="speaking"))
    assert calls == [("thinking", "speaking", turn), ("speaking", "listening", None),
                     ("listening", "speaking", None)]


@pytest.mark.asyncio
async def test_pipeline_wiring_holds_then_releases_on_an_interruption(host):
    pipeline = _bare_pipeline()
    pipeline._gaze_addressee = host.gaze
    # The pipeline forwards agent state instead of the harness.
    host.session.handlers["agent_state_changed"].remove(host._forward_agent_state)
    turn, _ = await ask(host, "k4", 1)
    pipeline._claim_agent_output_timeline(turn)
    pipeline._on_agent_state_changed(SimpleNamespace(old_state="thinking", new_state="speaking"))
    pipeline._finish_agent_output("interrupted_by_user")
    pipeline._on_agent_state_changed(SimpleNamespace(old_state="speaking", new_state="listening"))
    await settle()
    assert [(lock.phase, lock.turn_id, lock.track_id) for lock in host.locks()] == [
        ("hold", turn.turn_id, "k4"), ("release", turn.turn_id, "k4")]


def test_pipeline_without_the_module_is_unchanged():
    pipeline = _bare_pipeline()
    pipeline._claim_agent_output_timeline(TurnTimeline("00000000000000bb"))
    pipeline._on_agent_state_changed(SimpleNamespace(old_state="thinking", new_state="speaking"))
    pipeline._on_agent_state_changed(SimpleNamespace(old_state="speaking", new_state="listening"))
    assert getattr(pipeline, "_gaze_addressee", None) is None


@pytest.mark.asyncio
async def test_session_wiring_feeds_facts_from_the_evidence_vad_and_reports_to_the_addressee():
    from eidolon.livekit.agent.full_duplex.lifecycle import FullDuplexSessionLifecycle

    voice, session, room, output_room = FakeVoice(), FakeSession(), FakeRoom(), FakeRoom()
    plan = SessionOutputPlan(session_id=SESSION, policy_revision=1,
                             outputs=OutputSelection(speech=True, motion=True))
    pipeline = SimpleNamespace(
        _factory=SimpleNamespace(output_plan=plan), _speaker_observation=voice,
        _room_data=RoomDataHandler(get_timeline=lambda: None))
    FullDuplexSessionLifecycle(pipeline).attach_motion_observers(
        session, room, output_room, DEVICE)
    facts, gaze = pipeline._interaction_publisher, pipeline._gaze_addressee
    try:
        await settle()
        voice.set(True)
        await settle()
        # Facts go to the output room; the lock goes back to the input room.
        assert output_room.sent[-1][1]["user_speaking"] is True and not room.sent
        assert gaze.on_facts in facts.observers
        gaze.on_speech_started(TurnTimeline("00000000000000ab"))
        pipeline._room_data.handle_packet(SimpleNamespace(
            topic=SPEAKER_TOPIC, participant=SimpleNamespace(identity=DEVICE),
            data=AttentionSpeaker(
                session_id=SESSION, policy_revision=1, sequence=1,
                facts_sequence=output_room.sent[-1][1]["sequence"], final=False,
                track_id="k1", basis="voice", confidence=0.6, in_view=False,
            ).model_dump_json().encode()))
        gaze.on_agent_state("thinking", "speaking", TurnTimeline("00000000000000ab"))
        await settle()
        ((topic, body, destination),) = room.sent
        assert (topic, body["track_id"], destination) == (LOCK_TOPIC, "k1", [DEVICE])
    finally:
        await gaze.aclose()
        await facts.aclose()
    assert voice.listeners == []
