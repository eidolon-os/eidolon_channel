import asyncio
from types import SimpleNamespace

import pytest

from eidolon.livekit.agent.session.interaction_facts import attach_interaction_publisher
from eidolon_sdk.biz.presentation import OutputSelection, SessionOutputPlan
from eidolon_sdk.biz.presentation.interaction import InteractionFacts


@pytest.mark.asyncio
async def test_motion_observer_keeps_duplex_facts_independent_and_closes():
    handlers, packets = {}, []
    session = SimpleNamespace(on=lambda name, fn: handlers.update({name: fn}),
                              off=lambda name, fn: handlers.pop(name))
    async def publish(data, **kwargs):
        packets.append(InteractionFacts.model_validate_json(data))
    room = SimpleNamespace(local_participant=SimpleNamespace(publish_data=publish))
    plan = SessionOutputPlan(session_id="session", policy_revision=1,
        outputs=OutputSelection(motion=True))
    observer = attach_interaction_publisher(session, room, plan)
    handlers["user_state_changed"](SimpleNamespace(new_state="speaking"))
    handlers["agent_state_changed"](SimpleNamespace(new_state="speaking"))
    await asyncio.sleep(0)
    assert packets[-1].user_speaking and packets[-1].agent_speaking
    await observer.aclose()
    assert observer.task.done() and not handlers


def test_audio_only_and_text_agents_have_no_new_observer():
    plan = SessionOutputPlan(session_id="session", policy_revision=1,
        outputs=OutputSelection(speech=True))
    assert attach_interaction_publisher(None, None, plan) is None
    assert attach_interaction_publisher(None, None, None) is None


class FakeVoice:
    """ConversationSpeaker.add_voice_listener: the speaker-evidence VAD state."""

    def __init__(self):
        self.listeners = []

    def add_voice_listener(self, listener):
        self.listeners.append(listener)
        return lambda: self.listeners.remove(listener)

    def set(self, speaking):
        for listener in list(self.listeners):
            listener(speaking)


async def _settle():
    for _ in range(4):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_user_speaking_is_either_vad_state_and_the_other_ones_edges_add_no_packets():
    handlers, packets = {}, []
    session = SimpleNamespace(on=lambda name, fn: handlers.update({name: fn}),
                              off=lambda name, fn: handlers.pop(name))
    async def publish(data, **kwargs):
        packets.append(InteractionFacts.model_validate_json(data))
    room = SimpleNamespace(local_participant=SimpleNamespace(publish_data=publish))
    plan = SessionOutputPlan(session_id="session", policy_revision=1,
        outputs=OutputSelection(motion=True))
    voice = FakeVoice()
    observer = attach_interaction_publisher(session, room, plan, voice=voice)
    await _settle()

    def user(state):
        handlers["user_state_changed"](SimpleNamespace(new_state=state))

    steps = [
        # 2026-11-12: away after 15 s, then only the evidence VAD hears a question.
        (lambda: user("away"), False),
        (lambda: voice.set(True), True),
        (lambda: voice.set(False), False),
        # The usual utterance: both hear it, starts and ends a little apart.
        (lambda: voice.set(True), True),
        (lambda: user("speaking"), True),
        (lambda: user("listening"), True),
        (lambda: voice.set(False), False),
        # Only the session VAD hears one: unchanged from before.
        (lambda: user("speaking"), True),
        (lambda: voice.set(False), None),
        (lambda: user("listening"), False),
    ]
    for step, expected in steps:
        before = len(packets)
        step()
        await _settle()
        if expected is None:
            assert len(packets) == before
        else:
            assert len(packets) == before + 1 and packets[-1].user_speaking is expected
    assert [p.sequence for p in packets] == list(range(1, len(packets) + 1))
    await observer.aclose()
    assert voice.listeners == [] and not handlers


@pytest.mark.asyncio
async def test_no_voice_source_or_one_without_the_hook_is_accepted():
    handlers = {}
    session = SimpleNamespace(on=lambda name, fn: handlers.update({name: fn}),
                              off=lambda name, fn: handlers.pop(name))
    async def publish(data, **kwargs):
        pass
    room = SimpleNamespace(local_participant=SimpleNamespace(publish_data=publish))
    plan = SessionOutputPlan(session_id="session", policy_revision=1,
        outputs=OutputSelection(motion=True))
    for voice in (None, SimpleNamespace()):
        observer = attach_interaction_publisher(session, room, plan, voice=voice)
        await observer.aclose()


class Facts:
    """A publisher on a fake session and room, with the packets it sent."""

    def __init__(self):
        self.handlers, self.packets = {}, []
        session = SimpleNamespace(on=lambda name, fn: self.handlers.update({name: fn}),
                                  off=lambda name, fn: self.handlers.pop(name))

        async def publish(data, **kwargs):
            self.packets.append(InteractionFacts.model_validate_json(data))
        room = SimpleNamespace(local_participant=SimpleNamespace(publish_data=publish))
        plan = SessionOutputPlan(session_id="session", policy_revision=1,
            outputs=OutputSelection(motion=True))
        self.voice = FakeVoice()
        self.observer = attach_interaction_publisher(session, room, plan, voice=self.voice)

    def user(self, state):
        self.handlers["user_state_changed"](SimpleNamespace(new_state=state))

    def agent(self, state):
        self.handlers["agent_state_changed"](SimpleNamespace(new_state=state))

    def speaking(self):
        return [p.user_speaking for p in self.packets]


@pytest.mark.asyncio
async def test_a_turn_only_stt_heard_is_user_speech_from_its_first_transcript_to_its_final():
    facts = Facts()
    await _settle()
    # 2026-11-12 14:47:58-14:48:06: away after 15 s, then neither VAD state
    # rises for a quiet question; its first interim opens the turn.
    facts.user("away")
    await _settle()
    before = len(facts.packets)
    facts.observer.transcript_turn_started("turn-3")
    facts.observer.transcript_heard(final=False)
    await _settle()
    facts.observer.transcript_heard(final=False)
    await _settle()
    assert facts.speaking()[before:] == [True]
    facts.observer.transcript_heard(final=True)
    facts.user("listening")  # LiveKit's own fallback on the final: away -> listening
    await _settle()
    assert facts.speaking()[before:] == [True, False]
    assert not facts.observer.transcript_speaking
    facts.observer.transcript_heard(final=False)  # a later revision reopens nothing
    await _settle()
    assert facts.speaking()[-1] is False
    assert [p.sequence for p in facts.packets] == list(range(1, len(facts.packets) + 1))
    await facts.observer.aclose()


@pytest.mark.asyncio
async def test_a_vad_inside_a_transcript_turn_adds_no_edges():
    facts = Facts()
    await _settle()
    before = len(facts.packets)
    facts.observer.transcript_turn_started("turn")
    await _settle()
    facts.voice.set(True)  # the evidence VAD catches up mid-sentence
    await _settle()
    facts.observer.transcript_heard(final=True)
    await _settle()
    facts.voice.set(False)
    await _settle()
    assert facts.speaking()[before:] == [True, False]
    await facts.observer.aclose()


@pytest.mark.asyncio
async def test_transcript_speech_without_a_final_ends_after_the_quiet_window():
    facts = Facts()
    facts.observer.transcript_quiet_s = 0.05
    await _settle()
    facts.observer.transcript_turn_started("turn")
    await asyncio.sleep(0.03)
    facts.observer.transcript_heard(final=False)  # each revision extends it
    await asyncio.sleep(0.03)
    assert facts.observer.transcript_speaking
    await asyncio.sleep(0.05)  # the interim was withdrawn: no final ever comes
    await _settle()
    assert not facts.observer.transcript_speaking and facts.speaking()[-1] is False
    await facts.observer.aclose()


@pytest.mark.asyncio
async def test_no_transcript_speech_while_the_robot_speaks_and_the_robot_starting_ends_it():
    facts = Facts()
    await _settle()
    facts.agent("speaking")
    await _settle()
    before = len(facts.packets)
    # Words only STT heard during a reply: no leave for the device to look,
    # but a packet, so a claim made for that turn lapses with it.
    facts.observer.transcript_turn_started("during-reply")
    await _settle()
    assert facts.speaking()[before:] == [False]
    assert not facts.observer.transcript_speaking
    facts.agent("listening")
    await _settle()
    assert facts.speaking()[-1] is False
    facts.observer.transcript_turn_started("quiet-question")
    await _settle()
    assert facts.speaking()[-1] is True
    facts.agent("speaking")  # another reply starts before the final
    await _settle()
    assert facts.speaking()[-1] is False and not facts.observer.transcript_speaking
    await facts.observer.aclose()


@pytest.mark.asyncio
async def test_closing_cancels_the_quiet_window():
    facts = Facts()
    await _settle()
    facts.observer.transcript_turn_started("turn")
    timer = facts.observer._transcript_quiet
    await facts.observer.aclose()
    assert timer.cancelled() and facts.observer._transcript_quiet is None
