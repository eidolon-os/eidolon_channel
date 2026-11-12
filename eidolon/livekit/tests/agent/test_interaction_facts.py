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
