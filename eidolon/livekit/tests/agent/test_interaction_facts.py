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
