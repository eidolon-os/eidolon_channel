"""Optional motion behavior observer, independent of turn/EOT decisions."""

import asyncio
import logging
from collections.abc import Callable

from eidolon_sdk.biz.presentation.interaction import InteractionFacts, TOPIC

log = logging.getLogger(__name__)


class InteractionPublisher:
    def __init__(self, session, room, plan):
        self.session, self.room, self.plan = session, room, plan
        self.user = self.agent = False
        self.sequence = 0
        self.closed = False
        # Told of each packet as it is built, in sequence order, before it is
        # sent; e.g. gaze_addressee maps speech onsets to turns through it.
        self.observers: list[Callable[[InteractionFacts], None]] = []
        self.changed = asyncio.Event()
        session.on("user_state_changed", self.on_user)
        session.on("agent_state_changed", self.on_agent)
        self.task = asyncio.create_task(self.run())
        self.changed.set()

    def on_user(self, event):
        self.user = event.new_state == "speaking"
        self.changed.set()

    def on_agent(self, event):
        self.agent = event.new_state == "speaking"
        self.changed.set()

    async def run(self):
        while not self.closed:
            await self.changed.wait()
            self.changed.clear()
            self.sequence += 1
            facts = InteractionFacts(session_id=self.plan.session_id,
                policy_revision=self.plan.policy_revision, sequence=self.sequence,
                user_speaking=self.user, agent_speaking=self.agent)
            for observer in tuple(self.observers):
                try:
                    observer(facts)
                except Exception:
                    log.exception("interaction facts observer failed sequence=%s", self.sequence)
            try:
                async with asyncio.timeout(2):
                    await self.room.local_participant.publish_data(
                        facts.model_dump_json().encode(), topic=TOPIC, reliable=True)
            except Exception:
                log.warning("interaction facts delivery unavailable sequence=%s", self.sequence)
                await asyncio.sleep(0.25)
                self.changed.set()

    def add_observer(self, observer: Callable[[InteractionFacts], None]) -> None:
        self.observers.append(observer)

    def remove_observer(self, observer: Callable[[InteractionFacts], None]) -> None:
        if observer in self.observers:
            self.observers.remove(observer)

    async def aclose(self):
        self.closed = True
        self.session.off("user_state_changed", self.on_user)
        self.session.off("agent_state_changed", self.on_agent)
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)


def attach_interaction_publisher(session, room, plan):
    if plan is None or not plan.outputs.motion:
        return None
    return InteractionPublisher(session, room, plan)
