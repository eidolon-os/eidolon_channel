"""Optional motion behavior observer, independent of turn/EOT decisions."""

import asyncio
import logging
from collections.abc import Callable

from eidolon_sdk.biz.presentation.interaction import InteractionFacts, TOPIC

log = logging.getLogger(__name__)


class InteractionPublisher:
    def __init__(self, session, room, plan, voice=None):
        self.session, self.room, self.plan = session, room, plan
        # ``user_speaking`` is true while either VAD state hears the user: the
        # session's own (LiveKit ``user_state``) or the speaker-evidence one in
        # ``voice``. Both run the same model over the same audio with separate
        # state, so quiet speech can trip one and not the other. On 2026-11-12,
        # after 15 s of silence, only the evidence one heard a 2.6 s question;
        # the turn still ran (opened by its transcript) but the device never
        # saw speech, so it had nothing to attribute.
        self.user_state_speaking = self.voice_speaking = False
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
        add_listener = getattr(voice, "add_voice_listener", None)
        self._remove_voice = add_listener(self.on_voice) if callable(add_listener) else None

    def on_user(self, event):
        self.user_state_speaking = event.new_state == "speaking"
        self.user = self.user_state_speaking or self.voice_speaking
        self.changed.set()

    def on_voice(self, speaking: bool) -> None:
        # Only an edge of the combined value is news; the other VAD state's
        # own start or end inside one utterance publishes nothing.
        self.voice_speaking = speaking
        user = self.user_state_speaking or speaking
        if user != self.user:
            self.user = user
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
        if self._remove_voice is not None:
            self._remove_voice()
            self._remove_voice = None
        self.session.off("user_state_changed", self.on_user)
        self.session.off("agent_state_changed", self.on_agent)
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)


def attach_interaction_publisher(session, room, plan, voice=None):
    """``voice``: optional ``add_voice_listener`` source (the speaker observation)."""
    if plan is None or not plan.outputs.motion:
        return None
    return InteractionPublisher(session, room, plan, voice)
