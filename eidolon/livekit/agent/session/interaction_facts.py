"""Optional motion behavior observer, independent of turn/EOT decisions."""

import asyncio
import logging
from collections.abc import Callable

from eidolon_sdk.biz.presentation.interaction import InteractionFacts, TOPIC

log = logging.getLogger(__name__)


class InteractionPublisher:
    # A turn opened on its transcript ends its speech at the final; this long
    # without any transcript ends it too (a withdrawn interim never gets one).
    # Within an utterance the provider sends a revision at least every ~0.8 s.
    transcript_quiet_s = 2.0

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
        # And from the first transcript of a turn neither VAD opened to its
        # final, with the robot silent (see transcript_turn_started).
        self.transcript_speaking = False
        self._transcript_turn = None
        self._transcript_quiet = None
        self.user = self.agent = False
        # Between the committed turn and the first audio of its reply. A device
        # with a body shows it was heard instead of standing still meanwhile.
        self.thinking = False
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
        self.user = self._user_speaking()
        self.changed.set()

    def on_voice(self, speaking: bool) -> None:
        self.voice_speaking = speaking
        self._update_user()

    def on_agent(self, event):
        self.agent = event.new_state == "speaking"
        self.thinking = event.new_state == "thinking"
        if self.agent:
            self._end_transcript("agent_speaking")
        self.user = self._user_speaking()
        self.changed.set()

    def transcript_turn_started(self, turn_id: str = "") -> None:
        """A turn opened on its first transcript, with neither VAD hearing speech.

        2026-11-12 14:48:04: after 15 s of silence a quiet 1.2 s question
        reached STT (and the voiceprint) while neither VAD state rose. The
        turn ran, but without ``user_speaking`` the device opened no segment
        and the reply's hold carried no track. Such a turn's speech now runs
        from its first transcript, about 0.3 s into the words, to its final,
        about 0.4 s after them, or until the robot starts speaking.

        Not when the robot is speaking: its own voice mutes the device's
        sampling then, and ``user_speaking`` is also what lets the device turn
        toward a voice, so words only STT heard during a reply (often a
        bystander's) get no such leave. The packet still goes out, so the
        addressee's claim for that turn lapses at once instead of passing to
        whatever speech starts next.
        """
        if self.agent:
            log.info("interaction facts: transcript turn while the agent speaks, "
                     "no user speech turn=%s", turn_id)
            self.changed.set()
            return
        self.transcript_speaking = True
        self._transcript_turn = turn_id
        self._arm_transcript_quiet()
        log.info("interaction facts: user speech from a transcript no VAD heard turn=%s", turn_id)
        self._update_user()

    def transcript_heard(self, *, final: bool) -> None:
        """Any transcript: a final ends a transcript-opened turn's speech, others extend it."""
        if not self.transcript_speaking:
            return
        if final:
            self._end_transcript("final")
        else:
            self._arm_transcript_quiet()

    def _arm_transcript_quiet(self) -> None:
        if self._transcript_quiet is not None:
            self._transcript_quiet.cancel()
        self._transcript_quiet = asyncio.get_running_loop().call_later(
            self.transcript_quiet_s, self._end_transcript, "quiet")

    def _end_transcript(self, reason: str) -> None:
        if self._transcript_quiet is not None:
            self._transcript_quiet.cancel()
            self._transcript_quiet = None
        if not self.transcript_speaking:
            return
        self.transcript_speaking = False
        log.info("interaction facts: transcript speech ended turn=%s reason=%s",
                 self._transcript_turn, reason)
        self._update_user()

    def _user_speaking(self) -> bool:
        return self.user_state_speaking or self.voice_speaking or self.transcript_speaking

    def _update_user(self) -> None:
        # Only an edge of the combined value is news; one source's own start
        # or end inside an utterance another already covers publishes nothing.
        user = self._user_speaking()
        if user != self.user:
            self.user = user
            self.changed.set()

    async def run(self):
        while not self.closed:
            await self.changed.wait()
            self.changed.clear()
            self.sequence += 1
            facts = InteractionFacts(session_id=self.plan.session_id,
                policy_revision=self.plan.policy_revision, sequence=self.sequence,
                user_speaking=self.user, agent_speaking=self.agent,
                agent_thinking=self.thinking)
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
        if self._transcript_quiet is not None:
            self._transcript_quiet.cancel()
            self._transcript_quiet = None
        if self._remove_voice is not None:
            self._remove_voice()
            self._remove_voice = None
        self.session.off("user_state_changed", self.on_user)
        self.session.off("agent_state_changed", self.on_agent)
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)


def attach_interaction_publisher(session, room, plan, voice=None):
    """``voice``: optional ``add_voice_listener`` source (the speaker observation).

    The full-duplex speech lifecycle also calls ``transcript_turn_started`` and
    ``transcript_heard`` on the result when it has one.
    """
    if plan is None or not plan.outputs.motion:
        return None
    return InteractionPublisher(session, room, plan, voice)
