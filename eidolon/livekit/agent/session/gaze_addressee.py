"""Hold a device's gaze on whoever the current reply answers.

The device reports, per user speech segment, which of its own session-local
tracks spoke (``attention.speaker``). This module stores each report on the
turn its segment belongs to, and when the reply to an adopted turn starts
playing it tells the device to hold on that track (``attention.lock`` hold),
then to let go when the reply ends or is interrupted (release).

Not to be confused with ``attention_effects`` / ``turn_policy.attention``:
those decide whether an utterance was addressed to the assistant at all.

Which turn a report belongs to comes from ``facts_sequence``: the
``InteractionFacts`` sequence whose ``user_speaking=true`` opened the device's
segment. The facts publisher tells this module, packet by packet, where each
segment opened; the speech lifecycle tells it which turn the speech it hears
belongs to (``on_speech_started``, for a new turn, a merged continuation, or a
turn its transcript opened without any VAD edge). A segment belongs to the
turn that speech claimed while the segment was open, or just before its
opening packet was built.

The two do not arrive in a fixed order: ``user_speaking`` is true while either
of two VAD states hears the user, and the one that opens turns can start up to
about a second after the other or, for quiet speech, not at all. A segment no
speech claimed (a sound no turn was made of) never lends its report to a turn,
and a report that arrives before its segment is claimed waits on the segment.

The binding window of a turn closes when its reply first speaks: a report that
arrives later, or that refers to an earlier segment, never changes the lock.

Nothing is sent until the session has accepted at least one valid speaker
report, so a device that never reports sees no new traffic at all.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from eidolon_sdk.biz.presentation.attention import (
    LOCK_HOLD_LEASE_MS,
    LOCK_RELEASE_LEASE_MS,
    LOCK_TOPIC,
    SPEAKER_TOPIC,
    AttentionLock,
    AttentionSpeaker,
)
from eidolon_sdk.biz.presentation.interaction import InteractionFacts

from eidolon.livekit.agent.observability import TurnTimeline

log = logging.getLogger(__name__)

# A reply is ongoing in these agent states. Between two spoken parts of one
# reply (a tool call) the agent is "thinking"; the hold stays, and its lease
# bounds the wait. Any other state ends the reply.
_REPLY_STATES = frozenset({"speaking", "thinking"})
_REMEMBERED = 32
_SEND_TIMEOUT_SEC = 2.0


@dataclass
class _Turn:
    turn_id: str
    report: AttentionSpeaker | None = None
    bound: bool = False

    def take(self, report: AttentionSpeaker) -> None:
        # Report sequences only increase, so this keeps the latest received.
        if not self.bound and (self.report is None or report.sequence > self.report.sequence):
            self.report = report


@dataclass
class _Segment:
    opened_at: int
    turn: _Turn | None = None
    report: AttentionSpeaker | None = None

    def claim(self, turn: _Turn) -> None:
        self.turn = turn
        if self.report is not None:
            turn.take(self.report)


def _sender(packet: Any) -> str:
    participant = getattr(packet, "participant", None)
    return getattr(participant, "identity", "") or ""


class GazeAddressee:
    def __init__(
        self,
        room: Any,
        plan: Any,
        *,
        participant_identity: str | None = None,
    ) -> None:
        self._room = room
        self._plan = plan
        self._participant_identity = participant_identity or None
        self._segments: deque[_Segment] = deque(maxlen=_REMEMBERED)
        self._turns: OrderedDict[str, _Turn] = OrderedDict()
        self._user_speaking = False
        # Speech claimed while no segment was open: the packet that opens one
        # is being built. Taken by that packet, dropped by any other.
        self._claim: _Turn | None = None
        self._reporter: str | None = None
        self._report_sequence = 0
        self._locked: _Turn | None = None
        self._sequence = 0
        self._outbox: asyncio.Queue[tuple[AttentionLock, str]] = asyncio.Queue()
        self._detach: list[Callable[[], None]] = []
        self._task = asyncio.create_task(self._send_loop(), name="eidolon-gaze-addressee")
        self.closed = False

    @property
    def enabled(self) -> bool:
        """True once this session has accepted a speaker report."""
        return self._reporter is not None

    # -- inputs ---------------------------------------------------------

    def on_facts(self, facts: InteractionFacts) -> None:
        """Facts publisher observer: remember where each device segment opens."""
        rising = facts.user_speaking and not self._user_speaking
        self._user_speaking = facts.user_speaking
        claim, self._claim = self._claim, None
        if rising:
            self._segments.append(_Segment(facts.sequence, claim))

    def on_speech_started(self, timeline: TurnTimeline | None) -> None:
        """Speech lifecycle: the user speech now heard belongs to ``timeline``'s turn."""
        if timeline is None:
            return
        turn = self._turn(timeline.turn_id)
        if self._user_speaking and self._segments:
            self._segments[-1].claim(turn)
        else:
            self._claim = turn

    def on_packet(self, packet: Any) -> None:
        """Room data subscriber for ``attention.speaker``."""
        identity = _sender(packet)
        # A lock is addressed to whoever reported; an anonymous or foreign
        # sender has nowhere to receive it.
        if not identity or (
            self._participant_identity is not None and identity != self._participant_identity
        ):
            log.debug("[GazeAddressee] speaker report ignored from=%s", identity or "unknown")
            return
        try:
            report = AttentionSpeaker.model_validate_json(bytes(getattr(packet, "data", b"") or b""))
        except ValueError:
            log.debug("[GazeAddressee] malformed speaker report ignored", exc_info=True)
            return
        if (report.session_id != self._plan.session_id
                or report.policy_revision != self._plan.policy_revision
                or report.sequence <= self._report_sequence):
            log.debug("[GazeAddressee] stale speaker report ignored sequence=%s", report.sequence)
            return
        self._report_sequence = report.sequence
        if self._reporter is None:
            log.info("[GazeAddressee] speaker reports enabled from=%s", identity or "unknown")
        self._reporter = identity
        segment = self._segment(report.facts_sequence)
        if segment is None:
            self._dropped(report, "no_turn")
            return
        # Kept on the segment too: a later claim (a turn opened after this
        # report, or the next turn of one long segment) takes it from there.
        segment.report = report
        turn = segment.turn
        if turn is None:
            log.debug("[GazeAddressee] speaker report waits for its turn sequence=%s "
                      "facts_sequence=%s", report.sequence, report.facts_sequence)
        elif turn.bound:
            self._dropped(report, "reply_already_started")
        else:
            turn.take(report)

    def on_agent_state(self, old_state: str, new_state: str, timeline: TurnTimeline | None) -> None:
        """Agent state observer; ``timeline`` owns the current output, if any."""
        if new_state == "speaking":
            self._hold(timeline)
        elif new_state not in _REPLY_STATES and self._locked is not None:
            turn, self._locked = self._locked, None
            self._enqueue("release", turn, LOCK_RELEASE_LEASE_MS)

    # -- decisions ------------------------------------------------------

    def _hold(self, timeline: TurnTimeline | None) -> None:
        # Welcome lines and proactive reports speak without an adopted turn.
        if not self.enabled or timeline is None:
            return
        turn = self._turn(timeline.turn_id)
        if not turn.bound:
            turn.bound = True
            report = turn.report
            timeline.set_attr("gaze_addressee", {
                "track_id": report.track_id if report else None,
                "basis": report.basis if report else None,
                "speaker_sequence": report.sequence if report else None,
                "facts_sequence": report.facts_sequence if report else None,
                "final": report.final if report else None,
            })
        self._locked = turn
        self._enqueue("hold", turn, LOCK_HOLD_LEASE_MS)

    def _turn(self, turn_id: str) -> _Turn:
        turn = self._turns.get(turn_id)
        if turn is None:
            turn = self._turns[turn_id] = _Turn(turn_id)
            while len(self._turns) > _REMEMBERED:
                self._turns.popitem(last=False)
        return turn

    @staticmethod
    def _dropped(report: AttentionSpeaker, reason: str) -> None:
        log.info(
            "[GazeAddressee] speaker report dropped sequence=%s facts_sequence=%s reason=%s",
            report.sequence, report.facts_sequence, reason,
        )

    def _segment(self, facts_sequence: int) -> _Segment | None:
        for segment in reversed(self._segments):
            if segment.opened_at <= facts_sequence:
                return segment
        return None

    # -- delivery -------------------------------------------------------

    def _enqueue(self, phase: str, turn: _Turn, lease_ms: int) -> None:
        destination = self._reporter
        if self.closed or destination is None:
            return
        report = turn.report
        self._sequence += 1
        message = AttentionLock(
            session_id=self._plan.session_id,
            policy_revision=self._plan.policy_revision,
            sequence=self._sequence,
            phase=phase,
            turn_id=turn.turn_id,
            track_id=report.track_id if report else None,
            speaker_sequence=report.sequence if report else None,
            lease_ms=lease_ms,
        )
        log.info(
            "[GazeAddressee] lock phase=%s turn=%s track=%s speaker_sequence=%s "
            "sequence=%s lease_ms=%s to=%s",
            phase, message.turn_id, message.track_id, message.speaker_sequence,
            message.sequence, lease_ms, destination,
        )
        self._outbox.put_nowait((message, destination))

    async def _send_loop(self) -> None:
        # One sender keeps hold and release in sequence order. A failed send is
        # not retried: a late hold is worse than none, and the device falls
        # back to its own playback rule within the lease.
        while True:
            message, destination = await self._outbox.get()
            try:
                async with asyncio.timeout(_SEND_TIMEOUT_SEC):
                    await self._room.local_participant.publish_data(
                        message.model_dump_json().encode(),
                        topic=LOCK_TOPIC,
                        reliable=True,
                        destination_identities=[destination],
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning(
                    "[GazeAddressee] lock delivery unavailable phase=%s sequence=%s",
                    message.phase, message.sequence,
                )

    def bind(self, interaction: Any, room_data: Any) -> None:
        """Observe the facts publisher and subscribe to speaker reports."""
        interaction.add_observer(self.on_facts)
        self._detach.append(lambda: interaction.remove_observer(self.on_facts))
        self._detach.append(room_data.subscribe(SPEAKER_TOPIC, self.on_packet))

    async def aclose(self) -> None:
        self.closed = True
        while self._detach:
            self._detach.pop()()
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)


def attach_gaze_addressee(
    room: Any,
    interaction: Any,
    *,
    room_data: Any,
    participant_identity: str | None = None,
) -> GazeAddressee | None:
    """Attach only where the facts publisher runs, i.e. the plan includes motion."""
    if interaction is None or room_data is None:
        return None
    addressee = GazeAddressee(
        room, interaction.plan,
        participant_identity=participant_identity,
    )
    addressee.bind(interaction, room_data)
    return addressee
