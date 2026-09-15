from __future__ import annotations

import json
from types import SimpleNamespace

from eidolon.livekit.agent.observability import turn_events
from eidolon.livekit.agent.observability import (
    ChannelEventContext,
    ChannelTurnEventSink,
    TurnTimeline,
)


def _enabled_sink() -> ChannelTurnEventSink:
    observed = []
    sink = ChannelTurnEventSink(observer=observed.append)
    sink._test_observed = observed  # type: ignore[attr-defined]
    sink._context = ChannelEventContext(  # type: ignore[attr-defined]
        owner_id="owner-1",
        companion_id="companion-1",
        device_id="device-1",
        room_name="room-1",
        session_flow_id=None,
    )
    return sink


def test_phase_projection_stays_in_the_telemetry_lane() -> None:
    sink = _enabled_sink()
    timeline = TurnTimeline("channel-turn-1")
    timeline.mark_at("speech_started_at", 10.0)

    sink.phase_changed(
        timeline=timeline,
        previous_phase="idle",
        phase="user_speech_open",
        event="speech_started",
        reason="new_speech_started",
        side_effect="none",
        occurred_at=10.125,
        details={"eot_score": 0.7, "transcript": "must not leave Channel"},
    )

    assert len(sink._test_observed) == 1  # type: ignore[attr-defined]
    assert sink.telemetry_observed_count == 1


def test_terminal_projection_is_deduped_and_classifies_rejection() -> None:
    sink = _enabled_sink()
    timeline = TurnTimeline("channel-turn-rejected")
    timeline.mark("speech_started_at")
    timeline.set_attr(
        "full_duplex_state",
        {"phase": "user_turn_rejected", "transition_count": 2},
    )

    sink.terminal(timeline, "voiceprint_commit_blocked")
    sink.terminal(timeline, "duplicate_flush")

    assert len(sink._test_observed) == 1  # type: ignore[attr-defined]
    pending = sink._test_observed[0]  # type: ignore[attr-defined]
    assert pending.event_type == "channel.turn.rejected"
    assert pending.outcome == "denied"
    assert pending.payload["status"] == "rejected"
    assert "turn_committed" in pending.payload["missing_milestones"]


def test_terminal_projection_distinguishes_interrupted_response_from_failed_tts() -> None:
    sink = _enabled_sink()
    interrupted = TurnTimeline("channel-turn-interrupted")
    interrupted.mark("turn_committed_at")
    interrupted.mark("tts_first_audio_at")
    failed = TurnTimeline("channel-turn-tts-failed")
    failed.mark("turn_committed_at")
    failed.mark("tts_first_audio_at")
    failed.mark("tts_error_at")

    sink.terminal(interrupted, "interrupted_by_user")
    sink.terminal(failed, "nonrecoverable_tts_error")

    interrupted_event, failed_event = sink._test_observed  # type: ignore[attr-defined]
    assert interrupted_event is not None
    assert interrupted_event.event_type == "channel.turn.completed"
    assert interrupted_event.payload["status"] == "interrupted"
    assert interrupted_event.payload["terminal_reason"] == "interrupted_by_user"
    assert failed_event is not None
    assert failed_event.event_type == "channel.turn.failed"
    assert failed_event.outcome == "failure"
    assert failed_event.payload["status"] == "failed"


def test_telemetry_adapter_failure_does_not_escape_into_voice_work() -> None:
    def _fail(_event) -> None:
        raise RuntimeError("metrics backend unavailable")

    sink = ChannelTurnEventSink(observer=_fail)
    sink._context = ChannelEventContext(  # type: ignore[attr-defined]
        owner_id="owner-1",
        companion_id="companion-1",
        device_id=None,
        room_name="room-1",
        session_flow_id=None,
    )
    sink.terminal(TurnTimeline("turn-1"), "agent_playback_done")

    assert sink.dropped_count == 1


async def test_event_context_supports_companion_without_device() -> None:
    participant = SimpleNamespace(
        identity="companion-1",
        metadata=json.dumps(
            {
                "kind": "companion",
                "owner_id": "owner-1",
                "companion_id": "companion-1",
            }
        ),
    )
    room = SimpleNamespace(
        name="room-virtual",
        remote_participants={"companion-1": participant},
    )

    context = await turn_events._resolve_event_context(room)

    assert context.owner_id == "owner-1"
    assert context.companion_id == "companion-1"
    assert context.device_id is None


def _device_room(**metadata) -> SimpleNamespace:
    """A device in a room, carrying the metadata a real device token carries.

    Deliberately no ``companion_id``: the Channel Provider leaves it out of the
    credential on purpose — "server-side orchestration is declared where the
    room is declared, never routed through a credential handed to the device" —
    so which Companion answers is the Kernel mount's to say, not the device's.
    """

    base = {"kind": "device", "device_id": "device-1", "owner_id": "owner-1"}
    base.update(metadata)
    participant = SimpleNamespace(identity="device-1", metadata=json.dumps(base))
    return SimpleNamespace(
        name="room-device", remote_participants={"device-1": participant}
    )


def _mount_resolver(companion_id: str | None):
    """The runtime resolver, answering as it does for a mounted body."""

    async def resolve(room):  # noqa: ANN001 - mirrors ChannelRuntimeServices
        if companion_id is None:
            return SimpleNamespace(
                owner_id="owner-1", device_id="device-1", answering_companion_id=None
            )
        return SimpleNamespace(
            runtime=SimpleNamespace(
                owner_id="owner-1",
                companion_id=companion_id,
                device_id="device-1",
            ),
            mount_revision=3,
        )

    return resolve


async def test_device_context_takes_the_companion_from_the_kernel_mount() -> None:
    """A device session must be attributable without the token naming a Companion.

    This is what left every device trace on disk called
    ``unknown-owner__unknown-companion``: the context refused to resolve, the
    sink disabled itself, and the writer fell back to placeholders — so a
    listing filtered by Owner returned nothing while the recordings sat there.

    The Companion comes from the same place the device token's own resolver
    reads it, so the trace names the pair the brain was actually talking as.
    """

    context = await turn_events._resolve_event_context(
        _device_room(), context_resolver=_mount_resolver("companion-mounted")
    )

    assert context.owner_id == "owner-1"
    assert context.companion_id == "companion-mounted"
    assert context.device_id == "device-1"


async def test_a_token_that_names_a_companion_is_not_overridden_by_the_mount() -> None:
    """Explicit metadata still wins, and the mount is not consulted.

    Same precedence as the runtime resolver's own
    ``metadata.companion_id or connection.answering_companion_id``. Two readers
    of one fact must not disagree about which half is authoritative.
    """

    async def _must_not_be_called(room):  # noqa: ANN001
        raise AssertionError("the mount was consulted despite an explicit companion")

    context = await turn_events._resolve_event_context(
        _device_room(companion_id="companion-explicit"),
        context_resolver=_must_not_be_called,
    )

    assert context.companion_id == "companion-explicit"


async def test_a_body_nobody_answers_through_still_refuses_to_guess() -> None:
    """No Companion mounted is not a licence to invent one.

    The mount is the authority and it said none. Naming the trace after a guess
    would be worse than leaving it unattributed.
    """

    try:
        await turn_events._resolve_event_context(
            _device_room(), context_resolver=_mount_resolver(None)
        )
    except RuntimeError as exc:
        assert "companion" in str(exc).lower()
    else:
        raise AssertionError("Channel must not invent a Companion for an unmounted body")


async def test_a_refusal_says_which_half_is_missing_and_why() -> None:
    """The message has to identify the fault, not restate the requirement.

    This is the regression that cost a real diagnosis: the Kernel answered
    200 OK, the Companion was fetched, and one millisecond later the session
    was filed under `unknown-owner__unknown-companion` — with a log line that
    said only that a Companion was required. Every distinct cause produced that
    same sentence, so the logs could not tell them apart.
    """

    async def _resolver_explodes(room):  # noqa: ANN001
        raise RuntimeError("Companion runtime does not match mounted Device owner/target")

    try:
        await turn_events._resolve_event_context(
            _device_room(), context_resolver=_resolver_explodes
        )
    except RuntimeError as exc:
        message = str(exc)
        assert "companion_id" in message, message
        # the resolver's own words survive, which is the whole point
        assert "does not match mounted Device" in message, message
    else:
        raise AssertionError("an exploding resolver must not read as a resolved one")


async def test_an_unmounted_body_and_a_broken_resolver_do_not_read_alike() -> None:
    """Two different faults, two different sentences.

    A mount that names nobody is a fact about the domain; a resolver that threw
    is a fact about this Host. Collapsing them is what made the failure look
    exactly like tracing being switched off.
    """

    async def _resolver_explodes(room):  # noqa: ANN001
        raise RuntimeError("Kernel Body GET failed")

    async def _grab(resolver) -> str:
        try:
            await turn_events._resolve_event_context(
                _device_room(), context_resolver=resolver
            )
        except RuntimeError as exc:
            return str(exc)
        raise AssertionError("expected a refusal")

    unmounted = await _grab(_mount_resolver(None))
    broken = await _grab(_resolver_explodes)

    assert "no Companion answering" in unmounted, unmounted
    assert "resolver refused" in broken, broken
    assert unmounted != broken


async def test_a_missing_resolver_says_so_rather_than_blaming_the_mount() -> None:
    """Nobody handed one over — that is this process's fault, not the Kernel's."""

    try:
        await turn_events._resolve_event_context(_device_room())
    except RuntimeError as exc:
        assert "no runtime resolver" in str(exc), str(exc)
    else:
        raise AssertionError("expected a refusal")


async def test_owner_event_context_requires_explicit_companion_selection() -> None:
    participant = SimpleNamespace(
        identity="owner-a",
        metadata=json.dumps({"kind": "owner", "owner_id": "owner-a"}),
    )
    room = SimpleNamespace(
        name="room-owner",
        remote_participants={"owner-a": participant},
    )

    try:
        await turn_events._resolve_event_context(room)
    except RuntimeError as exc:
        assert "explicitly selected companion" in str(exc)
    else:
        raise AssertionError("Channel must not guess a Companion from Owner scope")


async def test_sink_keeps_session_and_turn_chain_in_telemetry_lane() -> None:
    participant = SimpleNamespace(
        identity="device-1",
        metadata=json.dumps(
            {
                "kind": "device",
                "device_id": "device-1",
                "owner_id": "owner-1",
                "companion_id": "companion-1",
            }
        ),
    )
    room = SimpleNamespace(
        name="room-1",
        remote_participants={"device-1": participant},
    )
    observed = []
    sink = ChannelTurnEventSink(observer=observed.append)
    await sink.start(room)
    assert sink.enabled

    timeline = TurnTimeline("channel-turn-1")
    timeline.mark_at("speech_started_at", 10.0)
    timeline.set_attr("full_duplex_state", {"phase": "turn_finished"})
    sink.phase_changed(
        timeline=timeline,
        previous_phase="idle",
        phase="user_speech_open",
        event="speech_started",
        reason="new_speech_started",
        side_effect="none",
        occurred_at=10.05,
    )
    sink.milestone(timeline, "brain_request_sent")
    sink.terminal(timeline, "agent_playback_done")
    await sink.close()

    event_types = [event.event_type for event in observed]
    assert event_types == [
        "channel.session.started",
        "channel.turn.phase_changed",
        "channel.turn.milestone",
        "channel.turn.completed",
        "channel.session.ended",
    ]
    turn_events_by_trace = [event for event in observed if event.trace_id == "channel-turn-1"]
    assert len(turn_events_by_trace) == 3
    assert all(event.payload.get("device_id") == "device-1" for event in observed)


def test_channel_telemetry_has_no_eidolon_data_dependency() -> None:
    source = turn_events.__file__
    assert source is not None
    assert "eidolon_data" not in open(source, encoding="utf-8").read()
