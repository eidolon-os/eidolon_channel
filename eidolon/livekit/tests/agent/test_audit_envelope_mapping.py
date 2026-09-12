"""The ``eidolon.audit.v1`` mapping, pinned before anything publishes it.

The envelope is a frozen pydantic model with ``extra="forbid"`` and length
limits, so every one of these is a ``ValidationError`` the first time a real
session hits it — after the turn, in the voice path, for a fact already gone.
Each test below is one field that can overrun or arrive in a shape the envelope
or its consumer refuses.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from eidolon_sdk.biz.audit import AuditEnvelope

from eidolon.livekit.agent.observability.audit_sink import PRODUCER, to_envelope
from eidolon.livekit.agent.observability.turn_events import (
    ChannelEventContext,
    ChannelTurnEventSink,
    _PendingEvent,
)
from eidolon.livekit.agent.observability.timeline import TurnTimeline


def _context(**overrides) -> ChannelEventContext:
    fields = {
        "owner_id": "owner-1",
        "companion_id": "companion-1",
        "device_id": "device-1",
        "room_name": "room-1",
        "session_flow_id": None,
    }
    fields.update(overrides)
    return ChannelEventContext(**fields)


def _event(**overrides) -> _PendingEvent:
    fields = {
        "event_type": "channel.turn.milestone",
        "subject_type": "turn",
        "subject_id": "turn-1",
        "trace_id": "turn-1",
        "severity": None,
        "outcome": None,
        "reason": "first_audio",
        "payload": {"channel_turn_id": "turn-1", "milestone": "first_audio"},
        "event_id": "evt_ch_mark_turn-1_1",
    }
    fields.update(overrides)
    return _PendingEvent(**fields)


def test_a_plain_milestone_maps_to_a_valid_envelope() -> None:
    envelope = to_envelope(_event(), _context(), producer_seq=1)

    assert isinstance(envelope, AuditEnvelope)
    assert envelope.contract == "eidolon.audit.v1"
    assert envelope.producer == PRODUCER
    assert envelope.category == "receipt"
    assert envelope.action == "channel.turn.milestone"
    assert envelope.subject_type == "turn"
    assert envelope.subject_id == "turn-1"
    assert envelope.owner_id == "owner-1"
    assert envelope.producer_seq == 1


def test_the_payload_says_whose_turn_it_was() -> None:
    """A turn's subject is the turn, so identity has to be in the payload.

    ``eidolon_admin``'s ``_event`` takes ``companion_id`` from the subject only
    when the subject is a Companion; otherwise it reads the payload. Without
    these keys every turn on the Owner's map belongs to nobody.
    """

    envelope = to_envelope(_event(), _context(), producer_seq=1)

    assert envelope.payload["owner_id"] == "owner-1"
    assert envelope.payload["companion_id"] == "companion-1"
    # Existing payload keys survive.
    assert envelope.payload["channel_turn_id"] == "turn-1"
    assert envelope.payload["milestone"] == "first_audio"


def test_an_event_without_an_id_still_gets_a_stable_one() -> None:
    """Session start/end carry no ``event_id``; the envelope requires one.

    Derived rather than random: a retried publish of the same fact must not
    become a second event in the index.
    """

    session_event = _event(
        event_type="channel.session.started",
        subject_type="device",
        subject_id="device-1",
        event_id=None,
        payload={},
    )
    first = to_envelope(session_event, _context(), producer_seq=1)
    second = to_envelope(session_event, _context(), producer_seq=2)

    assert first.event_id
    assert first.event_id == second.event_id
    assert "started" in first.event_id


def test_a_session_flow_id_longer_than_the_envelope_allows_is_clamped() -> None:
    """``normalize_session_flow_id`` admits 96 characters; ``trace_id`` stops at 64."""

    long_flow = "f" * 96
    envelope = to_envelope(
        _event(trace_id=long_flow), _context(session_flow_id=long_flow), producer_seq=1
    )

    assert len(envelope.trace_id or "") == 64
    assert (envelope.trace_id or "").startswith("f")


def test_a_long_decision_reason_is_clamped_not_refused() -> None:
    """Turn-policy reasons carry their evidence inline and do overrun 256."""

    reason = (
        "deadline_hold_max_suspend_elapsed suspend=2.34s>=2.00s last_reason="
        + "semantic_score_wait " * 20
    )
    assert len(reason) > 256
    envelope = to_envelope(_event(reason=reason), _context(), producer_seq=1)

    assert len(envelope.reason or "") == 256


def test_oversized_ids_are_clamped() -> None:
    envelope = to_envelope(
        _event(subject_id="t" * 200, event_id="e" * 200),
        _context(owner_id="o" * 200),
        producer_seq=1,
    )

    assert len(envelope.subject_id) == 128
    assert len(envelope.event_id) == 64
    assert len(envelope.owner_id or "") == 64


def test_severity_never_reaches_the_value_the_consumer_cannot_read() -> None:
    """The envelope admits ``critical``; ``eidolon_admin``'s RuntimeSeverity does not."""

    assert to_envelope(_event(severity="critical"), _context(), producer_seq=1).severity == "info"
    assert to_envelope(_event(severity="error"), _context(), producer_seq=1).severity == "error"
    assert to_envelope(_event(severity=None), _context(), producer_seq=1).severity == "info"
    assert to_envelope(_event(severity="nonsense"), _context(), producer_seq=1).severity == "info"


def test_classification_is_always_safe() -> None:
    """The one value this contract and the consumer's PrivacyMode agree on.

    The envelope's middle value is ``sensitive``; ``eidolon_admin`` spells it
    ``summary``, so anything but ``safe`` fails its projection. These payloads
    carry ids, phases, counts and durations — never transcript text.
    """

    envelope = to_envelope(_event(), _context(), producer_seq=1)
    assert envelope.data_classification == "safe"


def test_an_unknown_outcome_degrades_to_success() -> None:
    assert to_envelope(_event(outcome="failure"), _context(), producer_seq=1).outcome == "failure"
    assert to_envelope(_event(outcome=None), _context(), producer_seq=1).outcome == "success"
    assert to_envelope(_event(outcome="weird"), _context(), producer_seq=1).outcome == "success"


def test_producer_seq_is_forced_into_range() -> None:
    """``ge=1``. A caller with a zero-based counter must not fail a turn for it."""

    assert to_envelope(_event(), _context(), producer_seq=0).producer_seq == 1
    assert to_envelope(_event(), _context(), producer_seq=-5).producer_seq == 1


def test_occurred_at_can_be_supplied_for_a_spooled_record() -> None:
    """When a spool stands between observation and publication, the write time
    is the fact; publishing time would make every latency a measure of the
    outbox."""

    when = datetime(2026, 9, 12, 21, 0, 0, tzinfo=UTC)
    assert to_envelope(_event(), _context(), producer_seq=1, occurred_at=when).occurred_at == when
    assert to_envelope(_event(), _context(), producer_seq=1).occurred_at.tzinfo is not None


def test_empty_identity_does_not_produce_an_invalid_envelope() -> None:
    """Nothing about a degraded context may turn into a ValidationError."""

    envelope = to_envelope(
        _event(subject_type="", subject_id="", event_type="", reason="", trace_id=""),
        _context(owner_id="", companion_id=""),
        producer_seq=1,
    )

    assert envelope.subject_type == "turn"
    assert envelope.subject_id == "unknown"
    assert envelope.action
    assert envelope.reason is None
    assert envelope.trace_id is None
    assert envelope.owner_id is None


# --------------------------------------------------------------------------- #
# Against the sink's real output rather than hand-built events
# --------------------------------------------------------------------------- #


def test_every_event_the_sink_actually_emits_maps_cleanly() -> None:
    """The mapping is exercised by the vocabulary, not by a sample of it.

    A hand-written event proves the mapper; this proves the mapper against what
    the sink emits — which is the thing that will be published.
    """

    observed: list[_PendingEvent] = []
    sink = ChannelTurnEventSink(observer=observed.append)
    sink._context = _context(session_flow_id="flow-1")  # type: ignore[attr-defined]

    timeline = TurnTimeline("turn-abc")
    timeline.mark("speech_started_at")
    sink.phase_changed(
        timeline=timeline,
        previous_phase="idle",
        phase="user_turn_committed",
        event="commit",
        reason="committed",
        side_effect="none",
        occurred_at=timeline.timestamps["speech_started_at"] + 0.5,
    )
    sink.milestone(timeline, "first_audio", reason="first_audio")
    sink.terminal(timeline, "agent_audio_playback_done")

    assert len(observed) >= 3
    for index, event in enumerate(observed, start=1):
        envelope = to_envelope(event, sink.context, producer_seq=index)
        # Round-trips through the contract: what a transport would serialize.
        restored = AuditEnvelope.model_validate(envelope.model_dump())
        assert restored == envelope
        assert restored.producer == "channel"
        assert restored.payload["owner_id"] == "owner-1"
        assert restored.payload["companion_id"] == "companion-1"

    actions = {event.event_type for event in observed}
    assert "channel.turn.phase_changed" in actions
    assert "channel.turn.milestone" in actions


def test_the_envelope_still_refuses_what_it_should() -> None:
    """A guard on the guard: if these ever stop raising, the clamps above are
    protecting nothing and the tests would keep passing anyway."""

    base = to_envelope(_event(), _context(), producer_seq=1).model_dump()

    with pytest.raises(ValidationError):
        AuditEnvelope.model_validate({**base, "producer_seq": 0})
    with pytest.raises(ValidationError):
        AuditEnvelope.model_validate({**base, "trace_id": "x" * 65})
    with pytest.raises(ValidationError):
        AuditEnvelope.model_validate({**base, "subject_id": "x" * 129})
    with pytest.raises(ValidationError):
        AuditEnvelope.model_validate({**base, "data_classification": "summary"})
    with pytest.raises(ValidationError):
        AuditEnvelope.model_validate({**base, "unexpected": 1})
