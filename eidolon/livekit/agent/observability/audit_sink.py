"""Map Channel turn telemetry onto the ``eidolon.audit.v1`` envelope.

This is the mapping and nothing else. There is no outbox here, no dispatcher,
no transport, and no NATS dependency: publishing is a separate decision with
its own durability and ordering problems, and the Host's audit index has no
retention policy yet. What this module buys is that the contract is already
aligned when that decision is made — the shape is pinned by tests rather than
rediscovered.

The consumer already exists and was written against these events.
``eidolon_admin``'s ``_project_runtime_turns`` reads only ``source == "channel"``
envelopes, groups them by ``payload["channel_turn_id"]``, and builds its stage
breakdown from ``phase_changed.payload["phase"]`` and
``milestone.payload["milestone"]`` with ``elapsed_ms``. Its ``RuntimeTurn`` even
carries ``channel_turn_id`` and ``missing_milestones`` — this sink's payload
keys. So the vocabulary is not being invented; it is being honoured.

**Every constrained field is bounded here rather than assumed.** The envelope
is a pydantic model with ``extra="forbid"`` and length limits, so a value that
overruns one is a ``ValidationError`` at publish time — in the voice path, for a
turn that already happened. Mapping is therefore total: it clamps, substitutes
and never raises. The seven places that can overrun are each covered by a test.

``producer_seq`` is deliberately a required argument. It must be monotonic per
producer, and an Agent worker is spawned per job — a counter owned here would
restart at 1 in every process and produce a sequence the consumer cannot use to
detect a gap. Only a single long-lived allocator can answer it, which is the
Channel Provider, which is where the outbox belongs when it is written.
"""

from __future__ import annotations

from datetime import UTC, datetime

from eidolon_sdk.biz.audit import AuditEnvelope

from .turn_events import ChannelEventContext, _PendingEvent

#: What this Host calls us in an envelope. ``eidolon_admin``'s ``RuntimeSource``
#: literal already includes it, so nothing downstream has to learn a new name.
PRODUCER = "channel"

#: A turn observation is a receipt, not a governance fact. Governance is the
#: Owner/Companion/Device decisions the other authorities record; a turn that
#: took 800 ms is the residue of serving one.
CATEGORY = "receipt"

#: Envelope field limits, restated because exceeding one is a ValidationError
#: and the values that reach here are not all bounded at their source.
_MAX_EVENT_ID = 64
_MAX_OWNER_ID = 64
_MAX_SUBJECT_TYPE = 64
_MAX_SUBJECT_ID = 128
_MAX_ACTION = 128
_MAX_REASON = 256
#: ``session_flow_id`` is normalized to 96 characters by the SDK, and the
#: envelope's ``trace_id`` stops at 64. One of the two has to give, and it is
#: not the envelope.
_MAX_TRACE_ID = 64

_OUTCOMES = frozenset({"success", "failure", "denied", "deferred"})
#: The envelope admits ``critical``; ``eidolon_admin``'s ``RuntimeSeverity`` does
#: not, and its projection would fail validation on one. Nothing this sink
#: observes is critical anyway — a failed turn is an error, and the Host is fine.
_SEVERITIES = frozenset({"info", "warn", "error"})


def to_envelope(
    event: _PendingEvent,
    context: ChannelEventContext,
    *,
    producer_seq: int,
    occurred_at: datetime | None = None,
) -> AuditEnvelope:
    """Build one envelope. Total: clamps rather than raising.

    ``occurred_at`` defaults to now because the sink calls its observer
    synchronously as the event happens. When a spool stands between the two,
    the time the record was written must be passed in — publishing time would
    make every latency in the file a measure of the outbox.
    """

    return AuditEnvelope(
        event_id=_event_id(event, context),
        producer=PRODUCER,
        producer_seq=max(1, int(producer_seq)),
        category=CATEGORY,
        owner_id=_clamp(context.owner_id, _MAX_OWNER_ID) or None,
        subject_type=_clamp(event.subject_type, _MAX_SUBJECT_TYPE) or "turn",
        subject_id=_clamp(event.subject_id, _MAX_SUBJECT_ID) or "unknown",
        action=_clamp(event.event_type, _MAX_ACTION) or "channel.turn.milestone",
        outcome=event.outcome if event.outcome in _OUTCOMES else "success",
        severity=event.severity if event.severity in _SEVERITIES else "info",
        reason=_clamp(event.reason, _MAX_REASON) or None,
        trace_id=_clamp(event.trace_id, _MAX_TRACE_ID) or None,
        # Always ``safe``. These payloads carry ids, phases, counts and
        # durations — never transcript text, which is the Agent's to hold. It
        # is also the only value ``eidolon_admin``'s ``PrivacyMode`` shares with
        # this contract: it spells the middle one ``summary`` where the envelope
        # says ``sensitive``, so anything else fails its projection.
        data_classification="safe",
        payload=_payload(event, context),
        occurred_at=occurred_at or datetime.now(UTC),
    )


def _event_id(event: _PendingEvent, context: ChannelEventContext) -> str:
    """The sink's id when it has one, or a stable one derived from the session.

    Turn events carry ``evt_ch_phase_*`` / ``evt_ch_mark_*`` / ``evt_ch_terminal_*``.
    Session start and end do not, and the envelope requires an id — so derive
    one that is stable for the session rather than random, because a retried
    publish of the same fact must not become a second event.
    """

    if event.event_id:
        return _clamp(event.event_id, _MAX_EVENT_ID)
    suffix = event.event_type.rsplit(".", 1)[-1] or "event"
    room = context.room_name or "room"
    return _clamp(f"evt_ch_{suffix}_{room}", _MAX_EVENT_ID)


def _payload(event: _PendingEvent, context: ChannelEventContext) -> dict[str, object]:
    """The sink's payload, with the identity the consumer expands into fields.

    ``eidolon_admin``'s ``_event`` resolves ``companion_id`` from the subject
    when the subject *is* a Companion, and otherwise from ``payload``. A turn's
    subject is the turn, so without this key every turn on the Owner's map
    belongs to no Companion.
    """

    payload = dict(event.payload)
    payload.setdefault("owner_id", context.owner_id)
    payload.setdefault("companion_id", context.companion_id)
    return payload


def _clamp(value: str | None, limit: int) -> str:
    return (value or "").strip()[:limit]


__all__ = ["CATEGORY", "PRODUCER", "to_envelope"]
