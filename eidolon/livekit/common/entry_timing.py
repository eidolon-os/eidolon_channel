"""How long getting into a conversation took, and where the time went.

The Provider stamps when it heard a device ask, and passes on how long the
device says it took to ask -- the one part only the device can measure. The job
that serves the conversation reads both back from its dispatch and logs one line
when it confirms the start. Provider and worker run on the same Host, so their
wall clocks are one clock.

Measurement only: nothing is granted or refused on these numbers.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: Dispatch metadata key; travels Provider to agent, never to a participant.
DISPATCH_ENTRY_FIELD = "entry"
#: A device that reports more than this has not measured one press.
ACTION_TO_SEND_MS_LIMIT = 60_000


def action_to_send_ms(raw: object) -> int | None:
    """The device's figure if it is one; anything else is dropped, not refused."""

    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    if raw != raw or not 0 <= raw <= ACTION_TO_SEND_MS_LIMIT:
        return None
    return int(raw)


def dispatch_entry(*, requested_at_ms: int, action_to_send_ms: int | None) -> dict[str, int]:
    entry = {"requested_at_ms": requested_at_ms}
    if action_to_send_ms is not None:
        entry["action_to_send_ms"] = action_to_send_ms
    return entry


def entry_record(
    metadata: Mapping[str, Any], *, job_started_at: float, warmed_at: float | None, now: float
) -> dict[str, Any]:
    """One conversation's way in, in milliseconds.

    ``warm_process`` says whether the job process was ready before the request
    arrived. When it was not, the conversation waited for a process to be built
    for it -- the cold path a second conversation took on 11-01 (8.6 s).
    """

    record: dict[str, Any] = {"job_to_started_ms": round((now - job_started_at) * 1000)}
    entry = metadata.get(DISPATCH_ENTRY_FIELD)
    requested = entry.get("requested_at_ms") if isinstance(entry, Mapping) else None
    if isinstance(requested, int) and not isinstance(requested, bool):
        requested_at = requested / 1000
        record["request_to_started_ms"] = round((now - requested_at) * 1000)
        record["request_to_job_ms"] = round((job_started_at - requested_at) * 1000)
        if warmed_at is not None:
            record["warm_process"] = warmed_at <= requested_at
        action = action_to_send_ms(entry.get("action_to_send_ms"))
        if action is not None:
            record["action_to_send_ms"] = action
            record["action_to_started_ms"] = action + record["request_to_started_ms"]
    return record


def format_entry(record: Mapping[str, Any]) -> str:
    return " ".join(
        f"{key}={str(value).lower() if isinstance(value, bool) else value}"
        for key, value in record.items()
    )
