"""Pin the shape contract with admin's ``/api/resolve`` envelope.

These tests lock the contract: ``from_json`` must accept both the
envelope shape (the truth admin emits today) AND a flat dict (so future
internal refactors don't silently break the wrapper handling).
"""

from __future__ import annotations

from eidolon_sdk.biz.admin import ResolvedContext


def _expected_manson() -> dict:
    return {
        "owner_id": "owner-1",
        "companion_id": "companion-1",
        "memory_realm_id": "realm-1",
        "genome_id": "genome-1",
        "device_id": "dev-1",
    }


def test_from_json_unwraps_admin_envelope() -> None:
    """admin's real payload is ``{"context": {...}}``. from_json must
    reach into ``context`` instead of pulling fields off the envelope
    root — pulling off the root gave us every field == "" and broke
    every memory recall for companion-path sessions."""
    payload = {"context": _expected_manson()}
    ctx = ResolvedContext.from_json(payload)
    assert ctx.owner_id == "owner-1"
    assert ctx.companion_id == "companion-1"
    assert ctx.memory_realm_id == "realm-1"
    assert ctx.genome_id == "genome-1"
    assert ctx.device_id == "dev-1"


def test_from_json_accepts_flat_shape() -> None:
    """Backwards-compatible: if a caller (or future admin variant) ever
    hands us a flat dict, from_json should still build the right
    context. Cheap insurance against the next envelope reshape."""
    ctx = ResolvedContext.from_json(_expected_manson())
    assert ctx.owner_id == "owner-1"
    assert ctx.companion_id == "companion-1"


def test_from_json_envelope_with_blank_fields_does_not_silently_pass() -> None:
    """If the envelope unwraps but the inner fields are blank, the
    resulting context HAS empty strings. We don't raise here — callers
    decide what to do with empty fields — but we lock the behavior so
    a future change can't quietly start succeeding with empties again
    without someone updating this test.
    """
    ctx = ResolvedContext.from_json({"context": {}})
    assert ctx.owner_id == ""
    assert ctx.companion_id == ""
    assert ctx.memory_realm_id == ""
    assert ctx.genome_id == ""
    assert ctx.device_id is None


def test_envelope_takes_precedence_over_root() -> None:
    """If somehow both root and nested fields are present (legacy
    shape during a contract migration), the envelope wins."""
    payload = {
        # Root carries old/stale flat shape...
        "owner_id": "stale",
        "companion_id": "stale-companion",
        # ...but envelope is the truth.
        "context": _expected_manson(),
    }
    ctx = ResolvedContext.from_json(payload)
    assert ctx.owner_id == "owner-1"
    assert ctx.companion_id == "companion-1"
