"""Pin the shape contract with admin's ``/api/resolve`` envelope.

These tests lock the contract: ``from_json`` accepts only the admin
``{"context": ...}`` envelope shape.
"""

from __future__ import annotations

import pytest

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


def test_from_json_rejects_flat_shape() -> None:
    with pytest.raises(ValueError, match="missing context"):
        ResolvedContext.from_json(_expected_manson())


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
    """If root fields are present, the envelope remains the only source."""
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
