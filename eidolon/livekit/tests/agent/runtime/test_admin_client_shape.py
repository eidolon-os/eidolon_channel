"""Pin the shape contract with admin's ``/api/resolve`` envelope.

These tests lock the contract: ``from_resolve_response`` accepts only the admin
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
        "schema_version": "eidolon.persona_genome",
        "genome_hash": "pg_contract",
        "realizer_version": "eidolon.persona_realizer",
        "device_id": "dev-1",
    }


def test_from_resolve_response_unwraps_admin_envelope() -> None:
    """admin's real payload is ``{"context": {...}}``. from_json must
    reach into ``context`` instead of pulling fields off the envelope
    root — pulling off the root gave us every field == "" and broke
    every memory recall for companion-path sessions."""
    payload = {"context": _expected_manson()}
    ctx = ResolvedContext.from_resolve_response(payload)
    assert ctx.owner_id == "owner-1"
    assert ctx.companion_id == "companion-1"
    assert ctx.memory_realm_id == "realm-1"
    assert ctx.genome_id == "genome-1"
    assert ctx.device_id == "dev-1"


def test_from_resolve_response_rejects_flat_shape() -> None:
    with pytest.raises(ValueError, match="missing context"):
        ResolvedContext.from_resolve_response(_expected_manson())


def test_from_resolve_response_rejects_blank_runtime_identity() -> None:
    with pytest.raises(ValueError):
        ResolvedContext.from_resolve_response({"context": {}})


def test_envelope_takes_precedence_over_root() -> None:
    """If root fields are present, the envelope remains the only source."""
    payload = {
        # Root carries old/stale flat shape...
        "owner_id": "stale",
        "companion_id": "stale-companion",
        # ...but envelope is the truth.
        "context": _expected_manson(),
    }
    ctx = ResolvedContext.from_resolve_response(payload)
    assert ctx.owner_id == "owner-1"
    assert ctx.companion_id == "companion-1"
