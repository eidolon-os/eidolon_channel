"""Pin the shape contract with admin's ``/api/resolve`` envelope.

Regression hunt 2026-06-03: admin's response is wrapped
``{"context": {tenant_id, user_id, ...}}`` (ResolveUserResponse /
ResolveDeviceResponse) but ``ResolvedContext.from_json`` was reading
flat fields, producing an all-blank context. JWT then got signed with
``user_id=""``, agent's memory.search("") returned no hits, every
recall over the companion path looked like "the AI forgot my data".

These tests lock the contract: ``from_json`` must accept both the
envelope shape (the truth admin emits today) AND a flat dict (so future
internal refactors don't silently break the wrapper handling).
"""

from __future__ import annotations

from eidolon_sdk.biz.admin import ResolvedContext


def _expected_manson() -> dict:
    return {
        "tenant_id": "default",
        "user_id": "manson",
        "agent_id": "ag_5f3184c6b9ba",
        "template_id": "caretaker_jiezhi",
        "template_revision": 1,
        "agent_runtime_url": "",
        "memory_mcp_url": "http://127.0.0.1:8031/mcp",
        "soul_preview": "metadata:...",
        "device_id": None,
    }


def test_from_json_unwraps_admin_envelope() -> None:
    """admin's real payload is ``{"context": {...}}``. from_json must
    reach into ``context`` instead of pulling fields off the envelope
    root — pulling off the root gave us every field == "" and broke
    every memory recall for companion-path sessions."""
    payload = {"context": _expected_manson()}
    ctx = ResolvedContext.from_json(payload)
    assert ctx.tenant_id == "default"
    assert ctx.user_id == "manson"
    assert ctx.agent_id == "ag_5f3184c6b9ba"
    assert ctx.template_id == "caretaker_jiezhi"
    assert ctx.memory_mcp_url == "http://127.0.0.1:8031/mcp"
    assert ctx.device_id is None


def test_from_json_accepts_flat_shape() -> None:
    """Backwards-compatible: if a caller (or future admin variant) ever
    hands us a flat dict, from_json should still build the right
    context. Cheap insurance against the next envelope reshape."""
    ctx = ResolvedContext.from_json(_expected_manson())
    assert ctx.user_id == "manson"
    assert ctx.tenant_id == "default"


def test_from_json_envelope_with_blank_fields_does_not_silently_pass() -> None:
    """If the envelope unwraps but the inner fields are blank, the
    resulting context HAS empty strings. We don't raise here — callers
    decide what to do with empty fields — but we lock the behavior so
    a future change can't quietly start succeeding with empties again
    without someone updating this test.
    """
    ctx = ResolvedContext.from_json({"context": {}})
    assert ctx.user_id == ""
    assert ctx.tenant_id == ""
    assert ctx.agent_id == ""
    assert ctx.memory_mcp_url == ""
    assert ctx.template_id is None


def test_envelope_takes_precedence_over_root() -> None:
    """If somehow both root and nested fields are present (legacy
    shape during a contract migration), the envelope wins — matching
    admin's actual ResolveUserResponse where root is reserved for the
    envelope and never carries flat fields."""
    payload = {
        # Root carries old/stale flat shape...
        "tenant_id": "stale",
        "user_id": "stale-user",
        # ...but envelope is the truth.
        "context": _expected_manson(),
    }
    ctx = ResolvedContext.from_json(payload)
    assert ctx.user_id == "manson"
    assert ctx.tenant_id == "default"
