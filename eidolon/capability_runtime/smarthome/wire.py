"""The ``eidolon.control`` envelope a panel payload travels in.

Snapshots and deltas are best-effort: a lost one shows up at the panel as a
revision or seq it cannot apply, and the panel asks for a fresh snapshot. So
they go fire-and-forget, never through a receipt-and-retry path.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from eidolon_sdk.biz.control.protocol import build_command_envelope
from eidolon_sdk.biz.smarthome import CAPABILITY_VERSION, OP_DELTA, OP_RESULT, OP_SNAPSHOT

PANEL_OPS = frozenset({OP_SNAPSHOT, OP_DELTA, OP_RESULT})
PANEL_COMMAND_TTL_MS = 10_000
SOURCE_ID = "channel_provider"


def panel_command(
    *,
    device_ref: str,
    op: str,
    payload: dict[str, Any],
    command_id: str | None = None,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Wrap one already-validated panel payload for the panel at ``device_ref``."""
    if op not in PANEL_OPS:
        raise ValueError(f"not a smart home panel op: {op!r}")
    return build_command_envelope(
        command_id=command_id or f"{op}:{uuid4().hex}",
        device_id=device_ref,
        payload=payload,
        op=op,
        capability_version=CAPABILITY_VERSION,
        ttl_ms=PANEL_COMMAND_TTL_MS,
        qos="fire_and_forget",
        src_type="channel",
        src_id=SOURCE_ID,
        created_at=created_at,
    )
