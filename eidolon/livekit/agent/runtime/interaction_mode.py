"""Per-session interaction-mode contract (plan Phase 5).

The device declares its capability to hub via the ``X-Device-Interaction-Mode``
header; hub stamps the resolved mode into the LiveKit token's
``participant_metadata`` (see ``eidolon_hub`` ``.../system/config.py``). Channel
reads it here — once per session, from the joined participant — and derives a
per-session turn policy:

  - ``half_duplex`` → push-to-talk appliance: the mic is closed during playback
    and the turn boundary is explicit (button release / tap-to-stop). The agent
    must NOT barge in or run attention/evidence interrupt guessing, so we force
    ``allow_interruptions=False`` + ``attention.enabled=False``.
  - ``full_duplex`` → open mic with hardware AEC: the current behaviour
    (``allow_interruptions=True`` + attention soft-interrupt + content echo gate).

Defense default (plan §1): missing / unparseable / unknown metadata degrades to
``half_duplex`` — the safe mode that never barges in. This lets channel run the
contract end-to-end before the device + hub halves ship.

Relationship to the packet-level ``client_audio_state.input_mode == "ptt"``
signal (commit 9642a0c): token metadata is the SESSION-LEVEL authority for the
two config knobs above; the packet signal remains the RUNTIME signal for the
fine-grained PTT behaviours (no-defer commit, echo-suppression bypass, no
idle-disconnect). For a half_duplex device the two agree — it sends ``ptt``
packets — and they act on orthogonal concerns, so they never fight.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import Any

from eidolon.livekit.common.config.schema import TurnPolicyConfig

logger = logging.getLogger("agent.interaction_mode")

INTERACTION_MODE_HALF_DUPLEX = "half_duplex"
INTERACTION_MODE_FULL_DUPLEX = "full_duplex"
_VALID_MODES = frozenset(
    {INTERACTION_MODE_HALF_DUPLEX, INTERACTION_MODE_FULL_DUPLEX}
)


def resolve_interaction_mode(
    raw_metadata: str | dict[str, Any] | None,
    *,
    default: str = INTERACTION_MODE_HALF_DUPLEX,
) -> str:
    """Parse ``interaction_mode`` out of a participant's metadata.

    ``raw_metadata`` is the LiveKit ``participant.metadata`` — a JSON string
    (hub writes a dict) or an already-parsed dict. Anything missing,
    unparseable, or holding an unknown value degrades to ``default``.
    """
    meta: dict[str, Any] | None = None
    if isinstance(raw_metadata, dict):
        meta = raw_metadata
    elif isinstance(raw_metadata, str) and raw_metadata.strip():
        try:
            parsed = json.loads(raw_metadata)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, dict):
            meta = parsed
    if not meta:
        return default
    candidate = str(meta.get("interaction_mode") or "").strip().lower()
    return candidate if candidate in _VALID_MODES else default


def apply_interaction_mode(
    *,
    turn_policy: TurnPolicyConfig,
    allow_interruptions: bool,
    interaction_mode: str,
) -> tuple[TurnPolicyConfig, bool]:
    """Derive the per-session ``(turn_policy, allow_interruptions)`` for a mode.

    ``full_duplex`` leaves the configured policy untouched. ``half_duplex``
    disables barge-in entirely: framework interruption off + attention
    admission off (no evidence-gate / interrupt guessing). Returns fresh
    values via ``dataclasses.replace``; never mutates the shared global config
    (the dataclasses are frozen anyway).
    """
    if interaction_mode == INTERACTION_MODE_HALF_DUPLEX:
        half_policy = dataclasses.replace(
            turn_policy,
            attention=dataclasses.replace(turn_policy.attention, enabled=False),
        )
        return half_policy, False
    return turn_policy, allow_interruptions
