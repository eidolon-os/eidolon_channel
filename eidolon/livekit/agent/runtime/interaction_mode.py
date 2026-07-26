"""Per-session interaction-mode contract (plan Phase 5).

The device declares its capability to hub via the ``X-Device-Interaction-Mode``
header; hub stamps the resolved mode into the LiveKit token's
``participant_metadata`` (see ``eidolon_hub`` ``.../system/config.py``). Channel
reads it here — once per session, from the joined participant — and derives a
per-session turn policy. ``interaction_mode`` is one of three mutually-exclusive
modes (authoritative descriptions in ``eidolon_sdk.biz.contracts``):

  - ``full_duplex`` → open mic + device hardware AEC + barge-in: the user can
    interrupt the agent. Leaves the configured policy untouched
    (``allow_interruptions=True`` + attention soft-interrupt + content echo
    gate). For devices with a validated AEC reference (e.g. esp-box-3).
  - ``half_duplex`` → auto-record after session start (no button), NO device
    AEC. The mic is closed while the agent speaks, so the turn is not
    interruptible; STT is committed via the SAME end-of-turn (EOT) judgment as
    ``full_duplex``. Barge-in is off, so we force ``allow_interruptions=False``
    + ``attention.enabled=False``. For boards without a clean AEC reference
    (e.g. m5stack-stackchan). NOTE: ``half_duplex`` is NOT push-to-talk — that
    is now the separate ``ptt`` mode.
  - ``ptt`` → push-to-talk: the mic is open only while the device button is
    held; button release is the explicit end-of-turn (mic closed otherwise).
    Same no-barge-in knobs as ``half_duplex`` (``allow_interruptions=False`` +
    ``attention.enabled=False``). For wearables / button devices
    (e.g. waveshare 2.06).

``apply_interaction_mode`` (below) derives only the ``(turn_policy,
allow_interruptions)`` knobs above; pipeline routing lives in ``server.py``
(``_use_ptt_pipeline``): ``ptt`` → ``HalfDuplexPttPipeline`` (button segment
turns), while ``half_duplex`` and ``full_duplex`` both run the streaming
``StreamingPipeline`` and differ only in barge-in.

Defense default (plan §1): missing / unparseable / unknown metadata degrades to
``half_duplex`` — a safe mode that never barges in. This lets channel run the
contract end-to-end before the device + hub halves ship.

Relationship to the packet-level ``client_audio_state.input_mode`` signal
(commit 9642a0c): token metadata is the SESSION-LEVEL authority for the config
knobs above; the packet signal remains the RUNTIME signal for the fine-grained
PTT behaviours. A ``ptt`` device sends ``input_mode="ptt"`` packets and the two
agree; ``half_duplex`` / ``full_duplex`` devices auto-record and report
``input_mode="auto"``. They act on orthogonal concerns, so they never fight.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import Any

from eidolon.livekit.common.config.schema import IdlePolicyConfig, TurnPolicyConfig
from eidolon_sdk.biz.contracts import (
    INTERACTION_MODE_HALF_DUPLEX,
    INTERACTION_MODE_PTT,
    SESSION_END_IDLE_NORMAL,
    SESSION_END_PROACTIVE_DONE,
    SESSION_INTENT_FIELD,
    SESSION_INTENT_PRESENCE,
    SESSION_INTENT_PROACTIVE,
    SESSION_INTENT_USER_INITIATED,
    VALID_INTERACTION_MODES,
    normalize_session_intent,
)

logger = logging.getLogger("agent.interaction_mode")

# Session intent (plan §3.2) — why this voice session exists. Rides the SAME
# join-metadata bus as interaction_mode (resolved once, from the same
# participant.metadata, passed as an explicit param), and is orthogonal to it:
#   - user_initiated: explicit JOIN, canned welcome, normal idle window.
#   - presence_initiated: verified owner-presence wake, canned welcome, bounded
#     no-response idle window.
#   - proactive_initiated: a report opens the session, so the canned welcome is
#     suppressed and an unanswered report uses proactive_done teardown.
# The INTENT_* / INTERACTION_MODE_* names + validity sets are sourced from
# ``eidolon_sdk.biz.contracts`` (single source) and re-exported above.


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
    return candidate if candidate in VALID_INTERACTION_MODES else default


def resolve_session_intent(
    raw_metadata: str | dict[str, Any] | None,
    *,
    default: str = SESSION_INTENT_USER_INITIATED,
) -> str:
    """Parse ``session_intent`` out of a participant's metadata.

    Same source + parsing contract as ``resolve_interaction_mode`` (one
    participant.metadata read resolves both). Anything missing, unparseable, or
    holding an unknown value degrades to ``default`` (``user_initiated``) — the
    safe assumption that a session is user-driven.
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
    return normalize_session_intent(
        str(meta.get(SESSION_INTENT_FIELD) or ""),
        default=default,
    )


def resolve_device_id(
    raw_metadata: str | dict[str, Any] | None,
) -> str | None:
    """Parse the stable ``device_id`` out of a participant's metadata.

    Same source + parsing contract as the resolvers above (hub stamps
    ``device_id`` into the device/control token metadata). Returns the id as a
    string, or ``None`` when absent/unparseable. Used to key the conversation on
    the device rather than the per-session voice room name (which now carries a
    nonce and would otherwise fragment brain context across reconnects).
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
        return None
    device_id = str(meta.get("device_id") or "").strip()
    return device_id or None


# Participant-metadata key by which a display-capable client (web/app) declares
# it wants the digital-human video avatar for this session. Same join-metadata
# bus as interaction_mode/session_intent; read once, per participant. Absent /
# false → audio-only (the safe default, so existing sessions are unaffected).
AVATAR_METADATA_KEY = "avatar"


def _coerce_metadata_dict(raw_metadata: str | dict[str, Any] | None) -> dict[str, Any] | None:
    """Parse participant.metadata (JSON string or dict) into a dict, or None."""
    if isinstance(raw_metadata, dict):
        return raw_metadata
    if isinstance(raw_metadata, str) and raw_metadata.strip():
        try:
            parsed = json.loads(raw_metadata)
        except (ValueError, TypeError):
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def resolve_avatar_requested(
    raw_metadata: str | dict[str, Any] | None,
    *,
    default: bool = False,
) -> bool:
    """Whether the joining client requested a video avatar for this session.

    Accepts a bool or a truthy string (``"true"/"1"/"yes"/"on"``). Same source +
    parsing contract as the other join-metadata resolvers. Anything missing /
    unparseable degrades to ``default`` (audio-only).
    """
    meta = _coerce_metadata_dict(raw_metadata)
    if not meta:
        return default
    value = meta.get(AVATAR_METADATA_KEY)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return default


def apply_interaction_mode(
    *,
    turn_policy: TurnPolicyConfig,
    allow_interruptions: bool,
    interaction_mode: str,
) -> tuple[TurnPolicyConfig, bool]:
    """Derive the per-session ``(turn_policy, allow_interruptions)`` for a mode.

    ``full_duplex`` leaves the configured policy untouched. ``half_duplex`` and
    ``ptt`` both disable barge-in entirely: framework interruption off +
    attention admission off (no evidence-gate / interrupt guessing). Returns
    fresh values via ``dataclasses.replace``; never mutates the shared global
    config (the dataclasses are frozen anyway).
    """
    if interaction_mode in (INTERACTION_MODE_HALF_DUPLEX, INTERACTION_MODE_PTT):
        no_barge_in_policy = dataclasses.replace(
            turn_policy,
            attention=dataclasses.replace(turn_policy.attention, enabled=False),
        )
        return no_barge_in_policy, False
    return turn_policy, allow_interruptions


@dataclasses.dataclass(frozen=True)
class IdlePolicy:
    """Per-session idle behaviour derived from session_intent (plan §3.2)."""

    timeout_sec: float
    end_reason: str


def resolve_welcome_text(
    *, session_intent: str, welcome_message: str | None
) -> str | None:
    """Map session origin to its canned opening.

    Proactive report sessions already have opening content. Explicit user and
    verified-presence sessions use the configured welcome.
    """
    if session_intent == SESSION_INTENT_PROACTIVE:
        return None
    return welcome_message or None


def resolve_idle_policy(
    *, session_intent: str, idle_config: IdlePolicyConfig
) -> IdlePolicy:
    """Map ``session_intent`` → idle window + teardown reason.

    Single source for the intent→idle decision. A presence wake gets a bounded
    answer window but ends like a normal idle conversation. A proactive report
    gets its own short window and proactive_done reason. Explicit user sessions
    retain the normal long window.
    """
    if session_intent == SESSION_INTENT_PRESENCE:
        return IdlePolicy(
            timeout_sec=idle_config.presence_disconnect_after_idle_ms / 1000.0,
            end_reason=SESSION_END_IDLE_NORMAL,
        )
    if session_intent == SESSION_INTENT_PROACTIVE:
        return IdlePolicy(
            timeout_sec=idle_config.proactive_disconnect_after_idle_ms / 1000.0,
            end_reason=SESSION_END_PROACTIVE_DONE,
        )
    return IdlePolicy(
        timeout_sec=idle_config.disconnect_after_idle_ms / 1000.0,
        end_reason=SESSION_END_IDLE_NORMAL,
    )
