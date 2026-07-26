"""Phase 5: per-session interaction-mode contract.

Covers the two pure functions channel uses to turn the device-declared mode
(carried in the LiveKit participant metadata that hub stamps) into a
per-session turn policy:

  - ``resolve_interaction_mode`` — parse + defense default.
  - ``apply_interaction_mode`` — half_duplex disables barge-in; full_duplex
    is the unchanged status quo.
"""

from __future__ import annotations

import json

import pytest

from eidolon_sdk.biz.contracts import (
    INTERACTION_MODE_FULL_DUPLEX,
    INTERACTION_MODE_HALF_DUPLEX,
    SESSION_END_IDLE_NORMAL,
    SESSION_END_PROACTIVE_DONE,
    SESSION_INTENT_PRESENCE,
    SESSION_INTENT_PROACTIVE,
    SESSION_INTENT_USER_INITIATED,
)

from eidolon.livekit.agent.runtime import (
    apply_interaction_mode,
    resolve_interaction_mode,
    resolve_session_intent,
)
from eidolon.livekit.agent.runtime.interaction_mode import resolve_idle_policy
from eidolon.livekit.common.config.schema import TurnPolicyConfig


# ── resolve_interaction_mode ────────────────────────────────────────────


def test_resolve_from_json_string():
    raw = json.dumps({"kind": "device", "interaction_mode": "full_duplex"})
    assert resolve_interaction_mode(raw) == INTERACTION_MODE_FULL_DUPLEX


def test_resolve_from_dict():
    meta = {"kind": "device", "interaction_mode": "half_duplex"}
    assert resolve_interaction_mode(meta) == INTERACTION_MODE_HALF_DUPLEX


@pytest.mark.parametrize(
    "raw",
    [None, "", "   ", "not-json", "[1,2,3]", json.dumps({"kind": "device"})],
)
def test_resolve_missing_or_unparseable_defaults_half(raw):
    # Defense default (plan §1): anything we can't read → safe half_duplex.
    assert resolve_interaction_mode(raw) == INTERACTION_MODE_HALF_DUPLEX


def test_resolve_unknown_value_defaults_half():
    raw = json.dumps({"interaction_mode": "duplexish"})
    assert resolve_interaction_mode(raw) == INTERACTION_MODE_HALF_DUPLEX


def test_resolve_is_case_insensitive():
    raw = json.dumps({"interaction_mode": "FULL_DUPLEX"})
    assert resolve_interaction_mode(raw) == INTERACTION_MODE_FULL_DUPLEX


def test_resolve_explicit_default_override():
    # Web caller passes default=full_duplex.
    assert (
        resolve_interaction_mode(None, default=INTERACTION_MODE_FULL_DUPLEX)
        == INTERACTION_MODE_FULL_DUPLEX
    )


# ── resolve_session_intent ──────────────────────────────────────────────


def test_resolve_intent_from_metadata():
    raw = json.dumps({"interaction_mode": "half_duplex", "session_intent": "proactive_initiated"})
    assert resolve_session_intent(raw) == SESSION_INTENT_PROACTIVE


def test_resolve_presence_intent_from_metadata():
    raw = json.dumps({"session_intent": "presence_initiated"})
    assert resolve_session_intent(raw) == SESSION_INTENT_PRESENCE


@pytest.mark.parametrize(
    "raw",
    [None, "", "not-json", json.dumps({"interaction_mode": "half_duplex"}), json.dumps({"session_intent": "bogus"})],
)
def test_resolve_intent_defaults_user_initiated(raw):
    # Missing / unparseable / unknown → safe user_initiated (today nothing
    # stamps proactive; Phase 3 wires the wake path).
    assert resolve_session_intent(raw) == SESSION_INTENT_USER_INITIATED


def test_resolve_intent_is_case_insensitive():
    raw = json.dumps({"session_intent": "PROACTIVE_INITIATED"})
    assert resolve_session_intent(raw) == SESSION_INTENT_PROACTIVE


def test_mode_and_intent_resolve_from_one_metadata():
    # The session-metadata bus: one participant.metadata read yields both.
    meta = {"interaction_mode": "full_duplex", "session_intent": "proactive_initiated"}
    assert resolve_interaction_mode(meta) == INTERACTION_MODE_FULL_DUPLEX
    assert resolve_session_intent(meta) == SESSION_INTENT_PROACTIVE


# ── apply_interaction_mode ──────────────────────────────────────────────


def test_half_duplex_disables_barge_in():
    """The core 'half doesn't barge in' contract: framework interruption off
    AND attention guessing off."""
    base = TurnPolicyConfig()
    assert base.attention.enabled is True  # guard the precondition

    policy, allow = apply_interaction_mode(
        turn_policy=base,
        allow_interruptions=True,
        interaction_mode=INTERACTION_MODE_HALF_DUPLEX,
    )
    assert allow is False
    assert policy.attention.enabled is False
    # Only attention.enabled flips; the rest of the policy is preserved.
    assert policy.attention.client_state_max_age_ms == base.attention.client_state_max_age_ms
    assert policy.eot == base.eot
    assert policy.interrupt == base.interrupt


def test_full_duplex_is_status_quo():
    base = TurnPolicyConfig()
    policy, allow = apply_interaction_mode(
        turn_policy=base,
        allow_interruptions=True,
        interaction_mode=INTERACTION_MODE_FULL_DUPLEX,
    )
    assert allow is True
    assert policy is base  # unchanged object — no per-session override


def test_apply_does_not_mutate_input():
    base = TurnPolicyConfig()
    apply_interaction_mode(
        turn_policy=base,
        allow_interruptions=True,
        interaction_mode=INTERACTION_MODE_HALF_DUPLEX,
    )
    # Global config object is untouched (frozen dataclass + replace).
    assert base.attention.enabled is True


# ── resolve_idle_policy (centralized intent→idle mapping) ────────────────


def test_idle_policy_user_initiated():
    idle = TurnPolicyConfig().idle
    policy = resolve_idle_policy(
        session_intent=SESSION_INTENT_USER_INITIATED, idle_config=idle
    )
    assert policy.timeout_sec == idle.disconnect_after_idle_ms / 1000.0
    assert policy.end_reason == SESSION_END_IDLE_NORMAL


def test_idle_policy_proactive_is_short():
    idle = TurnPolicyConfig().idle
    policy = resolve_idle_policy(
        session_intent=SESSION_INTENT_PROACTIVE, idle_config=idle
    )
    assert policy.timeout_sec == idle.proactive_disconnect_after_idle_ms / 1000.0
    assert policy.end_reason == SESSION_END_PROACTIVE_DONE
    # Proactive window is strictly shorter than a user session's.
    assert (
        idle.proactive_disconnect_after_idle_ms < idle.disconnect_after_idle_ms
    )


def test_idle_policy_presence_is_bounded_and_returns_to_normal_standby():
    idle = TurnPolicyConfig().idle
    policy = resolve_idle_policy(
        session_intent=SESSION_INTENT_PRESENCE, idle_config=idle
    )
    assert policy.timeout_sec == idle.presence_disconnect_after_idle_ms / 1000.0
    assert policy.end_reason == SESSION_END_IDLE_NORMAL
    assert idle.presence_disconnect_after_idle_ms < idle.disconnect_after_idle_ms


def test_idle_policy_unknown_intent_defaults_user_like():
    # Defense default: anything that isn't proactive behaves like a user session
    # (long window) — never the aggressive short teardown.
    idle = TurnPolicyConfig().idle
    policy = resolve_idle_policy(session_intent="bogus", idle_config=idle)
    assert policy.end_reason == SESSION_END_IDLE_NORMAL
