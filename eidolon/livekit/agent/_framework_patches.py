# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Centralised monkey-patches for ``livekit-agents`` internal state.

============================================================================
PURPOSE
============================================================================

ALL code that touches livekit-agents *internal* (underscore-prefixed) APIs
MUST live in this module. The rest of the codebase may only call the
``apply_*`` helper functions defined here.

This restriction exists because internal APIs may break on SDK upgrade
without notice. Centralising them gives us:

  1. **One audit checklist** — when the SDK is upgraded, this file is the
     ONE place to re-validate. ``grep`` for ``_safe_disable_*`` shouldn't
     surface anything outside this module.

  2. **Clear documentation** — each patch explains:
        WHY public API is insufficient
        WHICH framework files/lines we're touching
        WHEN this was last validated
     so the next person upgrading knows exactly what to verify.

  3. **Defensive fallback** — each patch checks attribute existence
     before mutating. If the framework refactored away the attribute,
     the patch logs a loud WARNING and continues; we never silent-fail.

  4. **Version surveillance** — :func:`check_framework_version` logs at
     startup if the framework version isn't on our tested-against list.
     Untested versions are not blocked, but operators are alerted to
     test for regressions in the cancel/interrupt path.

============================================================================
WHEN UPGRADING ``livekit-agents``
============================================================================

1. Read the changelog of livekit-agents between the old version and new.
   Look for changes to ``voice/agent_activity.py``, ``voice/agent_session.py``,
   ``voice/turn.py``.

2. For each ``apply_*`` function in this module, manually verify:
     a. The framework attributes/methods named in the docstring still exist.
     b. The behaviour the patch alters still works the way the docstring claims.
     c. Whether the framework now offers a public API that obviates the patch
        (in which case: delete the patch + the call site, prefer public API).

3. Add the new version to ``TESTED_VERSIONS``.

4. Run the full test suite + at least one production smoke test (real LiveKit
   room + cancel test) before merging.

============================================================================
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("agent.framework_patches")


# ---------------------------------------------------------------------------
# Version surveillance
# ---------------------------------------------------------------------------

# livekit-agents versions whose internal APIs we have explicitly validated
# the patches in this module against. Add a new entry only after running the
# full upgrade audit (see module docstring above).
#
# Ordered newest-first for readability.
TESTED_VERSIONS: tuple[str, ...] = (
    "1.6.4",
    "1.5.8",
    "1.5.6",
)


def check_framework_version() -> None:
    """Log a warning at startup if running on an untested framework version.

    Non-fatal: untested versions still run, but operators are alerted that
    the cancel/interrupt path may behave unexpectedly until the patches in
    this module are revalidated.

    Call this once during agent server startup, NOT at module import time
    (so log routing is configured first).
    """
    try:
        import livekit.agents as _lk_agents
        current = getattr(_lk_agents, "__version__", "unknown")
    except Exception:
        logger.warning(
            "[framework_patches] could not detect livekit-agents version; "
            "proceeding without compatibility check"
        )
        return

    if current in TESTED_VERSIONS:
        logger.info(
            "[framework_patches] livekit-agents %s — internal-API patches "
            "validated against this version",
            current,
        )
        return

    logger.warning(
        "[framework_patches] livekit-agents %s has NOT been validated "
        "against our internal-API patches. Tested versions: %s. "
        "If the agent ignores user interrupts or has other unexpected "
        "behaviour, suspect a framework refactor that broke the patches in "
        "%s. See module docstring for upgrade audit checklist.",
        current, TESTED_VERSIONS, __name__,
    )


# ---------------------------------------------------------------------------
# Patch: disable framework's audio-activity-driven auto-interrupt path
# ---------------------------------------------------------------------------


def disable_audio_activity_interruption(session: Any) -> None:
    """Disable framework's automatic VAD/STT-driven interrupt path.

    What we want:
      - Framework should NOT auto-interrupt the agent based on raw VAD or
        STT events. Eidolon's interruption owner
        (:class:`InterruptionOrchestrator`, ``TurnPolicyRuntime``, and the
        attention/effect handlers) handles interruption decisions with
        multi-signal fusion (VAD, transcript, hard-stop intent,
        backchannel/echo guards, client state, conversation phase, and EOT
        scoring).
      - Framework should still call our ``eot_model.predict_end_of_turn``
        for endpointing (deciding when user turn ends).
      - Framework should still respect manual ``session.interrupt()`` calls
        from Eidolon's owner.

    Why no public API works:

      * ``turn_handling.interruption.enabled = False``
        → too broad: also blocks our own ``session.interrupt()``.
      * ``turn_handling.turn_detection = "manual"``
        → too narrow: framework also stops calling our custom turn-detector
        ``predict_end_of_turn`` for endpointing.
      * ``turn_handling.interruption.mode = "vad"`` / ``"adaptive"``
        → these tune *how* auto-interrupt fires, not whether it does.

    What this patch does:

      Sets BOTH internal flags on the AgentActivity:

      * ``_interruption_by_audio_activity_enabled``  (runtime flag, line 202)
        — gates ``_interrupt_by_audio_activity()`` call (line 1575).
      * ``_default_interruption_by_audio_activity_enabled`` (default, line 207)
        — value the runtime flag is restored to when framework calls
        ``_restore_interruption_by_audio_activity()`` (line 3451) on agent
        state transitions.

      Patching only the runtime flag is insufficient: framework restores it
      to default on every agent state change, requiring constant re-patching
      (the old approach in ``_on_agent_state_changed``). Patching BOTH means
      the disable sticks.

    Targeted file: ``livekit/agents/voice/agent_activity.py``
    Targeted lines (1.6.4): 246, 250, 1781-1792, 3926-3932
      Notes: 1.6.4 also keeps adaptive interruption/backchannel logic in the
      framework; ``turn.InterruptionOptions.backchannel_boundary`` now defaults
      to ``(1.0, 1.0)`` in the installed source. That matters for native
      LiveKit owner A/B profiles, but this patch is still required for the
      Eidolon-owner hybrid profile.
    Targeted lines (1.5.6): 202, 207, 1565-1576, 3450-3453

    Targeted attr on AgentSession (1.5.6): ``_activity`` (was ``_agent_activity``
    in earlier prerelease snapshots). We try the canonical name first then the
    legacy one — if neither exists, log a loud WARNING and bail.

    Last validated: livekit-agents 1.6.4 (2026-06-30), 1.5.8, 1.5.6

    Idempotent — safe to call multiple times.
    """
    # Try canonical name first (1.5.6+), then legacy fallback. Some early
    # 1.5.x prereleases used ``_agent_activity``; current GA uses ``_activity``.
    activity = getattr(session, "_activity", None) or getattr(session, "_agent_activity", None)
    if activity is None:
        logger.warning(
            "[framework_patches] session has neither _activity nor _agent_activity — "
            "framework refactor or version mismatch. Auto-interrupt path "
            "will REMAIN ACTIVE and may conflict with Eidolon's interruption owner. "
            "Re-validate ``disable_audio_activity_interruption`` in "
            "%s.",
            __name__,
        )
        return

    runtime_attr = "_interruption_by_audio_activity_enabled"
    default_attr = "_default_interruption_by_audio_activity_enabled"

    runtime_present = hasattr(activity, runtime_attr)
    default_present = hasattr(activity, default_attr)

    if not runtime_present and not default_present:
        logger.warning(
            "[framework_patches] AgentActivity is missing both "
            "%s and %s — framework refactor. "
            "Auto-interrupt patch SKIPPED. Re-validate this patch in %s.",
            runtime_attr, default_attr, __name__,
        )
        return

    if runtime_present:
        setattr(activity, runtime_attr, False)
    else:
        logger.warning(
            "[framework_patches] missing AgentActivity.%s — partial patch only. "
            "Re-validate.", runtime_attr,
        )

    if default_present:
        setattr(activity, default_attr, False)
    else:
        logger.warning(
            "[framework_patches] missing AgentActivity.%s — framework will "
            "re-enable auto-interrupt on each agent state transition. "
            "Re-validate this patch in %s.",
            default_attr, __name__,
        )

    logger.debug(
        "[framework_patches] disabled audio-activity auto-interrupt "
        "(runtime=%s default=%s)",
        getattr(activity, runtime_attr, "?"),
        getattr(activity, default_attr, "?"),
    )


# ---------------------------------------------------------------------------
# Patch registry — used by tests and docs
# ---------------------------------------------------------------------------

PATCHES_APPLIED: tuple[tuple[str, str], ...] = (
    (
        "disable_audio_activity_interruption",
        "Disables livekit-agents' built-in VAD/STT-driven auto-interrupt "
        "so Eidolon's InterruptionOrchestrator/turn policy is the sole authority "
        "on interruption decisions.",
    ),
)
"""(name, one-line summary) for each patch in this module.

Tests can iterate this list to verify all patches still apply on the
current framework version. ``ARCHITECTURE.md`` can render it as the
"framework internal-API surface" section.
"""
