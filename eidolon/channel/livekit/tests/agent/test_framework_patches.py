# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for ``agent/_framework_patches.py``.

These tests verify that:
1. The current ``livekit-agents`` version is on the tested-against list,
   OR a clear warning is logged.
2. ``disable_audio_activity_interruption`` patches both the runtime
   flag and the default-value flag so the framework's restore logic
   doesn't undo our patch.
3. The patch is defensive: missing attributes log a warning rather
   than crashing.
4. The patch is idempotent: calling twice is safe.

If livekit-agents is upgraded and these tests start failing, the
upgrade audit (see ``_framework_patches.py`` module docstring) is
overdue.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from eidolon.channel.livekit.agent import _framework_patches


# ──────────────────────────────────────────────────────────────────
# Version surveillance
# ──────────────────────────────────────────────────────────────────


class TestVersionCheck:
    def test_current_framework_version_is_tested(self) -> None:
        """The version we're running against MUST be in TESTED_VERSIONS,
        else the audit checklist hasn't been run for this version."""
        import livekit.agents as _lk_agents
        current = getattr(_lk_agents, "__version__", "unknown")
        assert current in _framework_patches.TESTED_VERSIONS, (
            f"livekit-agents {current} is not in TESTED_VERSIONS "
            f"{_framework_patches.TESTED_VERSIONS}. Either: (a) downgrade to "
            f"a tested version, or (b) run the upgrade audit (see "
            f"_framework_patches.py module docstring) and add this version."
        )

    def test_check_framework_version_is_idempotent(self, caplog) -> None:
        """Repeated calls are safe — only emit a single INFO/WARNING line each."""
        with caplog.at_level(logging.INFO, logger="agent.framework_patches"):
            _framework_patches.check_framework_version()
            _framework_patches.check_framework_version()
        # Either info (tested version) or warning (untested), but each
        # invocation must log exactly one line.
        records = [
            r for r in caplog.records
            if r.name == "agent.framework_patches"
        ]
        # Exactly two records (one per call), and they should be identical.
        assert len(records) == 2
        assert records[0].getMessage() == records[1].getMessage()


# ──────────────────────────────────────────────────────────────────
# disable_audio_activity_interruption
# ──────────────────────────────────────────────────────────────────


class TestDisableAudioActivityInterruption:
    def test_patches_both_flags(self) -> None:
        """The fix that mattered: BOTH the runtime flag AND the default-
        value flag are set to False. Without patching the default,
        framework's _restore_interruption_by_audio_activity() would
        re-enable the runtime flag on every agent state transition."""
        activity = MagicMock()
        activity._interruption_by_audio_activity_enabled = True
        activity._default_interruption_by_audio_activity_enabled = True
        session = MagicMock()
        session._activity = activity

        _framework_patches.disable_audio_activity_interruption(session)

        assert activity._interruption_by_audio_activity_enabled is False
        assert activity._default_interruption_by_audio_activity_enabled is False

    def test_idempotent(self) -> None:
        """Calling the patch twice is safe — no exceptions."""
        activity = MagicMock()
        activity._interruption_by_audio_activity_enabled = True
        activity._default_interruption_by_audio_activity_enabled = True
        session = MagicMock()
        session._activity = activity

        _framework_patches.disable_audio_activity_interruption(session)
        _framework_patches.disable_audio_activity_interruption(session)
        assert activity._interruption_by_audio_activity_enabled is False
        assert activity._default_interruption_by_audio_activity_enabled is False

    def test_session_without_activity_logs_warning(
        self, caplog
    ) -> None:
        """Defensive: if framework removed _agent_activity (refactor),
        we log a loud warning rather than crash."""
        session = MagicMock(spec=[])  # no _activity attr

        with caplog.at_level(
            logging.WARNING, logger="agent.framework_patches"
        ):
            _framework_patches.disable_audio_activity_interruption(session)

        assert any(
            "neither _activity nor _agent_activity" in r.getMessage()
            for r in caplog.records
        )

    def test_missing_runtime_flag_logs_warning(self, caplog) -> None:
        """Defensive: if framework renamed _interruption_by_audio_activity_enabled,
        log warning and continue (partial patch)."""

        class FakeActivity:
            # Only the default flag exists, not the runtime one.
            _default_interruption_by_audio_activity_enabled = True

        session = MagicMock()
        session._activity = FakeActivity()

        with caplog.at_level(
            logging.WARNING, logger="agent.framework_patches"
        ):
            _framework_patches.disable_audio_activity_interruption(session)

        assert any(
            "_interruption_by_audio_activity_enabled" in r.getMessage()
            for r in caplog.records
        )

    def test_missing_default_flag_logs_warning(self, caplog) -> None:
        """Same defensive check for the default-value flag."""

        class FakeActivity:
            _interruption_by_audio_activity_enabled = True

        session = MagicMock()
        session._activity = FakeActivity()

        with caplog.at_level(
            logging.WARNING, logger="agent.framework_patches"
        ):
            _framework_patches.disable_audio_activity_interruption(session)

        assert any(
            "_default_interruption_by_audio_activity_enabled" in r.getMessage()
            for r in caplog.records
        )

    def test_both_flags_missing_skips_patch(self, caplog) -> None:
        """If neither flag exists, framework was completely refactored;
        skip patch entirely (loudly), rather than mutate spurious attrs."""

        class FakeActivity:
            pass  # no flags at all

        session = MagicMock()
        session._activity = FakeActivity()

        with caplog.at_level(
            logging.WARNING, logger="agent.framework_patches"
        ):
            _framework_patches.disable_audio_activity_interruption(session)

        assert any(
            "missing both" in r.getMessage()
            for r in caplog.records
        )


# ──────────────────────────────────────────────────────────────────
# Integration: targeted attributes still exist on real framework
# ──────────────────────────────────────────────────────────────────


class TestRealFrameworkAttributesExist:
    """Live check against the installed livekit-agents — these tests
    fail if the framework has refactored away the attributes we patch.

    If they fail on a new framework version, that's the signal to:
    1. Read the framework changelog.
    2. Re-validate or rewrite the affected patch in
       ``_framework_patches.py``.
    3. Update ``TESTED_VERSIONS``.
    """

    def test_agent_activity_class_has_patched_attributes(self) -> None:
        """The two attributes we patch must exist on AgentActivity instances.
        We instantiate via attribute name lookup on the source rather than
        instantiating (which requires a session + agent setup)."""
        from livekit.agents.voice.agent_activity import AgentActivity

        # We can't easily instantiate AgentActivity without a full
        # session, but we CAN inspect the source for attribute
        # initializations.
        import inspect
        source = inspect.getsource(AgentActivity.__init__)

        # Both attributes must appear as ``self._<flag> = ...`` in __init__
        assert "_interruption_by_audio_activity_enabled" in source, (
            "AgentActivity.__init__ no longer initializes "
            "_interruption_by_audio_activity_enabled — patch is broken. "
            "Audit _framework_patches.py."
        )
        assert "_default_interruption_by_audio_activity_enabled" in source, (
            "AgentActivity.__init__ no longer initializes "
            "_default_interruption_by_audio_activity_enabled — patch is broken. "
            "Audit _framework_patches.py."
        )

    def test_restore_method_still_exists(self) -> None:
        """The framework's ``_restore_interruption_by_audio_activity`` is
        what motivates patching the *default* flag. If the method is gone,
        we may not need to patch the default anymore — re-evaluate."""
        from livekit.agents.voice.agent_activity import AgentActivity

        assert hasattr(AgentActivity, "_restore_interruption_by_audio_activity"), (
            "Framework removed _restore_interruption_by_audio_activity — "
            "patching the default flag may no longer be needed. "
            "Re-evaluate _framework_patches.disable_audio_activity_interruption."
        )


# ──────────────────────────────────────────────────────────────────
# Patch registry
# ──────────────────────────────────────────────────────────────────


class TestPatchRegistry:
    def test_registry_lists_all_apply_functions(self) -> None:
        """Anyone adding a new ``apply_*`` patch MUST also register it
        in ``PATCHES_APPLIED`` so the registry stays the single source
        of truth for "what internal APIs we touch"."""
        public_apply_fns = [
            name for name in dir(_framework_patches)
            if name.startswith("disable_") or name.startswith("apply_")
        ]
        registered_names = {n for n, _ in _framework_patches.PATCHES_APPLIED}

        for fn in public_apply_fns:
            assert fn in registered_names, (
                f"{fn} is a patch function but not registered in "
                f"PATCHES_APPLIED. Add an entry."
            )
