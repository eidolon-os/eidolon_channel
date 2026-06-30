"""Compatibility wrapper for :mod:`eidolon.livekit.agent.integration.framework_patches`."""

from __future__ import annotations

from eidolon.livekit.agent.integration.framework_patches import (
    PATCHES_APPLIED,
    TESTED_VERSIONS,
    check_framework_version,
    disable_audio_activity_interruption,
)

__all__ = [
    "PATCHES_APPLIED",
    "TESTED_VERSIONS",
    "check_framework_version",
    "disable_audio_activity_interruption",
]
