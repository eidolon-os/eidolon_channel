"""Integration package import boundary tests."""

from __future__ import annotations


def test_integration_package_exports_client_audio_state() -> None:
    from eidolon.livekit.agent.integration import ClientAudioState
    from eidolon.livekit.agent.integration.client_audio_state import (
        ClientAudioState as Direct,
    )

    assert ClientAudioState is Direct


def test_integration_package_has_no_framework_private_patch_exports() -> None:
    from eidolon.livekit.agent import integration

    assert "framework_patches" not in integration.__all__
    assert "disable_audio_activity_interruption" not in integration.__all__
