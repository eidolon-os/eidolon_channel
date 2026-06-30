"""Integration package import boundary tests."""

from __future__ import annotations


def test_integration_package_exports_client_audio_state() -> None:
    from eidolon.livekit.agent.integration import ClientAudioState
    from eidolon.livekit.agent.integration.client_audio_state import (
        ClientAudioState as Direct,
    )

    assert ClientAudioState is Direct


def test_integration_package_exports_framework_patches() -> None:
    from eidolon.livekit.agent.integration import framework_patches
    from eidolon.livekit.agent.integration.framework_patches import (
        disable_audio_activity_interruption,
    )

    assert framework_patches.disable_audio_activity_interruption is (
        disable_audio_activity_interruption
    )


def test_legacy_client_audio_state_path_reexports_integration_model() -> None:
    from eidolon.livekit.agent.client_audio_state import ClientAudioState as Legacy
    from eidolon.livekit.agent.integration import ClientAudioState

    assert Legacy is ClientAudioState


def test_legacy_framework_patch_path_reexports_integration_patch() -> None:
    from eidolon.livekit.agent import _framework_patches as Legacy
    from eidolon.livekit.agent.integration import framework_patches

    assert Legacy.disable_audio_activity_interruption is (
        framework_patches.disable_audio_activity_interruption
    )
