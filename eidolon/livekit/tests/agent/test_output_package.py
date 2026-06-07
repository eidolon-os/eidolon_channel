"""Output package import boundary tests."""

from __future__ import annotations


def test_output_package_exports_output_controller() -> None:
    from eidolon.livekit.agent.output import OutputController
    from eidolon.livekit.agent.output.controller import OutputController as Direct

    assert OutputController is Direct


def test_legacy_output_controller_path_reexports_new_output_controller() -> None:
    from eidolon.livekit.agent.output import OutputController
    from eidolon.livekit.agent.output_controller import OutputController as Legacy

    assert Legacy is OutputController


def test_legacy_filler_path_reexports_new_filler_manager() -> None:
    from eidolon.livekit.agent.filler import FillerManager as Legacy
    from eidolon.livekit.agent.output import FillerManager

    assert Legacy is FillerManager
