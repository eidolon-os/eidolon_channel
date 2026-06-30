"""Output package import boundary tests."""

from __future__ import annotations


def test_output_package_exports_output_controller() -> None:
    from eidolon.livekit.agent.output import OutputController
    from eidolon.livekit.agent.output.controller import OutputController as Direct

    assert OutputController is Direct
