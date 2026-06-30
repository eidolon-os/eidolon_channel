"""Bundled speaker-verification model paths."""

from __future__ import annotations

from pathlib import Path


def default_campplus_model_dir() -> Path:
    """Return the bundled 3D-Speaker CAM++ model directory."""

    return (
        Path(__file__).resolve().parent
        / "resources"
        / "3dspeaker"
        / "campplus_zh_16k_common"
    )
