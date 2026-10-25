"""Resolve the versioned Models asset; Channel owns no model weights."""

from __future__ import annotations

from pathlib import Path


def default_campplus_model_dir() -> Path:
    """Compatibility entry for offline tools, using the shared Host setting."""
    from ...common.config import load_voiceprint_config

    return require_speaker_model_dir(load_voiceprint_config().model_dir)


def require_speaker_model_dir(value: str) -> Path:
    """Ops selects the version. No environment or packaged-weight fallback."""
    if not value or not Path(value).expanduser().is_absolute():
        raise ValueError("voiceprint.model_dir must name the deployed absolute model directory")
    return Path(value).expanduser()
