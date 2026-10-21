"""Resolve the versioned Models asset; Channel owns no model weights."""

from __future__ import annotations

from pathlib import Path
import os


def default_campplus_model_dir() -> Path:
    """Use an explicit development path or the checksum-verified Host artifact."""
    value = os.environ.get("EIDOLON_3DSPEAKER_MODEL_DIR", "").strip()
    return Path(value).expanduser() if value else Path(
        "/var/lib/eidolon/models/campplus-zh-16k-624dcb0f"
    )
