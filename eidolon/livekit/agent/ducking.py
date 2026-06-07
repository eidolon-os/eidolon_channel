# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""DEPRECATED — Backwards-compat shim for the legacy ``ducking`` module.

G17b (2026-05-18): the class was renamed ``DuckingMixer`` → ``OutputController``.
The implementation now lives in :mod:`eidolon.livekit.agent.output`. The new
name better describes the post-G18a semantics: it's the output-side controller
that decides whether TTS audio reaches the user, not just a volume "ducker".

This file remains for one release as an import alias so that any out-of-tree
code or stale imports keep working. New code MUST do::

    from eidolon.livekit.agent.output import OutputController

instead of::

    from eidolon.livekit.agent.ducking import DuckingMixer  # DEPRECATED
"""

from __future__ import annotations

import warnings

from eidolon.livekit.agent.output import (
    OutputController as _OutputController,
)

# Backwards-compat alias. Mark the symbol but do NOT emit a DeprecationWarning
# at import-time — DeprecationWarning is silent by default in production but
# noisy in pytest, which would create test noise without driving behavior
# change. Static analyzers should flag the symbol via this docstring.
DuckingMixer = _OutputController
"""DEPRECATED alias for :class:`OutputController`. Will be removed after
the next release. Update imports to ``from eidolon.livekit.agent.output
import OutputController``."""


def _emit_module_deprecation_warning() -> None:
    """Emit a one-shot warning if anything actually constructs DuckingMixer
    via the old path. Off by default to keep tests quiet; flip on by setting
    ``EIDOLON_WARN_DUCKING_LEGACY=1``."""
    import os
    if os.environ.get("EIDOLON_WARN_DUCKING_LEGACY"):
        warnings.warn(
            "eidolon.livekit.agent.ducking is deprecated; "
            "import from eidolon.livekit.agent.output instead",
            DeprecationWarning,
            stacklevel=3,
        )


_emit_module_deprecation_warning()


__all__ = ["DuckingMixer"]
