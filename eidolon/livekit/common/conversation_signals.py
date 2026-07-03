"""Shared text-signal constants for realtime conversation control.

Canonical definitions live in ``eidolon_sdk.biz.dialogue_control`` (shared
with eidolon_agent); re-exported here under historical names.
"""

from __future__ import annotations

from eidolon_sdk.biz.dialogue_control import (
    BACKCHANNEL_WORDS,
    NOISE_LIKE_TRANSCRIPTIONS,
    REPEATED_NOISE_CHARS,
)
from eidolon_sdk.biz.dialogue_control.lexicon import BACKCHANNEL_COMPOUND_CHARS

__all__ = [
    "BACKCHANNEL_COMPOUND_CHARS",
    "BACKCHANNEL_WORDS",
    "NOISE_LIKE_TRANSCRIPTIONS",
    "REPEATED_NOISE_CHARS",
]
