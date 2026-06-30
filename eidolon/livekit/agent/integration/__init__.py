"""LiveKit/framework integration boundary for the channel worker.

This package holds code that touches external runtime contracts directly:
LiveKit framework internals, LiveKit data-channel payloads, and other boundary
adapters. Core turn policy and session logic should import structured types
from here instead of parsing framework payloads inline.
"""

from . import framework_patches
from .client_audio_state import ClientAudioState, parse_client_audio_state
from .framework_patches import (
    PATCHES_APPLIED,
    TESTED_VERSIONS,
    check_framework_version,
    disable_audio_activity_interruption,
)

__all__ = [
    "ClientAudioState",
    "PATCHES_APPLIED",
    "TESTED_VERSIONS",
    "check_framework_version",
    "disable_audio_activity_interruption",
    "framework_patches",
    "parse_client_audio_state",
]
