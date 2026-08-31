"""LiveKit/framework integration boundary for the channel worker.

This package holds code that touches external runtime contracts directly:
LiveKit public events/options, LiveKit data-channel payloads, and other boundary
adapters. Core turn policy and session logic should import structured types
from here instead of parsing framework payloads inline.
"""

from .client_audio_state import ClientAudioState, parse_client_audio_state

__all__ = [
    "ClientAudioState",
    "parse_client_audio_state",
]
