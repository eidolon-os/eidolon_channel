"""Host-side processing of a device microphone, complementing the device's own.

livekit-agents applies WebRTC AGC to every RoomIO microphone by default. A
device whose hardware already runs AGC (Reachy Mini's XVF3800) would get a
second, stacked gain stage that lifts its own echo and motor residue into the
VAD speech range. A device without AGC (ESP-BOX-3, StackChan) still needs the
Host's, so the decision follows what the device declares, never its name; an
undeclared device keeps the framework default.
"""

from __future__ import annotations

from collections.abc import Mapping

from livekit.agents.voice.room_io import AudioInputOptions

from eidolon_sdk.biz.audio_input import AUDIO_INPUT_AGC, AUDIO_INPUT_DEVICE


def inbound_audio_options(processing: Mapping[str, str]) -> AudioInputOptions:
    """RoomIO input options for a device's declared microphone processing."""
    return AudioInputOptions(auto_gain_control=processing.get(AUDIO_INPUT_AGC) != AUDIO_INPUT_DEVICE)
