"""Host-side processing of a device microphone, complementing the device's own.

livekit-agents applies WebRTC AGC to every RoomIO microphone by default. On a
device that already runs AGC (Reachy Mini's XVF3800, phone and browser WebRTC)
that is a second, stacked gain stage: it lifted Reachy's own echo and motor
residue by 10–18 dB into the VAD speech range and paused playback mid-sentence
(eidolon_client_reachy_mini docs/reports/motion-vs-native-acoustic-20261105).
A device without AGC (ESP-BOX-3, StackChan) still needs the Host's, so the
decision follows what the device declares, never its name.
"""

from __future__ import annotations

from collections.abc import Mapping

from livekit.agents.voice.room_io import AudioInputOptions

from eidolon_sdk.biz.audio_input import AUDIO_INPUT_DEVICE


def inbound_audio_options(processing: Mapping[str, str]) -> AudioInputOptions:
    """RoomIO input options for a device's declared microphone processing."""
    return AudioInputOptions(auto_gain_control=processing.get("agc") != AUDIO_INPUT_DEVICE)
