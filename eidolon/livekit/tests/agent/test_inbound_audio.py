"""Host microphone processing follows what the device declares it already does."""

from __future__ import annotations

import json

import pytest

from eidolon.livekit.agent.runtime import resolve_audio_input_processing
from eidolon.livekit.agent.session.inbound_audio import inbound_audio_options


def test_declared_processing_is_read_from_the_runtime_actor_metadata():
    metadata = json.dumps({"interaction_mode": "full_duplex", "audio_input": {"aec": "device", "agc": "device"}})

    assert resolve_audio_input_processing(metadata) == {"aec": "device", "agc": "device"}


@pytest.mark.parametrize(
    "raw",
    [None, "", "not json", json.dumps({"interaction_mode": "full_duplex"}), {"audio_input": "device"}],
)
def test_missing_or_malformed_declarations_stay_undeclared(raw):
    assert resolve_audio_input_processing(raw) == {}


def test_unknown_stages_and_values_are_dropped():
    raw = {"audio_input": {"agc": "loud", "ns": "none", "beamforming": "device", "aec": ["device"]}}

    assert resolve_audio_input_processing(raw) == {"ns": "none"}


def test_host_agc_stays_on_for_an_undeclared_device():
    assert inbound_audio_options({}).auto_gain_control is True


def test_host_agc_stays_on_for_a_device_without_its_own():
    # ESP-BOX-3 and StackChan: the Host gain stage is their only normalization.
    assert inbound_audio_options({"aec": "device", "agc": "none"}).auto_gain_control is True


def test_host_agc_is_not_stacked_on_device_agc():
    # Reachy Mini (XVF3800), phone and browser WebRTC already run AGC.
    assert inbound_audio_options({"aec": "device", "agc": "device"}).auto_gain_control is False
