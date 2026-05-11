# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for tests/_harness/audio.py."""

from __future__ import annotations

import math

import pytest

from eidolon.channel.livekit.tests._harness.audio import (
    DEFAULT_SAMPLE_RATE,
    SAMPLE_WIDTH_BYTES,
    frames_from_pcm,
    pcm_duration_sec,
    pcm_from_frames,
    pcm_rms,
    synth_silence,
    synth_tone,
    synth_voiced,
)


class TestSynthSilence:
    def test_length_matches_duration(self):
        # 1 second @ 16kHz mono 16-bit = 32000 bytes
        assert len(synth_silence(1.0)) == 16_000 * 2

    def test_zero_duration_zero_bytes(self):
        assert synth_silence(0.0) == b""

    def test_silence_is_zero_bytes(self):
        pcm = synth_silence(0.05)
        assert pcm == b"\x00" * len(pcm)

    def test_rms_is_zero(self):
        assert pcm_rms(synth_silence(0.1)) == 0.0


class TestSynthTone:
    def test_length_matches_duration(self):
        pcm = synth_tone(440, 0.5)
        assert len(pcm) == 16_000 * 2 // 2  # 0.5s at 16kHz mono 16-bit

    def test_amplitude_zero_yields_silence(self):
        assert pcm_rms(synth_tone(440, 0.1, amplitude=0.0)) == 0.0

    def test_amplitude_clamped_to_unit(self):
        # Amplitude > 1 should clamp, not blow up.
        pcm = synth_tone(440, 0.05, amplitude=2.0)
        # RMS of a unit-amplitude sine ≈ 1/√2 ≈ 0.707.
        assert 0.6 < pcm_rms(pcm) < 0.8

    def test_rms_scales_with_amplitude(self):
        small = pcm_rms(synth_tone(440, 0.1, amplitude=0.1))
        big = pcm_rms(synth_tone(440, 0.1, amplitude=0.9))
        assert big > 5 * small

    def test_higher_freq_does_not_change_rms(self):
        # Sine RMS depends on amplitude, not frequency.
        a = pcm_rms(synth_tone(200, 0.1, amplitude=0.5))
        b = pcm_rms(synth_tone(2000, 0.1, amplitude=0.5))
        assert math.isclose(a, b, rel_tol=0.05)


class TestSynthVoiced:
    def test_length_matches_duration(self):
        assert len(synth_voiced(0.25)) == 16_000 * 2 // 4

    def test_non_silent(self):
        # Voiced should produce non-zero RMS (carries audio energy).
        assert pcm_rms(synth_voiced(0.1)) > 0.05

    def test_below_typical_vad_amplitude(self):
        # We default to amplitude 0.4 — not full-scale, leaves headroom.
        rms = pcm_rms(synth_voiced(0.2))
        assert rms < 0.6


class TestFramesRoundTrip:
    def test_pcm_to_frames_and_back(self):
        # 60ms voiced @ 16kHz, 20ms frames → 3 frames
        pcm = synth_voiced(0.06)
        frames = list(frames_from_pcm(pcm, frame_ms=20))
        assert len(frames) == 3
        roundtrip = pcm_from_frames(frames)
        assert roundtrip == pcm

    def test_partial_trailing_frame_dropped(self):
        # 25ms PCM, 20ms frames → 1 frame (5ms remainder discarded)
        pcm = synth_voiced(0.025)
        frames = list(frames_from_pcm(pcm, frame_ms=20))
        assert len(frames) == 1

    def test_frame_metadata(self):
        frames = list(frames_from_pcm(synth_voiced(0.04), frame_ms=20))
        assert frames[0].sample_rate == DEFAULT_SAMPLE_RATE
        assert frames[0].num_channels == 1
        assert frames[0].samples_per_channel == 320  # 20ms @ 16kHz

    def test_invalid_frame_ms_raises(self):
        with pytest.raises(ValueError):
            list(frames_from_pcm(synth_voiced(0.04), frame_ms=0))


class TestPcmHelpers:
    def test_duration_for_known_length(self):
        # 32_000 bytes 16-bit mono @ 16kHz = 1 second
        assert pcm_duration_sec(b"\x00\x00" * 16_000) == pytest.approx(1.0)

    def test_duration_for_silence_matches_synth(self):
        pcm = synth_silence(0.345)
        assert pcm_duration_sec(pcm) == pytest.approx(0.345, abs=1e-3)
