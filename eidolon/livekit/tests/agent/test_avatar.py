"""Unit tests for the digital-human video avatar wiring (hermetic, no service)."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import wave

import pytest

from eidolon.livekit.agent.runtime import resolve_avatar_requested
from eidolon.livekit.agent.full_duplex.turn_handling import build_full_duplex_turn_handling
from eidolon.livekit.avatar.service_client import StreamVideoParams, pcm16_to_wav
from eidolon.livekit.common.config import AvatarConfig, load_agent_config
from eidolon.livekit.common.config.schema import TurnPolicyConfig


# --------------------------------------------------------------- resolver
@pytest.mark.parametrize(
    "meta,expected",
    [
        ({"avatar": True}, True),
        ({"avatar": "true"}, True),
        ({"avatar": "1"}, True),
        ({"avatar": "on"}, True),
        ({"avatar": False}, False),
        ({"avatar": "no"}, False),
        ({"interaction_mode": "full_duplex"}, False),
        ({}, False),
    ],
)
def test_resolve_avatar_requested_dict(meta, expected):
    assert resolve_avatar_requested(meta) is expected


def test_resolve_avatar_requested_json_string_and_garbage():
    assert resolve_avatar_requested(json.dumps({"avatar": True})) is True
    assert resolve_avatar_requested("not-json") is False
    assert resolve_avatar_requested(None) is False


# ------------------------------------------------------------- turn handling
def test_turn_handling_avatar_mode_disables_resume():
    th = build_full_duplex_turn_handling(
        turn_policy=TurnPolicyConfig(),
        allow_interruptions=True,
        false_interruption_timeout=6.0,
        avatar_mode=True,
    )
    # DataStreamAudioOutput can't pause → framework resume must be off.
    assert th["interruption"]["resume_false_interruption"] is False


def test_turn_handling_non_avatar_unchanged():
    th = build_full_duplex_turn_handling(
        turn_policy=TurnPolicyConfig(),
        allow_interruptions=True,
        false_interruption_timeout=6.0,
        avatar_mode=False,
    )
    # channel-owned (non-adaptive) mode does not force resume_false_interruption.
    assert "resume_false_interruption" not in th["interruption"]


# --------------------------------------------------------------------- config
def test_avatar_config_defaults_off():
    # The dataclass default is the "off unless explicitly configured" invariant —
    # audio-only sessions are unaffected when no avatar block is present. (The
    # loaded config may enable it via settings.yaml; that's the operator's choice.)
    default = AvatarConfig()
    assert default.enabled is False  # global kill-switch defaults off
    assert default.output_sample_rate == 24000
    # load path stays intact regardless of the enabled value
    assert isinstance(load_agent_config().avatar, AvatarConfig)


def test_device_info_shapes_resolution_and_fps():
    di = json.loads(StreamVideoParams(width=448, height=448, fps=25.0).device_info())
    assert di["max_video_resolution"] == "448x448"
    assert di["prefer_fps"] == 25.0


# ------------------------------------------------------------- pcm16 -> wav
def test_pcm16_to_wav_roundtrip():
    pcm = (b"\x01\x00" * 16000)  # 1s of 16-bit mono @ 16k
    wav = pcm16_to_wav(pcm, sample_rate=16000, num_channels=1)
    with wave.open(io.BytesIO(wav), "rb") as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.readframes(w.getnframes()) == pcm


# ------------------------------------------- decoder retiming (hermetic)
def _synth_clip(path: str, *, src_fps: int, dur: float) -> None:
    """A known clip: testsrc video @ src_fps + sine audio, muxed to mp4/h264/aac."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc=size=64x64:rate={src_fps}:duration={dur}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={dur}",
            "-c:v", "libx264", "-profile:v", "baseline", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ar", "24000", "-ac", "1", "-shortest", path,
        ],
        check=True,
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")
def test_decode_container_retimes_to_target_fps(tmp_path):
    from eidolon.livekit.avatar.decoder import decode_container
    from livekit import rtc

    clip = tmp_path / "clip.mp4"
    # Source at 10 fps, but decoder must retime to 25 fps spanning audio duration.
    _synth_clip(str(clip), src_fps=10, dur=2.0)
    frames = decode_container(clip.read_bytes(), audio_sample_rate=24000, target_fps=25.0)

    video = [f for f in frames if isinstance(f, rtc.VideoFrame)]
    audio = [f for f in frames if isinstance(f, rtc.AudioFrame)]
    assert audio, "expected decoded audio"
    dur = sum(a.samples_per_channel for a in audio) / 24000
    # Retimed to 25 fps against the true audio duration (not the 10 fps source).
    assert abs(len(video) - round(25.0 * dur)) <= 2, (len(video), dur)
    # I420 byte size is exact (tightly-packed Y+U+V).
    for v in video:
        assert len(v.data) == v.width * v.height * 3 // 2
    # Frames are interleaved by time (no long single-kind run).
    kinds = ["v" if isinstance(f, rtc.VideoFrame) else "a" for f in frames]
    run = mx = 1
    for i in range(1, len(kinds)):
        run = run + 1 if kinds[i] == kinds[i - 1] else 1
        mx = max(mx, run)
    assert mx <= 6, f"poor interleave: max run {mx}"


def test_avatar_identity_deterministic():
    from eidolon.livekit.avatar import avatar_identity_for

    assert avatar_identity_for("device-abc") == "avatar-device-abc"
    assert avatar_identity_for("r1", prefix="dh") == "dh-r1"
