"""Configuration compatibility, portable assets and bounded PCM caching."""

import shutil
import wave
from dataclasses import asdict

import pytest
import yaml

from eidolon.livekit.common.config import load_effective_config
from eidolon.livekit.common.welcome import (
    WelcomeAudio,
    parse_welcome_message,
    prepare_welcome_audio,
    welcome_audio_path,
)


@pytest.mark.parametrize("value", ["你好，可以开始了。", "", {"audio": "builtin:soft-ready"}])
def test_yaml_welcome_round_trip(tmp_path, monkeypatch, value):
    settings = tmp_path / "settings.yaml"
    settings.write_text(yaml.safe_dump({
        "behavior": {"welcome_message": value},
        "llm": {"base_url": "http://test.invalid/v1"},
    }))
    monkeypatch.setenv("EIDOLON_CHANNEL_SETTINGS_YAML", str(settings))
    monkeypatch.delenv("EIDOLON_CHANNEL_SETTINGS_OVERLAY_YAML", raising=False)
    cfg = load_effective_config()
    assert asdict(cfg.behavior)["welcome_message"] == value


def test_default_welcome_is_packaged_sound(tmp_path, monkeypatch):
    settings = tmp_path / "settings.yaml"
    settings.write_text("llm:\n  base_url: http://test.invalid/v1\n")
    monkeypatch.setenv("EIDOLON_CHANNEL_SETTINGS_YAML", str(settings))
    monkeypatch.delenv("EIDOLON_CHANNEL_SETTINGS_OVERLAY_YAML", raising=False)
    monkeypatch.chdir(tmp_path)
    assert load_effective_config().behavior.welcome_message == WelcomeAudio()
    pcm = prepare_welcome_audio(WelcomeAudio(), sample_rate=16000)
    assert len(pcm) == 7040 * 2  # The selected 440 ms cue, resampled from 24 kHz.


def test_custom_audio_is_relative_to_settings_not_cwd(tmp_path, monkeypatch):
    settings_dir = tmp_path / "config"
    settings_dir.mkdir()
    sound = settings_dir / "cue.wav"
    shutil.copyfile(welcome_audio_path(WelcomeAudio()), sound)
    monkeypatch.chdir(tmp_path)
    welcome = parse_welcome_message({"audio": "cue.wav"}, base_dir=settings_dir)
    assert welcome == WelcomeAudio(str(sound))
    first = prepare_welcome_audio(welcome, sample_rate=16000)
    assert prepare_welcome_audio(welcome, sample_rate=16000) is first
    with wave.open(str(sound), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * 1600)
    assert prepare_welcome_audio(welcome, sample_rate=16000) == b"\0\0" * 1600


@pytest.mark.parametrize("value", [
    False, 123, [], {"audio": ""}, {"audio": 5}, {"audio": "builtin:missing"},
    {"audio": "https://example.com/cue.wav"}, {"audio": "builtin:soft-ready", "text": "hi"},
])
def test_bad_welcome_config_is_rejected(tmp_path, value):
    with pytest.raises(ValueError):
        parse_welcome_message(value, base_dir=tmp_path)


def test_missing_audio_is_rejected_before_session_entry(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_welcome_message({"audio": "missing.wav"}, base_dir=tmp_path)


@pytest.mark.parametrize("channels,width,rate,count", [
    (2, 2, 16000, 100), (1, 1, 16000, 100), (1, 2, 96000, 100),
    (1, 2, 16000, 0), (1, 2, 16000, 160001),
])
def test_unsupported_wav_is_rejected(tmp_path, channels, width, rate, count):
    path = tmp_path / "cue.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setparams((channels, width, rate, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0" * count * channels * width)
    with pytest.raises(ValueError):
        parse_welcome_message({"audio": str(path)}, base_dir=tmp_path)
