from pathlib import Path

from eidolon.livekit.plugins.speaker_verification.model_paths import default_campplus_model_dir
from eidolon.livekit.common.config import VoiceprintConfig


def test_host_uses_versioned_models_asset_without_a_channel_copy(monkeypatch):
    monkeypatch.delenv('EIDOLON_3DSPEAKER_MODEL_DIR', raising=False)
    assert default_campplus_model_dir() == Path('/var/lib/eidolon/models/campplus-zh-16k-624dcb0f')


def test_explicit_local_model_path_is_respected(monkeypatch, tmp_path):
    monkeypatch.setenv('EIDOLON_3DSPEAKER_MODEL_DIR', str(tmp_path))
    assert default_campplus_model_dir() == tmp_path


def test_model_installation_does_not_enable_legacy_enforcement():
    assert VoiceprintConfig().enabled is False
