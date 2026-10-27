import pytest
from eidolon.livekit.plugins.speaker_verification.model_paths import require_speaker_model_dir
from eidolon.livekit.common.config import VoiceprintConfig


def test_missing_model_path_never_guesses_a_version_or_uses_old_env(monkeypatch):
    monkeypatch.setenv("EIDOLON_3DSPEAKER_MODEL_DIR", "/stale-model")
    for value in ("", "relative/model"):
        with pytest.raises(ValueError, match="model_dir"):
            require_speaker_model_dir(value)


def test_explicit_model_path_is_respected(tmp_path):
    assert require_speaker_model_dir(str(tmp_path)) == tmp_path


def test_model_installation_does_not_enable_legacy_enforcement():
    assert set(VoiceprintConfig.__dataclass_fields__) == {"model_dir"}
