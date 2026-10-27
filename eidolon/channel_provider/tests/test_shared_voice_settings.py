"""Both Channel processes consume the same Ops-rendered voice settings."""

import pytest

from eidolon.channel_provider.config import load_provider_config
from eidolon.livekit.common.config import load_effective_config
from .test_config import _environment


@pytest.mark.parametrize("host", ["mac", "pi5"])
def test_provider_and_worker_use_one_host_document(monkeypatch, tmp_path, host):
    _environment(
        monkeypatch,
        tmp_path,
        """
        storage:
          path: $STATE_ROOT_FOR_TEST/provider.sqlite3
        livekit:
          api_url: http://127.0.0.1:7880
          client_url: wss://livekit.example.test
    """,
    )
    root = tmp_path / host
    root.mkdir()
    document = root / "channel.yaml"
    worker_defaults = "llm:\n  base_url: http://127.0.0.1:9999/v1\n"
    document.write_text(
        worker_defaults + "voiceprint:\n  model_dir: $EIDOLON_STATE_ROOT/model\n"
    )
    monkeypatch.setenv("EIDOLON_CHANNEL_SETTINGS_YAML", str(document))
    monkeypatch.setenv("EIDOLON_CHANNEL_ENV_FILE", str(tmp_path / "empty.env"))
    monkeypatch.delenv("EIDOLON_CHANNEL_SETTINGS_OVERLAY_YAML", raising=False)
    monkeypatch.setenv("EIDOLON_STATE_ROOT", str(root))
    # A retired per-process shortcut cannot override the Host document.
    monkeypatch.setenv("EIDOLON_3DSPEAKER_MODEL_DIR", "/stale/model")
    expected = str(root / "model")
    assert load_provider_config().speaker_model_dir == expected
    assert load_effective_config().voiceprint.model_dir == expected
    document.write_text(worker_defaults + "voiceprint:\n  model_dir: ''\n")
    assert load_provider_config().speaker_model_dir == ""
    assert load_effective_config().voiceprint.model_dir == ""


def test_provider_reads_shared_voice_section_without_loading_worker_policy(monkeypatch, tmp_path):
    _environment(
        monkeypatch,
        tmp_path,
        """
        storage:
          path: $STATE_ROOT_FOR_TEST/provider.sqlite3
        livekit:
          api_url: http://127.0.0.1:7880
          client_url: wss://livekit.example.test
    """,
    )
    document = tmp_path / "channel.yaml"
    document.write_text(
        "providers:\n  stt_provider: worker-only-unknown\nvoiceprint:\n  model_dir: /host/model\n"
    )
    monkeypatch.setenv("EIDOLON_CHANNEL_SETTINGS_YAML", str(document))
    monkeypatch.setenv("EIDOLON_CHANNEL_ENV_FILE", str(tmp_path / "empty.env"))
    monkeypatch.delenv("EIDOLON_CHANNEL_SETTINGS_OVERLAY_YAML", raising=False)
    assert load_provider_config().speaker_model_dir == "/host/model"
    document.unlink()
    with pytest.raises(FileNotFoundError, match="EIDOLON_CHANNEL_SETTINGS_YAML missing"):
        load_provider_config()
