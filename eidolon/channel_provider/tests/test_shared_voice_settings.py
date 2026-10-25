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
        worker_defaults + "voiceprint:\n  calibration_path: $EIDOLON_STATE_ROOT/criterion.json\n"
    )
    monkeypatch.setenv("EIDOLON_CHANNEL_SETTINGS_YAML", str(document))
    monkeypatch.setenv("EIDOLON_CHANNEL_ENV_FILE", str(tmp_path / "empty.env"))
    monkeypatch.delenv("EIDOLON_CHANNEL_SETTINGS_OVERLAY_YAML", raising=False)
    monkeypatch.setenv("EIDOLON_STATE_ROOT", str(root))
    # A retired per-process shortcut cannot override the Host document.
    monkeypatch.setenv("EIDOLON_SPEAKER_CALIBRATION_PATH", "/stale/criterion.json")
    expected = str(root / "criterion.json")
    assert load_provider_config().speaker_calibration_path == expected
    assert load_effective_config().voiceprint.calibration_path == expected
    document.write_text(worker_defaults + "voiceprint:\n  calibration_path: ''\n")
    assert load_provider_config().speaker_calibration_path == ""
    assert load_effective_config().voiceprint.calibration_path == ""


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
        "providers:\n  stt_provider: worker-only-unknown\nvoiceprint:\n  calibration_path: /host/criterion.json\n"
    )
    monkeypatch.setenv("EIDOLON_CHANNEL_SETTINGS_YAML", str(document))
    monkeypatch.setenv("EIDOLON_CHANNEL_ENV_FILE", str(tmp_path / "empty.env"))
    monkeypatch.delenv("EIDOLON_CHANNEL_SETTINGS_OVERLAY_YAML", raising=False)
    assert load_provider_config().speaker_calibration_path == "/host/criterion.json"
    document.write_text("voiceprint:\n  calibration_path: /host/criterion.json\n  enabled: true\n")
    with pytest.raises(ValueError, match="cannot coexist"):
        load_provider_config()
    document.unlink()
    with pytest.raises(FileNotFoundError, match="EIDOLON_CHANNEL_SETTINGS_YAML missing"):
        load_provider_config()
