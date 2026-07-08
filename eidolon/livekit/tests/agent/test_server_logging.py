import os
from pathlib import Path

from eidolon.livekit.agent import server
from eidolon.livekit.plugins.eot.models import base as eot_base


def test_resolve_log_dir_defaults_to_channel_log_root(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("LOG_DIR", raising=False)
    monkeypatch.setenv("EIDOLON_LOG_ROOT", str(tmp_path / "logs"))

    assert server._resolve_log_dir() == tmp_path / "logs" / "channel"


def test_relative_eot_debug_log_is_anchored_under_channel_logs(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("EIDOLON_EOT_DEBUG_LOG", "eot-debug.jsonl")

    server._normalize_optional_log_file_env("EIDOLON_EOT_DEBUG_LOG", tmp_path)

    assert Path(os.environ["EIDOLON_EOT_DEBUG_LOG"]) == tmp_path / "eot-debug.jsonl"


def test_eot_model_resolves_relative_debug_log_with_default_channel_root(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("LOG_DIR", raising=False)
    monkeypatch.setenv("EIDOLON_LOG_ROOT", str(tmp_path / "logs"))
    monkeypatch.setenv("EIDOLON_EOT_DEBUG_LOG", "eot-debug.jsonl")

    assert (
        Path(eot_base._resolve_debug_log_path())
        == tmp_path / "logs" / "channel" / "eot-debug.jsonl"
    )
