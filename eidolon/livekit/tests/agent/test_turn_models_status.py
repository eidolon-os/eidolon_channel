"""The worker says whether its job processes could load the turn models.

livekit's own health surface answers for the worker process, not for what a
job process prewarms. On 2026-10-31 both ONNX files were LFS pointers: two
warnings in the log, a green worker, and no session able to take turns.
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from eidolon.livekit.agent import server
from eidolon.livekit.common import config
from eidolon.livekit.common.config.schema import EffectiveAgentConfig
from eidolon.livekit.plugins import eot
from eidolon.livekit.plugins.vad.firered import FireredPvadVAD


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("EIDOLON_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setattr(config, "load_effective_config", lambda: EffectiveAgentConfig())
    return tmp_path / "channel/turn-models.json"


def _published(path):
    return json.loads(path.read_text())


def test_loaded_models_are_published(runtime, monkeypatch) -> None:
    monkeypatch.setattr(FireredPvadVAD, "load", Mock())
    monkeypatch.setattr(eot, "ChineseModel", Mock())

    server._load_prewarm_models(SimpleNamespace(userdata={}))

    status = _published(runtime)
    assert status["state"] == "loaded"
    assert status["vad"] == {"state": "loaded", "provider": "firered_pvad"}
    assert status["eot"] == {"state": "loaded", "provider": "firered"}


def test_a_model_that_will_not_load_is_published_as_failed(runtime, monkeypatch) -> None:
    pointer = RuntimeError("[ONNXRuntimeError] : 7 : INVALID_PROTOBUF")
    monkeypatch.setattr(FireredPvadVAD, "load", Mock(side_effect=pointer))
    monkeypatch.setattr(eot, "ChineseModel", Mock())

    server._load_prewarm_models(SimpleNamespace(userdata={}))

    status = _published(runtime)
    assert status["state"] == "failed"
    assert status["vad"]["state"] == "failed"
    assert "INVALID_PROTOBUF" in status["vad"]["error"]
    assert status["eot"]["state"] == "loaded"


def test_a_starting_worker_replaces_the_last_runs_answer(runtime, monkeypatch) -> None:
    runtime.parent.mkdir(parents=True)
    runtime.write_text(json.dumps({"state": "loaded"}))
    seen = {}

    def stop(cfg):
        seen["status"] = _published(runtime)
        raise SystemExit

    monkeypatch.setattr(server, "load_agent_config", lambda: replace(EffectiveAgentConfig()))
    monkeypatch.setattr(server, "_validate_config", lambda cfg: None)
    monkeypatch.setattr(server, "_build_server", stop)
    import asyncio

    with pytest.raises(SystemExit):
        asyncio.run(server._serve())
    assert seen["status"]["state"] == "pending"


def test_an_unwritable_runtime_root_never_fails_prewarm(monkeypatch, tmp_path) -> None:
    blocked = tmp_path / "file"
    blocked.write_text("")
    monkeypatch.setenv("EIDOLON_RUNTIME_ROOT", str(blocked))
    server._publish_turn_models({"state": "pending"})
