"""The worker's setup budget must come from configuration, not the framework.

livekit-agents kills a worker process that spends more than 10s in ``setup_fnc``
and retries forever, leaving the unit "active" while never registering a usable
process. Our setup loads pVAD and the EOT tokenizer, which takes ~21s on a
Raspberry Pi 5, so the framework default is a permanent spawn-kill loop there.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from eidolon.livekit.common.config import AgentConfig, WorkerConfig
from eidolon.livekit.common.config.validators import validate_effective_config
from eidolon.livekit.agent import server


class _RecordingAgentServer:
    last_kwargs: dict = {}
    last_rtc_kwargs: dict = {}

    def __init__(self, **kwargs: object) -> None:
        type(self).last_kwargs = kwargs

    def rtc_session(self, **kwargs: object):
        type(self).last_rtc_kwargs = kwargs

        def decorator(fn):
            return fn

        return decorator


@pytest.fixture
def recorded_server(monkeypatch: pytest.MonkeyPatch):
    import livekit.agents as lk_agents

    monkeypatch.setattr(lk_agents, "AgentServer", _RecordingAgentServer)
    return _RecordingAgentServer


def test_setup_timeout_is_taken_from_config(recorded_server, monkeypatch) -> None:
    monkeypatch.setenv("EIDOLON_ENV", "dev")
    cfg = AgentConfig(worker=WorkerConfig(setup_timeout_sec=45.0))

    server._build_server(cfg)

    assert recorded_server.last_kwargs["initialize_process_timeout"] == 45.0


def test_agent_name_is_taken_from_config(recorded_server, monkeypatch) -> None:
    monkeypatch.setenv("EIDOLON_ENV", "dev")
    cfg = AgentConfig(worker=WorkerConfig(agent_name="eidolon-isolated-test"))

    server._build_server(cfg)

    assert recorded_server.last_rtc_kwargs["agent_name"] == "eidolon-isolated-test"


def test_default_setup_budget_covers_slow_hardware() -> None:
    # A Pi 5 needs ~21s for the pVAD import plus the EOT tokenizer, and that is
    # on an idle box. Anything close to the framework's 10s default would put
    # the worker back in the spawn-kill loop.
    assert WorkerConfig().setup_timeout_sec >= 60.0


def test_out_of_range_setup_timeout_fails_validation() -> None:
    with pytest.raises(ValueError, match="worker.setup_timeout_sec"):
        validate_effective_config(AgentConfig(worker=WorkerConfig(setup_timeout_sec=0.0)))


def test_empty_agent_name_fails_validation() -> None:
    with pytest.raises(ValueError, match="worker.agent_name"):
        validate_effective_config(AgentConfig(worker=WorkerConfig(agent_name=" ")))


@pytest.mark.asyncio
async def test_worker_shutdown_uses_public_server_lifecycle() -> None:
    livekit_server = AsyncMock()

    await server._close_server(livekit_server)

    livekit_server.aclose.assert_awaited_once_with()
