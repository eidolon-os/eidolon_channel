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


GIB = 1024**3


@pytest.mark.parametrize(
    ("cpu_count", "memory_gib", "expected"),
    [
        (12, 36, 3),  # the Mac Host on 11-01
        (8, 16, 3),  # RK3588 board
        (4, 8, 2),  # Raspberry Pi 5, 8 GB
        (4, 4, 1),  # a 4 GB board affords one
        (2, 64, 1),  # building a process competes with the conversation
    ],
)
def test_the_host_decides_how_many_processes_stay_warm(cpu_count, memory_gib, expected) -> None:
    assert server.host_warm_processes(cpu_count=cpu_count, memory_bytes=memory_gib * GIB) == expected


def test_a_product_host_keeps_more_than_one_warm(monkeypatch) -> None:
    # Pinned to 1 until 11-01: the second of two conversations starting within
    # a refill waited 8.6 s for a new process.
    monkeypatch.delenv("EIDOLON_CHANNEL_NUM_IDLE_PROCESSES", raising=False)
    cfg = AgentConfig()
    assert server._resolve_num_idle_processes(
        cfg, env="product", cpu_count=4, memory_bytes=8 * GIB
    ) == 2
    assert server._resolve_num_idle_processes(cfg, env="dev", cpu_count=12, memory_bytes=36 * GIB) == 1


def test_an_explicit_pool_size_still_wins(monkeypatch) -> None:
    monkeypatch.delenv("EIDOLON_CHANNEL_NUM_IDLE_PROCESSES", raising=False)
    pinned = AgentConfig(worker=WorkerConfig(num_idle_processes=0))
    assert server._resolve_num_idle_processes(pinned, env="product", cpu_count=12, memory_bytes=36 * GIB) == 0
    monkeypatch.setenv("EIDOLON_CHANNEL_NUM_IDLE_PROCESSES", "5")
    assert server._resolve_num_idle_processes(pinned, env="product") == 5


def test_the_product_template_does_not_pin_the_pool() -> None:
    from pathlib import Path

    import yaml

    template = Path(server.__file__).resolve().parents[3] / "config/settings.yaml"
    worker = yaml.safe_load(template.read_text(encoding="utf-8")).get("worker") or {}
    assert "num_idle_processes" not in worker
