"""Voice agent server — LiveKit agent worker.

This server runs as a LiveKit agent worker, connecting to LiveKit rooms as an
agent participant. When a user joins a room, this worker dispatches a job that
runs either StreamingPipeline (streaming mode, default) or BatchPipeline (manual
mode, configured via AGENT_MODE env var).

Recommended local startup: ``eidolon_admin`` supervisord / ``./deploy/dev/run_all.sh``
(loads ``config/.env`` via ``with-env.sh``).

Standalone worker after ``./deploy/dev/init.sh``::

    cd <repository-root> && source .venv/bin/activate
    EIDOLON_ENV=dev python -m eidolon.livekit.agent.server

Optional override of the env file path::

    export EIDOLON_CHANNEL_ENV_FILE=/path/to/config/.env
    EIDOLON_ENV=dev python -m eidolon.livekit.agent.server
"""

from __future__ import annotations

import asyncio
import logging
import logging.handlers
import os
import signal
import sys
import multiprocessing
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from livekit.agents import AgentServer

from eidolon.livekit.common.config import AgentConfig, load_agent_config

logger = logging.getLogger("agent_server")

# Suppress noisy debug logs from websockets library (BINARY frame dumps)
logging.getLogger("websockets").setLevel(logging.INFO)
logging.getLogger("websockets.client").setLevel(logging.INFO)
logging.getLogger("websockets.server").setLevel(logging.INFO)


def _configure_logging(
    env: str,
    log_dir: Path | None = None,
    file_level: str = "DEBUG",
    log_to_file: bool = True,
) -> None:
    level = logging.DEBUG if env == "dev" else logging.INFO
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(fmt))
    root_logger.addHandler(console_handler)

    if log_to_file and log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        flevel = getattr(logging, file_level.upper(), logging.DEBUG)
        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=log_dir / f"livekit-agent-{date.today().isoformat()}.log",
            when="midnight",
            interval=1,
            backupCount=0,
        )
        file_handler.setLevel(flevel)
        file_handler.setFormatter(logging.Formatter(fmt))
        root_logger.addHandler(file_handler)


def _register_plugins() -> None:
    try:
        from livekit.plugins import openai as _  # noqa: F401
    except ImportError:
        logger.warning("livekit-plugins-openai not installed, LLM support unavailable")

    try:
        from eidolon.livekit.plugins.vad.firered import register_plugin
        register_plugin()
    except ImportError:
        logger.warning("firered pvad plugin not available")

    try:
        from eidolon.livekit.plugins.eot import register_plugin
        register_plugin()
    except ImportError:
        logger.warning("eidolon eot plugin not available")


def _prewarm(proc) -> None:
    """Prewarm hook: load VAD and EOT models before the first job runs.

    This is called by WorkerOptions.prewarm_fnc when a worker process starts,
    before any job is dispatched. Loading models here avoids cold-start latency
    on the first user audio frame.
    """
    # Validate framework-internal patches are still applicable on this
    # SDK version. Logs WARNING (not fatal) on untested versions —
    # surfaces SDK upgrades that may have broken our patches before
    # users see weird behaviour. See _framework_patches.py for details.
    from eidolon.livekit.agent import _framework_patches
    _framework_patches.check_framework_version()

    try:
        from eidolon.livekit.plugins.vad.firered import FireredPvadVAD
        from eidolon.livekit.common.config import load_effective_config

        cfg = load_effective_config()
        vad_cfg = cfg.turn_policy.vad

        proc.userdata["vad"] = FireredPvadVAD.load(
            min_speech_duration=vad_cfg.min_speech_duration_ms / 1000.0,
            min_silence_duration=vad_cfg.min_silence_duration_ms / 1000.0,
            prefix_padding_duration=vad_cfg.prefix_padding_ms / 1000.0,
            max_buffered_speech=vad_cfg.max_buffered_speech_ms / 1000.0,
            activation_threshold=vad_cfg.activation_threshold,
        )
        logger.info("[Agent] prewarm: FireRed pVAD loaded")
    except Exception as e:
        logger.warning("[Agent] prewarm: VAD load failed: %s", e)

    try:
        from eidolon.livekit.plugins.eot import ChineseModel

        proc.userdata["eot_model"] = ChineseModel()
        logger.info("[Agent] prewarm: EOT model loaded")
    except Exception as e:
        logger.warning("[Agent] prewarm: EOT load failed: %s", e)


async def run_agent(ctx, cfg: AgentConfig) -> None:
    """Agent job entrypoint — runs the voice pipeline in the LiveKit room."""
    from eidolon.livekit.agent import (
        BatchPipeline,
        SharedStageFactory,
        StreamingPipeline,
    )

    logger.info(
        "[Agent] starting room=%s mode=%s stt=%s tts=%s vad=%s",
        ctx.room.name,
        cfg.behavior.agent_mode,
        cfg.providers.stt_provider,
        cfg.providers.tts_provider,
        cfg.providers.vad_provider,
    )

    prebuilt_vad = getattr(ctx.proc, "userdata", {}).get("vad")
    room = ctx.room
    # session_key still passed as a synchronous fallback (Room.sid is async,
    # Room.name is set pre-connect). D1: also pass the room reference so the
    # remote-agent adapter can lazily build conversation_id="<prefix>:<participant_identity>:<room_name>"
    # at chat() time, when the user has connected and we know who they are.
    session_key = room.name or ""
    factory = SharedStageFactory.from_config(
        cfg,
        prebuilt_vad=prebuilt_vad,
        livekit_session_key=session_key,
        livekit_room=room,
    )

    if cfg.behavior.agent_mode == "batch":
        pipeline = BatchPipeline(factory)
    else:
        pipeline = StreamingPipeline(
            factory,
            instructions=cfg.behavior.instructions,
            allow_interruptions=True,
            welcome_message=cfg.behavior.welcome_message,
            audio_sample_rate=cfg.behavior.audio_sample_rate,
            turn_policy=cfg.turn_policy,
            observability=cfg.observability,
        )

    from livekit import api as lk_api
    from livekit.api.twirp_client import TwirpError, TwirpErrorCode

    async def _delete_room_cb(_reason: str) -> None:
        try:
            await ctx.api.room.delete_room(lk_api.DeleteRoomRequest(room=room.name))
            logger.info("[Agent] room=%s deleted via shutdown callback", room.name)
        except TwirpError as e:
            if e.code == TwirpErrorCode.NOT_FOUND:
                logger.debug(
                    "[Agent] room=%s already deleted by LiveKit auto-cleanup",
                    room.name,
                )
                return
            logger.exception(
                "[Agent] failed to delete room=%s via shutdown callback",
                room.name,
            )
        except Exception:
            logger.exception(
                "[Agent] failed to delete room=%s via shutdown callback",
                room.name,
            )

    ctx.add_shutdown_callback(_delete_room_cb)
    await pipeline.run(room)


# Module-level config shared between main process and spawned workers
_agent_config: AgentConfig | None = None


async def _on_session(ctx) -> None:
    """Agent session handler — must be module-level for multiprocessing spawn compatibility."""
    global _agent_config

    cfg = _agent_config
    if cfg is None:
        cfg = load_agent_config()
        _agent_config = cfg
        logger.info(
            "[_on_session] config reloaded: llm=%s vad=%s stt=%s tts=%s",
            cfg.llm.model,
            cfg.providers.vad_provider,
            cfg.providers.stt_provider,
            cfg.providers.tts_provider,
        )
    else:
        logger.info("[_on_session] using cached _agent_config")

    await run_agent(ctx, cfg)
    logger.info("[_on_session] run_agent returned")


def _validate_config(cfg: AgentConfig) -> None:
    errors: list[str] = []
    if not cfg.core.livekit_url:
        errors.append("LIVEKIT_URL is not set")
    if not cfg.core.api_key:
        errors.append("LIVEKIT_API_KEY is not set")
    if cfg.providers.brain_provider == "direct_llm":
        if not cfg.llm.base_url:
            errors.append("LLM base_url is not set")
        if not cfg.llm.model:
            errors.append("LLM model is not set")
    else:
        if not cfg.remote_agent_rpc.target:
            errors.append("REMOTE_AGENT_RPC target is not set")
        # Phase 32.B: either path must produce a usable token:
        #   - runtime_admin.enabled + secret available (env or shared file)
        #   - remote_agent_rpc.device_token (legacy static)
        # If both empty, factory will refuse to open the gRPC session at
        # first chat() — but we fail loud here at startup instead.
        rt_admin_ok = (
            cfg.runtime_admin.enabled
            and (
                cfg.runtime_admin.jwt_secret
                or Path("~/eidolon/run/jwt-secret").expanduser().is_file()
            )
        )
        if not rt_admin_ok and not cfg.remote_agent_rpc.device_token:
            errors.append(
                "no device_token source available: enable runtime_admin "
                "(needs PAIRING_JWT_SECRET or ~/eidolon/run/jwt-secret) "
                "or set REMOTE_AGENT_RPC_DEVICE_TOKEN in .env (legacy)"
            )
    if not cfg.providers.stt_provider:
        errors.append("STT_PROVIDER is not set")
    if not cfg.providers.tts_provider:
        errors.append("TTS_PROVIDER is not set")
    if not cfg.providers.vad_provider:
        errors.append("VAD_PROVIDER is not set")

    if errors:
        raise ValueError("AgentConfig validation failed:\n  - " + "\n  - ".join(errors))


def _build_server(cfg: AgentConfig) -> "AgentServer":
    import multiprocessing
    import os

    from livekit.agents import AgentServer, JobExecutorType, WorkerType

    env = os.getenv("EIDOLON_ENV", "prod")
    is_dev = env == "dev"

    cpu_count = multiprocessing.cpu_count()
    num_idle = 0 if is_dev else min(cpu_count, 4)

    prometheus_port = int(os.getenv("AGENT_PROMETHEUS_PORT", "0") or "0")
    prometheus_multiproc_dir = os.getenv("AGENT_PROMETHEUS_MULTIPROC_DIR", "")
    if prometheus_port and not prometheus_multiproc_dir:
        prometheus_multiproc_dir = "/tmp/eidolon_prom_multiproc"

    server = AgentServer(
        ws_url=cfg.core.livekit_url,
        api_key=cfg.core.api_key,
        api_secret=cfg.core.api_secret,
        host=cfg.core.host,
        port=cfg.core.port,
        job_executor_type=JobExecutorType.PROCESS,
        num_idle_processes=num_idle,
        setup_fnc=_prewarm,
        # Memory guardrails
        job_memory_warn_mb=1024,
        job_memory_limit_mb=2048,
        # Graceful shutdown: 10 min drain + 30s hard kill per process
        drain_timeout=600,
        shutdown_process_timeout=30.0,
        # Prometheus metrics
        prometheus_port=prometheus_port or None,
        prometheus_multiproc_dir=prometheus_multiproc_dir or None,
        # Log level: DEBUG in dev, INFO in prod
        log_level="DEBUG" if is_dev else "INFO",
    )

    server.rtc_session(
        agent_name="eidolon",
        type=WorkerType.PUBLISHER,
    )(_on_session)

    return server


async def _serve() -> None:
    cfg = load_agent_config()
    _validate_config(cfg)

    global _agent_config
    _agent_config = cfg

    logger.info(
        "[Server] starting env=%s url=%s mode=%s llm=%s idle_procs=%s",
        os.getenv("EIDOLON_ENV", "prod"),
        cfg.core.livekit_url,
        cfg.behavior.agent_mode,
        cfg.llm.model,
        min(multiprocessing.cpu_count(), 4),
    )

    server = _build_server(cfg)

    is_dev = os.getenv("EIDOLON_ENV", "prod") == "dev"
    await server.run(devmode=is_dev)


def main() -> None:
    env = os.getenv("EIDOLON_ENV", "prod")
    log_dir_str = os.getenv("LOG_DIR", "")
    log_dir = Path(log_dir_str) if log_dir_str else None
    _configure_logging(env, log_dir=log_dir, log_to_file=bool(log_dir_str))
    _register_plugins()

    loop = asyncio.new_event_loop()
    server_task: asyncio.Task | None = None

    async def _run():
        nonlocal server_task
        server_task = asyncio.create_task(_serve())
        await server_task

    def _signal_handler(sig: signal.Signals):
        logger.info("[Server] received %s, initiating graceful shutdown...", sig.name)
        if server_task and not server_task.done():
            server_task.cancel()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler, sig)

    try:
        loop.run_until_complete(_run())
    except asyncio.CancelledError:
        logger.info("[Server] main task cancelled, exiting.")
    finally:
        loop.close()


if __name__ == "__main__":
    main()
