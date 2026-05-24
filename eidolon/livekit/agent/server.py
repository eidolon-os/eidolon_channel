"""Voice agent server — LiveKit agent worker.

This server runs as a LiveKit agent worker, connecting to LiveKit rooms as an
agent participant. When a user joins a room, this worker dispatches a job that
runs either StreamingPipeline (streaming mode, default) or BatchPipeline (manual
mode, configured via AGENT_MODE env var).

Recommended local startup (sets ``EIDOLON_CHANNEL_LIVEKIT_ENV``,
``EIDOLON_ENV=dev``, ``PYTHONPATH``) — see ``deploy/README.md``::

    ./deploy/run_livekit_channel.sh

Manual module run (``EIDOLON_CHANNEL_LIVEKIT_ENV`` is **required** and must
point to an existing env file, or ``AgentConfig.from_env()`` raises ``ValueError``)::

    cd <repository-root> && source .venv/bin/activate
    export EIDOLON_CHANNEL_LIVEKIT_ENV=/path/to/.livekit-channel.env
    python -m eidolon.livekit.agent.server

For ad-hoc devmode (0 warm processes, easier to debug) without the shell
script, still export the env file path::

    export EIDOLON_CHANNEL_LIVEKIT_ENV=/path/to/.livekit-channel.env
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

        proc.userdata["vad"] = FireredPvadVAD.load(
            min_speech_duration=0.2,
            min_silence_duration=0.5,
            activation_threshold=0.55,
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
        cfg.stt_provider,
        cfg.tts_provider,
        cfg.vad_provider,
    )

    prebuilt_vad = getattr(ctx.proc, "userdata", {}).get("vad")
    room = ctx.room
    # Room.sid is async on livekit-agents 1.5+; resolve here so the factory
    # receives a plain string and the brain gets a stable conversation_id.
    session_key = ""
    try:
        session_key = await room.sid
    except Exception as exc:
        logger.warning("[Agent] room.sid resolve failed: %r — falling back to room.name", exc)
    if not session_key:
        session_key = getattr(room, "name", "") or ""
    factory = SharedStageFactory.from_config(
        cfg, prebuilt_vad=prebuilt_vad, livekit_session_key=session_key
    )

    if cfg.behavior.agent_mode == "batch":
        pipeline = BatchPipeline(factory)
    else:
        pipeline = StreamingPipeline(
            factory,
            instructions=cfg.behavior.instructions,
            allow_interruptions=True,
            welcome_message=cfg.behavior.welcome_message,
            false_interruption_timeout=cfg.behavior.false_interruption_timeout,
            audio_sample_rate=cfg.behavior.audio_sample_rate,
            stt_commit_transcript_timeout=cfg.behavior.stt_commit_transcript_timeout,
            aec_warmup_duration=cfg.behavior.aec_warmup_duration,
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
            cfg.llm.model, cfg.vad_provider, cfg.stt_provider, cfg.tts_provider,
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
    if not cfg.llm.base_url:
        errors.append("LLM base_url is not set")
    if not cfg.llm.model:
        errors.append("LLM model is not set")
    if not cfg.stt_provider:
        errors.append("STT_PROVIDER is not set")
    if not cfg.tts_provider:
        errors.append("TTS_PROVIDER is not set")
    if not cfg.vad_provider:
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
    shutdown_event = asyncio.Event()

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
