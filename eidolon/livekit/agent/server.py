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

from eidolon_sdk.biz.contracts import (
    INTERACTION_MODE_HALF_DUPLEX,
    SESSION_CONTROL_TOPIC,
    SESSION_END_ERROR,
    SESSION_END_TYPE,
    SESSION_END_USER_LEFT,
    SESSION_INTENT_USER_INITIATED,
    WIRE_SCHEMA_VERSION,
)
from eidolon.livekit.common.config import AgentConfig, load_agent_config
from eidolon.livekit.plugins.speaker_verification import default_campplus_model_dir

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
    # users see weird behaviour. See integration/framework_patches.py for details.
    from eidolon.livekit.agent.integration import framework_patches

    framework_patches.check_framework_version()

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

    try:
        from eidolon.livekit.plugins.speaker_verification import (
            ModelScopeCampPlusSpeakerVerificationProvider,
        )
        from eidolon.livekit.common.config import load_effective_config

        cfg = load_effective_config()
        vp_cfg = cfg.voiceprint
        if not vp_cfg.enabled or not vp_cfg.prewarm:
            logger.info("[Agent] prewarm: voiceprint disabled")
            return
        model_dir = (
            Path(vp_cfg.model_dir).expanduser()
            if vp_cfg.model_dir.strip()
            else default_campplus_model_dir()
        )
        provider = ModelScopeCampPlusSpeakerVerificationProvider(
            model_dir=model_dir,
            voiceprint_root=Path(vp_cfg.root).expanduser(),
            threshold=vp_cfg.threshold,
            min_audio_ms=vp_cfg.min_audio_ms,
        )
        asyncio.run(provider.warm_up())
        proc.userdata["voiceprint_provider"] = provider
        logger.info("[Agent] prewarm: voiceprint model loaded")
    except Exception as e:
        logger.warning("[Agent] prewarm: voiceprint load failed: %s", e)


async def _resolve_session_metadata(ctx) -> tuple[str, str]:
    """Resolve ``(interaction_mode, session_intent)`` from the joined participant.

    Single resolution point for the session-metadata bus (plan §3.2): hub stamps
    both into the LiveKit token's ``participant_metadata``;
    ``wait_for_participant`` returns once the device/web client is present, so we
    read the authoritative values — from ONE metadata read — before building the
    pipeline (both are AgentSession-construction inputs). Any failure degrades to
    the safe defaults (``half_duplex`` / ``user_initiated``).
    """
    from eidolon.livekit.agent.runtime import (
        resolve_interaction_mode,
        resolve_session_intent,
    )

    try:
        participant = await ctx.wait_for_participant()
    except Exception:
        logger.exception(
            "[Agent] wait_for_participant failed; defaulting mode=%s intent=%s",
            INTERACTION_MODE_HALF_DUPLEX,
            SESSION_INTENT_USER_INITIATED,
        )
        return INTERACTION_MODE_HALF_DUPLEX, SESSION_INTENT_USER_INITIATED
    metadata = getattr(participant, "metadata", None)
    return (
        resolve_interaction_mode(metadata),
        resolve_session_intent(metadata),
    )


async def run_agent(ctx, cfg: AgentConfig) -> None:
    """Agent job entrypoint — runs the voice pipeline in the LiveKit room."""
    from eidolon.livekit.agent import (
        BatchPipeline,
        SharedStageFactory,
        StreamingPipeline,
    )
    from eidolon.livekit.agent.runtime import apply_interaction_mode

    logger.info(
        "[Agent] starting room=%s mode=%s stt=%s tts=%s vad=%s",
        ctx.room.name,
        cfg.behavior.pipeline_mode,
        cfg.providers.stt_provider,
        cfg.providers.tts_provider,
        cfg.providers.vad_provider,
    )

    prebuilt_vad = getattr(ctx.proc, "userdata", {}).get("vad")
    prebuilt_voiceprint_provider = getattr(ctx.proc, "userdata", {}).get("voiceprint_provider")
    room = ctx.room
    # session_key still passed as a synchronous fallback (Room.sid is async,
    # Room.name is set pre-connect). D1: also pass the room reference so the
    # remote-agent adapter can lazily build conversation_id="<prefix>:<participant_identity>:<room_name>"
    # at chat() time, when the user has connected and we know who they are.
    session_key = room.name or ""
    factory = SharedStageFactory.from_config(
        cfg,
        prebuilt_vad=prebuilt_vad,
        prebuilt_voiceprint_provider=prebuilt_voiceprint_provider,
        livekit_session_key=session_key,
        livekit_room=room,
    )

    from livekit import api as lk_api
    from livekit.api.twirp_client import TwirpError, TwirpErrorCode

    # session_end{reason} — the room is going away; tell the still-connected
    # client WHY so it can distinguish a normal end of conversation from a join
    # failure (plan §3.2). reason ∈ {idle_normal_end, user_left, error,
    # superseded, proactive_done}. idle_normal_end is routed here by the idle
    # watchdog; user_left/error come from the job-shutdown path; superseded and
    # proactive_done are reserved for Phase 3. Idempotent: the first reason wins,
    # so the shutdown callback that always follows an idle delete does not
    # overwrite idle_normal_end with user_left.
    _SESSION_CONTROL_TOPIC = SESSION_CONTROL_TOPIC
    session_end_state: dict[str, str | bool] = {"sent": False}

    async def _publish_session_end(reason: str) -> None:
        if session_end_state["sent"]:
            return
        session_end_state["sent"] = True
        local = getattr(room, "local_participant", None)
        if local is None:
            logger.info(
                "[lifecycle] session_end reason=%s room=%s skipped (no local participant)",
                reason,
                room.name,
            )
            return
        import json as _json

        try:
            await local.publish_data(
                _json.dumps(
                    {
                        "schema_v": WIRE_SCHEMA_VERSION,
                        "type": SESSION_END_TYPE,
                        "reason": reason,
                    }
                ).encode("utf-8"),
                reliable=True,
                topic=_SESSION_CONTROL_TOPIC,
            )
            logger.info("[lifecycle] session_end reason=%s room=%s sent", reason, room.name)
        except Exception:
            logger.debug(
                "[lifecycle] session_end reason=%s room=%s publish failed",
                reason,
                room.name,
                exc_info=True,
            )

    async def _delete_room(context: str) -> None:
        # [lifecycle] is the grep anchor that aligns server room-teardown with the
        # ESP32 controller's [lifecycle] logs by room_name + timestamp (plan Phase
        # 0). context distinguishes a normal idle end ("idle timeout") from the
        # job-shutdown path ("shutdown callback"); the pre-delete snapshot shows
        # whether anyone was still in the room when we tore it down.
        try:
            remote = getattr(room, "remote_participants", {}) or {}
            local = getattr(room, "local_participant", None)
            logger.info(
                "[lifecycle] deleting room=%s context=%s remote_participants=%d "
                "remote_identities=%s local_identity=%s",
                room.name,
                context,
                len(remote),
                list(remote.keys()),
                getattr(local, "identity", None),
            )
        except Exception:  # pragma: no cover - logging must never break teardown
            logger.debug("[lifecycle] pre-delete snapshot failed", exc_info=True)
        try:
            await ctx.api.room.delete_room(lk_api.DeleteRoomRequest(room=room.name))
            logger.info("[lifecycle] room=%s deleted (context=%s)", room.name, context)
        except TwirpError as e:
            if e.code == TwirpErrorCode.NOT_FOUND:
                logger.debug(
                    "[Agent] room=%s already deleted by LiveKit auto-cleanup",
                    room.name,
                )
                return
            logger.exception("[Agent] failed to delete room=%s (%s)", room.name, context)
        except Exception:
            logger.exception("[Agent] failed to delete room=%s (%s)", room.name, context)

    if cfg.behavior.pipeline_mode == "batch":
        pipeline = BatchPipeline(factory)
    else:
        # Phase 5 / Phase 2: resolve the session-metadata bus (interaction_mode +
        # session_intent) from the device-declared token metadata in one read,
        # then derive a per-session turn policy (half_duplex → no barge-in).
        # Never mutates the shared global cfg.
        interaction_mode, session_intent = await _resolve_session_metadata(ctx)
        session_turn_policy, allow_interruptions = apply_interaction_mode(
            turn_policy=cfg.turn_policy,
            allow_interruptions=True,
            interaction_mode=interaction_mode,
        )
        logger.info(
            "[Agent] interaction_mode=%s session_intent=%s allow_interruptions=%s "
            "attention_enabled=%s",
            interaction_mode,
            session_intent,
            allow_interruptions,
            session_turn_policy.attention.enabled,
        )
        pipeline = StreamingPipeline(
            factory,
            instructions=cfg.behavior.instructions,
            allow_interruptions=allow_interruptions,
            welcome_message=cfg.behavior.welcome_message,
            audio_sample_rate=cfg.behavior.audio_sample_rate,
            false_interruption_timeout=(
                session_turn_policy.interrupt.framework_false_interruption_timeout_ms / 1000.0
            ),
            stt_commit_transcript_timeout=(
                session_turn_policy.interrupt.stt_commit_transcript_timeout_ms / 1000.0
            ),
            aec_warmup_duration=(
                None
                if session_turn_policy.interrupt.aec_warmup_ms is None
                else session_turn_policy.interrupt.aec_warmup_ms / 1000.0
            ),
            turn_policy=session_turn_policy,
            interaction_mode=interaction_mode,
            session_intent=session_intent,
            observability=cfg.observability,
            voiceprint_config=cfg.voiceprint,
            # Idle watchdog disconnect: delete the room so the still-connected
            # client is actively kicked (ROOM_DELETED) and the job's
            # shutdown_fut resolves — session.aclose() alone leaves the client
            # in a dead room and the job hanging on shutdown_fut.
            on_idle_disconnect=lambda: _delete_room("idle timeout"),
            # The idle watchdog routes its client notice through this as
            # reason=idle_normal_end (sent before the grace + delete above).
            on_session_end=_publish_session_end,
            # On session close (device left / error), delete the room PROMPTLY —
            # before the slow STT/TTS shutdown drain — so this fixed-name room
            # (device-<id>) is gone before a rapid re-JOIN. Otherwise the old
            # agent + its audio track linger here for the whole drain; an
            # auto_subscribe=false client re-joining subscribes to that stale
            # track and gets "in room + agent_speaking state but NO audio"
            # (real-device confirmed: JOIN→X→quick JOIN → silent).
            on_session_closed=lambda: _delete_room("session closed (device left)"),
        )

    async def _delete_room_cb(reason: str) -> None:
        # The job is shutting down (user left, or an error tore the session down).
        # Send session_end first so a client that is still connected learns the
        # reason before ROOM_DELETED arrives. No-op if idle already sent
        # idle_normal_end (idempotent). Map the framework reason to our taxonomy.
        text = str(reason or "").lower()
        end_reason = (
            SESSION_END_ERROR if ("error" in text or "fail" in text) else SESSION_END_USER_LEFT
        )
        await _publish_session_end(end_reason)
        await _delete_room("shutdown callback")

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
        # The runtime_admin path is mandatory for eidolon_agent. Channel must
        # have a usable HMAC secret at startup; without one, every session
        # would die at first chat().
        if not cfg.runtime_admin.enabled:
            errors.append(
                "runtime_admin.enabled=false is not valid for eidolon_agent. Set enabled=true."
            )
        elif not (
            cfg.runtime_admin.jwt_secret or Path("~/eidolon/run/jwt-secret").expanduser().is_file()
        ):
            errors.append(
                "PAIRING_JWT_SECRET empty AND ~/eidolon/run/jwt-secret "
                "missing. Start eidolon-agent once (it persists the secret) "
                "or set PAIRING_JWT_SECRET in config/.env."
            )
    if not cfg.providers.stt_provider:
        errors.append("STT_PROVIDER is not set")
    if not cfg.providers.tts_provider:
        errors.append("TTS_PROVIDER is not set")
    if not cfg.providers.vad_provider:
        errors.append("VAD_PROVIDER is not set")

    if errors:
        raise ValueError("AgentConfig validation failed:\n  - " + "\n  - ".join(errors))


def _resolve_num_idle_processes(
    cfg: AgentConfig,
    *,
    env: str | None = None,
    cpu_count: int | None = None,
) -> int:
    raw_env = os.getenv("EIDOLON_CHANNEL_NUM_IDLE_PROCESSES", "").strip()
    if raw_env:
        value = int(raw_env)
        if value < 0:
            raise ValueError("EIDOLON_CHANNEL_NUM_IDLE_PROCESSES must be >= 0")
        return min(value, 64)

    configured = cfg.worker.num_idle_processes
    if configured is not None:
        return configured

    runtime_env = env or os.getenv("EIDOLON_ENV", "prod")
    runtime_cpu_count = cpu_count if cpu_count is not None else multiprocessing.cpu_count()
    return 0 if runtime_env == "dev" else min(runtime_cpu_count, 4)


def _build_server(cfg: AgentConfig) -> "AgentServer":
    from livekit.agents import AgentServer, JobExecutorType, WorkerType

    env = os.getenv("EIDOLON_ENV", "prod")
    is_dev = env == "dev"
    num_idle = _resolve_num_idle_processes(cfg, env=env)

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
        cfg.behavior.pipeline_mode,
        cfg.llm.model,
        _resolve_num_idle_processes(cfg),
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
