"""Voice agent server — LiveKit agent worker.

This server runs as a LiveKit agent worker, connecting to LiveKit rooms as an
agent participant. When a user joins a room, this worker dispatches a job and
selects the mode-specific pipeline from the session metadata stamped into the
LiveKit participant token.

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
import json
import logging
import logging.handlers
import os
import signal
import sys
import time
import multiprocessing
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from livekit.agents import AgentServer

from eidolon_sdk.biz.contracts import (
    INTERACTION_MODE_PTT,
    SESSION_CONVERSATION_ID_FIELD,
    SESSION_CONTROL_TOPIC,
    SESSION_END_ERROR,
    SESSION_END_TYPE,
    SESSION_END_USER_LEFT,
    SESSION_STARTED_TYPE,
    WIRE_SCHEMA_VERSION,
    normalize_conversation_id,
)
from eidolon.livekit.common.config import AgentConfig, load_agent_config

# Load the session-contract resolver with the worker, not lazily per job.
# LiveKit job processes inherit the worker's module snapshot; eager loading
# prevents a hot-updated Channel module from mixing with a stale SDK contract
# already resident in the parent process.
from eidolon.livekit.agent.runtime import (
    apply_interaction_mode,
    resolve_avatar_requested,
    resolve_interaction_mode,
    resolve_session_intent,
)
from eidolon.livekit.agent.runtime.resolver import (
    DeviceTokenResolverError,
    wait_for_runtime_participant_metadata,
)
from eidolon.livekit.plugins.speaker_verification import default_campplus_model_dir

logger = logging.getLogger("agent_server")


# Suppress noisy debug logs from websockets library (BINARY frame dumps)
logging.getLogger("websockets").setLevel(logging.INFO)
logging.getLogger("websockets.client").setLevel(logging.INFO)
logging.getLogger("websockets.server").setLevel(logging.INFO)


def _eidolon_log_root() -> Path:
    raw = os.getenv("EIDOLON_LOG_ROOT", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / "eidolon" / "logs"


def _default_log_dir() -> Path:
    return _eidolon_log_root() / "channel"


def _runtime_secret_file() -> Path:
    root = Path(os.getenv("EIDOLON_RUNTIME_ROOT", "~/eidolon/run")).expanduser()
    return root / "agent/jwt-secret"


def _resolve_log_dir() -> Path:
    raw = os.getenv("LOG_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    return _default_log_dir()


def _normalize_optional_log_file_env(name: str, log_dir: Path) -> None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = log_dir / path
    os.environ[name] = str(path)


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

    The framework kills a process that overruns ``worker.setup_timeout_sec``,
    so the elapsed time is logged: it is the only place the cost of this hook
    is observable, and a process that dies here dies silently as far as the
    unit is concerned.
    """
    started = time.monotonic()
    try:
        _load_prewarm_models(proc)
    finally:
        logger.info("[Agent] prewarm: finished in %.1fs", time.monotonic() - started)


def _load_prewarm_models(proc) -> None:
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


async def _resolve_session_metadata(ctx) -> tuple[str, str, bool]:
    """Resolve ``(interaction_mode, session_intent, avatar_requested)`` from the participant.

    Single resolution point for the session-metadata bus (plan §3.2): hub / web
    client stamps these into the LiveKit token's ``participant_metadata``.
    Infrastructure participants such as the Hub control bridge may join first,
    so select the explicitly typed runtime actor and read all values from that
    ONE metadata snapshot before building the pipeline. Connection and actor
    resolution failures are fatal; optional metadata fields retain their safe
    defaults.
    """
    # Remote participants are synchronized only after the job joins the room.
    await ctx.connect()
    try:
        _, metadata = await wait_for_runtime_participant_metadata(ctx.room)
    except DeviceTokenResolverError:
        logger.exception("[Agent] runtime participant unavailable")
        raise
    return (
        resolve_interaction_mode(metadata),
        resolve_session_intent(metadata),
        resolve_avatar_requested(metadata),
    )


def _resolve_runtime_session_id(ctx) -> str:
    """Read the per-entry interaction identity from named dispatch metadata.

    The external lifecycle wire still calls this field ``conversation_id``.
    Internally it is never the Agent brain's long-lived conversation identity.
    """

    try:
        metadata = json.loads(str(getattr(ctx.job, "metadata", "") or "{}"))
    except (TypeError, ValueError) as exc:
        raise ValueError("agent dispatch metadata is not valid JSON") from exc
    if not isinstance(metadata, dict):
        raise ValueError("agent dispatch metadata must be an object")
    runtime_session_id = normalize_conversation_id(metadata.get(SESSION_CONVERSATION_ID_FIELD))
    if runtime_session_id is None:
        raise ValueError("agent dispatch has no valid conversation_id")
    return runtime_session_id


def _session_lifecycle_payload(
    message_type: str, runtime_session_id: str, *, reason: str | None = None
) -> bytes:
    payload = {
        "schema_v": WIRE_SCHEMA_VERSION,
        "type": message_type,
        SESSION_CONVERSATION_ID_FIELD: runtime_session_id,
    }
    if reason is not None:
        payload["reason"] = reason
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


async def run_agent(ctx, cfg: AgentConfig) -> None:
    """Agent job entrypoint — runs the voice pipeline in the LiveKit room."""
    from eidolon.livekit.agent.factory import SharedStageFactory
    from eidolon.livekit.agent.half_duplex.pipeline import HalfDuplexPttPipeline
    from eidolon.livekit.agent.full_duplex import StreamingPipeline

    logger.info(
        "[Agent] starting room=%s stt=%s tts=%s vad=%s",
        ctx.room.name,
        cfg.providers.stt_provider,
        cfg.providers.tts_provider,
        cfg.providers.vad_provider,
    )

    prebuilt_vad = getattr(ctx.proc, "userdata", {}).get("vad")
    prebuilt_voiceprint_provider = getattr(ctx.proc, "userdata", {}).get("voiceprint_provider")
    room = ctx.room
    runtime_session_id = _resolve_runtime_session_id(ctx)
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
        runtime_session_id=runtime_session_id,
    )

    from livekit.api.twirp_client import TwirpError, TwirpErrorCode

    # session_end{reason} — the conversation is ending while the device stays
    # exactly where it is; tell it WHY so it can distinguish a normal end from a
    # failure to be served (plan §3.2). reason ∈ {idle_normal_end, user_left, error,
    # superseded, proactive_done}. idle_normal_end is routed here by the idle
    # watchdog; user_left/error come from the job-shutdown path; superseded and
    # proactive_done are reserved for Phase 3. Idempotent: the first reason wins,
    # so the shutdown callback that always follows an idle delete does not
    # overwrite idle_normal_end with user_left.
    _SESSION_CONTROL_TOPIC = SESSION_CONTROL_TOPIC
    session_end_state: dict[str, str | bool] = {"sent": False, "reason": ""}

    async def _publish_session_started() -> None:
        local = getattr(room, "local_participant", None)
        if local is None:
            raise RuntimeError("cannot confirm session start without a local participant")
        await local.publish_data(
            _session_lifecycle_payload(SESSION_STARTED_TYPE, runtime_session_id),
            reliable=True,
            topic=SESSION_CONTROL_TOPIC,
        )
        logger.info(
            "[lifecycle] session_started conversation_id=%s room=%s sent",
            runtime_session_id,
            room.name,
        )

    async def _publish_session_end(reason: str) -> None:
        if session_end_state["sent"]:
            return
        first_reason = str(session_end_state.get("reason") or "")
        if first_reason:
            reason = first_reason
        else:
            session_end_state["reason"] = reason
        local = getattr(room, "local_participant", None)
        if local is None:
            logger.info(
                "[lifecycle] session_end reason=%s room=%s skipped (no local participant)",
                reason,
                room.name,
            )
            return
        try:
            await local.publish_data(
                _session_lifecycle_payload(
                    SESSION_END_TYPE,
                    runtime_session_id,
                    reason=reason,
                ),
                reliable=True,
                topic=_SESSION_CONTROL_TOPIC,
            )
            session_end_state["sent"] = True
            logger.info("[lifecycle] session_end reason=%s room=%s sent", reason, room.name)
        except Exception:
            # Not debug: this is the failure of the notification itself. If the
            # device cannot be told the conversation ended, every remaining
            # surface — the phone, the Provider, this log — shows a healthy
            # session, so the silence has to be loud here or it is nowhere.
            logger.warning(
                "[lifecycle] session_end reason=%s room=%s publish FAILED — the device "
                "was not told the conversation ended",
                reason,
                room.name,
                exc_info=True,
            )

    async def _end_serving(context: str) -> None:
        """End this conversation by withdrawing our own dispatch, not the room.

        The room is not ours to delete. It is the device's channel: the device
        lives in it continuously and would lose its way back if we tore it down
        at the end of a conversation. What we withdraw is the standing order
        that put us here — which is also what makes this teardown terminal. A
        dispatch left behind would have LiveKit hand us straight back a new job,
        because the device is still sitting in the room, and the conversation we
        just ended would restart itself.

        [lifecycle] is the grep anchor that aligns server-side teardown with the
        ESP32 controller's [lifecycle] logs by room_name + timestamp. context
        distinguishes a normal idle end ("idle timeout") from the job-shutdown
        path ("shutdown callback").
        """
        job = getattr(ctx, "job", None)
        agent_name = getattr(job, "agent_name", "") or ""
        dispatch_id = getattr(job, "dispatch_id", "") or ""
        try:
            remote = getattr(room, "remote_participants", {}) or {}
            local = getattr(room, "local_participant", None)
            logger.info(
                "[lifecycle] ending serving room=%s agent=%s context=%s "
                "remote_participants=%d remote_identities=%s local_identity=%s",
                room.name,
                agent_name,
                context,
                len(remote),
                list(remote.keys()),
                getattr(local, "identity", None),
            )
        except Exception:  # pragma: no cover - logging must never break teardown
            logger.debug("[lifecycle] pre-teardown snapshot failed", exc_info=True)
        if not agent_name or not dispatch_id:
            # A job without its own dispatch identity cannot safely withdraw a
            # standing order: matching by agent name could delete a newer
            # conversation that has already superseded this one.
            logger.warning(
                "[lifecycle] room=%s has no dispatch identity; cannot end serving (context=%s)",
                room.name,
                context,
            )
            return
        try:
            await ctx.api.agent_dispatch.delete_dispatch(
                dispatch_id=dispatch_id,
                room_name=room.name,
            )
            logger.info(
                "[lifecycle] room=%s dispatch=%s conversation_id=%s withdrawn (context=%s)",
                room.name,
                dispatch_id,
                runtime_session_id,
                context,
            )
        except TwirpError as e:
            if e.code == TwirpErrorCode.NOT_FOUND:
                logger.debug("[Agent] room=%s dispatch already gone", room.name)
                return
            logger.exception("[Agent] failed to end serving room=%s (%s)", room.name, context)
        except Exception:
            logger.exception("[Agent] failed to end serving room=%s (%s)", room.name, context)

    # Resolve the session-metadata bus (interaction_mode + session_intent) from
    # the device-declared token metadata in one read, then derive a per-session
    # turn policy. The old process-wide "batch/streaming" pipeline mode was
    # retired; the Channel worker now always chooses half/full duplex at the
    # session boundary.
    interaction_mode, session_intent, avatar_requested = await _resolve_session_metadata(ctx)
    session_turn_policy, allow_interruptions = apply_interaction_mode(
        turn_policy=cfg.turn_policy,
        allow_interruptions=True,
        interaction_mode=interaction_mode,
    )
    # Video avatar is enabled for this session only if globally available AND the
    # client declared it (full-duplex path for M1). Default off → audio-only.
    avatar_enabled = bool(cfg.avatar.enabled and avatar_requested)
    logger.info(
        "[Agent] interaction_mode=%s session_intent=%s allow_interruptions=%s "
        "attention_enabled=%s avatar_requested=%s avatar_enabled=%s",
        interaction_mode,
        session_intent,
        allow_interruptions,
        session_turn_policy.attention.enabled,
        avatar_requested,
        avatar_enabled,
    )
    if _use_ptt_pipeline(interaction_mode):
        pipeline = HalfDuplexPttPipeline(
            factory,
            instructions=cfg.behavior.instructions,
            welcome_message=cfg.behavior.welcome_message,
            audio_sample_rate=cfg.behavior.audio_sample_rate,
            turn_policy=session_turn_policy,
            observability=cfg.observability,
            session_intent=session_intent,
            on_session_started=_publish_session_started,
            on_session_end=_publish_session_end,
            on_idle_disconnect=lambda: _end_serving("idle timeout"),
            on_session_closed=lambda: _end_serving("session closed"),
        )
    else:
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
            on_session_started=_publish_session_started,
            observability=cfg.observability,
            voiceprint_config=cfg.voiceprint,
            # Video avatar (per-session; audio routed to the avatar worker when on).
            avatar_enabled=avatar_enabled,
            avatar_config=cfg.avatar,
            core_config=cfg.core,
            # Idle watchdog disconnect: withdrawing the dispatch is what makes
            # the job's shutdown_fut resolve — session.aclose() alone would
            # leave the job hanging on it. The device is untouched and stays in
            # its channel, which is the point.
            on_idle_disconnect=lambda: _end_serving("idle timeout"),
            # The idle watchdog routes its client notice through this as
            # reason=idle_normal_end (sent before the grace + teardown above).
            on_session_end=_publish_session_end,
            # On session close (device left / error), withdraw PROMPTLY — before
            # the slow STT/TTS shutdown drain. LiveKit removes us from the room
            # as soon as the dispatch is gone, so the next session finds a clean
            # room even while this job is still draining. Leaving a stale agent
            # and its audio track in the room for the whole drain is what used
            # to give a re-entering client "agent_speaking state but NO audio"
            # (real-device confirmed).
            on_session_closed=lambda: _end_serving("session closed"),
        )

    async def _end_serving_cb(reason: str) -> None:
        # The job is shutting down (user left, or an error tore the session down).
        # Send session_end first so the device — which is still connected, and
        # stays connected — learns why the conversation ended. No-op if idle
        # already sent idle_normal_end (idempotent). Map the framework reason to
        # our taxonomy.
        text = str(reason or "").lower()
        end_reason = (
            SESSION_END_ERROR if ("error" in text or "fail" in text) else SESSION_END_USER_LEFT
        )
        await _publish_session_end(end_reason)
        await _end_serving("shutdown callback")

    ctx.add_shutdown_callback(_end_serving_cb)
    try:
        await pipeline.run(room)
    except Exception:
        # A job that dies before it has a running session still owes the device
        # an answer, and _end_serving_cb cannot give one: the framework
        # disconnects the room before shutdown callbacks run, so publish_data
        # there always raises and the device is left holding an open microphone
        # against a room with nothing in it. This is the last point at which the
        # room is still live, so send session_end{error} here — the same
        # "failure to be served" reason the contract already defines — and then
        # let the job fail exactly as it would have. _publish_session_end is
        # idempotent, so the shutdown callback stays a no-op backstop.
        await _publish_session_end(SESSION_END_ERROR)
        raise


def _use_ptt_pipeline(interaction_mode: str) -> bool:
    # Only push-to-talk uses the button-driven segment pipeline. full_duplex AND
    # half_duplex both run the streaming (EOT-commit) pipeline — they differ only
    # in barge-in (allow_interruptions), not in turn detection.
    return interaction_mode == INTERACTION_MODE_PTT


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
        # The runtime authority path is mandatory for eidolon_agent. Channel must
        # have a usable HMAC secret at startup; without one, every session
        # would die at first chat().
        if not cfg.runtime_authority.enabled:
            errors.append(
                "runtime_authority.enabled=false is not valid for eidolon_agent. Set enabled=true."
            )
        elif not (cfg.runtime_authority.jwt_secret or _runtime_secret_file().is_file()):
            errors.append(
                f"PAIRING_JWT_SECRET empty AND {_runtime_secret_file()} "
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
        # Model loading, not a liveness probe — see WorkerConfig.setup_timeout_sec.
        initialize_process_timeout=cfg.worker.setup_timeout_sec,
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
        agent_name=cfg.worker.agent_name,
        type=WorkerType.PUBLISHER,
    )(_on_session)

    return server


async def _serve(
    *, on_server_ready: Callable[["AgentServer"], None] | None = None
) -> None:
    cfg = load_agent_config()
    _validate_config(cfg)

    global _agent_config
    _agent_config = cfg

    logger.info(
        "[Server] starting env=%s url=%s llm=%s idle_procs=%s",
        os.getenv("EIDOLON_ENV", "prod"),
        cfg.core.livekit_url,
        cfg.llm.model,
        _resolve_num_idle_processes(cfg),
    )

    server = _build_server(cfg)
    if on_server_ready is not None:
        on_server_ready(server)

    is_dev = os.getenv("EIDOLON_ENV", "prod") == "dev"
    await server.run(devmode=is_dev)


async def _close_server(server: "AgentServer") -> None:
    """Close the LiveKit worker through its public lifecycle API."""

    await server.aclose()


def main() -> None:
    env = os.getenv("EIDOLON_ENV", "prod")
    log_dir = _resolve_log_dir()
    _normalize_optional_log_file_env("EIDOLON_EOT_DEBUG_LOG", log_dir)
    _configure_logging(env, log_dir=log_dir, log_to_file=True)
    _register_plugins()

    loop = asyncio.new_event_loop()
    server_task: asyncio.Task | None = None
    shutdown_task: asyncio.Task[None] | None = None
    active_server: AgentServer | None = None

    def _remember_server(server: AgentServer) -> None:
        nonlocal active_server
        active_server = server

    async def _run():
        nonlocal server_task
        server_task = asyncio.create_task(
            _serve(on_server_ready=_remember_server),
            name="eidolon_agent_server",
        )
        await server_task

    def _signal_handler(sig: signal.Signals):
        nonlocal shutdown_task
        logger.info("[Server] received %s, initiating graceful shutdown...", sig.name)
        if active_server is not None and (shutdown_task is None or shutdown_task.done()):
            shutdown_task = loop.create_task(
                _close_server(active_server),
                name="eidolon_agent_server_shutdown",
            )

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler, sig)

    try:
        loop.run_until_complete(_run())
    finally:
        if shutdown_task is not None:
            loop.run_until_complete(shutdown_task)
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.run_until_complete(loop.shutdown_default_executor())
        loop.close()


if __name__ == "__main__":
    main()
