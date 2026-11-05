"""Voice agent server — LiveKit agent worker.

This server runs as a LiveKit agent worker, connecting to LiveKit rooms as an
agent participant. The Channel Provider explicitly dispatches a job when a
session is requested; joining a standing room alone does not start processing.
The worker combines trusted dispatch settings with participant interaction mode
to assemble the session pipeline.

Recommended local startup: eidolon_ops ``./eidolon mac start`` (loads the
generated channel env via ``with-env.sh``); restart only this worker with
``./eidolon mac service restart channel``.

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
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from livekit.agents import AgentServer

from eidolon_sdk.biz.presentation import SessionOutputPlan
from eidolon_sdk.biz.contracts import (
    INTERACTION_MODE_PTT,
    INTERACTION_MODE_HALF_DUPLEX,
    SESSION_CONVERSATION_ID_FIELD,
    SESSION_APPLICATION_COMPANION,
    SESSION_APPLICATION_FIELD,
    SESSION_APPLICATION_HOME_COMMAND,
    VALID_SESSION_APPLICATIONS,
    SESSION_CONTROL_TOPIC,
    SESSION_END_ERROR,
    SESSION_END_TYPE,
    SESSION_END_USER_LEFT,
    SESSION_STARTED_TYPE,
    WIRE_SCHEMA_VERSION,
    normalize_conversation_id,
)
from eidolon.livekit.common.config import AgentConfig, load_agent_config
from eidolon.livekit.common.entry_timing import entry_record, format_entry
from eidolon.locked_environment import require_locked_environment

# Load the session-contract resolver with the worker, not lazily per job.
# LiveKit job processes inherit the worker's module snapshot; eager loading
# prevents a hot-updated Channel module from mixing with a stale SDK contract
# already resident in the parent process.
from eidolon.livekit.agent.runtime import (
    apply_interaction_mode,
    resolve_audio_input_processing,
    resolve_avatar_requested,
    resolve_interaction_mode,
    resolve_session_intent,
)
from eidolon.livekit.agent.runtime.resolver import (
    DeviceTokenResolverError,
    wait_for_runtime_participant_metadata,
)

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
        from eidolon.livekit.common.config import load_effective_config
        from eidolon.livekit.common.welcome import prepare_welcome_audio

        cfg = load_effective_config()
        prepare_welcome_audio(cfg.behavior.welcome_message, sample_rate=cfg.behavior.audio_sample_rate)
        _load_prewarm_models(proc)
    finally:
        logger.info("[Agent] prewarm: finished in %.1fs", time.monotonic() - started)


#: Where the worker publishes what its job processes could load, beside its
#: other runtime state. livekit's own ``/`` cannot say it: on 2026-10-31 the
#: VAD and end-of-turn ONNX were LFS pointers, prewarm logged two warnings,
#: and the worker answered healthy while no session could take turns.
TURN_MODELS_STATUS = "channel/turn-models.json"


def _turn_models_status_file() -> Path:
    root = Path(os.getenv("EIDOLON_RUNTIME_ROOT", "~/eidolon/run")).expanduser()
    return root / TURN_MODELS_STATUS


def _publish_turn_models(status: dict) -> None:
    """Replace the published status atomically; never fail the caller."""

    path = _turn_models_status_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        staged = path.with_name(f".{path.name}.{os.getpid()}")
        staged.write_text(json.dumps({**status, "pid": os.getpid()}), encoding="utf-8")
        os.replace(staged, path)
    except OSError:
        logger.warning("[Agent] cannot publish turn model status at %s", path)


def _job_process_cost() -> dict:
    """What building this job process cost, from its creation until now.

    Every conversation takes a warm process and starts building the next, so
    this is how long the next conversation would wait were none left. CPU time
    is the budgeted figure: wall time also counts everything else starting with
    it (a whole Host booting), CPU time only the work this process did --
    which is where compiling imports from source or loading torch shows up.
    """

    import psutil

    process = psutil.Process()
    times = process.cpu_times()
    return {
        "ready_seconds": round(time.time() - process.create_time(), 2),
        "ready_cpu_seconds": round(times.user + times.system, 2),
    }


def _load_prewarm_models(proc) -> None:
    loaded = {"vad": {"state": "failed"}, "eot": {"state": "failed"}}
    try:
        _load_turn_models(proc, loaded)
    finally:
        both = all(part["state"] == "loaded" for part in loaded.values())
        proc.userdata["warmed_at"] = time.time()
        try:
            from eidolon.livekit.common.config import load_effective_config

            warm = {"warm_processes": _resolve_num_idle_processes(load_effective_config())}
        except Exception:
            warm = {}
        _publish_turn_models({
            "state": "loaded" if both else "failed", **loaded, **_job_process_cost(), **warm,
        })


def _load_turn_models(proc, loaded: dict) -> None:
    provider = "unknown"
    try:
        from eidolon.livekit.agent.providers.vad import canonical_vad_provider, load_vad
        from eidolon.livekit.common.config import load_effective_config

        cfg = load_effective_config()
        provider = canonical_vad_provider(cfg.providers.vad_provider)
        proc.userdata["vad"] = load_vad(cfg.providers.vad_provider, cfg.turn_policy.vad)
        if proc.userdata["vad"] is None:
            raise ValueError("VAD disabled by configuration")
        logger.info("[Agent] prewarm: VAD %s loaded", provider)
        loaded["vad"] = {"state": "loaded", "provider": provider}
    except Exception as e:
        logger.warning("[Agent] prewarm: VAD load failed: %s", e)
        loaded["vad"] = {"state": "failed", "provider": provider, "error": str(e)[:200]}

    try:
        from eidolon.livekit.common.config import load_effective_config

        eot_policy = load_effective_config().turn_policy.eot
        if eot_policy.provider == "livekit_mini":
            from livekit.local_inference import init_eot

            init_eot()
        else:
            from eidolon.livekit.plugins.eot import ChineseModel

            proc.userdata["eot_model"] = ChineseModel()
        logger.info("[Agent] prewarm: EOT provider=%s loaded", eot_policy.provider)
        loaded["eot"] = {"state": "loaded", "provider": eot_policy.provider}
    except Exception as e:
        logger.warning("[Agent] prewarm: EOT load failed: %s", e)
        loaded["eot"] = {"state": "failed", "error": str(e)[:200]}



async def _resolve_session_metadata(ctx) -> tuple[str, bool, dict[str, str]]:
    """Resolve ``(interaction_mode, avatar_requested, audio_input_processing)``.

    What this bus carries is what the joining actor is entitled to say about
    itself: how its hardware takes turns and which microphone processing it
    already applies (device manifest facts the Channel Provider stamps into the
    channel credential) and whether it wants a face
    (a web client's own declaration). Neither can be overclaimed into a
    privilege — a wrong turn-taking mode only makes the claimant's own session
    worse, and the avatar is still gated by `cfg.avatar.enabled`, so asking is
    an opt-in within what the deployment already allows. That is why the actor
    gets to say them.

    Infrastructure participants such as the Hub control bridge may join first,
    so select the explicitly typed runtime actor and read every value from that
    ONE metadata snapshot before building the pipeline. Connection and actor
    resolution failures are fatal; optional metadata fields retain their safe
    defaults.

    ``session_intent`` used to be read here and is not any more — see
    ``_resolve_session_intent``. It is not a fact about the actor, it is a
    statement about one conversation, and the actor is not who gets to make it.
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
        resolve_avatar_requested(metadata),
        resolve_audio_input_processing(metadata),
    )


def _resolve_session_intent(ctx) -> str:
    """Read why this session exists from the Provider's job dispatch.

    Same authority as ``_resolve_output_plan``, for the same reason: this is
    not a description, it is what the session is permitted to do.
    ``presence_initiated`` is answered with an externally governed renewable
    Owner lease, and ``proactive_initiated`` changes both the opening and the
    teardown of an unanswered session. A participant that could name its own
    intent would be granting itself all of that, so the value is taken from the
    one bus in the room that no participant can write: the agent dispatch, which
    only the Channel Provider creates. The device's channel credential is minted
    with ``can_update_own_metadata=False`` and carries no intent at all, so a
    participant claiming one in its own metadata is ignored, not obeyed.

    Missing or unrecognised degrades to ``user_initiated``. The Provider's
    control contract rejects a typo outright, where an authenticated caller is
    there to be told; by the time a dispatch reaches this process the only way
    to say nothing is to be a Provider older than the field, and the safe
    reading of silence is an ordinary user-driven session.
    """
    return resolve_session_intent(getattr(ctx.job, "metadata", None))


def _resolve_session_target(ctx) -> str | None:
    """Only the Provider dispatch may choose this device session's temporary target."""
    from eidolon.interaction_context import validate_companion_target

    metadata = json.loads(str(getattr(ctx.job, "metadata", "") or "{}"))
    if "target_companion_id" not in metadata:
        return None
    return validate_companion_target(metadata["target_companion_id"])


def _resolve_personal_memory(ctx):
    from eidolon.interaction_context import PersonalMemorySelection
    from eidolon_sdk.biz.contracts import SESSION_APPLICATION_FIELD, SESSION_APPLICATION_COMPANION

    metadata = json.loads(str(getattr(ctx.job, "metadata", "") or "{}"))
    if "personal_memory" not in metadata:
        return None
    if (metadata.get("team_dispatch") is not None
            or metadata.get("presentation_endpoint") is not None
            or metadata.get(SESSION_APPLICATION_FIELD, SESSION_APPLICATION_COMPANION) != SESSION_APPLICATION_COMPANION):
        raise ValueError("personal memory requires an ordinary Companion session")
    return PersonalMemorySelection.model_validate(metadata["personal_memory"])


def _resolve_output_plan(ctx, session_id: str) -> SessionOutputPlan | None:
    # Only the Provider's authenticated job dispatch supplies the policy.
    # Participant metadata, speech and LLM output cannot grant output rights.
    metadata = json.loads(str(getattr(ctx.job, "metadata", "") or "{}"))
    if "output_plan" not in metadata:
        return None  # existing devices retain their established voice profile
    plan = SessionOutputPlan.model_validate(metadata["output_plan"])
    if plan.session_id != session_id:
        raise ValueError("OUTPUT_PLAN_SESSION_MISMATCH")
    return plan


def _resolve_presentation_endpoint(ctx):
    from eidolon.livekit.common.presentation_endpoint import PresentationEndpoint

    metadata = json.loads(str(getattr(ctx.job, "metadata", "") or "{}"))
    raw = metadata.get("presentation_endpoint")
    if raw is None:
        return None
    endpoint = PresentationEndpoint.model_validate(raw)
    if endpoint.room == ctx.room.name:
        raise ValueError("REMOTE_PRESENTATION_REQUIRES_DISTINCT_ROOM")
    return endpoint


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
    message_type: str, runtime_session_id: str, *, reason: str | None = None,
    output_plan: SessionOutputPlan | None = None,
) -> bytes:
    payload = {
        "schema_v": WIRE_SCHEMA_VERSION,
        "type": message_type,
        SESSION_CONVERSATION_ID_FIELD: runtime_session_id,
    }
    if reason is not None:
        payload["reason"] = reason
    if output_plan is not None:
        payload["output_plan"] = output_plan.model_dump(mode="json")
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


async def run_agent(ctx, cfg: AgentConfig, *, started: asyncio.Event | None = None) -> None:
    """Run the voice pipeline; signal readiness for the job-owned lifetime."""
    from eidolon.livekit.agent.factory import SharedStageFactory
    from eidolon.livekit.agent.half_duplex.pipeline import HalfDuplexPttPipeline
    from eidolon.livekit.agent.full_duplex import StreamingPipeline

    job_started_at = time.time()
    logger.info(
        "[Agent] starting room=%s stt=%s tts=%s vad=%s",
        ctx.room.name,
        cfg.providers.stt_provider,
        cfg.providers.tts_provider,
        cfg.providers.vad_provider,
    )

    metadata = json.loads(ctx.job.metadata or '{}')
    personal_memory = _resolve_personal_memory(ctx)
    if 'team_dispatch' in metadata:
        from eidolon.livekit.common.team_dispatch import TeamDispatch
        from eidolon.livekit.agent.coordination.entrypoint import run_team_dispatch
        team = TeamDispatch.model_validate(metadata['team_dispatch'])
        if (team.input_plan.session_id != _resolve_runtime_session_id(ctx)
                or team.input_plan != _resolve_output_plan(ctx, team.input_plan.session_id)
                or metadata.get('presentation_endpoint') is not None
                or metadata.get('target_companion_id') is not None):
            raise ValueError('TEAM_DISPATCH_SCOPE_MISMATCH')
        await run_team_dispatch(ctx, cfg, metadata['team_dispatch'])
        return

    prebuilt_vad = getattr(ctx.proc, "userdata", {}).get("vad")
    room = ctx.room
    runtime_session_id = _resolve_runtime_session_id(ctx)
    output_plan = _resolve_output_plan(ctx, runtime_session_id)
    target_companion_id = _resolve_session_target(ctx)
    presentation_endpoint = _resolve_presentation_endpoint(ctx)
    presentation_room = None
    if presentation_endpoint is not None:
        if output_plan is None or output_plan.outputs.can_respond or output_plan.outputs.audio_cue:
            raise ValueError("DIRECTED_SOURCE_MUST_BE_INPUT_ONLY")
        if not target_companion_id:
            raise ValueError("DIRECTED_COMPANION_REQUIRED")

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
        if presentation_endpoint is not None:
            await presentation_room.local_participant.publish_data(
                _session_lifecycle_payload(SESSION_STARTED_TYPE,
                    presentation_endpoint.plan.session_id, output_plan=presentation_endpoint.plan),
                reliable=True, topic=SESSION_CONTROL_TOPIC,
                destination_identities=[presentation_endpoint.participant_identity],
            )
        await local.publish_data(
            _session_lifecycle_payload(SESSION_STARTED_TYPE, runtime_session_id, output_plan=output_plan),
            reliable=True,
            topic=SESSION_CONTROL_TOPIC,
        )
        logger.info(
            "[lifecycle] session_started conversation_id=%s room=%s sent",
            runtime_session_id,
            room.name,
        )
        # One line per way in: request to start, where it went, and whether a
        # warm process was waiting (common.entry_timing).
        logger.info(
            "[entry] conversation_id=%s room=%s %s",
            runtime_session_id,
            room.name,
            format_entry(entry_record(
                metadata, job_started_at=job_started_at, now=time.time(),
                warmed_at=getattr(ctx.proc, "userdata", {}).get("warmed_at"),
            )),
        )
        if started is not None:
            started.set()

    async def _publish_session_end(reason: str) -> None:
        if session_end_state["sent"]:
            return
        first_reason = str(session_end_state.get("reason") or "")
        if first_reason:
            reason = first_reason
        else:
            session_end_state["reason"] = reason
        # LiveKit raises while the job has not connected yet. Startup failure
        # still needs to withdraw its dispatch, even when no notice can be sent.
        try:
            local = room.local_participant
        except Exception:
            local = None
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
            # Two situations reach here and only one of them is a defect.
            #
            # The room is still up: the device is sitting in it, reachable, and
            # was not told. Every remaining surface — the phone, the Provider,
            # this log — goes on showing a healthy session, so the silence has
            # to be loud here or it is nowhere.
            #
            # The room is already gone: nobody could have delivered this, and
            # the device knows the channel dropped because it is usually the
            # party that dropped it. The framework tears the room down before
            # shutdown callbacks run, so the backstop lands here on every
            # ordinary user-ended conversation — measured on hardware, an end
            # the Owner asked for arrives as ParticipantRemoved first and
            # `engine is closed` second. Warning on that teaches the reader to
            # skip the warning, and then it is not there for the case above.
            #
            # `reason` cannot make this call: `user_left` is what
            # _end_serving_cb assigns to every non-error shutdown, not a signal
            # that the device asked to stop. The channel's own state can.
            try:
                reachable = bool(room.isconnected())
            except Exception:
                reachable = False
            if reachable:
                logger.warning(
                    "[lifecycle] session_end reason=%s room=%s publish FAILED — the device "
                    "was not told the conversation ended",
                    reason,
                    room.name,
                    exc_info=True,
                )
            else:
                logger.info(
                    "[lifecycle] session_end reason=%s room=%s not sent: the channel was "
                    "already gone",
                    reason,
                    room.name,
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

    application = metadata.get(SESSION_APPLICATION_FIELD, SESSION_APPLICATION_COMPANION)
    if application not in VALID_SESSION_APPLICATIONS:
        raise ValueError("unsupported voice session application")
    if application == SESSION_APPLICATION_HOME_COMMAND:
        owner_id = metadata.get("smarthome_owner")
        device_ref = metadata.get("smarthome_device")
        if not isinstance(owner_id, str) or not owner_id or not isinstance(device_ref, str) or not device_ref:
            raise ValueError("invalid smart-home panel dispatch")
        from eidolon.livekit.agent.smarthome import run_smarthome_session

        async def _home_shutdown(reason: str) -> None:
            await _publish_session_end(
                SESSION_END_ERROR if "error" in str(reason).lower() else SESSION_END_USER_LEFT
            )
            await _end_serving("shutdown callback")

        ctx.add_shutdown_callback(_home_shutdown)
        try:
            # Room participant metadata is unavailable until the job joins.
            await ctx.connect()
            await run_smarthome_session(
                room=room,
                cfg=cfg,
                prebuilt_vad=prebuilt_vad,
                owner_id=owner_id,
                device_ref=device_ref,
                session_id=runtime_session_id,
                target_companion_id=target_companion_id,
                on_started=_publish_session_started,
                on_end=_publish_session_end,
                on_closed=lambda: _end_serving("session closed"),
                on_idle=lambda: _end_serving("idle timeout"),
                session_intent=_resolve_session_intent(ctx),
            )
        except Exception:
            try:
                await _publish_session_end(SESSION_END_ERROR)
            finally:
                await _end_serving("smart-home startup failed")
            raise
        return

    # Two buses, because two different parties are entitled to speak on them.
    # What the actor declares about itself (turn-taking, avatar) comes off its
    # own token metadata; why this session exists comes off the Provider's
    # dispatch, which the actor cannot write. Both are read before the pipeline
    # is built. The old process-wide "batch/streaming" pipeline mode was
    # retired; the Channel worker now always chooses half/full duplex at the
    # session boundary.
    session_intent = _resolve_session_intent(ctx)
    interaction_mode, avatar_requested, audio_input_processing = await _resolve_session_metadata(ctx)
    # A remote speaker cannot supply the input board with its local AEC reference.
    # Preserve explicit PTT; directed open-mic sessions are half-duplex.
    if presentation_endpoint is not None and interaction_mode != INTERACTION_MODE_PTT:
        interaction_mode = INTERACTION_MODE_HALF_DUPLEX
    session_turn_policy, allow_interruptions = apply_interaction_mode(
        turn_policy=cfg.turn_policy,
        allow_interruptions=True,
        interaction_mode=interaction_mode,
    )
    # Construct optional stages from the same effective policy as the pipeline.
    # Non-barge-in sessions must not allocate a channel intent model.
    # Preserve the room reference for the existing lazy brain identity resolver.
    session_key = room.name or ""
    stage_output_plan = output_plan
    if presentation_endpoint is not None:
        stage_output_plan = SessionOutputPlan(
            session_id=runtime_session_id,
            policy_revision=presentation_endpoint.plan.policy_revision,
            inputs=output_plan.inputs, outputs=presentation_endpoint.plan.outputs,
        )
    runtime_services = SharedStageFactory.runtime_services_for(
        cfg, target_companion_id=target_companion_id, personal_memory=personal_memory
    )
    session_voice = None
    if runtime_services is not None and (stage_output_plan is None or stage_output_plan.outputs.speech):
        from eidolon.speech_voice.dispatch import dispatched_voice
        from .session.speech_voice import resolve_session_voice, shielded_companion_id
        session_voice = await resolve_session_voice(
            cfg, companion_id=lambda: shielded_companion_id(runtime_services.resolve_room(room)),
            authority=runtime_services.runtime, dispatched=dispatched_voice(metadata),
        )
    try:
        factory = SharedStageFactory.from_config(
            replace(cfg, turn_policy=session_turn_policy),
            prebuilt_vad=prebuilt_vad,
            livekit_session_key=session_key,
            livekit_room=room,
            runtime_session_id=runtime_session_id,
            output_plan=stage_output_plan,
            target_companion_id=target_companion_id,
            personal_memory=personal_memory,
            runtime_services=runtime_services,
            speech_voice=session_voice.resolved.ref if session_voice else None,
        )
    except BaseException:
        if runtime_services is not None:
            await runtime_services.aclose()
        raise

    speaker_observer = speaker_http = None
    if getattr(factory, "runtime_services", None) is not None:
        from .session.conversation_speaker import create_conversation_speaker
        try:
            speaker_observer, speaker_http = await create_conversation_speaker(
                factory, cfg, room)
        except Exception:
            logger.warning("Conversation speaker initialization unavailable")

    # Video avatar is enabled for this session only if globally available AND the
    # client declared it (full-duplex path for M1). Default off → audio-only.
    avatar_enabled = bool(cfg.avatar.enabled and avatar_requested)
    logger.info(
        "[Agent] interaction_mode=%s session_intent=%s allow_interruptions=%s "
        "attention_enabled=%s avatar_requested=%s avatar_enabled=%s audio_input=%s",
        interaction_mode,
        session_intent,
        allow_interruptions,
        session_turn_policy.attention.enabled,
        avatar_requested,
        avatar_enabled,
        audio_input_processing or "undeclared",
    )
    # The existing manually driven pipeline also serves typed/output-only
    # sessions. No audio turn detector is needed when microphone input is off.
    if _use_ptt_pipeline(interaction_mode) or (output_plan is not None and not output_plan.inputs.microphone):
        pipeline = HalfDuplexPttPipeline(
            factory,
            interaction_mode=interaction_mode,
            instructions=cfg.behavior.instructions,
            welcome_message=(None if presentation_endpoint is not None else cfg.behavior.welcome_message),
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
            speaker_observation=speaker_observer,
            instructions=cfg.behavior.instructions,
            allow_interruptions=allow_interruptions,
            welcome_message=(None if presentation_endpoint is not None else cfg.behavior.welcome_message),
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
            audio_input_processing=audio_input_processing,
            session_intent=session_intent,
            on_session_started=_publish_session_started,
            observability=cfg.observability,
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
    if session_voice is not None:
        pipeline.session_mark("speech_voice", **session_voice.trace_fields())

    runtime_llm = getattr(getattr(factory, "llm", None), "llm", None)
    if getattr(factory, "runtime_services", None) is not None and hasattr(runtime_llm, "prepare_memory_turn"):
        from .runtime.automatic_memory import AutomaticMemorySession
        from .runtime.resolver import _participant_identity_and_metadata
        from .session.playback_stop import stop_playback
        from eidolon_sdk.biz.runtime import resolve_shared_secret

        async def stop_revoked_memory_outputs():
            session = pipeline._session
            if session is None:
                raise RuntimeError("output session is unavailable")
            await session.interrupt(force=True)
            if presentation_endpoint is not None:
                if presentation_room is None:
                    raise RuntimeError("presentation room is unavailable")
                await stop_playback(presentation_room, presentation_endpoint.participant_identity,
                    presentation_endpoint.plan.session_id, presentation_endpoint.plan.policy_revision)
            else:
                participant = _participant_identity_and_metadata(room)
                if participant is None:
                    raise RuntimeError("output participant is unavailable")
                await stop_playback(room, participant[0], runtime_session_id, 0,
                                    sdk_output_stopped=True)

        async def publish_memory_state(payload):
            await room.local_participant.publish_data(json.dumps(payload).encode(),
                topic="eidolon.memory_scope", reliable=True)

        pipeline._prepare_memory_turn = runtime_llm.prepare_memory_turn
        runtime_llm.memory_session = AutomaticMemorySession(
            services=factory.runtime_services, room=room,
            source_session_id=runtime_session_id,
            secret=resolve_shared_secret(cfg.runtime_authority.jwt_secret),
            llm=runtime_llm, stop_outputs=stop_revoked_memory_outputs, publish=publish_memory_state,
            ttl_seconds=cfg.runtime_authority.device_token_ttl_seconds)

    if speaker_observer is not None and getattr(runtime_llm, "memory_session", None) is not None:
        speaker_observer.on_observation = runtime_llm.memory_session.observe

    pipeline._speaker_observation = speaker_observer
    pipeline._speaker_observation_http = speaker_http

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
        if presentation_endpoint is None:
            await pipeline.run(room)
        else:
            if factory.runtime_services is None:
                raise RuntimeError("directed conversation requires authoritative runtime identity")
            await factory.runtime_services.resolve_room(room)
            presentation_room = await presentation_endpoint.connect(
                core=cfg.core, worker_identity=f"presentation-{ctx.job.id}",
            )
            await pipeline.run(room, output_room=presentation_room,
                output_participant_identity=presentation_endpoint.participant_identity)
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
    finally:
        if presentation_room is not None:
            try:
                await presentation_room.local_participant.publish_data(
                    _session_lifecycle_payload(SESSION_END_TYPE,
                        presentation_endpoint.plan.session_id,
                        reason=str(session_end_state.get("reason") or SESSION_END_USER_LEFT)),
                    reliable=True, topic=SESSION_CONTROL_TOPIC,
                    destination_identities=[presentation_endpoint.participant_identity],
                )
            except Exception:
                logger.exception("could not send presentation session end room=%s", presentation_endpoint.room)
            finally:
                await presentation_room.disconnect()


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

    from .session.job_lifecycle import start_job_pipeline

    await start_job_pipeline(ctx, lambda ready: run_agent(ctx, cfg, started=ready))
    logger.info("[_on_session] pipeline started; lifecycle owned by job")


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


#: Every conversation takes a warm job process, and its replacement starts only
#: then, so this many conversations can start together before one waits for a
#: process to be built. A household Host has a few voice bodies.
WARM_PROCESS_CEILING = 3
#: A warmed job process, measured 11-01 once it no longer imported torch:
#: ~0.65 GB resident (1.03 GB before).
WARM_PROCESS_BYTES = 700 * 1024 * 1024
#: Idle warm processes may hold at most this much of the Host's memory.
WARM_PROCESS_MEMORY_SHARE = 0.25


def host_warm_processes(*, cpu_count: int, memory_bytes: int) -> int:
    """How many job processes this Host keeps warm when nobody says otherwise.

    One for every conversation that may start at once, as far as the Host can
    afford: idle processes keep to a quarter of memory and one per two cores,
    because building one competes with the conversation already running. Never
    none — the first conversation would then wait for a process.
    """

    by_memory = int(memory_bytes * WARM_PROCESS_MEMORY_SHARE) // WARM_PROCESS_BYTES
    return max(1, min(WARM_PROCESS_CEILING, cpu_count // 2, by_memory))


def _resolve_num_idle_processes(
    cfg: AgentConfig,
    *,
    env: str | None = None,
    cpu_count: int | None = None,
    memory_bytes: int | None = None,
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
    if runtime_env == "dev":
        return 1
    if memory_bytes is None:
        import psutil

        memory_bytes = psutil.virtual_memory().total
    return host_warm_processes(
        cpu_count=cpu_count if cpu_count is not None else multiprocessing.cpu_count(),
        memory_bytes=memory_bytes,
    )


def host_conversation_capacity(*, cpu_count: float, memory_bytes: int) -> int:
    """User-selected upper reference: 25 jobs per 4 CPUs and 8 GiB.

    This is a resource budget, not a measured audio quality guarantee. Idle
    prewarming remains independently bounded by host_warm_processes.
    """
    return max(1, min(int(cpu_count * 25 / 4), memory_bytes * 25 // (8 * 1024**3)))


def _resolve_max_concurrent_jobs(cfg: AgentConfig) -> int | None:
    configured = cfg.worker.max_concurrent_jobs
    if configured != "auto":
        return configured
    import psutil
    from livekit.agents.utils.hw.cpu import get_cpu_monitor

    cpu_count = get_cpu_monitor().cpu_count()
    memory_bytes = psutil.virtual_memory().total
    # psutil reports physical memory on Linux too; respect container budgets.
    for limit_path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            limit = int(Path(limit_path).read_text().strip())
        except (OSError, ValueError):
            continue
        if limit > 0:
            memory_bytes = min(memory_bytes, limit)
    capacity = host_conversation_capacity(cpu_count=cpu_count, memory_bytes=memory_bytes)
    logger.info("[worker] capacity=auto cpus=%s memory_bytes=%d max_jobs=%d reference=25/4cpu/8GiB",
                cpu_count, memory_bytes, capacity)
    return capacity


def _build_server(cfg: AgentConfig) -> "AgentServer":
    from livekit.agents import AgentServer, JobExecutorType, WorkerType

    env = os.getenv("EIDOLON_ENV", "prod")
    is_dev = env == "dev"
    num_idle = _resolve_num_idle_processes(cfg, env=env)

    admission = {}
    max_jobs = _resolve_max_concurrent_jobs(cfg)
    if max_jobs is not None:
        # LiveKit includes accepted-but-not-launched reservations in its load.
        # Bound the idle pool too, so its first-job reservation estimate cannot
        # admit more jobs than the configured capacity. CPU-based default stays
        # available for dedicated Hosts; shared workstations opt in explicitly.
        num_idle = min(num_idle, max_jobs)
        # Binary-exact units keep the SDK reservation arithmetic exact when
        # the idle pool equals capacity (e.g. 0.9 / 3 * 3 is below 0.9).
        load_scale = 1 << max_jobs.bit_length()
        admission = {
            "load_threshold": max_jobs / load_scale,
            "load_fnc": lambda worker: min(len(worker.active_jobs) / load_scale, 1.0),
        }
        logger.info("[worker] admission=concurrency max_jobs=%d idle_processes=%d", max_jobs, num_idle)

    prometheus_port = int(os.getenv("AGENT_PROMETHEUS_PORT", "0") or "0")
    prometheus_multiproc_dir = os.getenv("AGENT_PROMETHEUS_MULTIPROC_DIR", "")
    if prometheus_port and not prometheus_multiproc_dir:
        prometheus_multiproc_dir = "/tmp/eidolon_prom_multiproc"

    server = AgentServer(
        **admission,
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
        type=WorkerType.ROOM,
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

    # Until a job process has loaded them, nobody knows; a file from the last
    # run must not answer for this one.
    _publish_turn_models({"state": "pending"})
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
    # Right after logging, before plugins and before any room is joined: a venv
    # below the declared floor shows up as a TypeError 1.5 s into every session
    # otherwise, which is the one place nobody is watching.
    require_locked_environment(logger=logger)
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
