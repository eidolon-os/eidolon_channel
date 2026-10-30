"""Authenticated aiohttp entry surface for Hub Channel Provider control."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
from datetime import datetime
from typing import Any

from aiohttp import web
from eidolon_sdk.biz.smarthome import ExecuteRequest, VoiceResult
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .contracts import (
    CLOSE_SESSION,
    OPEN_SESSION,
    BackendUnavailable,
    ContractError,
    CurrentRequest,
    DomainError,
    ProvisionRequest,
    RevokeRequest,
    SessionRequest,
    Unauthenticated,
)
from .device_capabilities import DeviceCommandRequest
from .service import ChannelProviderService
from .session_traces import SessionTraceReader, TraceQuery
from .shared_session_contract import CloseSharedSession, OpenSharedSession

logger = logging.getLogger("eidolon.channel_provider.http")

#: How long a state stream may stay silent before a comment line is written.
#: Proxies and mobile radios drop an idle connection; a change minutes later
#: would then have nowhere to go, and the client would look merely stale.
#: Kept well below the shortest idle timeout any client is likely to impose —
#: Dart's HttpClient defaults to 15s, and a heartbeat at exactly that interval
#: races it, losing the stream every time the device stays quiet.
_SSE_HEARTBEAT_SECONDS = 5.0


class _HomeScope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    owner_id: str = Field(min_length=1, max_length=128)


class _HomeExecute(_HomeScope):
    request: ExecuteRequest


class _HomeResult(_HomeScope):
    device_ref: str = Field(min_length=1, max_length=128)
    result: VoiceResult


class _DeviceCommand(_HomeScope):
    #: Optional because a command need not come from a Companion's session: the
    #: Owner's own app addresses a body by device id. An absent value cannot
    #: match "serving a different Companion", which is the check that matters.
    companion_id: str = Field(default="", max_length=128)
    idempotency_key: str = Field(min_length=1, max_length=128)
    device_id: str = Field(min_length=1, max_length=128)
    op: str = Field(min_length=3, max_length=128)
    capability_version: int = Field(ge=1, le=65535)
    payload: dict[str, Any] = Field(default_factory=dict)
    ttl_ms: int = Field(default=30_000, ge=1_000, le=600_000)


def create_app(
    *,
    service: ChannelProviderService,
    bearer_token: str,
    traces: SessionTraceReader | None = None,
    speaker_engine=None,
    voice_capture=None,
    speech_voices=None,
) -> web.Application:
    app = web.Application(client_max_size=256 * 1024)
    from .speaker_extraction import register_speaker_extraction, speaker_recognition_status
    register_speaker_extraction(app, engine=speaker_engine,
                               authorized=lambda request: _authorized(request, bearer_token),
                               problem=_problem_body)
    from .voice_capture import register_voice_capture
    register_voice_capture(app, capture=voice_capture,
                           authorized=lambda request: _authorized(request, bearer_token))
    if speech_voices is not None:
        from .speech_voices import register_speech_voices
        catalog, previews = speech_voices
        register_speech_voices(app, catalog=catalog, previews=previews,
                               authorized=lambda request: _authorized(request, bearer_token),
                               problem=_problem_body)
    trace_reader = traces if traces is not None else SessionTraceReader(None)

    async def health(_request: web.Request) -> web.Response:
        try:
            await service.healthcheck()
        except Exception:
            logger.exception("Channel Provider health check failed")
            return _json({"status": "unavailable"}, status=503)
        return _json(
            {
                "status": "ok",
                "service": "eidolon-channel-provider",
                "contract_version": "v1",
                # A capability's own state, beside the service's: recognition
                # being off never makes the Provider unhealthy.
                "speaker_recognition": speaker_recognition_status(speaker_engine),
            }
        )

    async def provision(request: web.Request) -> web.Response:
        if not _authorized(request, bearer_token):
            return _problem(Unauthenticated("bearer credential was not accepted"))
        if request.content_type != "application/json":
            return _contract_problem("content-type must be application/json", status=415)
        try:
            result = await service.provision(ProvisionRequest.parse(await request.read()))
            return web.Response(text=result, content_type="application/json")
        except ContractError as exc:
            return _contract_problem(str(exc), status=422)
        except DomainError as exc:
            return _problem(exc)

    async def revoke(request: web.Request) -> web.Response:
        if not _authorized(request, bearer_token):
            return _problem(Unauthenticated("bearer credential was not accepted"))
        if request.content_type != "application/json":
            return _contract_problem("content-type must be application/json", status=415)
        try:
            result = await service.revoke(RevokeRequest.parse(await request.read()))
            return web.Response(text=result, content_type="application/json")
        except ContractError as exc:
            return _contract_problem(str(exc), status=422)
        except DomainError as exc:
            return _problem(exc)

    async def current(request: web.Request) -> web.Response:
        if not _authorized(request, bearer_token):
            return _problem(Unauthenticated("bearer credential was not accepted"))
        if request.content_type != "application/json":
            return _contract_problem("content-type must be application/json", status=415)
        try:
            result = await service.current(CurrentRequest.parse(await request.read()))
            return web.Response(text=result, content_type="application/json")
        except ContractError as exc:
            return _contract_problem(str(exc), status=422)
        except DomainError as exc:
            return _problem(exc)

    async def session(request: web.Request, *, expected: str) -> web.Response:
        if not _authorized(request, bearer_token):
            return _problem(Unauthenticated("bearer credential was not accepted"))
        if request.content_type != "application/json":
            return _contract_problem("content-type must be application/json", status=415)
        try:
            parsed = SessionRequest.parse(await request.read(), expected=expected)
            result = (
                await service.open_session(parsed)
                if expected == OPEN_SESSION
                else await service.close_session(parsed)
            )
            return web.Response(text=result, content_type="application/json")
        except ContractError as exc:
            return _contract_problem(str(exc), status=422)
        except DomainError as exc:
            return _problem(exc)

    async def start(_app: web.Application) -> None:
        # Channels outlive this process, so resuming what we were listening to
        # is part of coming up, not something a later request can stand in for.
        await service.start()

    async def close(_app: web.Application) -> None:
        await service.shutdown()

    app.on_startup.append(start)
    async def presence(request: web.Request) -> web.Response:
        """Which bodies are on their channel. A GET, because it asks nothing.

        Every other route here takes a contract body naming one device, because
        every other route acts on one device. This one has no input: nothing on
        this Host knows which bodies to ask about — the composition that wants
        presence has no device list of its own, which is the reason it had none
        to join presence onto. So the channel answers for the channels it
        granted, and the caller keeps only the rows it has standing for.
        """

        if not _authorized(request, bearer_token):
            return _problem(Unauthenticated("bearer credential was not accepted"))
        try:
            return web.Response(text=await service.presence(), content_type="application/json")
        except DomainError as exc:
            return _problem(exc)

    async def session_traces(request: web.Request) -> web.Response:
        """What voice sessions this Host recorded. A GET, because it asks nothing.

        Reads, like ``presence``, rather than acting: the worker wrote these
        files and this process is the same authority, so serving them needs no
        state of its own. Filters are bounds on the answer — an absent trace
        root means none were recorded, which is an empty list and not an error.
        """

        if not _authorized(request, bearer_token):
            return _problem(Unauthenticated("bearer credential was not accepted"))
        try:
            query = _trace_query(request)
        except ValueError as exc:
            return _contract_problem(str(exc), status=422)
        sessions = trace_reader.list(query)
        return _json_body(
            {
                "operation": "channel.session-traces",
                "recording": trace_reader.available,
                "sessions": sessions,
            }
        )

    async def session_trace(request: web.Request) -> web.Response:
        """One session's records, in the order they were written."""

        if not _authorized(request, bearer_token):
            return _problem(Unauthenticated("bearer credential was not accepted"))
        session_id = request.match_info.get("session_id", "")
        kinds = _trace_kinds(request)
        found = trace_reader.read(session_id, kinds=kinds)
        if found is None:
            return _problem_body(
                code="NOT_FOUND",
                category="not_found",
                retryable=False,
                status=404,
                detail=f"no session trace for {session_id!r}",
            )
        return _json_body({"operation": "channel.session-trace", **found})

    app.router.add_get("/health", health)
    app.router.add_get("/v1/session-traces", session_traces)
    app.router.add_get("/v1/session-traces/{session_id}", session_trace)
    app.router.add_get("/v1/device-channels/presence", presence)

    async def capability_directory(request: web.Request) -> web.Response:
        """What each body on this Host says it can do, for the Agent's tools.

        A read like ``presence``: it acts on nothing and answers only for the
        channels this Provider still holds.
        """

        if not _authorized(request, bearer_token):
            return _problem(Unauthenticated("bearer credential was not accepted"))
        try:
            return web.Response(
                text=await service.capability_directory(), content_type="application/json"
            )
        except DomainError as exc:
            return _problem(exc)

    app.router.add_get("/v1/device-channels/capabilities", capability_directory)

    async def device_state(request: web.Request) -> web.Response:
        """What a device last reported changing without being asked.

        A read with no side effect, and only a hint: a key it has never reported
        is absent rather than zero.
        """

        if not _authorized(request, bearer_token):
            return _problem(Unauthenticated("bearer credential was not accepted"))
        try:
            return web.Response(
                text=await service.device_state_snapshot(request.match_info["device_id"]),
                content_type="application/json",
            )
        except DomainError as exc:
            return _problem(exc)

    app.router.add_get("/v1/device-channels/{device_id}/state", device_state)

    async def device_state_stream(request: web.Request) -> web.StreamResponse:
        """Long-lived stream of a device's own state changes.

        The subscriber is registered before the response headers go out, so a
        change landing between "connected" and "listening" is not lost. Events
        carry only what moved; a reader that needs a value it never saw must
        still ask for it.
        """

        if not _authorized(request, bearer_token):
            return _problem(Unauthenticated("bearer credential was not accepted"))
        device_id = request.match_info["device_id"]
        queue: asyncio.Queue = asyncio.Queue(maxsize=64)

        def _observe(key: str, value: int) -> None:
            try:
                queue.put_nowait({"key": key, "value": value})
            except asyncio.QueueFull:
                logger.warning("device state stream dropped an event device=%s", device_id)

        try:
            stop = await service.watch_device_state(device_id, _observe)
        except DomainError as exc:
            return _problem(exc)

        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        try:
            await response.prepare(request)
            await response.write(b"event: ready\ndata: {}\n\n")
            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=_SSE_HEARTBEAT_SECONDS)
                except TimeoutError:
                    await response.write(b": keep-alive\n\n")
                    continue
                body = json.dumps(payload, separators=(",", ":"))
                await response.write(f"event: state\ndata: {body}\n\n".encode())
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            stop()
        return response

    app.router.add_get("/v1/device-channels/{device_id}/state/stream", device_state_stream)

    async def deliver_device_command(request: web.Request) -> web.Response:
        if not _authorized(request, bearer_token):
            return _problem(Unauthenticated("bearer credential was not accepted"))
        try:
            body = _DeviceCommand.model_validate(await request.json())
        except (ValidationError, ValueError) as exc:
            return _contract_problem(str(exc), status=422)
        try:
            result = await service.deliver_device_command(
                DeviceCommandRequest(
                    owner_id=body.owner_id,
                    companion_id=body.companion_id,
                    idempotency_key=body.idempotency_key,
                    device_id=body.device_id,
                    op=body.op,
                    capability_version=body.capability_version,
                    payload=body.payload,
                    ttl_ms=body.ttl_ms,
                )
            )
        except DomainError as exc:
            return _problem(exc)
        return _json_body({"operation": "channel.device-command", **result.to_json()})

    app.router.add_post("/v1/device-channels/commands", deliver_device_command)
    app.router.add_post("/v1/device-channels/provision", provision)
    app.router.add_post("/v1/device-channels/revoke", revoke)
    app.router.add_post("/v1/device-channels/current", current)

    async def open_session(request: web.Request) -> web.Response:
        return await session(request, expected=OPEN_SESSION)
    async def close_session(request: web.Request) -> web.Response:
        return await session(request, expected=CLOSE_SESSION)

    app.router.add_post("/v1/device-channels/sessions/open", open_session)
    app.router.add_post("/v1/device-channels/sessions/close", close_session)

    async def device_conversation(request: web.Request) -> web.Response:
        from eidolon_sdk.biz.control.device_conversation import DeviceConversationSelection
        from pydantic import BaseModel, ConfigDict, Field

        class OpenRequest(BaseModel):
            model_config = ConfigDict(extra="forbid")
            owner_id: str = Field(min_length=1, max_length=255)
            selection: DeviceConversationSelection

        if not _authorized(request, bearer_token):
            return _problem(Unauthenticated("bearer credential was not accepted"))
        if request.content_type != "application/json":
            return _contract_problem("content-type must be application/json", status=415)
        try:
            if request.match_info["action"] == "open":
                payload = OpenRequest.model_validate_json(await request.read())
                result = await service.open_device_conversation(payload.selection,
                    authenticated_owner_id=payload.owner_id)
            else:
                payload = CloseSharedSession.model_validate_json(await request.read())
                result = await service.device_conversation(payload.session_id,
                    authenticated_owner_id=payload.owner_id,
                    close=request.match_info["action"] == "close")
            return _json_body(result)
        except ValidationError as exc:
            return _contract_problem(str(exc), status=422)
        except DomainError as exc:
            return _problem(exc)

    app.router.add_post("/v1/device-conversations/{action:open|status|close}",
                        device_conversation)

    async def device_session(request: web.Request) -> web.Response:
        from eidolon_sdk.biz.control.device_session import DeviceSessionSelection
        from pydantic import BaseModel, ConfigDict, Field

        class OpenRequest(BaseModel):
            model_config = ConfigDict(extra="forbid")
            owner_id: str = Field(min_length=1, max_length=255)
            selection: DeviceSessionSelection

        if not _authorized(request, bearer_token):
            return _problem(Unauthenticated("bearer credential was not accepted"))
        if request.content_type != "application/json":
            return _contract_problem("content-type must be application/json", status=415)
        try:
            if request.match_info["action"] == "open":
                payload = OpenRequest.model_validate_json(await request.read())
                result = await service.open_device_session(payload.selection,
                    authenticated_owner_id=payload.owner_id)
            else:
                payload = CloseSharedSession.model_validate_json(await request.read())
                result = await service.device_session(payload.session_id,
                    authenticated_owner_id=payload.owner_id,
                    close=request.match_info["action"] == "close")
            return _json_body(result)
        except ValidationError as exc:
            return _contract_problem(str(exc), status=422)
        except DomainError as exc:
            return _problem(exc)

    app.router.add_post("/v1/device-sessions/{action:open|status|close}",
                        device_session)


    async def team_conversation(request):
        from eidolon_sdk.biz.control.coordination_stream import OpenScene
        if not _authorized(request, bearer_token):
            return _problem(Unauthenticated("bearer credential was not accepted"))
        if request.content_type != "application/json":
            return _contract_problem("content-type must be application/json", status=415)
        try:
            if request.match_info['action'] == 'open':
                opened = OpenScene.model_validate_json(await request.read())
                result = await service.open_team_conversation(opened,
                    authenticated_owner_id=opened.owner_id)
            else:
                payload = CloseSharedSession.model_validate_json(await request.read())
                result = await service.device_conversation(payload.session_id,
                    authenticated_owner_id=payload.owner_id, team=True,
                    close=request.match_info['action'] == 'close')
            return _json_body(result)
        except ValidationError as exc:
            return _contract_problem(str(exc), status=422)
        except DomainError as exc:
            return _problem(exc)

    app.router.add_post('/v1/role-groups/{action:open|status|close}', team_conversation)

    async def shared_session(request: web.Request) -> web.Response:
        if not _authorized(request, bearer_token):
            return _problem(Unauthenticated("bearer credential was not accepted"))
        if request.content_type != "application/json":
            return _contract_problem("content-type must be application/json", status=415)
        try:
            if request.match_info["action"] == "open":
                command = OpenSharedSession.model_validate_json(await request.read())
                result = await service.open_shared_session(
                    command.selection,
                    tuple(s.model_dump(mode="json") for s in command.specifications),
                    authenticated_owner_id=command.owner_id,
                )
            else:
                command = CloseSharedSession.model_validate_json(await request.read())
                result = await service.close_shared_session(
                    command.session_id, authenticated_owner_id=command.owner_id,
                )
            return _json(result)
        except (ValidationError, ContractError):
            return _contract_problem("invalid shared session command", status=422)
        except DomainError as exc:
            # Domain messages are authored by this service, never request bodies
            # or transport credentials. Keep the exact refusal at its boundary.
            logger.warning("shared action=%s code=%s detail=%s",
                           request.match_info["action"], exc.code, str(exc))
            return _problem(exc)
        except TimeoutError:
            logger.warning("shared action=%s admission deadline elapsed",
                           request.match_info["action"])
            return _problem_body(code="PROVIDER_UNAVAILABLE", category="unavailable", retryable=True,
                                 status=503, detail="shared admission deadline elapsed")

    app.router.add_post("/v1/shared-sessions/{action:open|close}", shared_session)

    async def smarthome(request: web.Request) -> web.Response:
        if not _authorized(request, bearer_token):
            return _problem(Unauthenticated("bearer credential was not accepted"))
        if request.content_type != "application/json":
            return _contract_problem("content-type must be application/json", status=415)
        try:
            action = request.match_info["action"]
            body = await request.read()
            command = _HomeResult.model_validate_json(body)
            await service.smarthome_result(
                command.owner_id, command.device_ref, command.result
            )
            return _json_body({"status": "sent"})
        except ValidationError as exc:
            return _contract_problem(str(exc), status=422)
        except DomainError as exc:
            return _problem(exc)
        except Exception:
            logger.exception("smart home %s failed", request.match_info["action"])
            return _problem(BackendUnavailable("smart home service unavailable"))

    app.router.add_post("/v1/smarthome/{action:result}", smarthome)
    app.on_cleanup.append(close)
    return app


def _authorized(request: web.Request, expected: str) -> bool:
    prefix = "Bearer "
    value = request.headers.get("Authorization", "")
    return value.startswith(prefix) and hmac.compare_digest(value[len(prefix) :], expected)


def _trace_query(request: web.Request) -> TraceQuery:
    """Parse the listing filters, refusing a malformed one rather than ignoring it.

    A filter silently dropped is worse than a rejected request: the caller reads
    the unfiltered answer as the filtered one.
    """

    fields: dict[str, Any] = {
        "owner_id": request.query.get("owner_id", ""),
        "companion_id": request.query.get("companion_id", ""),
    }
    raw_limit = request.query.get("limit", "").strip()
    if raw_limit:
        if not raw_limit.isdigit():
            raise ValueError("limit must be a positive integer")
        # Left unset when absent so the dataclass default applies. Reading the
        # default off the class does not work here: ``slots=True`` replaces it
        # with a slot descriptor, which then fails the bounds comparison.
        fields["limit"] = int(raw_limit)
    raw_since = request.query.get("since", "").strip()
    if raw_since:
        try:
            since = datetime.fromisoformat(raw_since)
        except ValueError as exc:
            raise ValueError("since must be an ISO-8601 date or datetime") from exc
        fields["since"] = since if since.tzinfo else since.astimezone()
    return TraceQuery(**fields)


def _trace_kinds(request: web.Request) -> frozenset[str] | None:
    raw = request.query.get("kinds", "").strip()
    if not raw:
        return None
    kinds = {piece.strip() for piece in raw.split(",") if piece.strip()}
    return frozenset(kinds) or None


def _json_body(value: dict[str, Any], *, status: int = 200) -> web.Response:
    return web.Response(
        text=json.dumps(value, ensure_ascii=False, separators=(",", ":")),
        status=status,
        content_type="application/json",
    )


def _json(value: dict[str, str], *, status: int = 200) -> web.Response:
    return web.Response(
        text=json.dumps(value, ensure_ascii=False, separators=(",", ":")),
        status=status,
        content_type="application/json",
    )


def _contract_problem(detail: str, *, status: int) -> web.Response:
    return _problem_body(
        code="INVALID_ARGUMENT",
        category="invalid",
        retryable=False,
        status=status,
        detail=detail,
    )


def _problem(error: DomainError) -> web.Response:
    return _problem_body(
        code=error.code,
        category=error.category,
        retryable=error.retryable,
        status=error.http_status,
        detail=str(error),
    )


def _problem_body(
    *, code: str, category: str, retryable: bool, status: int, detail: str,
    audio_quality: dict | None = None,
) -> web.Response:
    value = {
        "type": f"https://problems.eidolon.live/channel/{code.lower()}",
        "title": code.replace("_", " ").title(),
        "status": status,
        "detail": detail,
        "code": code,
        "category": category,
        "retryable": retryable,
        "authority": "eidolon-channel-provider",
    }
    if audio_quality is not None:
        value["audio_quality"] = audio_quality
    return web.Response(
        text=json.dumps(value, ensure_ascii=False, separators=(",", ":")),
        status=status,
        content_type="application/problem+json",
    )
