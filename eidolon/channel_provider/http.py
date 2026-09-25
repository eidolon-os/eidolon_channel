"""Authenticated aiohttp entry surface for Hub Channel Provider control."""

from __future__ import annotations

import hmac
import json
import logging
from datetime import datetime
from typing import Any

from aiohttp import web

from .contracts import (
    CLOSE_SESSION,
    OPEN_SESSION,
    ContractError,
    DomainError,
    ProvisionRequest,
    CurrentRequest,
    RevokeRequest,
    SessionRequest,
    Unauthenticated,
)
from .service import ChannelProviderService
from .session_traces import SessionTraceReader, TraceQuery

logger = logging.getLogger("eidolon.channel_provider.http")


def create_app(
    *,
    service: ChannelProviderService,
    bearer_token: str,
    traces: SessionTraceReader | None = None,
) -> web.Application:
    app = web.Application(client_max_size=256 * 1024)
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
    app.router.add_post("/v1/device-channels/provision", provision)
    app.router.add_post("/v1/device-channels/revoke", revoke)
    app.router.add_post("/v1/device-channels/current", current)

    async def open_session(request: web.Request) -> web.Response:
        return await session(request, expected=OPEN_SESSION)

    async def close_session(request: web.Request) -> web.Response:
        return await session(request, expected=CLOSE_SESSION)

    app.router.add_post("/v1/device-channels/sessions/open", open_session)
    app.router.add_post("/v1/device-channels/sessions/close", close_session)
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
    *, code: str, category: str, retryable: bool, status: int, detail: str
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
    return web.Response(
        text=json.dumps(value, ensure_ascii=False, separators=(",", ":")),
        status=status,
        content_type="application/problem+json",
    )
