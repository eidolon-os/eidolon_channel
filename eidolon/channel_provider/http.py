"""Authenticated aiohttp entry surface for Hub Channel Provider control."""

from __future__ import annotations

import hmac
import json
import logging

from aiohttp import web

from .contracts import (
    BackendUnavailable,
    ContractError,
    IdempotencyConflict,
    ProvisionRequest,
    RevokeRequest,
)
from .service import ChannelProviderService

logger = logging.getLogger("eidolon.channel_provider.http")


def create_app(
    *,
    service: ChannelProviderService,
    bearer_token: str,
) -> web.Application:
    app = web.Application(client_max_size=256 * 1024)

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
            return _json({"error": "unauthorized"}, status=401)
        if request.content_type != "application/json":
            return _json({"error": "content-type must be application/json"}, status=415)
        try:
            result = await service.provision(ProvisionRequest.parse(await request.read()))
            return web.Response(text=result, content_type="application/json")
        except ContractError:
            return _json({"error": "invalid channel provision request"}, status=422)
        except IdempotencyConflict:
            return _json({"error": "channel provision conflict"}, status=409)
        except BackendUnavailable:
            logger.warning("LiveKit unavailable during channel provision")
            return _json({"error": "channel backend unavailable"}, status=503)

    async def revoke(request: web.Request) -> web.Response:
        if not _authorized(request, bearer_token):
            return _json({"error": "unauthorized"}, status=401)
        if request.content_type != "application/json":
            return _json({"error": "content-type must be application/json"}, status=415)
        try:
            result = await service.revoke(RevokeRequest.parse(await request.read()))
            return web.Response(text=result, content_type="application/json")
        except ContractError:
            return _json({"error": "invalid channel revocation request"}, status=422)
        except IdempotencyConflict:
            return _json({"error": "channel revocation conflict"}, status=409)
        except BackendUnavailable:
            logger.warning("LiveKit unavailable during channel revocation")
            return _json({"error": "channel backend unavailable"}, status=503)

    async def close(_app: web.Application) -> None:
        await service.shutdown()

    app.router.add_get("/health", health)
    app.router.add_post("/v1/device-channels/provision", provision)
    app.router.add_post("/v1/device-channels/revoke", revoke)
    app.on_cleanup.append(close)
    return app


def _authorized(request: web.Request, expected: str) -> bool:
    prefix = "Bearer "
    value = request.headers.get("Authorization", "")
    return value.startswith(prefix) and hmac.compare_digest(value[len(prefix) :], expected)


def _json(value: dict[str, str], *, status: int = 200) -> web.Response:
    return web.Response(
        text=json.dumps(value, ensure_ascii=False, separators=(",", ":")),
        status=status,
        content_type="application/json",
    )
