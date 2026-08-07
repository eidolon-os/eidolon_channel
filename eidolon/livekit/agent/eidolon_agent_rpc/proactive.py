"""Consume ``EidolonAgent.SubscribeProactive`` so the agent can speak unprompted.

The Chat() path (``session.py``) is request-driven: the user says something, the
brain answers. Proactive reports are the opposite — the brain decides on its own
that it has something to say (e.g. a long background task finished) and pushes a
:class:`ProactiveEvent` down a server-streaming RPC. This subscriber owns a
dedicated gRPC channel for that stream, kept separate from the Chat() channel so
its reconnect/backoff loop can't disturb in-flight turns.

It is deliberately thin: open the stream, hand each event's spoken text to the
``on_event`` callback (the pipeline turns it into TTS), and reconnect with
backoff if the stream drops. Delivery is best-effort core NATS on the server
side — if the client is offline when the event fires, it is lost (acceptable for
the demo; durable delivery would need JetStream).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable

import grpc
import grpc.aio
from eidolon_sdk.core.grpc import (
    GrpcTlsConfig,
    authorization_metadata,
    build_channel_credentials,
    create_aio_channel_with_credentials,
)

from eidolon.livekit.agent.eidolon_agent_rpc.v1.grpc_gen import (
    eidolon_pb2 as pb,
)
from eidolon.livekit.agent.eidolon_agent_rpc.v1.grpc_gen import (
    eidolon_pb2_grpc as pbg,
)

logger = logging.getLogger("eidolon_agent_rpc.proactive")

# Reconnect backoff bounds. The stream is long-lived and idle most of the time;
# a dropped connection (agent restart, network blip) should re-establish quickly
# but not hammer a down server.
_RECONNECT_DELAY_MIN_SEC = 1.0
_RECONNECT_DELAY_MAX_SEC = 30.0


@dataclass(frozen=True, slots=True)
class ProactiveReport:
    """One proactive announcement pushed by the brain."""

    instance_id: str
    intent: str
    text: str
    style_hint: str


ProactiveHandler = Callable[[ProactiveReport], Awaitable[None]]


class ProactiveSubscriber:
    """Owns a dedicated ``SubscribeProactive`` stream with reconnect/backoff."""

    def __init__(
        self,
        *,
        target: str,
        device_token: str,
        on_event: ProactiveHandler,
        tls: GrpcTlsConfig | None = None,
    ) -> None:
        self._target = target.strip()
        if not self._target:
            raise ValueError("ProactiveSubscriber: target must be non-empty")
        self._metadata = authorization_metadata(device_token)
        self._tls = tls or GrpcTlsConfig()
        self._credentials = build_channel_credentials(self._tls)
        self._on_event = on_event
        self._channel: grpc.aio.Channel | None = None
        self._closed = False

    async def run(self) -> None:
        """Consume the stream forever, reconnecting with backoff until closed."""
        delay = _RECONNECT_DELAY_MIN_SEC
        while not self._closed:
            try:
                await self._consume_once()
                # Clean server-side stream end (EOF) — reset backoff and reopen.
                delay = _RECONNECT_DELAY_MIN_SEC
            except asyncio.CancelledError:
                raise
            except grpc.aio.AioRpcError as exc:
                logger.warning(
                    "[ProactiveSubscriber] stream ended code=%s; reconnecting in %.1fs",
                    exc.code(),
                    delay,
                )
            except Exception:
                logger.exception(
                    "[ProactiveSubscriber] stream crashed; reconnecting in %.1fs",
                    delay,
                )
            if self._closed:
                break
            await asyncio.sleep(delay)
            delay = min(delay * 2, _RECONNECT_DELAY_MAX_SEC)

    async def _consume_once(self) -> None:
        channel = await self._ensure_channel()
        stub = pbg.EidolonAgentStub(channel)
        call = stub.SubscribeProactive(
            pb.SubscribeRequest(),
            metadata=self._metadata,
        )
        logger.info("[ProactiveSubscriber] subscribed target=%s", self._target)
        async for event in call:
            if self._closed:
                break
            report = ProactiveReport(
                instance_id=event.instance_id,
                intent=event.intent,
                text=event.text,
                style_hint=event.style_hint,
            )
            try:
                await self._on_event(report)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "[ProactiveSubscriber] on_event handler failed intent=%s",
                    report.intent,
                )

    async def _ensure_channel(self) -> grpc.aio.Channel:
        if self._channel is None:
            self._channel = create_aio_channel_with_credentials(
                self._target, self._credentials
            )
        return self._channel

    async def aclose(self) -> None:
        self._closed = True
        if self._channel is not None:
            try:
                await self._channel.close()
            except Exception:  # noqa: BLE001
                logger.debug("[ProactiveSubscriber] channel close ignored", exc_info=True)
            self._channel = None
