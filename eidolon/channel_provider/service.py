"""Idempotent Channel Provider application service.

This layer owns idempotency, persistence and the Hub-facing response shape. It
does not know how a channel is realised: it derives what the device declared it
needs, asks the registry which adapter can carry that, and stores whatever
handle the adapter hands back without ever looking inside it.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import time
from collections.abc import Callable

from .contracts import (
    IdempotencyConflict,
    InvalidTransition,
    ProvisionRequest,
    CurrentRequest,
    RevokeRequest,
    SessionRequest,
    UnknownChannel,
    canonical_json,
)
from .ports import ChannelGrant, ServingAction, ServingRequest, ServingRequestSink
from .selection import AdapterRegistry
from .spec import ChannelSpec, MediaFlow, derive_spec
from .store import ChannelProviderStore, StoredProvision

logger = logging.getLogger("eidolon.channel_provider.service")


class ChannelProviderService:
    def __init__(
        self,
        *,
        store: ChannelProviderStore,
        registry: AdapterRegistry,
        agent_name: str,
        now_ms: Callable[[], int] | None = None,
    ) -> None:
        self._store = store
        self._registry = registry
        self._agent_name = agent_name
        self._now_ms = now_ms or (lambda: time.time_ns() // 1_000_000)
        self._lock = asyncio.Lock()

    def initialize(self) -> None:
        self._store.initialize()

    async def start(self) -> None:
        """Begin listening to every channel that is already open.

        A device's channel outlives this process, and so does its right to ask
        to be heard. Nothing tells us on restart which devices are mid-silence,
        so we re-state what we want watched from the only durable record there
        is. Failing to reach one channel must not cost the others theirs.
        """
        self._store.expire_credentials(self._now_ms())
        for stored in self._store.active_provisions():
            try:
                await self._accept_requests(stored)
            except Exception:
                logger.exception(
                    "channel for device=%s could not be watched; it can still be "
                    "served by request, but the device cannot ask",
                    stored.device_id,
                )

    async def healthcheck(self) -> None:
        self._store.healthcheck()
        await self._registry.healthcheck()

    async def shutdown(self) -> None:
        await self._registry.shutdown()

    async def provision(self, request: ProvisionRequest) -> str:
        response = await self._provision(request)
        # Outside the lock: listening opens a connection of the adapter's own,
        # and a device asking to talk on one channel must not wait behind
        # another device's enrollment.
        stored = self._store.active_device(request.device_ref)
        if stored is not None:
            try:
                await self._accept_requests(stored)
            except Exception:
                # The channel is real and can still be served by request; only
                # the device's own way of asking is missing. Losing the grant
                # over that would be the worse trade.
                logger.exception(
                    "channel for device=%s provisioned but cannot be watched",
                    stored.device_id,
                )
        return response

    async def _provision(self, request: ProvisionRequest) -> str:
        async with self._lock:
            now = self._now_ms()
            self._store.expire_credentials(now)
            self._store.assert_not_stale(request.device_ref)
            stored = self._store.operation(
                request.device_ref, request.operation, request.operation_id
            )
            if stored is not None:
                self._require_same_provision(stored, request)
                if stored.status in {"fenced", "revoked"}:
                    raise InvalidTransition("the operation belongs to a terminal fenced lifecycle")
                return stored.response_json

            previous = self._store.current_channel(request.device_ref)
            spec = derive_spec(
                request.device,
                device_instance_id=request.device_ref.device_instance_id,
                agent_name=self._agent_name,
            )
            adapter = (
                self._registry.get(previous.adapter_name)
                if request.operation == "channel.refresh-device" and previous is not None
                else self._registry.select(spec)
            )
            # This is observed by the transport adapter, never asserted by a
            # caller. The ledger checks the exact observed row still owns the
            # active generation when committing the replacement.
            runtime_refresh_of = None
            if (request.operation == "channel.refresh-device" and previous is not None
                    and not adapter.binding_current(json.loads(previous.handle_json))):
                runtime_refresh_of = previous
            grant = await adapter.open(spec, issued_at_ms=now)
            channel_id = self._channel_id(request)
            response = self._response(
                request=request,
                spec=spec,
                grant=grant,
                channel_id=channel_id,
                issued_at_ms=now,
            )
            value = StoredProvision(
                operation_id=request.operation_id,
                operation_kind=request.operation,
                request_fingerprint=request.fingerprint,
                device_ref=request.device_ref,
                owner_id=str(request.device.owner_id),
                manifest_revision=request.device.manifest_revision,
                adapter_name=adapter.name,
                handle_json=json.dumps(grant.handle, sort_keys=True, separators=(",", ":")),
                channel_id=channel_id,
                response_json=response,
                expires_at_ms=grant.expires_at_ms,
                status="active",
                created_at_ms=now,
                updated_at_ms=now,
            )
            try:
                committed, replayed = self._store.create_provision(
                    value, now_ms=now, runtime_refresh_of=runtime_refresh_of
                )
            except Exception:
                if not self._same_transport_resource(
                    adapter,
                    grant.handle,
                    (
                        self._registry.get(previous.adapter_name)
                        if previous is not None
                        else None
                    ),
                    json.loads(previous.handle_json) if previous is not None else None,
                ):
                    await adapter.close(grant.handle)
                raise
            if replayed:
                committed_adapter = self._registry.get(committed.adapter_name)
                committed_handle = json.loads(committed.handle_json)
                if not self._same_transport_resource(
                    adapter,
                    grant.handle,
                    committed_adapter,
                    committed_handle,
                ):
                    await adapter.close(grant.handle)
                return committed.response_json
            if previous is not None and (
                previous.device_ref != committed.device_ref
                or previous.operation_kind != committed.operation_kind
                or previous.operation_id != committed.operation_id
            ):
                previous_adapter = self._registry.get(previous.adapter_name)
                previous_handle = json.loads(previous.handle_json)
                if not self._same_transport_resource(
                    previous_adapter,
                    previous_handle,
                    adapter,
                    grant.handle,
                ):
                    await previous_adapter.stop_accepting(previous_handle)
                    await previous_adapter.close(previous_handle)
            return committed.response_json

    @staticmethod
    def _same_transport_resource(
        left_adapter,
        left_handle: dict | None,
        right_adapter,
        right_handle: dict | None,
    ) -> bool:
        """Whether two credential generations retain one standing resource."""

        if (
            left_adapter is None
            or right_adapter is None
            or left_adapter.name != right_adapter.name
            or left_handle is None
            or right_handle is None
        ):
            return False
        left_identity = left_adapter.resource_identity(left_handle)
        right_identity = right_adapter.resource_identity(right_handle)
        return bool(left_identity and left_identity == right_identity)

    async def current(self, request: CurrentRequest) -> str:
        """Report the device's current binding. Reads only; issues nothing.

        Expiry is applied first so the answer is about now, not about when the
        row was written — an Authority deciding whether to advance a generation
        must not be told a lapsed credential is current.
        """

        async with self._lock:
            self._store.expire_credentials(self._now_ms())
            stored = self._store.current_channel(request.device_ref)
        binding = json.loads(stored.response_json) if stored is not None else None
        refresh_required = stored is not None and not self._registry.get(
            stored.adapter_name
        ).binding_current(json.loads(stored.handle_json))
        return json.dumps(
            {"operation": "channel.current-device", "binding": binding,
             **({"refresh_required": True} if refresh_required else {})},
            separators=(",", ":"),
        )

    async def presence(self) -> str:
        """Which of this Host's bodies are on their channel right now.

        Presence had no producer anywhere on this Host. Hub publishes existence
        and lifecycle and refuses liveness by contract; Kernel commits an
        assignment and says so about the assignment, not the body; the runtime
        blackboard's reader was withdrawn. So every body on the Owner's map read
        「未探测」 while its speaker was plainly in a call.

        This is the smallest thing that can answer it truthfully, and it is a
        read: for each channel this provider granted, ask its adapter whether
        the device itself is on it. No heartbeat to keep alive, no session state
        to hold — ``_converge_serving`` deliberately writes nothing — and no new
        authority. The channel is the only thing that ever knew, and this
        provider is what granted the channel.

        The lock is taken only to read the provisions; the adapter calls happen
        outside it. A transport that has gone slow must not hold up a device
        asking to be served.
        """

        async with self._lock:
            self._store.expire_credentials(self._now_ms())
            active = self._store.active_provisions()
        bodies = []
        for stored in active:
            adapter = self._registry.get(stored.adapter_name)
            reader = getattr(adapter, "device_is_on_channel", None)
            on_channel = await reader(json.loads(stored.handle_json)) if reader else None
            bodies.append(
                {
                    # The device's stable instance id, which is what a body is
                    # called everywhere else on this Host — see the request
                    # contracts' own `device_id` property, which reads the same
                    # field. Read directly: a `DeviceRef` that does not carry it
                    # is a shape this method does not understand.
                    "device_id": stored.device_ref.device_instance_id,
                    "owner_id": stored.owner_id,
                    "channel_id": stored.channel_id,
                    # Three states. `null` is "this channel cannot say", which
                    # is not the same answer as "the body is not there".
                    "on_channel": on_channel,
                }
            )
        return canonical_json({"operation": "channel.presence", "bodies": bodies})

    async def revoke(self, request: RevokeRequest) -> str:
        async with self._lock:
            now = self._now_ms()
            self._store.expire_credentials(now)
            self._store.assert_not_stale(request.device_ref)
            prior = self._store.revocation(request.device_ref, request.operation_id)
            if prior is not None:
                if (
                    prior.request_fingerprint != request.fingerprint
                    or prior.device_id != request.device_id
                ):
                    raise IdempotencyConflict(
                        "revocation operation_id was reused with different content"
                    )
                return prior.response_json

            active = self._store.active_device(request.device_ref)
            if active is not None:
                adapter = self._registry.get(active.adapter_name)
                handle = json.loads(active.handle_json)
                # Stop listening before the channel goes: a revoked device has
                # no standing to ask for anything, least of all on the way out.
                await adapter.stop_accepting(handle)
                await adapter.close(handle)
            response = canonical_json(
                {
                    "operation": "channel.revoked-device",
                    "operation_id": request.operation_id,
                    "device_ref": request.device_ref.model_dump(mode="json"),
                }
            )
            completed, _replayed = self._store.complete_revocation(
                operation_id=request.operation_id,
                request_fingerprint=request.fingerprint,
                device_ref=request.device_ref,
                response_json=response,
                now_ms=now,
            )
            return completed.response_json

    async def open_session(self, request: SessionRequest) -> str:
        """Serve the device's channel, because the device asked to talk."""
        return await self._serve(request, serving=True)

    async def close_session(self, request: SessionRequest) -> str:
        """Stop serving the device's channel; the channel itself survives."""
        return await self._serve(request, serving=False)

    async def _serve(self, request: SessionRequest, *, serving: bool) -> str:
        channel = await self._converge_serving(
            request.device_ref,
            serving=serving,
            conversation_id=request.conversation_id,
        )
        return canonical_json(
            {
                "operation": "channel.opened-session" if serving else "channel.closed-session",
                "device_ref": request.device_ref.model_dump(mode="json"),
                "channel_id": channel.channel_id,
                "conversation_id": request.conversation_id,
                "serving": serving,
            }
        )

    async def _converge_serving(
        self, device_ref, *, serving: bool, conversation_id: str
    ) -> StoredProvision:
        """Converge one device's channel onto served or unserved.

        The single place a conversation starts or stops, whether the device
        asked over its own channel or something else asked on its behalf. Both
        arrive here saying only which device and which way, because that is all
        either of them knows.

        Takes the same lock as provision and revocation so that reading the
        channel and acting on it cannot straddle a revocation — otherwise a
        device could be granted a conversation on a channel that was withdrawn
        a moment earlier. Nothing is written: this is a statement of desired
        state the adapter converges onto, not an event to record.
        """
        async with self._lock:
            self._store.expire_credentials(self._now_ms())
            active = self._store.active_device(device_ref)
            if active is None:
                raise UnknownChannel("device has no active channel")
            adapter = self._registry.get(active.adapter_name)
            handle = json.loads(active.handle_json)
            if serving:
                await adapter.open_session(handle, conversation_id)
            else:
                await adapter.close_session(handle, conversation_id)
            return active

    def _sink_for(self, device_ref) -> ServingRequestSink:
        """Bind a channel's requests to the device it can only ever speak for.

        The adapter reports what was asked, never who asked it: a request that
        arrived on this channel is by construction this device's, so identity
        comes from the channel we chose to listen to rather than from anything
        the message claims.
        """

        async def _requested(request: ServingRequest) -> None:
            await self._converge_serving(
                device_ref,
                serving=request.action is ServingAction.START,
                conversation_id=request.conversation_id,
            )

        return _requested

    async def _accept_requests(self, stored: StoredProvision) -> None:
        adapter = self._registry.get(stored.adapter_name)
        await adapter.accept_requests(
            json.loads(stored.handle_json),
            sink=self._sink_for(stored.device_ref),
        )

    def _response(
        self,
        *,
        request: ProvisionRequest,
        spec: ChannelSpec,
        grant: ChannelGrant,
        channel_id: str,
        issued_at_ms: int,
    ) -> str:
        return canonical_json(
            {
                "operation": "channel.provisioned-device",
                "operation_id": request.operation_id,
                "device_ref": request.device_ref.model_dump(mode="json"),
                "manifest_revision": request.device.manifest_revision,
                "channels": [
                    {
                        "channel_id": channel_id,
                        "purpose": "device-session",
                        "kinds": _channel_kinds(spec),
                        "binding_format": grant.binding_format,
                        "issued_at_ms": issued_at_ms,
                        "expires_at_ms": grant.expires_at_ms,
                        "opaque_binding": base64.b64encode(grant.payload).decode("ascii"),
                    }
                ],
            }
        )

    @staticmethod
    def _channel_id(request: ProvisionRequest) -> str:
        digest = hashlib.sha256(
            f"{request.owner_domain_id}\0{request.device_id}".encode()
        ).hexdigest()[:24]
        return f"chan-{digest}"

    @staticmethod
    def _require_same_provision(stored: StoredProvision, request: ProvisionRequest) -> None:
        if (
            stored.request_fingerprint != request.fingerprint
            or stored.device_ref != request.device_ref
            or stored.operation_kind != request.operation
        ):
            raise IdempotencyConflict("provision operation_id was reused with different content")


def _channel_kinds(spec: ChannelSpec) -> list[str]:
    """Advertise what this channel actually carries for this device.

    A camera that never speaks should not be told its channel carries audio.
    """
    kinds = ["reliable-data", "realtime-data"]
    if spec.audio is not MediaFlow.NONE:
        kinds.append("audio")
    if spec.video is not MediaFlow.NONE:
        kinds.append("video")
    return kinds
