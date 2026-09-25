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
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from uuid import uuid4

from eidolon_sdk.biz.contracts import SESSION_INTENT_USER_INITIATED
from eidolon_sdk.biz.control.shared_session import (
    SharedSessionInvitation,
    SharedChannelSnapshot,
    SharedSessionSelection,
)

from .contracts import (
    BackendUnavailable,
    Forbidden,
    IdempotencyConflict,
    InvalidTransition,
    ProvisionRequest,
    CurrentRequest,
    RevokeRequest,
    SessionRequest,
    UnknownChannel,
    canonical_json,
    request_fingerprint,
)
from .shared_admission import SharedAdmission
from .shared_invitation import invitation_command, SHARED_VISIT_MAX_SECONDS
from .ports import ChannelGrant, ServingAction, ServingRequest, ServingRequestSink
from .selection import AdapterRegistry
from .spec import ChannelSpec, MediaFlow, derive_spec
from .store import ChannelProviderStore, StoredProvision

logger = logging.getLogger("eidolon.channel_provider.service")
PRESENCE_READ_TIMEOUT_SECONDS = 3.0


class _SharedTransitionRequired(InvalidTransition):
    """Release the lifecycle lock before cancelling a shared caller scope."""


@dataclass
class _SharedScope:
    task: asyncio.Task
    cleanup: Callable[[], Awaitable[None]] | None = None


@dataclass
class _SharedVisit:
    fingerprint: str
    ready: asyncio.Future
    devices: tuple[str, ...]
    task: asyncio.Task | None = None


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
        self._shared_scopes: dict[str, _SharedScope] = {}
        self._shared_visits: dict[tuple[str, str], _SharedVisit] = {}

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
        for owner, session_id in tuple(self._shared_visits):
            await self.close_shared_session(session_id, authenticated_owner_id=owner)
        for device_id in tuple(self._shared_scopes):
            await self._retire_shared_scope(device_id)
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
        try:
            return await self._provision_once(request)
        except _SharedTransitionRequired:
            await self._retire_shared_scope(request.device_id)
            return await self._provision_once(request)

    async def _provision_once(self, request: ProvisionRequest) -> str:
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
            if previous is not None and previous.device_ref == request.device_ref:
                old, new = previous.output_policy, request.device.output_policy
                if old is not None and (new is None or new.revision < old.revision
                        or (new.revision == old.revision and new != old)):
                    raise InvalidTransition("output policy cannot regress or reuse a revision")
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
            if request.device_id in self._shared_scopes:
                raise _SharedTransitionRequired("selected device configuration is changing")
            grant = await adapter.open(
                spec, issued_at_ms=now, observed_host_address=request.observed_host_address
            )
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

    async def _retire_shared_scope(self, device_id: str) -> None:
        scope = self._shared_scopes.get(device_id)
        if scope is None:
            return
        task = scope.task
        if task is asyncio.current_task():
            raise InvalidTransition("leave shared transport before changing its device configuration")
        if not task.done():
            if not task.cancelling():
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        # A failed close retains the reservation so revocation/refresh cannot
        # silently leave an authorized temporary room behind. Retry the close.
        if self._shared_scopes.get(device_id) is scope and scope.cleanup is not None:
            await scope.cleanup()

    async def open_shared_session(
        self, selection: SharedSessionSelection, specifications: tuple[dict, ...],
        *, authenticated_owner_id: str,
    ) -> dict:
        """Converge a bounded shared transport to open, independent of HTTP lifetime."""
        if not authenticated_owner_id:
            raise Forbidden("authenticated Owner is required")
        # Snapshot mutable input before handing it to a service-owned task.
        specifications = tuple(json.loads(json.dumps(specifications)))
        fingerprint = request_fingerprint({
            "selection": selection.model_dump(mode="json"), "specifications": specifications,
        })
        key = (authenticated_owner_id, selection.session_id)
        async with self._lock:
            visit = self._shared_visits.get(key)
            if visit is not None:
                if visit.fingerprint != fingerprint:
                    raise IdempotencyConflict("shared session already has another selection")
                if visit.task.done() or visit.task.cancelling():
                    raise InvalidTransition("shared transport cleanup must complete before reopening")
            else:
                visit = _SharedVisit(fingerprint, asyncio.get_running_loop().create_future(),
                                    tuple(ref.device_instance_id for ref in selection.devices))
                # Retrieve failures even if the HTTP caller has disconnected.
                visit.ready.add_done_callback(lambda f: None if f.cancelled() else f.exception())
                self._shared_visits[key] = visit

                async def run():
                    try:
                        async with self.shared_transport_from_specifications(
                            selection, specifications, authenticated_owner_id=authenticated_owner_id,
                        ) as ready:
                            visit.ready.set_result(ready)
                            await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        if not visit.ready.done():
                            visit.ready.set_exception(InvalidTransition("shared start was cancelled"))
                        raise
                    except Exception as exc:
                        if not visit.ready.done():
                            visit.ready.set_exception(exc)
                        raise
                    finally:
                        # Failed room deletion retains the existing reservation;
                        # an explicit close retries it through the same cleanup.
                        if not any(self._shared_scopes.get(d) is not None
                                   and self._shared_scopes[d].task is asyncio.current_task()
                                   for d in visit.devices):
                            if self._shared_visits.get(key) is visit:
                                self._shared_visits.pop(key)

                visit.task = asyncio.create_task(run())
                visit.task.add_done_callback(lambda t: None if t.cancelled() else t.exception())
        return dict(await asyncio.shield(visit.ready))

    async def close_shared_session(self, session_id: str, *, authenticated_owner_id: str) -> dict:
        if not authenticated_owner_id:
            raise Forbidden("authenticated Owner is required")
        key = (authenticated_owner_id, session_id)
        async with self._lock:
            visit = self._shared_visits.get(key)
        if visit is not None:
            if not visit.task.done():
                if not visit.task.cancelling():
                    visit.task.cancel()
                try:
                    await asyncio.shield(visit.task)
                except asyncio.CancelledError:
                    if asyncio.current_task().cancelling():
                        raise
            if not visit.ready.done():
                visit.ready.set_exception(InvalidTransition("shared start was cancelled"))
            for device in visit.devices:
                scope = self._shared_scopes.get(device)
                if scope is not None and scope.task is visit.task:
                    await self._retire_shared_scope(device)
            if self._shared_visits.get(key) is visit:
                self._shared_visits.pop(key)
        return {"session_id": session_id, "state": "closed"}

    @asynccontextmanager
    async def shared_transport_from_specifications(
        self, selection: SharedSessionSelection, specifications: tuple[dict, ...],
        *, authenticated_owner_id: str,
    ):
        # Operation identity belongs to this ledger, never to a business caller.
        # The existing scope revalidates these records and specification hashes
        # under its lock before allocating any transport resources.
        by_ref = {ref.device_instance_id: ref for ref in selection.devices}
        requests = []
        async with self._lock:
            for specification in specifications:
                if not isinstance(specification, dict) or set(specification) != {"device_ref", "device"}:
                    raise InvalidTransition("invalid specification fields")
                ref_value = specification["device_ref"]
                if not isinstance(ref_value, dict) or not isinstance(ref_value.get("device_instance_id"), str):
                    raise InvalidTransition("invalid specification DeviceRef")
                ref = by_ref.get(ref_value.get("device_instance_id"))
                if ref is None or ref.model_dump(mode="json") != ref_value:
                    raise InvalidTransition("specification is outside selected lifecycle")
                stored = self._store.active_device(ref)
                if stored is None:
                    raise UnknownChannel("selected device has no current channel")
                if stored.owner_id != authenticated_owner_id:
                    raise Forbidden("selected device belongs to another Owner")
                requests.append(ProvisionRequest.parse(json.dumps({
                    "operation": stored.operation_kind,
                    "operation_id": stored.operation_id,
                    **specification,
                }).encode()))
        async with self.shared_transport(
            selection, tuple(requests), authenticated_owner_id=authenticated_owner_id,
        ) as ready:
            yield ready

    @asynccontextmanager
    async def shared_transport(
        self, selection: SharedSessionSelection, provisions: tuple[ProvisionRequest, ...],
        *, authenticated_owner_id: str,
    ):
        """Lifetime of one pre-authorized transport attempt, owned by its caller.

        Agent/application owns the business session; this scope owns only the
        temporary transport. It emits no credentials and dispatches no Agent.
        Original provision fingerprints fence caller-supplied specifications.
        The caller must not suppress cancellation inside the scope.
        """
        if not authenticated_owner_id:
            raise Forbidden("authenticated Owner is required")
        by_id = {p.device_id: p for p in provisions}
        device_ids = tuple(ref.device_instance_id for ref in selection.devices)
        if len(by_id) != len(provisions) or set(by_id) != set(device_ids):
            raise InvalidTransition("specifications must match the complete selected device set")

        def current_records():
            records = []
            for ref in selection.devices:
                stored = self._store.active_device(ref)
                if stored is None or stored.expires_at_ms <= self._now_ms():
                    raise UnknownChannel("selected device has no current channel")
                if stored.owner_id != authenticated_owner_id:
                    raise Forbidden("selected device belongs to another Owner")
                request = by_id[ref.device_instance_id]
                device = request.device
                # Recompute from the actual specification, not the caller's
                # fingerprint attribute. Nested manifests are mutable Python data.
                wire = {
                    "operation": request.operation, "operation_id": request.operation_id,
                    "device_ref": request.device_ref.model_dump(mode="json"),
                    "device": {"owner_id": str(device.owner_id), "display_name": device.display_name,
                               "device_kind": device.manifest_id, "manifest": device.manifest,
                               "manifest_revision": device.manifest_revision,
                               **({"output_policy": device.output_policy.model_dump(mode="json", exclude_unset=True)}
                                  if device.output_policy is not None else {})},
                }
                if request_fingerprint(wire) != stored.request_fingerprint:
                    raise IdempotencyConflict("specification differs from the current provision")
                self._require_same_provision(stored, request)
                if request.operation_id != stored.operation_id:
                    raise InvalidTransition("specification is not from the current provision")
                records.append(stored)
            return tuple(records)

        async with self._lock:
            records = current_records()
            if any(device_id in self._shared_scopes for device_id in device_ids):
                raise InvalidTransition("a selected device already belongs to a shared transport")
            adapters = {record.adapter_name for record in records}
            if len(adapters) != 1:
                raise InvalidTransition("selected channels cannot share one transport")
            adapter = self._registry.get(records[0].adapter_name)
            if not all(callable(getattr(adapter, name, None)) for name in
                       ("open_shared", "deliver_shared_invitation", "device_is_on_channel")):
                raise InvalidTransition("selected transport does not support shared invitations")
            specs = tuple(derive_spec(by_id[device_id].device,
                          device_instance_id=device_id, agent_name=self._agent_name)
                          for device_id in device_ids)
            scope = _SharedScope(asyncio.current_task())
            for device_id in device_ids:
                self._shared_scopes[device_id] = scope

        grants = {}
        close_lock = asyncio.Lock()
        closed = False

        async def cleanup():
            nonlocal closed
            async with close_lock:
                if closed:
                    return
                if grants:
                    await adapter.close(next(iter(grants.values())).handle)
                closed = True
                for device_id in device_ids:
                    if self._shared_scopes.get(device_id) is scope:
                        self._shared_scopes.pop(device_id)

        scope.cleanup = cleanup
        try:
            async with asyncio.timeout(25):
                # Deleting the previous temporary room starts device recovery;
                # it does not mean the standing control channel is already back.
                # Observe readiness within the same bounded admission operation.
                recovery_deadline = self._now_ms() + 20000
                while True:
                    present = await asyncio.gather(*(self._read_channel_presence(r) for r in records))
                    async with self._lock:
                        if current_records() != records:
                            raise InvalidTransition("selected channels changed during control recovery")
                    if all(value is True for value in present):
                        break
                    if self._now_ms() >= recovery_deadline:
                        raise InvalidTransition("selected devices did not become reachable before invitation")
                    await asyncio.sleep(0.1)
                async with self._lock:
                    if current_records() != records:
                        raise InvalidTransition("selected channels changed before shared creation")
                issued_at_ms = self._now_ms()
                grants = await adapter.open_shared(specs, input_device_id=selection.input_device_id,
                                                   issued_at_ms=issued_at_ms)
                deadline = min(issued_at_ms + 20000, *(g.expires_at_ms for g in grants.values()))
                ids = {device_id: uuid4().hex for device_id in device_ids}
                admission = SharedAdmission(ids, deadline_ms=deadline)
                async with self._lock:
                    if current_records() != records:
                        raise InvalidTransition("selected channels changed during shared creation")

                async def invite(record):
                    device_id = record.device_id
                    command = invitation_command(
                        grants[device_id], device_ref=record.device_ref,
                        session_id=selection.session_id, command_id=ids[device_id],
                        channel_id=adapter.resource_identity(grants[device_id].handle),
                        kinds=tuple(_channel_kinds(specs[device_ids.index(device_id)])),
                        issued_at_ms=issued_at_ms, deadline_ms=deadline,
                    )
                    payload = SharedSessionInvitation.model_validate_json(json.dumps(command["payload"]))
                    status = await adapter.deliver_shared_invitation(
                        json.loads(record.handle_json), payload, command_id=ids[device_id])
                    admission.acknowledge(device_id, ids[device_id], status)

                # TaskGroup cancels unfinished sends on any error before cleanup.
                try:
                    async with asyncio.TaskGroup() as group:
                        for record in records:
                            group.create_task(invite(record))
                except ExceptionGroup as exc:
                    raise BackendUnavailable("shared invitation delivery failed") from exc
                if admission.failed_devices:
                    raise InvalidTransition("a shared invitation was refused")
                while True:
                    members = await asyncio.gather(*(adapter.device_is_on_channel(grants[d].handle)
                                                     for d in device_ids))
                    admission.observe_members({d for d, present in zip(device_ids, members, strict=True)
                                               if present is True})
                    async with self._lock:
                        if current_records() != records:
                            raise InvalidTransition("selected channels changed during admission")
                    if admission.ready(now_ms=self._now_ms()):
                        break
                    if self._now_ms() >= deadline:
                        raise InvalidTransition("shared admission deadline elapsed")
                    await asyncio.sleep(0.1)
            # Bound the scope even if a consumer forgets to finish a visit. The
            # device enforces its own credential lease if this process crashes.
            remaining = min(SHARED_VISIT_MAX_SECONDS, (min(g.expires_at_ms for g in grants.values()) - self._now_ms()) / 1000)
            async with asyncio.timeout(max(0, remaining)):
                yield {"session_id": selection.session_id, "state": "transport_ready",
                       "device_ids": list(device_ids)}
        finally:
            await cleanup()

    async def inspect_shared_selection(
        self,
        request: SharedSessionSelection,
        *,
        authenticated_owner_id: str,
    ) -> tuple[SharedChannelSnapshot, ...]:
        """Observe selected lifecycle records without granting room admission.

        The caller supplies the authenticated business Owner separately from
        the selection. Invitation must revalidate authority, binding, capability
        and presence; these credential-free observations reserve no resources.
        """
        if not authenticated_owner_id:
            raise Forbidden("authenticated Owner is required")

        def current_records() -> tuple[StoredProvision, ...]:
            now = self._now_ms()
            records = []
            for ref in request.devices:
                stored = self._store.active_device(ref)
                if stored is None or stored.expires_at_ms <= now:
                    raise UnknownChannel("selected device has no current channel")
                if stored.owner_id != authenticated_owner_id:
                    raise Forbidden("selected device belongs to another Owner")
                records.append(stored)
            return tuple(records)

        async with self._lock:
            records = current_records()

        async def observe(stored: StoredProvision) -> tuple[bool | None, int]:
            present = await self._read_channel_presence(stored)
            return present, self._now_ms()

        # The bounded selection probes concurrently, outside the lifecycle lock.
        observations = await asyncio.gather(*(observe(record) for record in records))
        async with self._lock:
            if current_records() != records:
                raise InvalidTransition("selected channels changed during inspection; retry")
            return tuple(
                SharedChannelSnapshot(
                    device_ref=stored.device_ref,
                    channel_id=stored.channel_id,
                    manifest_revision=stored.manifest_revision,
                    expires_at_ms=stored.expires_at_ms,
                    on_channel=present,
                    observed_at_ms=observed_at,
                )
                for stored, (present, observed_at) in zip(records, observations, strict=True)
            )

    async def _read_channel_presence(self, stored: StoredProvision) -> bool | None:
        adapter = self._registry.get(stored.adapter_name)
        reader = getattr(adapter, "device_is_on_channel", None)
        if reader is None:
            return None
        try:
            async with asyncio.timeout(PRESENCE_READ_TIMEOUT_SECONDS):
                present = await reader(json.loads(stored.handle_json))
            return present if type(present) is bool else None
        except Exception:
            # Do not log the opaque handle or adapter exception: either may
            # contain credentials. Cancellation still propagates to the caller.
            logger.warning("presence unavailable for channel=%s", stored.channel_id)
            return None

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
            active = self._store.observable_provisions()
        bodies = []
        for stored in active:
            on_channel = await self._read_channel_presence(stored)
            async with self._lock:
                if self._store.current_channel(stored.device_ref) != stored:
                    continue  # A retired handle cannot speak for a new lifecycle.
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
        try:
            return await self._revoke_once(request)
        except _SharedTransitionRequired:
            await self._retire_shared_scope(request.device_id)
            return await self._revoke_once(request)

    async def _revoke_once(self, request: RevokeRequest) -> str:
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

            if request.device_id in self._shared_scopes:
                raise _SharedTransitionRequired("selected device is being revoked")
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
        """Serve the device's channel, because someone with standing asked.

        The name is older than the two ways in. This is now the orchestrated
        road — an authenticated caller waking a body on the Owner's behalf, and
        the only one that may say the wake is anything other than user-driven.
        """
        return await self._serve(request, serving=True)

    async def close_session(self, request: SessionRequest) -> str:
        """Stop serving the device's channel; the channel itself survives."""
        return await self._serve(request, serving=False)

    async def _serve(self, request: SessionRequest, *, serving: bool) -> str:
        channel = await self._converge_serving(
            request.device_ref,
            serving=serving,
            conversation_id=request.conversation_id,
            session_intent=request.session_intent,
            target_companion_id=request.target_companion_id,
        )
        return canonical_json(
            {
                "operation": "channel.opened-session" if serving else "channel.closed-session",
                "device_ref": request.device_ref.model_dump(mode="json"),
                "channel_id": channel.channel_id,
                "conversation_id": request.conversation_id,
                "serving": serving,
                # Echoed so the caller can see which intent was actually
                # honoured. An orchestrator that asked for a presence wake and
                # got an ordinary session has to be able to tell.
                **({"session_intent": request.session_intent} if serving else {}),
                **({"target_companion_id": request.target_companion_id}
                   if serving and request.target_companion_id is not None else {}),
            }
        )

    async def _converge_serving(
        self, device_ref, *, serving: bool, conversation_id: str, session_intent: str,
        target_companion_id: str | None = None,
    ) -> StoredProvision:
        """Converge one device's channel onto served or unserved.

        The single place a conversation starts or stops, whether the device
        asked over its own channel or something else asked on its behalf. Both
        arrive here saying only which device, which way, and why — and the two
        roads differ in exactly that last part, because only one of them has
        the standing to name it. `session_intent` is read only when opening;
        ending a conversation has no intent to state.

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
            if serving and active.device_id in self._shared_scopes:
                raise InvalidTransition("device is reserved by a shared transport")
            adapter = self._registry.get(active.adapter_name)
            handle = json.loads(active.handle_json)
            if serving:
                await adapter.open_session(
                    handle, conversation_id, session_intent=session_intent,
                    target_companion_id=target_companion_id,
                )
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
                # Stated here, not carried from the packet, and that is the
                # whole authority rule in one line: a device asking over its
                # own channel IS a user-initiated session. The privileged
                # intents describe a wake somebody else decided on, so they can
                # only come from the authenticated road above. This is written
                # as a literal rather than defaulted so that a later reader
                # sees a decision instead of an omission.
                session_intent=SESSION_INTENT_USER_INITIATED,
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
                **({"output_policy": request.device.output_policy.model_dump(mode="json")}
                   if request.device.output_policy is not None else {}),
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
