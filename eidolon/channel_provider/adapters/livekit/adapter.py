"""Realising a device channel as a LiveKit room.

Everything LiveKit-shaped lives behind this file: room naming, grants, agent
dispatch and the binding's wire shape. The service above it never learns that
rooms exist.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import socket
import time
import ipaddress
from ...shared_invitation import SHARED_VISIT_MAX_SECONDS

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import timedelta
from typing import Any
from uuid import uuid4

import psutil
from urllib.parse import urlparse, urlunparse

from eidolon_sdk.biz.control.shared_session import SharedSessionInvitation
from eidolon_sdk.biz.smarthome import PANEL_REQUEST_TOPIC
from eidolon_sdk.biz.control.protocol import command_status_from_ack
from eidolon.livekit.control_receipts import read_control_receipt
from eidolon_sdk.biz.presentation import SessionOutputPlan, FACE_PROFILE
from eidolon_sdk.biz.contracts import (
    CHANNEL_PROVIDER_IDENTITY_PREFIX,
    CONTROL_OP_PLAYBACK_STOP,
    CONTROL_TOPIC,
    SESSION_CLOSE_TYPE,
    SESSION_CONVERSATION_ID_FIELD,
    SESSION_APPLICATION_COMPANION,
    SESSION_APPLICATION_FIELD,
    SESSION_APPLICATION_HOME_COMMAND,
    VALID_SESSION_APPLICATIONS,
    SESSION_CONTROL_REQUEST_ID_FIELD,
    SESSION_CONTROL_TOPIC,
    SESSION_END_ERROR,
    SESSION_END_TYPE,
    SESSION_INTENT_FIELD,
    SESSION_INTENT_USER_INITIATED,
    SESSION_OPEN_TYPE,
    SESSION_REJECTED_TYPE,
    SESSION_REJECTION_CONFLICT,
    WIRE_SCHEMA_VERSION,
    normalize_conversation_id,
)
from eidolon_sdk.system import on_product_link
from livekit import api, rtc
from livekit.api.twirp_client import TwirpError, TwirpErrorCode
from livekit.protocol.agent import JobStatus
from livekit.protocol.models import ParticipantInfo

from ...contracts import BackendUnavailable, ChannelNotServable, InvalidTransition, canonical_json
from ...ports import ChannelGrant, PanelRequestSink, ServingAction, ServingRequest, ServingRequestSink
from ...spec import ChannelSpec, MediaFlow
from .config import LiveKitConfig

logger = logging.getLogger("eidolon.channel_provider.livekit")

ADAPTER_NAME = "livekit"
BINDING_FORMAT = "application/vnd.eidolon.livekit-session+json;v=2"

def _client_addresses(
    management_networks: Sequence[ipaddress.IPv4Network | ipaddress.IPv6Network] = (),
) -> list[str]:
    """Current LAN candidates; the default route is only an ordering hint.

    Candidates, not answers — the firmware works down the list. That is why a
    link the device cannot reach is not a harmless extra: the device pays a
    full LiveKit connect attempt for it, seven internal retries at an eleven
    second TLS timeout each, about 77 seconds of nothing behind a screen that
    only says it is retrying. An operator's bench cable is exactly such a link
    and looks like any other /24 from here, so Ops declares it and it is
    dropped before it can cost a device a round of that.
    """
    stats = psutil.net_if_stats()
    addresses = set()
    for name, entries in psutil.net_if_addrs().items():
        if name not in stats or not stats[name].isup:
            continue
        for entry in entries:
            if entry.family != socket.AF_INET:
                continue
            address = ipaddress.ip_address(entry.address)
            if not (address.is_loopback or address.is_link_local or
                    address.is_unspecified or address.is_multicast):
                if on_product_link(address, management_networks):
                    addresses.add(str(address))
    preferred = None
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
            connection.connect(("192.0.2.1", 9))
            preferred = connection.getsockname()[0]
    except OSError:
        pass
    return sorted(addresses, key=lambda address: (address != preferred, int(ipaddress.ip_address(address))))


def _observed_first(urls: list[str], observed: str) -> list[str]:
    """The same candidates, led by the one this device is known to reach.

    Ordering only, never widening. The candidates are what this Host offers at
    all, and that question is answered elsewhere — by what Ops declared about
    this machine's links. What an observation answers is the narrower one of
    which of them to try first, and it answers it with evidence rather than a
    guess: an address that carried the device's own request is, by
    construction, an address that device can reach. So an observation naming
    something absent from the list orders nothing and adds nothing, which is
    also what makes it safe to pass one through unexamined — a Host answering
    a request that arrived over loopback hands on `127.0.0.1`, and `127.0.0.1`
    is not a candidate, so nothing happens.

    Worth the trouble because the list is tried in order and a wrong guess is
    not free: the firmware pays a full LiveKit connect for each candidate,
    seven internal retries at an eleven second TLS timeout, about 77 seconds
    behind a screen that only says it is retrying.
    """

    if not observed:
        return urls
    try:
        wanted = ipaddress.ip_address(observed)
    except ValueError:
        return urls
    for index, url in enumerate(urls):
        host = urlparse(url).hostname
        try:
            reached = host is not None and ipaddress.ip_address(host) == wanted
        except ValueError:
            continue
        if reached:
            return [url, *urls[:index], *urls[index + 1:]]
    return urls


_HEALTHCHECK_ROOM = "__eidolon_channel_provider_healthcheck__"
_ENDED_JOB_STATUSES = frozenset({JobStatus.JS_SUCCESS, JobStatus.JS_FAILED})
_SESSION_REQUESTS = {
    SESSION_OPEN_TYPE: ServingAction.START,
    SESSION_CLOSE_TYPE: ServingAction.STOP,
}
_REJOIN_BASE_DELAY = 1.0
_REJOIN_MAX_DELAY = 30.0
# How long a dispatch has to become a serving agent before we say it did
# not. Generous on purpose: a cold worker measured 3.6s of prewarm and
# then joined in under a second, so this is far outside the normal spread
# and only fires when nothing arrived at all.
_SERVING_CONFIRM_DELAY = 10.0
# How much longer an unserved dispatch is kept after that report before it is
# withdrawn. The two together (30s) outlast the device's own 25s wait for
# `session_started`, after which a device that is still there has already sent
# its close; what is left by then is a dispatch nobody will ever end. Waiting
# past the device's deadline, rather than acting at the report, is what keeps
# a slow start from being torn down under a device that is about to be served.
_UNSERVED_RECLAIM_AFTER = 20.0
# Written into a dispatch opened because the device asked on its own channel.
# Only this adapter writes dispatch metadata, so the device cannot claim it.
_REQUESTED_BY_FIELD = "requested_by"
_REQUESTED_BY_DEVICE = "device"


@dataclass
class _Listening:
    """One channel this adapter has undertaken to carry requests for.

    Outlives any particular connection to it, which is the point: the promise is
    to the channel, and a connection is only how it is currently being kept.
    """

    device: str
    sink: ServingRequestSink
    panel_sink: PanelRequestSink | None = None
    connection: Any = None
    retry: asyncio.Task[None] | None = None
    attempt: int = 0
    input_handle: dict[str, Any] = field(default_factory=dict)


def _is_spent(dispatch: Any) -> bool:
    """Whether this dispatch's work is over rather than under way.

    An empty job list is deliberately *not* spent. LiveKit publishes the job a
    beat after it starts running, so a dispatch created moments ago reads as
    having no jobs at all — treating that as finished would tear down the very
    session that is starting up.
    """
    jobs = list(dispatch.state.jobs)
    return bool(jobs) and all(job.state.status in _ENDED_JOB_STATUSES for job in jobs)


def _failure_of(dispatch: Any) -> str:
    """What this dispatch's failed jobs say for themselves, or "" if none did.

    The one place `JS_FAILED` must not be interchangeable with `JS_SUCCESS`.
    `_is_spent` is right to fold them together — a finished job cannot serve a
    new conversation whichever way it finished — but that answer is about the
    dispatch's usefulness, and this one is about a device that was not served,
    which is the only question the two statuses answer differently. `error` is
    the job's own account of why, and on this side of the edge it is the only
    description of the failure that exists.
    """
    return "; ".join(
        f"job={getattr(job, 'id', '') or '?'} failed: "
        f"{getattr(job.state, 'error', '') or 'no reason reported'}"
        for job in dispatch.state.jobs
        if job.state.status == JobStatus.JS_FAILED
    )


def _why_unserved(dispatch: Any) -> str:
    """Say what the dispatch knows about a conversation nobody is serving.

    Three answers worth telling apart, because each sends the reader somewhere
    else: a job that died, and its own error; no job at all, meaning nothing
    ever took the standing order — what an absent or misnamed worker looks like
    from here; or a job LiveKit still holds as live, meaning the agent left the
    room without its job being marked.
    """
    failure = _failure_of(dispatch)
    if failure:
        return failure
    jobs = list(dispatch.state.jobs)
    if not jobs:
        return "no worker took the dispatch"
    return "dispatch still holds " + ", ".join(
        JobStatus.Name(job.state.status) for job in jobs
    )


@dataclass
class _ControlReceipt:
    op: str
    future: asyncio.Future[str]
    terminal: bool = False


class LiveKitChannelAdapter:
    """Opens one LiveKit room per device and mints that device's token."""

    def __init__(
        self,
        config: LiveKitConfig,
        management_networks: Sequence[ipaddress.IPv4Network | ipaddress.IPv6Network] = (),
    ) -> None:
        self._config = config
        #: Which of this Host's links belong to the operator rather than to the
        #: product. Taken at construction and not re-read: it is a declaration
        #: Ops made about this machine, not an observation, and unlike the
        #: addresses below it does not change while the process runs.
        self._management_networks = tuple(management_networks)
        self._api: api.LiveKitAPI | None = None
        # One connection per channel we are listening to, keyed by room so that
        # re-stating a channel converges instead of stacking up connections.
        self._listeners: dict[str, _Listening] = {}
        self._requests: set[asyncio.Task[None]] = set()
        self._input_locks: dict[str, asyncio.Lock] = {}
        self._input_revisions: dict[str, int] = {}
        self._input_applied: dict[str, int] = {}
        self._session_end_observers: dict[str, tuple[str, Any]] = {}
        self._control_receipts: dict[tuple[str, str], _ControlReceipt] = {}

    @property
    def name(self) -> str:
        return ADAPTER_NAME

    @property
    def carries_media(self) -> bool:
        return True

    @property
    def serves_dataonly(self) -> bool:
        """No: a room for a Body with nothing to say is a room nobody serves.

        LiveKit could carry such a channel technically. It is refused because a
        Manifest that declares no media produces no serving spec, so no agent
        is dispatched — and that combination is what silently handed a device a
        room where nothing would ever answer it. When a sensor-only Body is a
        real product, this becomes a real answer, and everything it needs
        beyond a room can be built behind it.
        """

        return False

    def resource_identity(self, handle: dict[str, Any]) -> str:
        room = str(handle.get("room") or "")
        return f"livekit:room:{room}" if room else ""

    def _client(self) -> api.LiveKitAPI:
        # LiveKitAPI owns an aiohttp session and must be constructed in the
        # running service loop, not while the synchronous composition root loads.
        if self._api is None:
            self._api = api.LiveKitAPI(
                url=self._config.api_url,
                api_key=self._config.api_key,
                api_secret=self._config.api_secret,
            )
        return self._api

    async def healthcheck(self) -> None:
        try:
            await self._client().room.list_rooms(api.ListRoomsRequest(names=[_HEALTHCHECK_ROOM]))
        except Exception as exc:
            raise BackendUnavailable("LiveKit health check failed") from exc

    # -- naming -----------------------------------------------------------
    # Resource names are this adapter's business. Another adapter names topics
    # or endpoints instead, and nothing above this line has to care.

    def _client_urls(self) -> list[str]:
        """Explicit remote policy stays explicit; local addresses are observed now.

        A Host cannot know which interface a Body can reach. The binding carries
        candidates for that Body to try, with one credential and one room.
        """
        parsed = urlparse(self._config.client_url)
        if parsed.hostname is not None:
            return [self._config.client_url]
        try:
            addresses = _client_addresses(self._management_networks)
        except (OSError, psutil.Error, ValueError) as exc:
            raise BackendUnavailable("LiveKit signalling interfaces cannot be observed") from exc
        if not addresses:
            raise BackendUnavailable("LiveKit has no usable local signalling address")
        return [urlunparse((parsed.scheme, f"{address}:{parsed.port}", "", "", "", ""))
                for address in addresses]

    def binding_current(self, handle: dict[str, Any]) -> bool:
        return (handle.get("server_urls") == self._client_urls()
                and ("input_revision" not in handle or
                     self._input_applied.get(str(handle.get("room"))) == handle["input_revision"]))

    def _client_url(self) -> str:
        return self._client_urls()[0]

    def _room_name(self, spec: ChannelSpec) -> str:
        digest = hashlib.sha256(
            f"{self._config.room_prefix}\0{spec.device_id}".encode()
        ).hexdigest()[:24]
        return f"{self._config.room_prefix}-{digest}"

    # -- lifecycle --------------------------------------------------------

    async def open(
        self, spec: ChannelSpec, *, issued_at_ms: int, observed_host_address: str = ""
    ) -> ChannelGrant:
        urls = self._client_urls()
        room = self._room_name(spec)
        await self._declare_room(room)
        return self._grant(
            spec, room=room, urls=urls, issued_at_ms=issued_at_ms,
            observed_host_address=observed_host_address,
        )

    def _grant(
        self, spec: ChannelSpec, *, room: str, urls: list[str], issued_at_ms: int,
        observed_host_address: str = "", ttl_limit_seconds: int | None = None,
    ) -> ChannelGrant:
        """Encode both standing and temporary bindings through the same codec."""
        offered = _observed_first(urls, observed_host_address)
        ttl = self._config.grant_ttl_seconds
        if ttl_limit_seconds is not None:
            ttl = min(ttl, ttl_limit_seconds)
        payload = canonical_json(
            {
                "schema_version": 2,
                "session": {
                    "server_url": offered[0],
                    **({"server_urls": offered} if len(offered) > 1 else {}),
                    "token": self._token(room, spec, ttl_seconds=ttl),
                    "identity": spec.device_id,
                    "room_name": room,
                },
                "audio": {
                    "sample_rate": self._config.sample_rate,
                    "channels": self._config.channels,
                },
            }
        ).encode()
        # The handle carries the agent name because opening a session later must
        # not depend on re-deriving the spec: by then the manifest that produced
        # it is the Hub's, not ours, and may already have moved on.
        # `device` is the identity the token above was minted for, so a request
        # arriving here can be checked against the only participant entitled to
        # make one — the room also holds the agent, which speaks on the same
        # topic in the other direction.
        # `server_urls` here is what this Host has, not the order this device
        # was told to try. `binding_current` reads it to ask whether the Host's
        # own addresses have moved since, and an observation is not the Host
        # moving — it is one device's account of how it arrived. Record the
        # ordering instead and every binding minted from an observation reads
        # as stale to the next `current`, which answers `refresh_required` to
        # an Authority that then refreshes it, on every configuration pull,
        # forever.
        handle: dict[str, Any] = {"room": room, "device": spec.device_id, "server_urls": urls}
        if spec.smarthome_panel:
            handle["smarthome_owner"] = spec.owner_id
        if spec.output_policy is not None and spec.selected_outputs is not None:
            handle["output_template"] = {
                "policy_revision": spec.output_policy.revision,
                "inputs": {"microphone": spec.audio.publishes},
                "outputs": spec.selected_outputs.model_dump(mode="json"),
                "expression_profile": FACE_PROFILE if spec.selected_outputs.expression else None,
                "motion_profile": spec.motion_profile,
            }
        if spec.output_policy is not None and spec.output_policy.inputs is not None:
            handle["input_revision"] = spec.output_policy.revision
            handle["input_permissions"] = {
                "microphone": spec.audio.publishes,
                "camera": spec.video.publishes,
                "subscribe": spec.audio.subscribes or spec.video.subscribes,
            }
        if spec.serving is not None:
            handle["agent"] = spec.serving.agent_name
            handle[SESSION_APPLICATION_FIELD] = spec.serving.application
        return ChannelGrant(
            binding_format=BINDING_FORMAT,
            payload=payload,
            expires_at_ms=issued_at_ms + ttl * 1000,
            handle=handle,
        )

    async def open_shared(
        self, specs: Sequence[ChannelSpec], *, input_device_id: str, issued_at_ms: int,
    ) -> dict[str, ChannelGrant]:
        """Create one temporary transport for a pre-authorized participant set.

        This is not a public admission API. The caller verifies current device
        identities and Owner authority, owns the returned room lifetime, and
        closes it on invitation failure, timeout or conversation completion.
        Each invocation is a new attempt; retries reuse the returned grants.
        No standing provision is changed and no serving agent is dispatched.
        """
        specs = tuple(specs)
        devices = {spec.device_id for spec in specs}
        owners = {spec.owner_id for spec in specs}
        if (not 2 <= len(specs) <= 16 or len(devices) != len(specs)
                or input_device_id not in devices or not all(devices)
                or len(owners) != 1 or not all(owners)):
            raise ValueError("shared transport requires distinct devices of one Owner and an input")
        room = f"{self._config.room_prefix}-shared-{uuid4().hex}"
        urls = self._client_urls()
        grants: dict[str, ChannelGrant] = {}
        for spec in specs:
            # Selection may narrow permissions, never widen the manifest/policy.
            audio = spec.audio
            if spec.device_id != input_device_id:
                audio = MediaFlow.SUBSCRIBE if audio.subscribes else MediaFlow.NONE
            narrowed = replace(spec, audio=audio, video=MediaFlow.NONE)
            grant = self._grant(narrowed, room=room, urls=urls, issued_at_ms=issued_at_ms,
                                ttl_limit_seconds=SHARED_VISIT_MAX_SECONDS)
            # A shared room cannot enter the original per-device dispatch or
            # live input-policy update path. Team orchestration owns those steps.
            for key in ("agent", "input_revision", "input_permissions"):
                grant.handle.pop(key, None)
            grants[spec.device_id] = grant
        # Sign everything before creating a resource; a signing failure leaves
        # no partial room. If creation has an uncertain outcome, try to reclaim
        # this unique attempt without touching any standing device channel.
        try:
            await self._declare_room(room)
        except BaseException:
            try:
                await self.close({"room": room})
            except Exception:
                logger.warning("Failed to reclaim temporary shared room %s", room)
            raise
        return grants

    async def deliver_shared_invitation(
        self, handle: dict[str, Any], invitation: SharedSessionInvitation, *, command_id: str,
    ) -> str:
        """Send on the device's standing channel and await its first receipt.

        The caller has already checked Owner/DeviceRef authority. This is a
        bounded transport exchange, not a durable queue or an admission grant.
        In particular, accepted does not mean the device joined the new room.
        """
        now_ms = int(time.time() * 1000)
        if invitation.deadline_ms <= now_ms or invitation.channel.issued_at_ms > now_ms:
            raise InvalidTransition("invitation is outside its delivery window")
        return await self.deliver_control(
            handle, invitation.command(command_id=command_id), wait_for_terminal=False,
        )

    async def deliver_control(
        self, handle: dict[str, Any], command: dict, *, wait_for_terminal: bool,
    ) -> str:
        """Use the standing listener for one bounded, correlated control exchange.

        Preparation needs the completed room.join receipt; a shared invitation
        needs its first receipt followed by independent admission observation.
        Neither operation receives a second control connection or retry queue.
        """
        room = str(handle.get("room") or "")
        device = str(handle.get("device") or "")
        if command.get("dst") != {"type": "device", "id": device}:
            raise InvalidTransition("command target does not match channel")
        command_id, op = command.get("id"), command.get("op")
        if not isinstance(command_id, str) or not command_id or not isinstance(op, str) or not op:
            raise InvalidTransition("command identity is required")
        issued, ttl = command.get("ts"), command.get("ttl_ms")
        now_ms = int(time.time() * 1000)
        if (type(issued) is not int or type(ttl) is not int or ttl <= 0
                or issued > now_ms or issued + ttl <= now_ms):
            raise InvalidTransition("command is outside its delivery window")
        watch = self._listeners.get(room)
        if watch is None or watch.device != device or watch.connection is None:
            raise BackendUnavailable("device control channel is not connected")
        key = (room, command_id)
        if key in self._control_receipts:
            raise InvalidTransition("control command is already in flight")
        # Stop playback immediately even when an ordinary command is awaiting
        # its receipt. Keep both exchanges correlated; do not replace or cancel
        # the earlier operation's future.
        if op != CONTROL_OP_PLAYBACK_STOP and any(
            pending_room == room for pending_room, _ in self._control_receipts
        ):
            raise InvalidTransition("device already has a control delivery in flight")
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._control_receipts[key] = _ControlReceipt(op, future, wait_for_terminal)
        logger.info("control send room=%s device=%s op=%s command=%s", room, device, op, command_id)
        try:
            async with asyncio.timeout(min((issued + ttl - now_ms) / 1000, 30.0)):
                await watch.connection.local_participant.publish_data(
                    canonical_json(command).encode(), reliable=True, topic=CONTROL_TOPIC,
                    destination_identities=[device],
                )
                return await future
        except TimeoutError as exc:
            logger.warning("control receipt timed out room=%s device=%s op=%s command=%s",
                           room, device, op, command_id)
            raise BackendUnavailable("control receipt timed out") from exc
        except asyncio.CancelledError:
            raise
        except BackendUnavailable:
            raise
        except Exception as exc:
            raise BackendUnavailable("control delivery failed") from exc
        finally:
            self._control_receipts.pop(key, None)
            if not future.done():
                future.cancel()
            elif not future.cancelled():
                future.exception()

    async def send_panel_control(
        self, handle: dict[str, Any], command: dict[str, Any]
    ) -> None:
        room = str(handle.get("room") or "")
        device = str(handle.get("device") or "")
        watch = self._listeners.get(room)
        if (
            command.get("dst") != {"type": "device", "id": device}
            or watch is None
            or watch.device != device
            or watch.connection is None
        ):
            raise BackendUnavailable("panel control channel is not connected")
        wire = canonical_json(command).encode()
        payload = command.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        logger.info(
            "panel publish room=%s device=%s command=%s op=%s bytes=%s "
            "revision=%s seq=%s reliable=true",
            room, device, command.get("id"), command.get("op"), len(wire),
            payload.get("revision"), payload.get("seq"),
        )
        try:
            await watch.connection.local_participant.publish_data(
                wire,
                reliable=True,
                topic=CONTROL_TOPIC,
                destination_identities=[device],
            )
            logger.info("panel published room=%s command=%s", room, command.get("id"))
        except Exception as exc:
            raise BackendUnavailable("panel control delivery failed") from exc

    def _control_receipt(self, packet: Any, *, room: str, device: str) -> None:
        body = read_control_receipt(packet, device=device)
        if body is None:
            return
        pending = self._control_receipts.get((room, body["ref"]))
        if pending is None or pending.future.done() or body.get("op") != pending.op:
            return
        status = command_status_from_ack(body["status"])
        if pending.terminal and status in {"accepted", "running"}:
            return
        code = body.get("code")
        if not isinstance(code, str) or len(code) > 64 or not code.replace("_", "").isalnum():
            code = "UNSPECIFIED"
        logger.info("control receipt room=%s device=%s op=%s command=%s status=%s code=%s",
                    room, device, pending.op, body["ref"], status, code)
        pending.future.set_result(status)

    def _fail_control_delivery(self, room: str) -> None:
        for (pending_room, _), pending in self._control_receipts.items():
            if pending_room == room and not pending.future.done():
                pending.future.set_exception(BackendUnavailable("control channel disconnected"))

    async def close(self, handle: dict[str, Any]) -> None:
        room = handle.get("room")
        if not room:
            return
        try:
            await self._client().room.delete_room(api.DeleteRoomRequest(room=room))
        except TwirpError as exc:
            if exc.code != TwirpErrorCode.NOT_FOUND:
                raise BackendUnavailable("LiveKit room revocation failed") from exc
        except Exception as exc:
            raise BackendUnavailable("LiveKit room revocation failed") from exc
        self._input_revisions.pop(room, None)
        self._input_applied.pop(room, None)

    # -- hearing the device -----------------------------------------------

    async def accept_requests(
        self, handle: dict[str, Any], *, sink: ServingRequestSink,
        panel_sink: PanelRequestSink | None = None,
    ) -> None:
        """Sit in the room so the device can say when it wants to be served.

        The device is already connected and already authenticated here, by the
        very credential this adapter minted for it, so its own channel is the
        one place it can ask for a conversation without being issued a second
        identity anywhere else.

        Joins as visible, non-publishing infrastructure so device SDKs can
        authenticate rejection receipts by participant identity. Runtime actor
        selection excludes infrastructure; this participant is never an Agent.
        """
        room = str(handle.get("room") or "")
        device = str(handle.get("device") or "")
        if not room or not device:
            # The one refusal: nothing about this channel says who may speak for
            # it, so no arrival could ever be attributed. Trying harder cannot
            # change that, and the caller needs to hear so.
            raise ChannelNotServable("channel handle cannot identify its device")
        if room in self._listeners:
            watch = self._listeners[room]
            if handle.get("input_revision", 0) >= watch.input_handle.get("input_revision", 0):
                watch.input_handle = handle
                watch.sink = sink
                watch.panel_sink = panel_sink
            await self._reconcile_input_permissions(watch.input_handle)
            await self._reconcile_output_dispatches(watch.input_handle)
            return
        watch = _Listening(device=device, sink=sink, panel_sink=panel_sink)
        watch.input_handle = handle
        self._listeners[room] = watch
        try:
            await self._join(room, watch)
        except Exception as exc:
            # Undertaking to carry a channel is not the same as being connected
            # to it this instant. A device provisioned in the seconds after the
            # transport restarts found exactly this gap on a real Host: the
            # first join timed out, and because the failure was final the
            # channel stayed deaf until something else provisioned it again —
            # while a connection lost one second later would have been rebuilt.
            # Same promise, so the same persistence at both ends.
            self._rejoin_later(room, watch, exc)
            return
        logger.info("listening to room=%s for device=%s", room, device)

    async def _reconcile_input_permissions(self, handle: dict[str, Any]) -> None:
        """Apply committed permissions to connected participants as well as new tokens."""
        permissions = handle.get("input_permissions")
        if permissions is None:
            return
        room = str(handle["room"])
        revision = int(handle["input_revision"])
        async with self._input_locks.setdefault(room, asyncio.Lock()):
            if revision < self._input_revisions.get(room, 0):
                return
            self._input_revisions[room] = revision
            self._input_applied.pop(room, None)
            await self._apply_input_permissions(handle, permissions)
            self._input_applied[room] = revision

    async def _apply_input_permissions(self, handle: dict[str, Any], permissions: dict[str, bool]) -> None:
        sources = []
        if permissions["microphone"]:
            sources.append(api.TrackSource.MICROPHONE)
        if permissions["camera"]:
            sources.append(api.TrackSource.CAMERA)
        try:
            await self._client().room.update_participant(api.UpdateParticipantRequest(
                room=handle["room"], identity=handle["device"],
                permission=api.ParticipantPermission(
                    can_publish=bool(sources), can_publish_sources=sources,
                    can_subscribe=permissions["subscribe"], can_publish_data=True,
                    can_update_metadata=False,
                ),
            ))
        except TwirpError as exc:
            if exc.code != TwirpErrorCode.NOT_FOUND:
                raise BackendUnavailable("LiveKit input permission update failed") from exc

    async def _reconcile_output_dispatches(self, handle: dict[str, Any]) -> None:
        """A committed policy change rebuilds the session; old TTS cannot linger."""
        template = handle.get("output_template")
        agent = handle.get("agent")
        if template is None or not agent:
            return
        room = str(handle["room"])
        for dispatch in await self._client().agent_dispatch.list_dispatch(room_name=room):
            if dispatch.agent_name != agent:
                continue
            metadata = self._dispatch_metadata(dispatch)
            # A standing-channel renewal must not delete a currently owned
            # scene. After process recovery there is no live observer, so an
            # orphaned worker is still withdrawn, never resurrected.
            if metadata.get("presentation_endpoint") is not None or metadata.get("team_dispatch") is not None:
                observer = self._session_end_observers.get(room)
                if observer is None or metadata.get(SESSION_CONVERSATION_ID_FIELD) != observer[0]:
                    await self._client().agent_dispatch.delete_dispatch(dispatch_id=dispatch.id, room_name=room)
                continue
            plan = metadata.get("output_plan")
            compatible = (isinstance(plan, dict) and all(plan.get(k) == v for k,v in template.items()))
            if not compatible:
                await self._client().agent_dispatch.delete_dispatch(dispatch_id=dispatch.id, room_name=room)

    async def _join(self, room: str, watch: _Listening) -> None:
        # Policy reconciliation belongs to the existing listener retry lifecycle.
        # LiveKit's control API may still be starting when this process starts.
        await self._reconcile_input_permissions(watch.input_handle)
        connection = rtc.Room()

        @connection.on("participant_connected")
        def _participant_connected(participant: Any) -> None:
            if participant.identity == watch.device:
                async def reconcile() -> None:
                    try:
                        await self._reconcile_input_permissions(watch.input_handle)
                    except Exception:
                        logger.exception("input permissions pending for room=%s", room)
                task = asyncio.create_task(reconcile())
                self._requests.add(task)
                task.add_done_callback(self._requests.discard)

        @connection.on("data_received")
        def _received(packet: Any) -> None:
            if self._listeners.get(room) is watch and watch.connection is connection:
                self._control_receipt(packet, room=room, device=watch.device)
            self._note_serving_failure(packet, device=watch.device, room=room)
            self._observe_session_end(packet, room=room)
            request = self._requested(packet, device=watch.device, room=room)
            if request is not None:
                self._dispatch_request(watch.sink, request, room=room)
            if (
                self._listeners.get(room) is watch
                and watch.connection is connection
                and watch.panel_sink is not None
                and getattr(packet, "topic", None) == PANEL_REQUEST_TOPIC
                and getattr(getattr(packet, "participant", None), "identity", None) == watch.device
                and len(bytes(packet.data)) <= 16 * 1024
            ):
                try:
                    body = json.loads(bytes(packet.data))
                except (TypeError, ValueError, UnicodeDecodeError):
                    logger.warning("room=%s sent an unreadable panel request", room)
                else:
                    self._dispatch_panel_request(watch.panel_sink, body, room=room)

        @connection.on("participant_disconnected")
        def _participant_disconnected(participant: Any) -> None:
            if self._listeners.get(room) is not watch or watch.connection is not connection:
                return
            observer = self._session_end_observers.get(room)
            if observer is not None and (
                participant.identity == watch.device
                or getattr(participant, "kind", None) == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
            ):
                observer[1]("TEAM_DEVICE_DISCONNECTED" if participant.identity == watch.device
                            else "TEAM_WORKER_DISCONNECTED")

        @connection.on("disconnected")
        def _dropped(reason: Any = None) -> None:
            # Losing this connection is silent in the worst way: the device goes
            # on asking to be heard and nothing anywhere reports that nobody is
            # listening. Measured against a real server — a listener dropped from
            # its room does not come back on its own — so getting back in is this
            # adapter's job for as long as it has said it is carrying the channel.
            if self._listeners.get(room) is watch and watch.connection is connection:
                self._fail_control_delivery(room)
                observer = self._session_end_observers.get(room)
                if observer is not None:
                    observer[1]("CHANNEL_LISTENER_DISCONNECTED")
                self._rejoin_later(room, watch, reason)

        try:
            await connection.connect(self._rtc_url(), self._listener_token(room))
        except Exception as exc:
            raise BackendUnavailable("LiveKit channel could not be listened to") from exc
        # Close the gap between applying a policy and joining its event stream.
        try:
            await self._reconcile_input_permissions(watch.input_handle)
            await self._reconcile_output_dispatches(watch.input_handle)
        except Exception:
            await connection.disconnect()
            raise
        watch.connection = connection
        watch.attempt = 0

    def _rejoin_later(self, room: str, watch: _Listening, reason: Any) -> None:
        if watch.retry is not None and not watch.retry.done():
            return
        watch.connection = None
        logger.warning("lost room=%s (%s); rejoining", room, reason)

        async def _rejoin() -> None:
            while self._listeners.get(room) is watch:
                # Recomputed every time round, because a transport that is down
                # rather than blinking must be asked less often, not forever at
                # the same rate. Observed on a real Host holding at the base
                # delay for minutes while the sandbox denied it a route out.
                delay = min(_REJOIN_MAX_DELAY, _REJOIN_BASE_DELAY * (2**watch.attempt))
                watch.attempt += 1
                await asyncio.sleep(delay)
                if self._listeners.get(room) is not watch:
                    return
                try:
                    await self._join(room, watch)
                except Exception:
                    logger.warning(
                        "room=%s still unreachable after %.0fs; will keep trying",
                        room, delay,
                    )
                    continue
                logger.info("listening to room=%s again", room)
                return

        watch.retry = asyncio.create_task(_rejoin())

    async def _dispatch_for(
        self, room: str, *, agent: str, conversation_id: str
    ) -> Any | None:
        """The dispatch standing for one conversation, or None if none is.

        Matched the way `close_session` matches it — by agent name and the
        conversation in its own metadata — rather than by an id we remembered:
        LiveKit keeps dispatch records of its own in the same room, and a
        session an earlier process opened must still be recognisable here.
        """
        for dispatch in await self._client().agent_dispatch.list_dispatch(room_name=room):
            if (
                dispatch.agent_name == agent
                and self._dispatch_metadata(dispatch).get(SESSION_CONVERSATION_ID_FIELD)
                == conversation_id
            ):
                return dispatch
        return None

    def _confirm_serving_later(
        self, room: str, *, agent: str, conversation_id: str
    ) -> None:
        """Check that the dispatch we just placed actually became a serving agent.

        This port promises to "bring this channel's agent to it" and what it
        does is place a standing order. Those are not the same thing, and the
        gap between them is invisible: a job that dies on startup leaves the
        dispatch accepted, this log saying `opened session`, the channel record
        answering 200, and the device holding an open microphone against a room
        with nothing in it. Measured on 2026-09-07: the agent was a participant
        for 1.5s and no surface anywhere reported that the conversation had no
        one in it.

        So we go and look, and the order of the three questions is the method.

        Is this conversation still wanted? A dispatch that is no longer there
        was withdrawn by `close_session`, and the agent leaves promptly after
        that — so every exchange shorter than this window leaves the device
        alone in its room, which is indistinguishable from a dispatch that
        never served. Asking this first is the difference between a report and
        an alarm that fires on healthy traffic, which is the same disease as
        the silence it was added to cure.

        Is anyone serving it? `Kind.AGENT` is the exact question — not a head
        count, which includes the device and any infrastructure participant,
        and not an identity guess, since LiveKit names the agent itself. It also
        outranks the dispatch record, which publishes a job a beat late: an
        agent demonstrably in the room is serving whatever the bookkeeping says.

        And if nobody is — why, from the dispatch already in hand. See
        `_why_unserved`: "nothing came" and "the job died with this error" are
        different reports to be woken by.

        This never fails a request. The device has already been answered by the
        time this runs; what it changes is that the Provider stops being the
        last component to know.

        A dispatch still unserved `_UNSERVED_RECLAIM_AFTER` later is withdrawn.
        By then the device has stopped waiting for it, so nothing can still be
        started by it, and left in place it is an occupied room: its job may
        read as running forever with no agent anywhere (2026-09-28: a job that
        crashed on startup kept its dispatch "JS_RUNNING" for 21 minutes, and
        every conversation the device asked for in that time was refused).
        """

        async def _unserved() -> Any | None:
            """The dispatch, if it is still wanted and nobody is serving it."""
            try:
                dispatch = await self._dispatch_for(
                    room, agent=agent, conversation_id=conversation_id
                )
            except Exception:
                logger.debug("could not read dispatch of room=%s", room, exc_info=True)
                return None
            if dispatch is None:
                # Withdrawn while we waited: this conversation ended, which is
                # what a short exchange looks like from here.
                return None
            try:
                participants = await self._client().room.list_participants(
                    api.ListParticipantsRequest(room=room)
                )
            except Exception:
                # Nobody looked, which is not the same as nothing being there.
                logger.debug(
                    "could not confirm serving on room=%s", room, exc_info=True
                )
                return None
            listed = getattr(participants, "participants", []) or []
            if any(
                getattr(p, "kind", None) == ParticipantInfo.Kind.AGENT
                for p in listed
            ):
                return None
            return dispatch

        async def _confirm() -> None:
            await asyncio.sleep(_SERVING_CONFIRM_DELAY)
            dispatch = await _unserved()
            if dispatch is None:
                return
            logger.error(
                "dispatch=%s on room=%s agent=%s conversation_id=%s produced no serving "
                "agent within %.0fs — the device asked for a conversation and nothing "
                "is in the room to hold it (%s)",
                dispatch.id,
                room,
                agent,
                conversation_id,
                _SERVING_CONFIRM_DELAY,
                _why_unserved(dispatch),
            )
            await asyncio.sleep(_UNSERVED_RECLAIM_AFTER)
            dispatch = await _unserved()
            if dispatch is None:
                return
            try:
                await self._client().agent_dispatch.delete_dispatch(
                    dispatch_id=dispatch.id, room_name=room
                )
            except Exception:
                logger.warning(
                    "could not withdraw unserved dispatch=%s on room=%s", dispatch.id, room,
                    exc_info=True,
                )
                return
            logger.warning(
                "withdrew dispatch=%s on room=%s conversation_id=%s: still nobody serving it "
                "%.0fs after it was placed (%s)",
                dispatch.id,
                room,
                conversation_id,
                _SERVING_CONFIRM_DELAY + _UNSERVED_RECLAIM_AFTER,
                _why_unserved(dispatch),
            )

        task = asyncio.create_task(_confirm())
        self._requests.add(task)
        task.add_done_callback(self._requests.discard)

    async def device_is_on_channel(self, handle: dict[str, Any]) -> bool | None:
        """Is the device's own participant in its room?

        Exact rather than inferred, and that distinction is the whole method.
        A room's participant count includes the agent, which sits in the room
        whether or not anything is listening — so counting would report every
        provisioned body as present forever. The device's identity is its
        ``device_id`` (see ``_token``), so this asks for that one identity and
        answers about that one thing.

        Never raises. LiveKit being unreachable is not an offline speaker; it is
        nobody having looked, which is ``None``.
        """

        room = str(handle.get("room") or "")
        device = str(handle.get("device") or "")
        if not room or not device:
            return None
        try:
            participants = await self._client().room.list_participants(
                api.ListParticipantsRequest(room=room)
            )
        except Exception:
            # Includes the ordinary case of a room LiveKit has already reaped,
            # which is indistinguishable here from a transport failure — and
            # guessing between them is exactly what `None` exists to avoid.
            logger.debug("could not read participants of room=%s", room, exc_info=True)
            return None
        return any(
            getattr(participant, "identity", "") == device
            and getattr(participant, "state", None) == api.ParticipantInfo.ACTIVE
            for participant in getattr(participants, "participants", []) or []
        )

    async def stop_accepting(self, handle: dict[str, Any]) -> None:
        # Dropped from the registry first: the disconnect below fires the same
        # event a failure does, and this is what tells them apart.
        room = str(handle.get("room") or "")
        self._fail_control_delivery(room)
        watch = self._listeners.pop(room, None)
        if watch is None:
            return
        if watch.retry is not None:
            watch.retry.cancel()
        if watch.connection is None:
            return
        try:
            await watch.connection.disconnect()
        except Exception:  # pragma: no cover - defensive; the channel is going anyway
            logger.debug("failed to leave room=%s cleanly", handle.get("room"), exc_info=True)

    def _requested(self, packet: Any, *, device: str, room: str) -> ServingRequest | None:
        """Read one packet as a request, or decide it is not one.

        Everything here is untrusted input, and the topic carries traffic in
        both directions, so anything that is not one of exactly two things said
        by someone entitled to say it is silently not a request.

        Rejects senders known to be someone else, rather than admitting only the
        sender known to be the device — because for roughly the first three
        seconds after a device connects, packets arrive attributed to nobody
        (measured against a real server: delivered immediately, sender resolved
        at +3s). Admitting only the known device would throw away exactly the
        requests of a device that connects and immediately wants to talk.

        That leniency is bounded by who can be in this room at all: every
        credential for it is minted by this adapter, for one device and one
        agent, and the agent only ever speaks the other direction of this topic
        — which is not a request type and is rejected below on that ground.
        """
        if getattr(packet, "topic", None) != SESSION_CONTROL_TOPIC:
            return None
        identity = getattr(getattr(packet, "participant", None), "identity", None)
        if identity is not None and identity != device:
            return None
        try:
            body = json.loads(bytes(packet.data))
        except (TypeError, ValueError, UnicodeDecodeError):
            logger.warning("room=%s sent an unreadable session request", room)
            return None
        if not isinstance(body, dict):
            return None
        if body.get("schema_v") != WIRE_SCHEMA_VERSION:
            return None
        action = _SESSION_REQUESTS.get(body.get("type"))
        conversation_id = normalize_conversation_id(body.get(SESSION_CONVERSATION_ID_FIELD))
        if action is None or conversation_id is None:
            return None
        control_request_id = body.get(SESSION_CONTROL_REQUEST_ID_FIELD)
        if control_request_id is not None and (
            not isinstance(control_request_id, str) or not control_request_id.strip()
            or len(control_request_id) > 128
        ):
            return None
        return ServingRequest(action=action, conversation_id=conversation_id,
                              control_request_id=control_request_id)

    def _note_serving_failure(self, packet: Any, *, device: str, room: str) -> None:
        """Hear the agent say it could not serve, since we are already in the room.

        The other half of the same edge. `session_end{error}` is published from
        the job's failure path while the room is still live, so it arrives here,
        on this connection, seconds before any confirmation window closes — and
        until now `_requested` dropped it as "not a request", which was true and
        was the end of it. The component that placed the dispatch was the last
        one to learn it had died.

        Read as news, never as an instruction. The device has already been told
        by the agent itself, and ending the session on the strength of a packet
        would take the channel away from the only party that can ask for another
        conversation. So this says so, and does nothing.

        Only `error`. `session_end` also carries the reasons an ordinary
        conversation ends with, and warning on those would teach the reader to
        skip the warning — which is how it would fail to be there for this.
        """
        if getattr(packet, "topic", None) != SESSION_CONTROL_TOPIC:
            return
        identity = getattr(getattr(packet, "participant", None), "identity", None)
        if identity == device:
            # The mirror of `_requested`'s gate: this is the direction the
            # device does not speak, and an unresolved sender is still heard
            # for the reason given there.
            return
        try:
            body = json.loads(bytes(packet.data))
        except (TypeError, ValueError, UnicodeDecodeError):
            # `_requested` has already said so about this same packet.
            return
        if not isinstance(body, dict) or body.get("schema_v") != WIRE_SCHEMA_VERSION:
            return
        if body.get("type") != SESSION_END_TYPE or body.get("reason") != SESSION_END_ERROR:
            return
        logger.warning(
            "room=%s agent reported session_end reason=%s conversation_id=%s — the "
            "conversation this channel opened is not being served",
            room,
            SESSION_END_ERROR,
            normalize_conversation_id(body.get(SESSION_CONVERSATION_ID_FIELD)) or "",
        )

    def _dispatch_request(
        self, sink: ServingRequestSink, request: ServingRequest, *, room: str
    ) -> None:
        """Hand the request on without blocking the room's callback.

        Held in a set until done: a bare task is only weakly referenced, and one
        collected mid-flight would drop a conversation the device asked for.
        """

        watch = self._listeners.get(room)

        async def _carry() -> None:
            try:
                await sink(request)
            except InvalidTransition:
                logger.info("room=%s rejected %s due to session conflict", room, request.action.value)
                if (request.action is ServingAction.START and watch is not None
                        and self._listeners.get(room) is watch and watch.connection is not None):
                    try:
                        await watch.connection.local_participant.publish_data(
                            canonical_json({
                                "schema_v": WIRE_SCHEMA_VERSION,
                                "type": SESSION_REJECTED_TYPE,
                                SESSION_CONVERSATION_ID_FIELD: request.conversation_id,
                                "reason": SESSION_REJECTION_CONFLICT,
                            }),
                            reliable=True, topic=SESSION_CONTROL_TOPIC,
                            destination_identities=[watch.device],
                        )
                    except Exception:
                        logger.exception("room=%s could not deliver session rejection", room)
            except Exception:
                logger.exception("room=%s could not act on %s", room, request.action.value)

        task = asyncio.create_task(_carry())
        self._requests.add(task)
        task.add_done_callback(self._requests.discard)

    def _dispatch_panel_request(
        self, sink: PanelRequestSink, body: object, *, room: str
    ) -> None:
        async def _carry() -> None:
            try:
                await sink(body)
            except Exception:
                logger.exception("room=%s could not act on panel request", room)

        task = asyncio.create_task(_carry())
        self._requests.add(task)
        task.add_done_callback(self._requests.discard)

    def _rtc_url(self) -> str:
        """Reach LiveKit the short way, not by the address devices are given.

        The client URL is a public name that has to resolve and terminate TLS
        from wherever a device happens to be; this process sits next to the
        server and has no reason to go out and come back.
        """
        api_url = self._config.api_url
        for scheme, replacement in (("https://", "wss://"), ("http://", "ws://")):
            if api_url.startswith(scheme):
                return replacement + api_url[len(scheme) :]
        return api_url

    def _listener_token(self, room: str) -> str:
        return (
            api.AccessToken(self._config.api_key, self._config.api_secret)
            .with_identity(f"{CHANNEL_PROVIDER_IDENTITY_PREFIX}{room}")
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room,
                    can_publish=False,
                    can_subscribe=False,
                    can_publish_data=True,
                    hidden=False,
                )
            )
            .to_jwt()
        )

    async def shutdown(self) -> None:
        for room in list(self._listeners):
            await self.stop_accepting({"room": room})
        for task in list(self._requests):
            task.cancel()
        if self._requests:
            await asyncio.gather(*self._requests, return_exceptions=True)
        if self._api is not None:
            await self._api.aclose()
            self._api = None

    # -- internals --------------------------------------------------------

    async def _declare_room(self, room: str) -> None:
        """Create the room the device will live in, and deliberately nothing more.

        The room is born with no agent dispatch. A dispatch is a standing order:
        LiveKit acts on it the moment anyone is in the room, so a room born with
        one would summon an agent — with its speech models and its metered
        upstream services — the instant the device connected, and keep one for
        as long as the device stayed. Since the device now stays permanently,
        that is a session that never ends. Serving is therefore not part of
        declaring the room; it is `open_session`, asked for when there is
        actually something to say.
        """
        try:
            await self._client().room.create_room(api.CreateRoomRequest(name=room))
        except TwirpError as exc:
            if exc.code == TwirpErrorCode.ALREADY_EXISTS:
                return
            raise BackendUnavailable("LiveKit room creation failed") from exc
        except Exception as exc:
            raise BackendUnavailable("LiveKit room creation failed") from exc

    # -- serving ----------------------------------------------------------

    async def open_session(
        self, handle: dict[str, Any], conversation_id: str, *, session_intent: str,
        target_companion_id: str | None = None, presentation_endpoint: dict | None = None,
        team_dispatch: dict | None = None, device_requested: bool = False,
    ) -> None:
        """Place the standing order, and say on it why this session exists.

        The intent rides the dispatch, not the device's token, because the
        dispatch is the only thing here that is per-session AND unwritable by
        the device: this adapter is what creates dispatches, while the token is
        handed to the device once per provision and lives in its flash for
        hours. It joins `conversation_id` and `output_plan`, which are on this
        bus for the same reason — the agent must be able to trust them.

        Another live conversation refuses the open, with one exception: when
        both it and this request came from the device's own channel. A device
        holds one conversation at a time and only picks a new id when it holds
        none, so a new id on its channel means it has let the old one go — most
        often because it restarted and the old id died with that run, which is
        also why it can never close the old one itself (2026-09-29: every
        conversation refused until the host restarted). Such a conversation is
        superseded. One that anybody else opened — an orchestrator, a team, a
        directed scene — still refuses the open: replacing another party's
        conversation is what 466e78b removed, and this does not bring it back.
        """
        if target_companion_id is not None:
            from eidolon.interaction_context import validate_companion_target
            validate_companion_target(target_companion_id)
        application = handle.get(SESSION_APPLICATION_FIELD, SESSION_APPLICATION_COMPANION)
        if application not in VALID_SESSION_APPLICATIONS:
            raise InvalidTransition("unsupported voice session application")
        if application == SESSION_APPLICATION_HOME_COMMAND and "smarthome_owner" not in handle:
            raise InvalidTransition("home command session has no authorized panel scope")
        if application == SESSION_APPLICATION_HOME_COMMAND and (
            team_dispatch is not None or presentation_endpoint is not None
        ):
            raise InvalidTransition("home command session cannot join a Companion team")
        if team_dispatch is not None:
            from eidolon.livekit.common.team_dispatch import TeamDispatch
            team = TeamDispatch.model_validate(team_dispatch)
            if (presentation_endpoint is not None or target_companion_id is not None
                    or team.input_plan.session_id != conversation_id
                    or team.opened.selection.input_device.device_instance_id != handle['device']):
                raise InvalidTransition("inconsistent team dispatch")
            team_dispatch = team.model_dump(mode='json')
        room, agent = self._serving(handle)
        if presentation_endpoint is not None:
            from eidolon.livekit.common.presentation_endpoint import PresentationEndpoint
            endpoint = PresentationEndpoint.model_validate(presentation_endpoint)
            if endpoint.room == room:
                raise InvalidTransition("presentation endpoint must keep a distinct device room")
            presentation_endpoint = endpoint.model_dump(mode="json")
        output_plan = (SessionOutputPlan(session_id=conversation_id, **handle["output_template"])
                       if "output_template" in handle else None)
        try:
            await self._require_no_remote_presentation(room)
            dispatches = [
                dispatch for dispatch in await self._client().agent_dispatch.list_dispatch(room_name=room)
                if dispatch.agent_name == agent
            ]
            already_serving = False
            superseded: list[Any] = []
            # Inspect the entire snapshot before any mutation, including when
            # a matching retry appears before a conflicting live dispatch.
            for dispatch in dispatches:
                metadata = self._dispatch_metadata(dispatch)
                same_conversation = metadata.get(SESSION_CONVERSATION_ID_FIELD) == conversation_id
                if same_conversation and metadata.get("target_companion_id") != target_companion_id:
                    raise InvalidTransition("changing a Companion target requires a new conversation_id")
                if _is_spent(dispatch):
                    continue
                if not same_conversation:
                    if device_requested and metadata.get(_REQUESTED_BY_FIELD) == _REQUESTED_BY_DEVICE:
                        superseded.append(dispatch)
                        continue
                    raise InvalidTransition("device is serving another conversation; close it explicitly first")
                if (
                    metadata.get(SESSION_INTENT_FIELD, SESSION_INTENT_USER_INITIATED) != session_intent
                    or metadata.get("output_plan") != (output_plan.model_dump(mode="json") if output_plan else None)
                    or metadata.get("presentation_endpoint") != presentation_endpoint
                    or metadata.get("team_dispatch") != team_dispatch
                    or metadata.get(SESSION_APPLICATION_FIELD, SESSION_APPLICATION_COMPANION) != application
                ):
                    raise InvalidTransition("an active conversation cannot change intent or output plan")
                already_serving = True
            for dispatch in superseded:
                logger.warning(
                    "superseding dispatch=%s on room=%s previous_conversation=%s "
                    "conversation_id=%s — the device opened a new conversation on its "
                    "own channel, so it no longer holds the previous one",
                    dispatch.id,
                    room,
                    self._dispatch_metadata(dispatch).get(SESSION_CONVERSATION_ID_FIELD, ""),
                    conversation_id,
                )
                await self._client().agent_dispatch.delete_dispatch(
                    dispatch_id=dispatch.id, room_name=room
                )
            if already_serving:
                return
            # Only terminal jobs may be recovered without an explicit close.
            # Preserve failure evidence in logs before deleting their records.
            superseded_ids = {dispatch.id for dispatch in superseded}
            for dispatch in dispatches:
                if dispatch.id in superseded_ids:
                    continue
                metadata = self._dispatch_metadata(dispatch)
                failure = _failure_of(dispatch)
                logger.log(
                    logging.WARNING if failure else logging.INFO,
                    "clearing dispatch=%s on room=%s previous_conversation=%s%s",
                    dispatch.id,
                    room,
                    metadata.get(SESSION_CONVERSATION_ID_FIELD, ""),
                    f" — it had {failure}" if failure else "",
                )
                await self._client().agent_dispatch.delete_dispatch(
                    dispatch_id=dispatch.id, room_name=room
                )
            await self._client().agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(
                    room=room,
                    agent_name=agent,
                    metadata=canonical_json(
                        {
                            "schema_v": WIRE_SCHEMA_VERSION,
                            SESSION_CONVERSATION_ID_FIELD: conversation_id,
                            SESSION_INTENT_FIELD: session_intent,
                            SESSION_APPLICATION_FIELD: application,
                            **({"smarthome_owner": handle["smarthome_owner"],
                                "smarthome_device": handle["device"]}
                               if application == SESSION_APPLICATION_HOME_COMMAND else {}),
                            **({"target_companion_id": target_companion_id}
                               if target_companion_id is not None else {}),
                            **({"output_plan": output_plan.model_dump(mode="json")} if output_plan else {}),
                            **({"presentation_endpoint": presentation_endpoint} if presentation_endpoint else {}),
                            **({"team_dispatch": team_dispatch} if team_dispatch else {}),
                            **({_REQUESTED_BY_FIELD: _REQUESTED_BY_DEVICE} if device_requested else {}),
                        }
                    ),
                )
            )
        except InvalidTransition:
            raise
        except Exception as exc:
            raise BackendUnavailable("LiveKit agent dispatch failed") from exc
        logger.info(
            "opened session on room=%s agent=%s intent=%s", room, agent, session_intent
        )
        self._confirm_serving_later(room, agent=agent, conversation_id=conversation_id)

    def observe_session_end(self, handle: dict, session_id: str, callback):
        room = str(handle["room"])
        observer = (session_id, callback)
        if room in self._session_end_observers:
            raise InvalidTransition("session end already observed")
        self._session_end_observers[room] = observer
        def remove():
            if self._session_end_observers.get(room) is observer:
                self._session_end_observers.pop(room)
        return remove

    def _observe_session_end(self, packet, *, room: str) -> None:
        observer = self._session_end_observers.get(room)
        if observer is None or getattr(packet, "topic", None) != SESSION_CONTROL_TOPIC:
            return
        participant = getattr(packet, "participant", None)
        if getattr(participant, "kind", None) != rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
            return
        try:
            body = json.loads(bytes(packet.data))
        except (ValueError, TypeError, UnicodeDecodeError):
            return
        if (isinstance(body, dict) and type(body.get("schema_v")) is int
                and body["schema_v"] == WIRE_SCHEMA_VERSION
                and body.get("type") == SESSION_END_TYPE
                and body.get(SESSION_CONVERSATION_ID_FIELD) == observer[0]):
            observer[1]()

    async def require_team_input(self, handle: dict) -> None:
        # Read the same Provider-stamped actor metadata the worker validates,
        # before waking any endpoint. Board names never imply capabilities.
        result = await self._client().room.list_participants(
            api.ListParticipantsRequest(room=handle['room']))
        participant = next((p for p in result.participants
                            if p.identity == handle['device']), None)
        if participant is None:
            raise BackendUnavailable('TEAM_INPUT_OFFLINE: input device is not on its channel')
        try:
            metadata = json.loads(participant.metadata or '{}')
        except (TypeError, ValueError):
            metadata = {}
        if not isinstance(metadata, dict) or metadata.get('interaction_mode') != 'ptt':
            raise InvalidTransition('TEAM_INPUT_REQUIRES_PTT: select a device configured for PTT')

    async def require_idle(self, handle: dict) -> None:
        room, agent = self._serving(handle)
        await self._require_no_remote_presentation(room)
        if await self.device_is_on_channel(handle) is not True:
            raise BackendUnavailable("selected device is not present on its channel")
        for dispatch in await self._client().agent_dispatch.list_dispatch(room_name=room):
            if dispatch.agent_name == agent and not _is_spent(dispatch):
                raise InvalidTransition("end the existing conversation before preparing another")

    async def _require_no_remote_presentation(self, room: str) -> None:
        watch = self._listeners.get(room)
        connection = watch.connection if watch is not None else None
        participants = getattr(connection, "remote_participants", {})
        for participant in participants.values():
            if (getattr(participant, "kind", None) == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
                    and str(getattr(participant, "identity", "")).startswith("presentation-")):
                raise InvalidTransition("previous presentation is still closing")

    async def resume_prepared_channels(self, handles) -> None:
        """Recover only Host receipt listeners, never device credentials or IO grants.

        A prepared scene may outlive its join credential. Its immutable closure
        correlations still need a receipt path after Provider restart.
        """
        async def reject_start(request):
            raise InvalidTransition("recovered channel is for scene cleanup only")
        for handle in handles:
            if handle['room'] not in self._listeners:
                await self.accept_requests({k: handle[k] for k in ('room', 'device')},
                    sink=reject_start)

    async def prepared_endpoints_present(self, handles) -> bool:
        """Observe current transport, not a cached preparation acknowledgement."""
        async def present(handle):
            try:
                result = await self._client().room.list_participants(
                    api.ListParticipantsRequest(room=handle['room']))
                return any(p.identity == handle['device'] for p in result.participants)
            except TwirpError as exc:
                if exc.code == TwirpErrorCode.NOT_FOUND:
                    return False
                raise
        try:
            async with asyncio.timeout(3):
                return all(await asyncio.gather(*(present(h) for h in handles)))
        except Exception as exc:
            raise BackendUnavailable("prepared endpoint presence is unknown") from exc

    async def quiesce_prepared_sessions(self, handles, sessions) -> None:
        """Withdraw generation ownership before asking any Body to leave.

        Dispatch deletion revokes the source job. Its presentation connections
        must disappear before Body cleanup: those workers drain/cancel using
        still-live device controls. A missing device is NOT a silence receipt.
        """
        await asyncio.gather(*(self.close_session(h, sessions[h['device']])
            for h in handles if h['device'] in sessions))
        try:
            async with asyncio.timeout(15):
                while True:
                    active = False
                    for handle in handles:
                        try:
                            result = await self._client().room.list_participants(
                                api.ListParticipantsRequest(room=handle['room']))
                        except TwirpError as exc:
                            if exc.code == TwirpErrorCode.NOT_FOUND:
                                continue
                            raise
                        active |= any(p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
                            and p.identity.startswith('presentation-')
                            for p in result.participants)
                    if not active:
                        return
                    await asyncio.sleep(0.1)
        except Exception as exc:
            raise BackendUnavailable("scene producers have not confirmed teardown") from exc

    async def end_prepared_session(self, handle: dict, session_id: str | None,
                                   *, control_request_id: str | None = None) -> None:
        """Revoke device IO with a correlated terminal receipt before releasing it.

        The Provider cannot impersonate an Agent's session_end. Use the existing
        room.leave Body capability over the authenticated device-control path.
        A join correlation also closes preparation that failed before a native
        conversation request arrived. Never close an unrelated conversation.
        """
        from eidolon_sdk.biz.body.capabilities import BODY_OP_ROOM_LEAVE
        from eidolon_sdk.biz.control.protocol import build_command_envelope
        if not session_id and not control_request_id:
            raise InvalidTransition("session cleanup requires an owned correlation")
        payload = {}
        if session_id:
            payload[SESSION_CONVERSATION_ID_FIELD] = session_id
        if control_request_id:
            payload['control_request_id'] = control_request_id
        command = build_command_envelope(command_id=f"leave:{uuid4().hex}",
            device_id=handle['device'], op=BODY_OP_ROOM_LEAVE, payload=payload,
            capability_version=1, src_type='channel', src_id='channel-provider',
            priority='urgent', ttl_ms=10_000)
        result = await self.deliver_control(handle, command, wait_for_terminal=True)
        if result != 'succeeded':
            raise BackendUnavailable(f"device did not confirm conversation cleanup: {result}")

    async def open_team_session(self, opened, handles, sessions):
        from eidolon_sdk.biz.presentation import InputSelection, OutputSelection
        from eidolon.livekit.common.presentation_endpoint import PresentationEndpoint
        from eidolon.livekit.common.team_dispatch import TeamDispatch
        source, *targets = handles
        plans = [SessionOutputPlan(session_id=sessions[h['device']], **h['output_template'])
                 for h in handles]
        if not plans[0].inputs.microphone or any(not p.outputs.speech for p in plans[1:]):
            raise InvalidTransition("team endpoint capabilities do not match scene")
        source_plan = plans[0].model_copy(update={'outputs': OutputSelection(),
                                                'expression_profile': None})
        endpoints = tuple(PresentationEndpoint(room=h['room'], participant_identity=h['device'],
            plan=p.model_copy(update={'inputs': InputSelection(microphone=False),
                'outputs': OutputSelection(speech=True, dialogue_text=p.outputs.dialogue_text),
                'expression_profile': None})) for h, p in zip(targets, plans[1:]))
        team = TeamDispatch(opened=opened, input_plan=source_plan, endpoints=endpoints)
        handle = dict(source)
        handle['output_template'] = source_plan.model_dump(mode='json', exclude={'session_id'})
        await self.open_session(handle, source_plan.session_id,
            session_intent=SESSION_INTENT_USER_INITIATED,
            team_dispatch=team.model_dump(mode='json'))

    async def open_directed_session(
        self, source: dict, target: dict, *, source_session_id: str,
        target_session_id: str, target_companion_id: str,
    ) -> None:
        """Dispatch one inference worker, presenting in the other standing room.

        Handles are from the Provider ledger after Owner/DeviceRef validation.
        This adapter only narrows each endpoint's physical input/output rights.
        """
        from eidolon_sdk.biz.presentation import InputSelection, OutputSelection
        from eidolon.livekit.common.presentation_endpoint import PresentationEndpoint

        source_template = source.get("output_template")
        target_template = target.get("output_template")
        if source_template is None or target_template is None:
            raise InvalidTransition("directed sessions require explicit endpoint policies")
        source_plan = SessionOutputPlan(session_id=source_session_id, **source_template)
        target_plan = SessionOutputPlan(session_id=target_session_id, **target_template)
        if not source_plan.inputs.microphone or not target_plan.outputs.speech:
            raise InvalidTransition("selected endpoints do not allow microphone input and speech output")
        target_plan = target_plan.model_copy(update={
            "inputs": InputSelection(microphone=False),
            "outputs": OutputSelection(speech=True, dialogue_text=target_plan.outputs.dialogue_text),
            "expression_profile": None,
        })
        endpoint = PresentationEndpoint(room=str(target["room"]),
            participant_identity=str(target["device"]), plan=target_plan)
        source_handle = dict(source)
        source_handle["output_template"] = {
            "policy_revision": source_plan.policy_revision,
            "inputs": source_plan.inputs.model_dump(mode="json"),
            "outputs": OutputSelection().model_dump(mode="json"),
            "expression_profile": None,
        }
        await self.open_session(source_handle, source_session_id,
            session_intent=SESSION_INTENT_USER_INITIATED,
            target_companion_id=target_companion_id,
            presentation_endpoint=endpoint.model_dump(mode="json"))

    async def close_session(
        self, handle: dict[str, Any], conversation_id: str | None
    ) -> list[str]:
        """Withdraw the standing order, which is what ends the agent's job.

        Deleting the dispatch — rather than deleting the room — is what keeps
        the device's channel intact across the end of a conversation. LiveKit
        removes the agent from the room promptly, so the room is clean for the
        next session even while the old job is still draining.

        Matches on the agent name rather than a remembered dispatch id: LiveKit
        also keeps dispatch records of its own, and a session that a previous
        process opened must still be closable by this one.

        `conversation_id=None` ends whatever ordinary conversation this device
        is being served, for a caller that cannot know the id — typically
        because the device run that picked it is gone. Team and directed
        scenes are left alone: they have their own close, which also owns
        their cleanup. Returns the conversations actually ended.
        """
        room, agent = self._serving(handle)
        closed: list[str] = []
        try:
            for dispatch in await self._client().agent_dispatch.list_dispatch(room_name=room):
                if dispatch.agent_name != agent:
                    continue
                metadata = self._dispatch_metadata(dispatch)
                if conversation_id is None:
                    if (metadata.get("team_dispatch") is not None
                            or metadata.get("presentation_endpoint") is not None):
                        continue
                elif metadata.get(SESSION_CONVERSATION_ID_FIELD) != conversation_id:
                    continue
                await self._client().agent_dispatch.delete_dispatch(
                    dispatch_id=dispatch.id, room_name=room
                )
                closed.append(str(metadata.get(SESSION_CONVERSATION_ID_FIELD, "")))
        except TwirpError as exc:
            if exc.code == TwirpErrorCode.NOT_FOUND:
                return closed
            raise BackendUnavailable("LiveKit agent dispatch withdrawal failed") from exc
        except Exception as exc:
            raise BackendUnavailable("LiveKit agent dispatch withdrawal failed") from exc
        if closed:
            logger.info(
                "closed session on room=%s agent=%s conversation_ids=%s", room, agent, closed
            )
        else:
            # A close for a conversation nobody is serving is still a success,
            # but it ended nothing. Logged as "closed", four of these on
            # 2026-09-28 read like conversations ending while the one blocking
            # the device stayed six more minutes.
            logger.info(
                "no session to close on room=%s agent=%s conversation_id=%s",
                room, agent, conversation_id or "*",
            )
        return closed

    @staticmethod
    def _dispatch_metadata(dispatch: Any) -> dict[str, Any]:
        try:
            value = json.loads(str(getattr(dispatch, "metadata", "") or "{}"))
        except (TypeError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _serving(handle: dict[str, Any]) -> tuple[str, str]:
        room = handle.get("room")
        agent = handle.get("agent")
        if not room:
            raise ChannelNotServable("channel handle names no room")
        if not agent:
            raise ChannelNotServable("this channel was not provisioned to be served")
        return room, agent

    def _token(self, room: str, spec: ChannelSpec, *, ttl_seconds: int) -> str:
        """Mint a credential, and nothing more.

        The token says who the device is and what it may do. It carries no room
        configuration: server-side orchestration is declared where the room is
        declared, never routed through a credential handed to the device.

        Everything in this metadata is true for as long as the channel is —
        who the body is, whose it is, what it declared it can do about taking
        turns. Deliberately absent is `session_intent`, which is true of one
        conversation rather than of the channel, and which decides what that
        conversation is allowed to do. It travels on the agent dispatch (see
        `open_session`), where this adapter writes it per session and the
        device cannot reach it. `can_update_own_metadata=False` below is the
        other half of that: a body may not rewrite what it was issued, and the
        one thing worth rewriting is the thing that is no longer here.
        """
        sources: list[str] = []
        if spec.audio.publishes:
            sources.append("microphone")
        if spec.video.publishes:
            sources.append("camera")
        metadata: dict[str, str] = {
            "kind": "device",
            "device_id": spec.device_id,
            "owner_id": spec.owner_id,
            "manifest_id": spec.manifest_id,
        }
        if spec.serving is not None:
            metadata["interaction_mode"] = spec.serving.interaction_mode
        return (
            api.AccessToken(self._config.api_key, self._config.api_secret)
            .with_identity(spec.device_id)
            .with_name(spec.device_id)
            .with_ttl(timedelta(seconds=ttl_seconds))
            .with_grants(
                api.VideoGrants(
                    room_create=False,
                    room_join=True,
                    room=room,
                    can_publish=bool(sources),
                    can_subscribe=spec.audio.subscribes or spec.video.subscribes,
                    can_publish_data=True,
                    can_publish_sources=sources or None,
                    can_update_own_metadata=False,
                )
            )
            .with_metadata(json.dumps(metadata, ensure_ascii=False, separators=(",", ":")))
            .to_jwt()
        )
