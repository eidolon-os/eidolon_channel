"""Strict wire bindings for Hub's v1 Channel Provider control contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import rfc8785
from eidolon_sdk.device_foundation.v1 import (
    BusinessOwnerId,
    DeviceCapabilityManifest,
    DeviceRef,
)
from eidolon_sdk.biz.contracts import normalize_conversation_id
from pydantic import ValidationError

MAX_REQUEST_BYTES = 256 * 1024


class ContractError(ValueError):
    """The caller did not send the exact v1 Provider contract."""


class DomainError(RuntimeError):
    """A stable Channel domain problem, independent of its transport mapping."""

    code = "INTERNAL"
    category = "internal"
    retryable = False
    http_status = 500


class IdempotencyConflict(DomainError):
    code = "IDEMPOTENCY_CONFLICT"
    category = "conflict"
    http_status = 409


class StaleGeneration(DomainError):
    code = "STALE_GENERATION"
    category = "conflict"
    http_status = 409


class InvalidTransition(DomainError):
    code = "INVALID_TRANSITION"
    category = "conflict"
    http_status = 409


class Unauthenticated(DomainError):
    code = "UNAUTHENTICATED"
    category = "auth"
    http_status = 401


class Forbidden(DomainError):
    code = "FORBIDDEN"
    category = "forbidden"
    http_status = 403


class ProviderUnavailable(DomainError):
    code = "PROVIDER_UNAVAILABLE"
    category = "unavailable"
    retryable = True
    http_status = 503


class BackendUnavailable(ProviderUnavailable):
    """The selected transport could not satisfy a control-plane operation."""


class UnknownChannel(DomainError):
    """The device named in the request has no channel to act on."""

    code = "NOT_FOUND"
    category = "missing"
    http_status = 404


class ChannelNotServable(DomainError):
    """The channel exists but was never provisioned to carry a conversation."""

    code = "INVALID_TRANSITION"
    category = "conflict"
    http_status = 409


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ContractError(f"duplicate JSON field: {key}")
        value[key] = item
    return value


def decode_json_object(raw: bytes) -> dict[str, Any]:
    if not raw or len(raw) > MAX_REQUEST_BYTES:
        raise ContractError("request body is empty or exceeds 256KiB")
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("request body must be one UTF-8 JSON object") from exc
    if not isinstance(value, dict):
        raise ContractError("request body must be one JSON object")
    return value


def canonical_json(value: Any) -> str:
    return rfc8785.dumps(value).decode("utf-8")


def request_fingerprint(value: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(rfc8785.dumps(value)).hexdigest()


def _exact_object(
    value: Any,
    *,
    name: str,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{name} must be an object")
    allowed = required | (optional or set())
    missing = required - value.keys()
    extra = value.keys() - allowed
    if missing:
        raise ContractError(f"{name} is missing fields: {','.join(sorted(missing))}")
    if extra:
        raise ContractError(f"{name} has unknown fields: {','.join(sorted(extra))}")
    return value


def _text(value: Any, *, name: str, minimum: int = 1, maximum: int) -> str:
    if not isinstance(value, str) or not minimum <= len(value) <= maximum:
        raise ContractError(f"{name} must contain {minimum}..{maximum} characters")
    if minimum and not value.strip():
        raise ContractError(f"{name} cannot be blank")
    return value


def _manifest_problem(exc: ValidationError) -> str:
    """The first thing wrong with the document, as a place and a reason."""

    first = exc.errors()[0]
    where = ".".join(str(part) for part in first["loc"]) or "<document>"
    return f"device.manifest.{where}: {first['msg']}"


def _manifest(value: Any) -> dict[str, Any]:
    """Check the declaration against the one definition of its shape.

    This used to be a hand-written copy of the vocabulary, and it was the
    strictest of five: it required `codecs`, which Hub's own binding made
    optional and which nothing anywhere reads, so one document could pass the
    Authority and be refused here. Reading the canonical definition instead is
    what makes the entry gate and this consumer accept the same set — the gap
    between them was the whole defect, because a document admitted upstream and
    refused here becomes a Claim the Owner approved and a channel that never
    arrives.

    The parsed model is deliberately discarded and the decoded object returned
    unchanged: the Manifest's digest is over exactly these bytes, and `spec.py`
    reads the document, not a binding.
    """

    if not isinstance(value, dict):
        raise ContractError("device.manifest must be an object")
    try:
        DeviceCapabilityManifest.model_validate(value)
    except ValidationError as exc:
        raise ContractError(_manifest_problem(exc)) from exc
    return value


@dataclass(frozen=True, slots=True)
class ProvisionDevice:
    owner_id: BusinessOwnerId
    display_name: str
    manifest_id: str
    manifest: dict[str, Any] = field(repr=False)
    manifest_revision: str = ""


@dataclass(frozen=True, slots=True)
class ProvisionRequest:
    operation: str
    operation_id: str
    device_ref: DeviceRef
    device: ProvisionDevice
    fingerprint: str

    @classmethod
    def parse(cls, raw: bytes) -> ProvisionRequest:
        value = decode_json_object(raw)
        root = _exact_object(
            value,
            name="provision request",
            required={"operation", "operation_id", "device_ref", "device"},
        )
        if root["operation"] not in {"channel.provision-device", "channel.refresh-device"}:
            raise ContractError(
                "operation must be channel.provision-device or channel.refresh-device"
            )
        device_value = _exact_object(
            root["device"],
            name="device",
            required={
                "owner_id",
                "display_name",
                "device_kind",
                "manifest",
                "manifest_revision",
            },
        )
        try:
            device_ref = DeviceRef.model_validate(root["device_ref"])
            owner_id = BusinessOwnerId.model_validate(device_value["owner_id"])
        except ValidationError as exc:
            raise ContractError("device_ref or business owner id is invalid") from exc
        device = ProvisionDevice(
            owner_id=owner_id,
            display_name=_text(
                device_value["display_name"],
                name="device.display_name",
                minimum=0,
                maximum=256,
            ),
            manifest_id=_text(
                device_value["device_kind"], name="device.device_kind", maximum=96
            ),
            manifest=_manifest(device_value["manifest"]),
            manifest_revision=_text(
                device_value["manifest_revision"],
                name="device.manifest_revision",
                maximum=96,
            ),
        )
        return cls(
            operation=root["operation"],
            operation_id=_text(root["operation_id"], name="operation_id", maximum=128),
            device_ref=device_ref,
            device=device,
            fingerprint=request_fingerprint(value),
        )

    @property
    def owner_domain_id(self) -> str:
        return str(self.device_ref.owner_domain_id)

    @property
    def device_id(self) -> str:
        return self.device_ref.device_instance_id


OPEN_SESSION = "channel.open-session"
CLOSE_SESSION = "channel.close-session"


@dataclass(frozen=True, slots=True)
class SessionRequest:
    """Start or end one stretch of conversation on an already-open channel.

    Deliberately carries no `operation_id`. Provision and revocation are events
    whose outcome must be replayable, so they are recorded and keyed. A session
    request is not an event but a statement of desired state — served, or not —
    and the adapter converges onto it. Two "open" requests mean one session, and
    closing a channel nobody is serving is a success, so there is nothing a
    replay key would protect.
    """

    operation: str
    device_ref: DeviceRef
    conversation_id: str

    @classmethod
    def parse(cls, raw: bytes, *, expected: str) -> SessionRequest:
        value = decode_json_object(raw)
        root = _exact_object(
            value,
            name="session request",
            required={"operation", "device_ref", "conversation_id"},
        )
        if root["operation"] != expected:
            raise ContractError(f"operation must be {expected}")
        try:
            device_ref = DeviceRef.model_validate(root["device_ref"])
        except ValidationError as exc:
            raise ContractError("device_ref is invalid") from exc
        conversation_id = normalize_conversation_id(root["conversation_id"])
        if conversation_id is None:
            raise ContractError("conversation_id is invalid")
        return cls(
            operation=expected,
            device_ref=device_ref,
            conversation_id=conversation_id,
        )

    @property
    def owner_domain_id(self) -> str:
        return str(self.device_ref.owner_domain_id)

    @property
    def device_id(self) -> str:
        return self.device_ref.device_instance_id


@dataclass(frozen=True, slots=True)
class CurrentRequest:
    """Ask what binding a device has, without asking for one to change.

    The Authority decides between beginning a generation and advancing one, and
    that decision needs the Provider's own answer about what exists. Without a
    read it had to guess by issuing ``provision`` every time — which replays a
    spent idempotency key once the first refresh has fenced it, and left every
    device permanently without a channel about two hours after enrolment.
    """

    device_ref: DeviceRef

    @classmethod
    def parse(cls, raw: bytes) -> CurrentRequest:
        value = decode_json_object(raw)
        root = _exact_object(
            value,
            name="current binding request",
            required={"operation", "device_ref"},
        )
        if root["operation"] != "channel.current-device":
            raise ContractError("operation must be channel.current-device")
        try:
            device_ref = DeviceRef.model_validate(root["device_ref"])
        except ValidationError as exc:
            raise ContractError("device_ref is invalid") from exc
        return cls(device_ref=device_ref)

    @property
    def owner_domain_id(self) -> str:
        return str(self.device_ref.owner_domain_id)

    @property
    def device_id(self) -> str:
        return self.device_ref.device_instance_id


@dataclass(frozen=True, slots=True)
class RevokeRequest:
    operation_id: str
    device_ref: DeviceRef
    reason: str = field(repr=False)
    fingerprint: str = ""

    @classmethod
    def parse(cls, raw: bytes) -> RevokeRequest:
        value = decode_json_object(raw)
        root = _exact_object(
            value,
            name="revoke request",
            required={"operation", "operation_id", "device_ref", "reason"},
        )
        if root["operation"] != "channel.revoke-device":
            raise ContractError("operation must be channel.revoke-device")
        try:
            device_ref = DeviceRef.model_validate(root["device_ref"])
        except ValidationError as exc:
            raise ContractError("device_ref is invalid") from exc
        return cls(
            operation_id=_text(root["operation_id"], name="operation_id", maximum=128),
            device_ref=device_ref,
            reason=_text(root["reason"], name="reason", maximum=256),
            fingerprint=request_fingerprint(value),
        )

    @property
    def owner_domain_id(self) -> str:
        return str(self.device_ref.owner_domain_id)

    @property
    def device_id(self) -> str:
        return self.device_ref.device_instance_id
