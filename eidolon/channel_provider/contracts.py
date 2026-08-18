"""Strict wire bindings for Hub's v1 Channel Provider control contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

MAX_REQUEST_BYTES = 256 * 1024


class ContractError(ValueError):
    """The caller did not send the exact v1 Provider contract."""


class IdempotencyConflict(RuntimeError):
    """An operation or device identity was reused with different authority."""


class BackendUnavailable(RuntimeError):
    """The selected transport could not satisfy a control-plane operation."""


class UnknownChannel(RuntimeError):
    """The device named in the request has no channel to act on."""


class ChannelNotServable(RuntimeError):
    """The channel exists but was never provisioned to carry a conversation."""


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
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def request_fingerprint(value: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode()).hexdigest()


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


def _array(value: Any, *, name: str, maximum: int) -> list[Any]:
    if not isinstance(value, list) or len(value) > maximum:
        raise ContractError(f"{name} must be an array with at most {maximum} items")
    return value


def _schema_object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{name} must be a JSON Schema object")
    return value


def _manifest(value: Any) -> dict[str, Any]:
    manifest = _exact_object(
        value,
        name="device.manifest",
        required={"schema_version", "title", "properties", "actions", "events", "media"},
    )
    if manifest["schema_version"] != 1 or isinstance(manifest["schema_version"], bool):
        raise ContractError("device.manifest.schema_version must be 1")
    _text(manifest["title"], name="device.manifest.title", maximum=128)

    for index, item in enumerate(
        _array(manifest["properties"], name="device.manifest.properties", maximum=64)
    ):
        prop = _exact_object(
            item,
            name=f"device.manifest.properties[{index}]",
            required={"name", "schema", "observable", "writable"},
        )
        _text(prop["name"], name=f"device.manifest.properties[{index}].name", maximum=128)
        _schema_object(prop["schema"], name=f"device.manifest.properties[{index}].schema")
        if not isinstance(prop["observable"], bool) or not isinstance(prop["writable"], bool):
            raise ContractError("manifest property flags must be booleans")

    for index, item in enumerate(
        _array(manifest["actions"], name="device.manifest.actions", maximum=64)
    ):
        action = _exact_object(
            item,
            name=f"device.manifest.actions[{index}]",
            required={"name", "version", "input_schema", "output_schema", "idempotent"},
        )
        _text(action["name"], name=f"device.manifest.actions[{index}].name", maximum=128)
        version = action["version"]
        if isinstance(version, bool) or not isinstance(version, int) or not 1 <= version <= 65535:
            raise ContractError("manifest action version must be an integer in 1..65535")
        _schema_object(action["input_schema"], name="manifest action input_schema")
        _schema_object(action["output_schema"], name="manifest action output_schema")
        if not isinstance(action["idempotent"], bool):
            raise ContractError("manifest action idempotent must be boolean")

    for index, item in enumerate(
        _array(manifest["events"], name="device.manifest.events", maximum=64)
    ):
        event = _exact_object(
            item,
            name=f"device.manifest.events[{index}]",
            required={"name", "data_schema"},
        )
        _text(event["name"], name=f"device.manifest.events[{index}].name", maximum=128)
        _schema_object(event["data_schema"], name="manifest event data_schema")

    for index, item in enumerate(
        _array(manifest["media"], name="device.manifest.media", maximum=16)
    ):
        media = _exact_object(
            item,
            name=f"device.manifest.media[{index}]",
            required={"kind", "direction", "codecs"},
        )
        if media["kind"] not in {"audio", "video"}:
            raise ContractError("manifest media kind must be audio or video")
        if media["direction"] not in {"publish", "subscribe", "bidirectional"}:
            raise ContractError("manifest media direction is invalid")
        codecs = _array(media["codecs"], name="manifest media codecs", maximum=32)
        for codec in codecs:
            _text(codec, name="manifest media codec", maximum=64)
    return manifest


@dataclass(frozen=True, slots=True)
class ProvisionDevice:
    device_id: str
    owner_id: str
    display_name: str
    device_kind: str
    manifest: dict[str, Any] = field(repr=False)
    manifest_revision: str = ""


@dataclass(frozen=True, slots=True)
class ProvisionRequest:
    operation_id: str
    owner_domain_id: str
    device: ProvisionDevice
    fingerprint: str

    @classmethod
    def parse(cls, raw: bytes) -> ProvisionRequest:
        value = decode_json_object(raw)
        root = _exact_object(
            value,
            name="provision request",
            required={"operation", "operation_id", "owner_domain_id", "device"},
        )
        if root["operation"] != "channel.provision-device":
            raise ContractError("operation must be channel.provision-device")
        device_value = _exact_object(
            root["device"],
            name="device",
            required={
                "device_id",
                "owner_id",
                "display_name",
                "device_kind",
                "manifest",
                "manifest_revision",
            },
        )
        device = ProvisionDevice(
            device_id=_text(device_value["device_id"], name="device.device_id", maximum=128),
            owner_id=_text(device_value["owner_id"], name="device.owner_id", maximum=128),
            display_name=_text(
                device_value["display_name"],
                name="device.display_name",
                minimum=0,
                maximum=256,
            ),
            device_kind=_text(
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
            operation_id=_text(root["operation_id"], name="operation_id", maximum=128),
            owner_domain_id=_text(root["owner_domain_id"], name="owner_domain_id", maximum=128),
            device=device,
            fingerprint=request_fingerprint(value),
        )


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
    owner_domain_id: str
    device_id: str

    @classmethod
    def parse(cls, raw: bytes, *, expected: str) -> SessionRequest:
        value = decode_json_object(raw)
        root = _exact_object(
            value,
            name="session request",
            required={"operation", "owner_domain_id", "device_id"},
        )
        if root["operation"] != expected:
            raise ContractError(f"operation must be {expected}")
        return cls(
            operation=expected,
            owner_domain_id=_text(root["owner_domain_id"], name="owner_domain_id", maximum=128),
            device_id=_text(root["device_id"], name="device_id", maximum=128),
        )


@dataclass(frozen=True, slots=True)
class RevokeRequest:
    operation_id: str
    owner_domain_id: str
    device_id: str
    reason: str = field(repr=False)
    fingerprint: str = ""

    @classmethod
    def parse(cls, raw: bytes) -> RevokeRequest:
        value = decode_json_object(raw)
        root = _exact_object(
            value,
            name="revoke request",
            required={"operation", "operation_id", "owner_domain_id", "device_id", "reason"},
        )
        if root["operation"] != "channel.revoke-device":
            raise ContractError("operation must be channel.revoke-device")
        return cls(
            operation_id=_text(root["operation_id"], name="operation_id", maximum=128),
            owner_domain_id=_text(root["owner_domain_id"], name="owner_domain_id", maximum=128),
            device_id=_text(root["device_id"], name="device_id", maximum=128),
            reason=_text(root["reason"], name="reason", maximum=256),
            fingerprint=request_fingerprint(value),
        )
