"""Device capability directory and command delivery exposed to the Agent.

The Agent projects a device's declared capabilities into LLM tools, but it is
not a transport: it cannot reach a LiveKit room and must not learn one. These
two operations are the whole contract it needs — what a body can do right now,
and one bounded command delivered to it.

A device appears in the directory only while this Provider holds a live channel
for it. Its capabilities come from the manifest the Host accepted and this
Provider validated at provision time, never from a transport probe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from eidolon_sdk.biz.body import CapabilityDeclaration

CommandOutcome = Literal["succeeded", "failed", "accepted", "running", "rejected"]

#: Bumped when a field changes meaning on this contract. The Agent pins the
#: version it implements and refuses a Provider that answers a different one,
#: rather than reading a response shape it did not agree to.
DEVICE_CAPABILITY_CONTRACT_VERSION = 1


class CapabilityContractVersionMismatch(RuntimeError):
    pass


def require_contract_version(value: object) -> int:
    if value != DEVICE_CAPABILITY_CONTRACT_VERSION:
        raise CapabilityContractVersionMismatch(
            f"device capability contract version {value!r} is not "
            f"{DEVICE_CAPABILITY_CONTRACT_VERSION}"
        )
    return DEVICE_CAPABILITY_CONTRACT_VERSION


@dataclass(frozen=True, slots=True)
class DeviceCommandRequest:
    owner_id: str
    companion_id: str
    device_id: str
    op: str
    capability_version: int
    payload: dict[str, object]
    ttl_ms: int
    #: Chosen by the caller and stable across retries of one logical command.
    #: A Provider that already completed this key answers with that outcome
    #: instead of sending the device a second command.
    idempotency_key: str


@dataclass(frozen=True, slots=True)
class DeviceCommandResult:
    device_id: str
    op: str
    capability_version: int
    status: CommandOutcome
    result: object = None
    code: str = ""
    message: str = ""

    def to_json(self) -> dict[str, object]:
        return {
            "contract_version": DEVICE_CAPABILITY_CONTRACT_VERSION,
            "device_id": self.device_id,
            "op": self.op,
            "capability_version": self.capability_version,
            "status": self.status,
            "result": self.result,
            "code": self.code,
            "message": self.message,
        }


def capabilities_from_manifest(manifest: dict[str, Any]) -> tuple[CapabilityDeclaration, ...]:
    """Map a manifest's `actions` onto the Agent's capability contract.

    An action carries a schema pair and an idempotency flag; the Agent needs a
    description and a result schema to hand the model. Falls back to the action
    name, because an empty description fails the Agent's own validation and
    would hide a capability the device really has.
    """

    actions = manifest.get("actions")
    if actions is None:
        return ()
    if not isinstance(actions, list):
        raise ValueError("manifest actions must be a list")
    declarations: list[CapabilityDeclaration] = []
    for action in actions:
        if not isinstance(action, dict):
            raise ValueError("manifest action must be an object")
        name = action.get("name")
        version = action.get("version", 1)
        if not isinstance(name, str) or not name:
            raise ValueError("manifest action has no name")
        if not isinstance(version, int) or isinstance(version, bool):
            raise ValueError(f"manifest action version must be an integer: {name}")
        declarations.append(
            CapabilityDeclaration(
                name=name,
                version=version,
                description=str(action.get("description") or name),
                input_schema=action.get("input_schema") or {"type": "object"},
                result_schema=action.get("output_schema") or {},
            )
        )
    return tuple(declarations)
