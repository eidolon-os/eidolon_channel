import json
from pathlib import Path

import httpx
import pytest

from eidolon.livekit.agent.runtime import kernel_bodies
from eidolon.livekit.agent.runtime.kernel_bodies import (
    KernelBodyContractError,
    KernelBodyHttpClient,
    body_endpoint_id,
)

from eidolon_sdk.device_foundation.v1.testing import named_device_instance_id

# Tests name the device they mean; the name becomes a real device
# instance id, which is a digest of a key and never a chosen string.
_DEVICE_1 = named_device_instance_id("device-1")
_DEVICE_2 = named_device_instance_id("device-2")


pytestmark = pytest.mark.asyncio


def assignment(**overrides):
    value = {
        "operation": "kernel.body-assignment",
        "assignment_id": f"assignment:{body_endpoint_id(_DEVICE_1)}",
        "body_endpoint_id": body_endpoint_id(_DEVICE_1),
        "device_id": _DEVICE_1,
        "endpoint_id": "body",
        "owner_id": "owner-1",
        "companion_id": "companion-1",
        "selection_provenance": "user_selected",
        "change_reason": None,
        "mode": "default",
        "policy_refs": [],
        "revision": 1,
        "generation": 1,
        "updated_at": "2026-08-05T00:00:00Z",
        "status": {
            "observed_generation": 1,
            "effective_companion_id": "companion-1",
            "conditions": ["Realized"],
        },
    }
    value.update(overrides)
    return value


def document(**overrides):
    value = {
        "operation": "kernel.body-endpoint",
        "body_endpoint_id": body_endpoint_id(_DEVICE_1),
        "device_id": _DEVICE_1,
        "owner_id": "owner-1",
        "endpoint_id": "body",
        "device_ref": {
            "device_instance_id": _DEVICE_1,
            "owner_domain_id": "owner-domain-1",
            "owner_domain_generation": 2,
            "claim_generation": 3,
            "trust_epoch": 4,
        },
        "mount_revision": 3,
        "roles": ["body"],
        "assignment_policy": "optional",
        "risk_class": "safe",
        "concurrency": "exclusive",
        "source": "derived",
        "present": True,
        "assignment": None,
    }
    value.update(overrides)
    return value


async def _resolve(handler, *, device_id=_DEVICE_1, owner_id="owner-1"):
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        client = KernelBodyHttpClient(
            base_url="http://kernel.test/api/kernel/v1", http_client=http
        )
        return await client.resolve(owner_id=owner_id, device_id=device_id)
    finally:
        await http.aclose()


async def test_the_body_client_is_owner_scoped_and_accepts_nobody_answering():
    async def handler(request):
        assert request.headers["X-Eidolon-Owner"] == "owner-1"
        assert request.url.path.endswith(f"/body-endpoints/{body_endpoint_id(_DEVICE_1)}")
        return httpx.Response(200, json=document())

    context = await _resolve(handler)

    assert context.device_id == _DEVICE_1
    assert str(context.device_ref.owner_domain_id) == "owner-domain-1"
    assert context.owner_id == "owner-1"
    assert context.mount_revision == 3
    assert context.answering_companion_id is None


async def test_one_read_answers_both_who_owns_the_device_and_who_answers_through_it():
    """The reason the Body endpoint is what this consumer reads.

    Splitting the Companion out of the mount would otherwise have made every
    device session start with two round trips.
    """

    calls: list[str] = []

    async def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json=document(assignment=assignment()))

    context = await _resolve(handler)

    assert len(calls) == 1
    assert context.answering_companion_id == "companion-1"
    assert context.mount_revision == 3


async def test_an_assignment_the_authority_says_is_not_in_force_answers_as_nobody():
    """``effective_companion_id``, not the spec's ``companion_id``.

    A Body whose device is no longer mounted keeps its assignment on purpose, so
    the device can come back to the same Eidolon. Reading the spec here would
    start a session as an Eidolon on hardware that is not there.
    """

    async def handler(_request):
        return httpx.Response(
            200,
            json=document(
                assignment=assignment(
                    status={
                        "observed_generation": 1,
                        "effective_companion_id": None,
                        "conditions": ["CapabilityMissing"],
                    }
                )
            ),
        )

    context = await _resolve(handler)
    assert context.answering_companion_id is None


async def test_the_body_client_rejects_contract_drift():
    async def handler(_request):
        return httpx.Response(200, json=document(unexpected="drift"))

    with pytest.raises(KernelBodyContractError, match="fields"):
        await _resolve(handler)


async def test_the_body_client_rejects_a_device_ref_for_another_device():
    async def handler(_request):
        return httpx.Response(
            200,
            json=document(
                device_ref={
                    **document()["device_ref"],
                    "device_instance_id": _DEVICE_2,
                }
            ),
        )

    with pytest.raises(KernelBodyContractError, match="values"):
        await _resolve(handler)


async def test_a_body_whose_device_is_not_mounted_is_not_a_session():
    async def handler(_request):
        return httpx.Response(200, json=document(present=False))

    with pytest.raises(KernelBodyContractError, match="values"):
        await _resolve(handler)


async def test_consumed_shape_matches_kernel_normative_body_endpoint_schema():
    workspace = Path(__file__).resolve().parents[6]
    schema_path = (
        workspace
        / "eidolon_kernel/eidolon_kernel/contracts/schemas/body-mesh/endpoint.schema.json"
    )
    if not schema_path.is_file():
        pytest.skip("sibling eidolon_kernel checkout is unavailable")

    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert set(schema["properties"]) == kernel_bodies._FIELDS
    assert set(schema["required"]) == kernel_bodies._FIELDS
    assert schema["additionalProperties"] is False
    assert (
        schema["properties"]["operation"]["const"] == "kernel.body-endpoint"
    )


async def test_the_derived_endpoint_id_this_consumer_composes_matches_the_producer():
    """A mirrored constant, checked against the authority that defines it.

    This package deliberately does not depend on the Kernel, so the one word it
    has to know — which Body a device's single endpoint is called — is mirrored
    and compared rather than imported.
    """

    workspace = Path(__file__).resolve().parents[6]
    body = workspace / "eidolon_kernel/eidolon_kernel/domain/body.py"
    if not body.is_file():
        pytest.skip("sibling eidolon_kernel checkout is unavailable")
    source = body.read_text(encoding="utf-8")
    assert f'DERIVED_ENDPOINT_ID = "{kernel_bodies._DERIVED_ENDPOINT_ID}"' in source
