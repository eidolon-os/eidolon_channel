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


async def test_the_shape_this_consumer_pins_is_the_producers_own_definition():
    """Imported, and no longer skippable.

    This was two cross-repo tests. One compared the pinned field set to the
    Kernel's JSON Schema — a schema that said ``status: {"type": "object"}``, so
    it could not have caught a change to the one field read below. The other
    substring-matched the Kernel's source for a constant. Both skipped when the
    sibling checkout was absent, which is how a check disappears by passing.

    There is nothing left to compare. The type this consumer validates with is
    the type the producer builds its response from.
    """

    from eidolon_sdk.device_foundation.v1 import DERIVED_ENDPOINT_ID, BodyEndpoint

    assert kernel_bodies.BodyEndpoint is BodyEndpoint
    assert body_endpoint_id(_DEVICE_1) == f"{_DEVICE_1}:{DERIVED_ENDPOINT_ID}"


async def test_a_status_that_stopped_saying_who_answers_is_refused_not_read_as_nobody():
    """The drift this consumer could not previously see.

    Reading ``status`` out of an untyped dictionary, a producer that dropped
    ``effective_companion_id`` was indistinguishable from a Body that answers as
    nobody: ``.get()`` returns None for both. That is the failure mode the
    original incident had — a read that failed and reported itself as an answer.
    """

    async def handler(_request):
        broken = assignment()
        broken["status"] = {"observed_generation": 1, "conditions": ["Realized"]}
        return httpx.Response(200, json=document(assignment=broken))

    with pytest.raises(KernelBodyContractError, match="fields"):
        await _resolve(handler)


async def test_a_condition_outside_the_authoritys_vocabulary_is_drift():
    """``conditions`` is a closed vocabulary, and now it is enforced as one.

    A word this consumer has never been told the meaning of must not arrive
    looking like one it has.
    """

    async def handler(_request):
        drifted = assignment()
        drifted["status"] = {
            "observed_generation": 1,
            "effective_companion_id": "companion-1",
            "conditions": ["InForce"],
        }
        return httpx.Response(200, json=document(assignment=drifted))

    with pytest.raises(KernelBodyContractError, match="fields"):
        await _resolve(handler)
