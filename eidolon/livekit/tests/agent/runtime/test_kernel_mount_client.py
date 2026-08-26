import json
from pathlib import Path

import httpx
import pytest

from eidolon.livekit.agent.runtime import kernel_mounts
from eidolon.livekit.agent.runtime.kernel_mounts import (
    KernelMountContractError,
    KernelMountHttpClient,
)

from eidolon_sdk.device_foundation.v1.testing import named_device_instance_id

# Tests name the device they mean; the name becomes a real device
# instance id, which is a digest of a key and never a chosen string.
_DEVICE_1 = named_device_instance_id("device-1")
_DEVICE_2 = named_device_instance_id("device-2")


pytestmark = pytest.mark.asyncio


def document(**overrides):
    value = {
        "operation": "kernel.device-mount",
        "device_id": _DEVICE_1,
        "owner_id": "owner-1",
        "device_ref": {
            "device_instance_id": _DEVICE_1,
            "owner_domain_id": "owner-domain-1",
            "owner_domain_generation": 2,
            "claim_generation": 3,
            "trust_epoch": 4,
        },
        "attached_companion_id": None,
        "revision": 3,
        "created_at": "2026-08-05T00:00:00Z",
        "updated_at": "2026-08-05T00:00:00Z",
        "request_id": "mount-1",
        "fingerprint": "sha256:" + "a" * 64,
        "active": True,
    }
    value.update(overrides)
    return value


async def test_kernel_mount_client_is_owner_scoped_and_accepts_no_attachment():
    async def handler(request):
        assert request.headers["X-Eidolon-Owner"] == "owner-1"
        assert request.url.path.endswith(f"/resolve/{_DEVICE_1}")
        return httpx.Response(200, json=document())

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        client = KernelMountHttpClient(
            base_url="http://kernel.test/api/kernel/v1", http_client=http
        )
        context = await client.resolve(owner_id="owner-1", device_id=_DEVICE_1)
    finally:
        await http.aclose()

    assert context.device_id == _DEVICE_1
    assert str(context.device_ref.owner_domain_id) == "owner-domain-1"
    assert context.owner_id == "owner-1"
    assert context.mount_revision == 3
    assert context.attached_companion_id is None


async def test_kernel_mount_client_rejects_contract_drift():
    async def handler(_request):
        return httpx.Response(200, json=document(unexpected="drift"))

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        client = KernelMountHttpClient(base_url="http://kernel.test", http_client=http)
        with pytest.raises(KernelMountContractError, match="fields"):
            await client.resolve(owner_id="owner-1", device_id=_DEVICE_1)
    finally:
        await http.aclose()


async def test_kernel_mount_client_rejects_device_ref_for_another_device():
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

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = KernelMountHttpClient(base_url="http://kernel.test", http_client=http)
        with pytest.raises(KernelMountContractError, match="values"):
            await client.resolve(owner_id="owner-1", device_id=_DEVICE_1)


async def test_consumed_shape_matches_kernel_normative_mount_schema():
    workspace = Path(__file__).resolve().parents[6]
    schema_path = (
        workspace / "eidolon_kernel/eidolon_kernel/contracts/schemas/device-mount/mount.schema.json"
    )
    if not schema_path.is_file():
        pytest.skip("sibling eidolon_kernel checkout is unavailable")

    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert set(schema["properties"]) == kernel_mounts._FIELDS
    assert set(schema["required"]) == kernel_mounts._FIELDS
    assert schema["additionalProperties"] is False
