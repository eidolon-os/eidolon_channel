from __future__ import annotations

import json

import pytest

from eidolon.channel_provider.contracts import (
    ContractError,
    ProvisionRequest,
    RevokeRequest,
)

from .helpers import encoded, provision_payload, revoke_payload


def test_provision_request_matches_hub_v1_contract() -> None:
    request = ProvisionRequest.parse(encoded(provision_payload()))

    assert request.operation_id == "enrollment-1"
    assert request.hub_id == "hub-1"
    assert request.device.device_id == "device-1"
    assert request.device.owner_id == "owner-1"
    assert request.device.manifest["media"][0]["kind"] == "audio"
    assert request.fingerprint.startswith("sha256:")


def test_revoke_request_matches_hub_v1_contract() -> None:
    request = RevokeRequest.parse(encoded(revoke_payload()))

    assert request.operation_id == "revoke-1"
    assert request.device_id == "device-1"
    assert request.reason == "owner-request"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"unexpected": True}),
        lambda value: value["device"].update({"unexpected": True}),
        lambda value: value["device"]["manifest"].update({"unexpected": True}),
        lambda value: value["device"]["manifest"].update({"schema_version": 2}),
        lambda value: value["device"]["manifest"]["media"][0].update(
            {"direction": "sideways"}
        ),
    ],
)
def test_provision_rejects_contract_drift(mutation) -> None:
    value = provision_payload()
    mutation(value)

    with pytest.raises(ContractError):
        ProvisionRequest.parse(encoded(value))


def test_contract_rejects_duplicate_json_keys() -> None:
    value = json.dumps(provision_payload(), separators=(",", ":"))
    raw = value.replace('"hub_id":"hub-1"', '"hub_id":"hub-1","hub_id":"other"')

    with pytest.raises(ContractError, match="duplicate JSON field"):
        ProvisionRequest.parse(raw.encode())
