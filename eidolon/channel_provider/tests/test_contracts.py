from __future__ import annotations

import json

import pytest

from eidolon.channel_provider.contracts import (
    CLOSE_SESSION,
    OPEN_SESSION,
    ContractError,
    ProvisionRequest,
    RevokeRequest,
    SessionRequest,
)

from .helpers import encoded, provision_payload, revoke_payload, session_payload


def test_provision_request_matches_hub_v1_contract() -> None:
    request = ProvisionRequest.parse(encoded(provision_payload()))

    assert request.operation_id == "enrollment-1"
    assert request.owner_domain_id == "owner-1"
    assert request.device.device_id == "device-1"
    assert request.device.owner_id == "owner-1"
    assert request.device.manifest["media"][0]["kind"] == "audio"
    assert request.fingerprint.startswith("sha256:")


def test_revoke_request_matches_hub_v1_contract() -> None:
    request = RevokeRequest.parse(encoded(revoke_payload()))

    assert request.operation_id == "revoke-1"
    assert request.device_id == "device-1"
    assert request.reason == "owner-request"


def test_session_request_matches_hub_v1_contract() -> None:
    request = SessionRequest.parse(
        encoded(session_payload(operation=OPEN_SESSION)), expected=OPEN_SESSION
    )

    assert request.operation == OPEN_SESSION
    assert request.owner_domain_id == "owner-1"
    assert request.device_id == "device-1"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"unexpected": True}),
        # An operation_id would imply this is a replayable event; it is not.
        lambda value: value.update({"operation_id": "session-1"}),
        lambda value: value.update({"operation": CLOSE_SESSION}),
        lambda value: value.pop("device_id"),
    ],
)
def test_session_rejects_contract_drift(mutation) -> None:
    value = session_payload(operation=OPEN_SESSION)
    mutation(value)

    with pytest.raises(ContractError):
        SessionRequest.parse(encoded(value), expected=OPEN_SESSION)


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
    raw = value.replace(
        '"owner_domain_id":"owner-1"',
        '"owner_domain_id":"owner-1","owner_domain_id":"other"',
    )

    with pytest.raises(ContractError, match="duplicate JSON field"):
        ProvisionRequest.parse(raw.encode())
