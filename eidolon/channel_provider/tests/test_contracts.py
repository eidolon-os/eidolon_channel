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

from eidolon_sdk.biz.contracts import (
    SESSION_INTENT_PRESENCE,
    SESSION_INTENT_USER_INITIATED,
    VALID_SESSION_INTENTS,
)
from eidolon_sdk.device_foundation.v1.testing import named_device_instance_id

# Tests name the device they mean; the name becomes a real device
# instance id, which is a digest of a key and never a chosen string.
_DEVICE_1 = named_device_instance_id("device-1")


def test_provision_request_matches_hub_v1_contract() -> None:
    request = ProvisionRequest.parse(encoded(provision_payload()))

    assert request.operation_id == "enrollment-1"
    assert request.owner_domain_id == "owner-domain-1"
    assert request.device_ref.device_instance_id == _DEVICE_1
    assert str(request.device.owner_id) == "owner_1"
    assert request.device.manifest["media"][0]["kind"] == "audio"
    assert request.fingerprint.startswith("sha256:")


def test_revoke_request_matches_hub_v1_contract() -> None:
    request = RevokeRequest.parse(encoded(revoke_payload()))

    assert request.operation_id == "revoke-1"
    assert request.device_ref.device_instance_id == _DEVICE_1
    assert request.reason == "owner-request"


def test_session_request_matches_hub_v1_contract() -> None:
    request = SessionRequest.parse(
        encoded(session_payload(operation=OPEN_SESSION)), expected=OPEN_SESSION
    )

    assert request.operation == OPEN_SESSION
    assert request.owner_domain_id == "owner-domain-1"
    assert request.device_id == _DEVICE_1
    # An open request that says nothing about why is a user-driven session —
    # the same answer anyone without standing to say otherwise would get.
    assert request.session_intent == SESSION_INTENT_USER_INITIATED


@pytest.mark.parametrize("intent", sorted(VALID_SESSION_INTENTS))
def test_an_open_request_may_name_any_valid_session_intent(intent: str) -> None:
    """The authenticated road is how a privileged wake becomes expressible.

    Before this the Provider had no field for it at all, so `presence_initiated`
    and `proactive_initiated` were behaviour the agent implemented and nothing
    could ask for.
    """
    request = SessionRequest.parse(
        encoded(session_payload(operation=OPEN_SESSION, session_intent=intent)),
        expected=OPEN_SESSION,
    )

    assert request.session_intent == intent


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"unexpected": True}),
        # An operation_id would imply this is a replayable event; it is not.
        lambda value: value.update({"operation_id": "session-1"}),
        lambda value: value.update({"operation": CLOSE_SESSION}),
        lambda value: value.pop("device_ref"),
        # Rejected rather than quietly demoted to user_initiated: an
        # orchestrator that misspells a presence wake must be told, not handed
        # an ordinary session while believing it holds an Owner lease.
        lambda value: value.update({"session_intent": "presence-initiated"}),
        lambda value: value.update({"session_intent": "PRESENCE_INITIATED"}),
        lambda value: value.update({"session_intent": ""}),
        lambda value: value.update({"session_intent": None}),
    ],
)
def test_session_rejects_contract_drift(mutation) -> None:
    value = session_payload(operation=OPEN_SESSION)
    mutation(value)

    with pytest.raises(ContractError):
        SessionRequest.parse(encoded(value), expected=OPEN_SESSION)


def test_a_close_request_cannot_name_a_session_intent() -> None:
    """Ending a conversation has no intent to state, so naming one is drift."""
    value = session_payload(
        operation=CLOSE_SESSION, session_intent=SESSION_INTENT_PRESENCE
    )

    with pytest.raises(ContractError, match="session_intent"):
        SessionRequest.parse(encoded(value), expected=CLOSE_SESSION)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"unexpected": True}),
        lambda value: value["device"].update({"unexpected": True}),
        lambda value: value["device"]["manifest"].update({"unexpected": True}),
        lambda value: value["device"]["manifest"].update({"schema_version": 2}),
        lambda value: value["device"]["manifest"]["media"][0].update({"direction": "sideways"}),
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
        '"owner_domain_id":"owner-domain-1"',
        '"owner_domain_id":"owner-domain-1","owner_domain_id":"other"',
    )

    with pytest.raises(ContractError, match="duplicate JSON field"):
        ProvisionRequest.parse(raw.encode())


def test_the_observed_host_address_is_optional_and_absent_means_nothing_was_seen() -> None:
    """Most reconciles have no device request in flight to observe."""

    request = ProvisionRequest.parse(encoded(provision_payload()))

    assert request.observed_host_address == ""


def test_the_observed_host_address_is_read_from_the_root_not_the_device() -> None:
    payload = provision_payload()
    payload["observed_host_address"] = "192.168.100.19"

    assert ProvisionRequest.parse(encoded(payload)).observed_host_address == "192.168.100.19"


def test_an_observed_host_address_that_is_not_an_address_is_refused() -> None:
    payload = provision_payload()
    payload["observed_host_address"] = "eidolon-hub-f89c0ecca5d0070a7989.local"

    with pytest.raises(ContractError, match="observed_host_address"):
        ProvisionRequest.parse(encoded(payload))


def test_the_observed_host_address_stays_out_of_the_idempotency_fingerprint() -> None:
    """A device that moved between two deliveries must not lose its channel.

    The Authority derives one operation id from the DeviceRef and the Manifest
    and re-sends it until it converges. Neither of those moves when a device
    changes network, but the address it reaches this Host on does — so if the
    observation counted as part of the ask, the second delivery of the *same*
    pending operation would read as that id reused for a different payload,
    and be refused IdempotencyConflict, which is terminal.
    """

    without = ProvisionRequest.parse(encoded(provision_payload()))
    wifi = provision_payload()
    wifi["observed_host_address"] = "192.168.100.19"
    cable = provision_payload()
    cable["observed_host_address"] = "10.42.0.2"

    seen = {
        without.fingerprint,
        ProvisionRequest.parse(encoded(wifi)).fingerprint,
        ProvisionRequest.parse(encoded(cable)).fingerprint,
    }
    assert len(seen) == 1

    # The control: something that *is* part of the ask still moves it, so the
    # assertion above is about this one field and not about a dead fingerprint.
    moved = provision_payload()
    moved["device"]["manifest_revision"] = "sha256:manifest-2"
    assert ProvisionRequest.parse(encoded(moved)).fingerprint != without.fingerprint
