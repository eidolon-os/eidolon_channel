"""Which device Manifests this Provider will carry, stated as cases.

This module exists because the Manifest is the only document in the system whose
*content* this Provider reads — `media[].kind/direction` becomes the channel's
media capability and the device's LiveKit publish grants, and the
`interaction_mode` property becomes the serving spec's turn-taking. Everything
upstream of here moves the Manifest by digest and never looks inside it: Hub
Admission checks only that the digest matches the bytes, the Kernel carries a
`manifest_revision` string, and the phone shows a `manifest_ref`.

So this file is where a shape either becomes a working channel or does not, and
the cases below name the shapes that have to keep working and the two that fail
without saying so. It started as a one-shot probe answering a single question
for the software-Body work — *will a phone-shaped Manifest be carried at all*
(`docs/设备与Body/纯软件Body准入身份裁决.md` W5, left unverified when that
decision was written) — and is kept because the answer is a precondition for the
mobile client's channel, not a fact anyone should have to re-derive.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import eidolon_sdk
import pytest

from eidolon.channel_provider.contracts import ContractError, ProvisionRequest
from eidolon.channel_provider.selection import AdapterRegistry, NoAdapterAvailable
from eidolon.channel_provider.spec import MediaFlow, derive_spec

from .helpers import device_ref, encoded


class _Adapter:
    """Just enough adapter to be selected, or not."""

    def __init__(self, name: str, *, carries_media: bool, serves_dataonly: bool = False) -> None:
        self.name = name
        self.carries_media = carries_media
        self.serves_dataonly = serves_dataonly


def _registry() -> AdapterRegistry:
    return AdapterRegistry(
        [_Adapter("livekit", carries_media=True), _Adapter("data-only", carries_media=False)],
        preference=["livekit", "data-only"],
    )


def _interaction_mode(mode: str) -> dict[str, Any]:
    """Turn-taking as the device states it: a property whose schema pins one value."""

    return {
        "name": "interaction_mode",
        "observable": False,
        "schema": {"const": mode, "type": "string"},
        "writable": False,
    }


def _golden(case_id: str) -> dict[str, Any]:
    """One of the canonical Manifest vectors, read rather than transcribed.

    These shapes used to be copied into this file by hand, which is the same
    mistake as the five hand-written definitions this module now reads instead
    of one: renaming a field in the canonical vocabulary left this file green
    while every real producer stopped being admissible. Located from the
    installed `eidolon_sdk` so it is the corpus the entry gate is checked
    against, whatever that package was installed from.
    """

    corpus = (
        Path(eidolon_sdk.__file__).resolve().parents[1]
        / "contracts/device_foundation/v1/examples/valid/common.json"
    )
    assert corpus.exists(), f"the canonical contract corpus is not at {corpus}"
    for case in json.loads(corpus.read_text(encoding="utf-8"))["cases"]:
        if case["case_id"] == case_id:
            return case["value"]
    raise AssertionError(f"the canonical corpus has no vector {case_id}")


#: The manifest a real BOX-3 build emits, held in the contract corpus as the
#: transcription of `eidolon-client-esp32/main/eidolon/hub_onboarding_protocol.cc`.
#: The control case: whatever else changes here, this one must keep provisioning.
BOX3_MANIFEST: dict[str, Any] = _golden("DF-MANIFEST-BOX3-DOCUMENT-VALID")

#: A phone: publishes a microphone, subscribes to the Companion's face, and
#: declares full duplex because the client turns on WebRTC AEC/NS/AGC explicitly.
#: It deliberately does not declare `video: publish` — that would grant a camera
#: source the client never publishes.
PHONE_MANIFEST: dict[str, Any] = _golden("DF-MANIFEST-SOFTWARE-BODY-DOCUMENT-VALID")


def _provision(manifest: dict[str, Any], *, manifest_id: str = "waveshare-box3"):
    return ProvisionRequest.parse(
        encoded(
            {
                "operation": "channel.provision-device",
                "operation_id": "manifest-shape-1",
                "device_ref": device_ref(),
                "device": {
                    "owner_id": "owner_1",
                    "display_name": "probe",
                    "device_kind": manifest_id,
                    "manifest": manifest,
                    "manifest_revision": "sha256:manifest-1",
                },
            }
        )
    )


def _spec(manifest: dict[str, Any], *, manifest_id: str = "waveshare-box3"):
    request = _provision(manifest, manifest_id=manifest_id)
    return derive_spec(
        request.device,
        device_instance_id=request.device_id,
        agent_name="eidolon-agent",
    )


def test_box3_manifest_is_carried() -> None:
    """The control. A real board's declaration provisions a media channel."""

    spec = _spec(BOX3_MANIFEST)

    assert spec.audio is MediaFlow.DUPLEX
    assert spec.video is MediaFlow.NONE
    assert spec.serving is not None
    assert spec.serving.interaction_mode == "full_duplex"
    assert _registry().select(spec).name == "livekit"


def test_phone_manifest_is_carried() -> None:
    """A phone-shaped Manifest is carried, and gets an agent.

    The unverified assumption behind the software-Body plan: nothing in
    provisioning is shaped for hardware. A phone publishing audio and
    subscribing to video reaches the same adapter a board does, and — because
    it publishes audio — is given a serving spec, which is what dispatches a
    conversational agent to it.
    """

    spec = _spec(PHONE_MANIFEST, manifest_id="eidolon-mobile-android")

    assert spec.audio is MediaFlow.DUPLEX
    assert spec.video is MediaFlow.SUBSCRIBE
    assert spec.serving is not None
    assert spec.serving.interaction_mode == "full_duplex"
    assert _registry().select(spec).name == "livekit"


def test_the_manifest_id_does_not_gate_provisioning() -> None:
    """Not an allowlist, and it must not quietly become one.

    A new Body type must never need to be registered somewhere before it can
    be given a channel. It is read nowhere that decides anything: from here it
    only travels into the LiveKit token's metadata, which no consumer reads.
    """

    spec = _spec(PHONE_MANIFEST, manifest_id="a-kind-nobody-has-ever-configured")

    assert spec.serving is not None
    assert _registry().select(spec).name == "livekit"


def test_the_wire_still_calls_it_device_kind() -> None:
    """The field is misnamed on the wire, and renaming it there is not free.

    Hub copies the Manifest id into a field called `device_kind` and has since
    the beginning; it is not a kind and never was, so every name on this side
    says `manifest_id` instead. The wire key is deliberately left alone:
    `_exact_object` accepts no unknown field and no missing one, so an Authority
    and a Provider that disagree about this name do not degrade — every device
    loses its channel until both are deployed. Renaming it is a coordinated
    breaking correction across Hub, this Provider, Kernel, Admin and the phone,
    and this test is what stops it happening by accident on one side.
    """

    request = _provision(BOX3_MANIFEST, manifest_id="esp-box-3")

    assert request.device.manifest_id == "esp-box-3"
    # The message a renamed sender would actually get: not a warning, a
    # refusal naming the field.
    with pytest.raises(ContractError, match="missing fields: device_kind"):
        ProvisionRequest.parse(
            encoded(
                {
                    "operation": "channel.provision-device",
                    "operation_id": "manifest-shape-1",
                    "device_ref": device_ref(),
                    "device": {
                        "owner_id": "owner_1",
                        "display_name": "probe",
                        "manifest_id": "esp-box-3",
                        "manifest": BOX3_MANIFEST,
                        "manifest_revision": "sha256:manifest-1",
                    },
                }
            )
        )


def test_media_declarations_decide_publish_grants() -> None:
    """What the device declares is what it may send — subscribe grants nothing.

    Asserted on the spec rather than the token because this is the fact the
    adapter reads: `spec.video.publishes` is what puts `camera` in the token's
    allowed sources. A Manifest that declares `video: subscribe` must not.
    """

    spec = _spec(PHONE_MANIFEST, manifest_id="eidolon-mobile-android")

    assert spec.audio.publishes and spec.audio.subscribes
    assert spec.video.subscribes
    assert not spec.video.publishes


def test_a_manifest_without_codecs_is_carried() -> None:
    """`codecs` no longer refuses a device, because nothing reads one.

    It used to be required here and optional in Hub's own binding, so a single
    document could pass the Authority and be refused at provisioning — which
    is a Claim an Owner approved and a channel that never arrives. Nothing in
    any repository reads a codec value, and the values shipped by producers
    that both work already disagree: a real board sends `opus` where this
    package's fixtures said `audio/opus`. What decides the device's token
    grants is `kind` and `direction`, and those stay required.
    """

    manifest = {
        "schema_version": 1,
        "title": "Sparse",
        "properties": [_interaction_mode("full_duplex")],
        "actions": [],
        "events": [],
        "media": [{"kind": "audio", "direction": "bidirectional"}],
    }

    spec = _spec(manifest)

    assert spec.audio is MediaFlow.DUPLEX
    assert spec.serving is not None
    assert _registry().select(spec).name == "livekit"


def test_the_canonical_shape_is_what_this_provider_accepts() -> None:
    """One definition, read here rather than copied here.

    This module used to hold a hand-written copy of the vocabulary, and it was
    the strictest of five separate definitions. The gap between the entry gate
    and this consumer was the defect itself, so the property worth pinning is
    not any single rule but that both ends read the same thing: the golden
    vectors in `eidolon_sdk/contracts/device_foundation/v1` are checked at the
    entry, and this provisioning path accepts exactly them.
    """

    from eidolon_sdk.device_foundation.v1 import DeviceCapabilityManifest

    for manifest in (BOX3_MANIFEST, PHONE_MANIFEST):
        assert DeviceCapabilityManifest.model_validate(manifest)
        assert _provision(manifest).device.manifest == manifest

    for refused in (
        {**BOX3_MANIFEST, "media": [{"kind": "audio"}]},
        {**BOX3_MANIFEST, "media": [{"kind": "audio", "direction": "duplex"}]},
        {**BOX3_MANIFEST, "schema_version": 2},
        {**BOX3_MANIFEST, "title": "  "},
        {**BOX3_MANIFEST, "endpoints": []},
    ):
        with pytest.raises(ContractError):
            _provision(refused)


def _silent_manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "title": "Silent",
        "properties": [],
        "actions": [],
        "events": [],
        "media": [],
    }


def test_a_body_that_declares_no_media_is_a_legal_document() -> None:
    """The contract does not decide whether a Body must have a voice.

    Refusing a media-less Manifest at the entry would burn "every Body speaks"
    into a wire contract four repositories share, and a future sensor-only Body
    would need a coordinated breaking correction to exist. So the document is
    admissible, and what it does not get is a channel — see below.
    """

    request = _provision(_silent_manifest())

    assert request.device.manifest["media"] == []
    assert not _spec(_silent_manifest()).needs_media


def test_a_body_that_declares_no_media_is_not_quietly_given_a_channel() -> None:
    """The failure that did not announce itself, and no longer happens.

    With no media there is nothing to carry, so `needs_media` was false, so
    *every* adapter "supported" it — including one carrying no media at all —
    while the same absence of audio meant no serving spec and so no agent. The
    device was handed a channel, joined it, and was never answered by anyone,
    with nothing raised on any side.

    Selection now asks the narrower question, and no shipped adapter answers
    yes, so the request is refused with a non-retryable domain problem: the
    Authority records it as refused rather than as a binding still pending.
    """

    spec = _spec(_silent_manifest())
    assert spec.serving is None

    for registry in (
        _registry(),
        AdapterRegistry([_Adapter("data-only", carries_media=False)], preference=["data-only"]),
    ):
        with pytest.raises(NoAdapterAvailable) as refusal:
            registry.select(spec)
        assert refusal.value.retryable is False, "waiting cannot make this succeed"

    # And it stays possible to carry one deliberately, without the contract or
    # this module changing: an adapter says so.
    willing = AdapterRegistry(
        [_Adapter("sensor-bus", carries_media=False, serves_dataonly=True)],
        preference=["sensor-bus"],
    )
    assert willing.select(spec).name == "sensor-bus"


def test_legacy_endpoints_manifest_is_refused_here_though_admission_accepted_it() -> None:
    """The shape that got in, and the reason validation at this end is too late.

    `{"endpoints": []}` is what the first canonically claimed device sent
    (`hub/domain/devices/manifest.py`). Admission used to store the document
    without checking its shape — the admission schema typed it
    `{"type": "object"}` — so a Manifest like this reached an approved Claim
    and failed only here, at provisioning, where the Authority recorded a
    binding still pending while the Claim, the mount and the Companion binding
    all kept reading healthy.

    The entry now refuses it too, from the same definition this reads
    (`DF-ADMISSION-CREATE-REJECTS-THE-DOCUMENT-THAT-STRANDED-A-CLAIM`), so this
    case has stopped being the first refusal. It stays because being the last
    line of defence is still worth asserting.
    """

    with pytest.raises(ContractError, match="Field required"):
        _provision({"endpoints": []})


def test_data_only_deployment_cannot_carry_a_speaking_device() -> None:
    """A device that needs media is refused rather than given a mute channel."""

    spec = _spec(PHONE_MANIFEST, manifest_id="eidolon-mobile-android")
    registry = AdapterRegistry(
        [_Adapter("data-only", carries_media=False)], preference=["data-only"]
    )

    with pytest.raises(NoAdapterAvailable):
        registry.select(spec)
