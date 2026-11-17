"""What a device needs from a channel, derived from what the device declared.

The Hub forwards the device manifest verbatim and treats the resulting binding
as opaque: it deliberately does not know how a channel is realised. This module
is the one place that reads the manifest and turns it into a transport-neutral
statement of need, so that no adapter has to parse a manifest and no adapter's
vocabulary leaks back into the service.
"""

from __future__ import annotations

from eidolon_sdk.biz.capabilities import declared_observable_properties, installed_registry


from dataclasses import dataclass
from enum import Enum
from typing import Any

from eidolon_sdk.biz.audio_input import AUDIO_INPUT_STAGE_PROPERTIES, VALID_AUDIO_INPUT_PROCESSING
from eidolon_sdk.biz.contracts import (
    SESSION_APPLICATION_COMPANION,
    SESSION_APPLICATION_HOME_COMMAND,
    VALID_SESSION_APPLICATIONS,
    VOICE_APPLICATION_PROPERTY,
)
from eidolon_sdk.biz.presentation import DeviceOutputPolicy, OutputSelection
from eidolon_sdk.biz.presentation.negotiation import (
    manifest_outputs,
    output_policy_required,
    select_outputs,
    validate_output_contract,
)
from eidolon_sdk.biz.smarthome import PANEL_PROFILE, PANEL_PROFILE_PROPERTY
from eidolon_sdk.biz.tools import DeviceToolRestriction

from .contracts import ContractError, ProvisionDevice

# Turn-taking is a device constraint, not a deployment preference: whether a
# device may listen while it speaks depends on its echo cancellation, not on
# what the operator would like. Devices declare it; the provider never guesses.
INTERACTION_MODES = frozenset({"full_duplex", "half_duplex", "ptt"})
DEFAULT_INTERACTION_MODE = "half_duplex"

_MANIFEST_INTERACTION_MODE = "interaction_mode"


class MediaFlow(Enum):
    """Which way a media kind travels, from the device's point of view."""

    NONE = "none"
    PUBLISH = "publish"
    SUBSCRIBE = "subscribe"
    DUPLEX = "duplex"

    @property
    def publishes(self) -> bool:
        return self in (MediaFlow.PUBLISH, MediaFlow.DUPLEX)

    @property
    def subscribes(self) -> bool:
        return self in (MediaFlow.SUBSCRIBE, MediaFlow.DUPLEX)


_DIRECTION_TO_FLOW = {
    "publish": MediaFlow.PUBLISH,
    "subscribe": MediaFlow.SUBSCRIBE,
    "bidirectional": MediaFlow.DUPLEX,
}


@dataclass(frozen=True, slots=True)
class ServingSpec:
    """The conversational agent this channel must be served by, if any.

    Serving belongs to the device session, independently of audio publication.
    A session with its microphone disabled can still accept text and emit its
    selected outputs.
    """

    agent_name: str
    interaction_mode: str
    application: str = SESSION_APPLICATION_COMPANION
    # Declared microphone processing as (stage, "device"|"none") pairs; an
    # undeclared stage is absent and the agent keeps its own default for it.
    # A tuple keeps the spec hashable like every other field.
    audio_input: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class ChannelSpec:
    """A transport-neutral statement of what one device's channel must carry."""

    device_id: str
    owner_id: str
    manifest_id: str
    audio: MediaFlow
    video: MediaFlow
    serving: ServingSpec | None
    output_policy: DeviceOutputPolicy | None = None
    selected_outputs: OutputSelection | None = None
    media_declared: bool = False
    smarthome_panel: bool = False
    motion_profile: str | None = None
    motion_capabilities: tuple[str, ...] = ()
    device_capabilities: tuple[str, ...] = ()
    #: Installed contracts this device declared that the Owner's own app may
    #: invoke outside a session. Not gated by the session selection: the Owner
    #: is the authority that selection answers to.
    owner_capabilities: tuple[str, ...] = ()
    #: Installed observable properties this device declared exactly.
    observable_properties: tuple[str, ...] = ()
    #: What the accepted Manifest declares whatever the Owner selected: every
    #: installed device contract and the head profile. Beside what a session
    #: may use, it tells "the device cannot" from "the Owner has not allowed it".
    declared_capabilities: tuple[str, ...] = ()
    declared_motion_profile: str | None = None
    #: The Owner's restriction on this device, already taken out of the two
    #: session tables above; carried so the grant can say which one it applied.
    tool_restriction: DeviceToolRestriction | None = None

    @property
    def needs_media(self) -> bool:
        return self.media_declared or self.audio is not MediaFlow.NONE or self.video is not MediaFlow.NONE


def _media_flow(manifest: dict[str, Any], kind: str) -> MediaFlow:
    """Collapse every declaration of one media kind into a single flow.

    A manifest may declare a kind more than once (say, publish and subscribe as
    separate entries); the channel has to satisfy all of them at once, so the
    declarations union rather than override.
    """
    publishes = False
    subscribes = False
    for entry in manifest.get("media", ()):
        if not isinstance(entry, dict) or entry.get("kind") != kind:
            continue
        flow = _DIRECTION_TO_FLOW.get(entry.get("direction", ""))
        if flow is None:
            continue
        publishes = publishes or flow.publishes
        subscribes = subscribes or flow.subscribes
    if publishes and subscribes:
        return MediaFlow.DUPLEX
    if publishes:
        return MediaFlow.PUBLISH
    if subscribes:
        return MediaFlow.SUBSCRIBE
    return MediaFlow.NONE


def _declared_interaction_mode(manifest: dict[str, Any]) -> str:
    """Read the device's declared turn-taking mode.

    The manifest schema carries immutable device attributes as properties whose
    schema pins a single value, so an `interaction_mode` property with a `const`
    (or a single-entry `enum`) is the device stating a fact about itself. A
    device that stays silent gets the conservative mode: half duplex never
    assumes echo cancellation that may not exist.
    """
    for prop in manifest.get("properties", ()):
        if not isinstance(prop, dict) or prop.get("name") != _MANIFEST_INTERACTION_MODE:
            continue
        declared = _pinned_value(prop.get("schema"))
        if isinstance(declared, str) and declared in INTERACTION_MODES:
            return declared
    return DEFAULT_INTERACTION_MODE


def _pinned_value(schema: object) -> object:
    """The single value a property schema pins (`const` or one-entry `enum`), or None."""
    if not isinstance(schema, dict):
        return None
    declared = schema.get("const")
    if declared is None:
        choices = schema.get("enum")
        if isinstance(choices, list) and len(choices) == 1:
            declared = choices[0]
    return declared


def _declared_audio_input(manifest: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Microphone processing the device states it already applies.

    Like `interaction_mode`, this is a fact about the hardware, so only a
    read-only const property counts; anything else leaves the stage undeclared.
    """
    by_name = {
        prop.get("name"): prop for prop in manifest.get("properties", ()) if isinstance(prop, dict)
    }
    declared: list[tuple[str, str]] = []
    for stage, name in AUDIO_INPUT_STAGE_PROPERTIES.items():
        prop = by_name.get(name)
        if prop is None or prop.get("writable") is not False:
            continue
        value = _pinned_value(prop.get("schema"))
        if isinstance(value, str) and value in VALID_AUDIO_INPUT_PROCESSING:
            declared.append((stage, value))
    return tuple(declared)


def _declared_voice_application(manifest: dict[str, Any]) -> str:
    """The device's voice use case; display properties never select it."""
    for prop in manifest.get("properties", ()):
        if not isinstance(prop, dict) or prop.get("name") != VOICE_APPLICATION_PROPERTY:
            continue
        schema = prop.get("schema")
        application = schema.get("const") if isinstance(schema, dict) else None
        if application not in VALID_SESSION_APPLICATIONS:
            raise ContractError("unsupported voice.application")
        return application
    return SESSION_APPLICATION_COMPANION


def derive_spec(
    device: ProvisionDevice, *, device_instance_id: str, agent_name: str,
    tool_restriction: DeviceToolRestriction | None = None,
) -> ChannelSpec:
    """Turn one device's declaration into what its channel must provide.

    ``tool_restriction`` only narrows what a session may use. What the device
    declared, and what the Owner's own app may invoke on it, stay as they are:
    keeping a Companion from the volume must not take the Owner's slider away.
    """
    manifest = device.manifest
    smarthome_panel = any(
        isinstance(prop, dict)
        and prop.get("name") == PANEL_PROFILE_PROPERTY
        and isinstance(prop.get("schema"), dict)
        and prop["schema"].get("const") == PANEL_PROFILE
        for prop in manifest.get("properties", ())
    )
    audio = _media_flow(manifest, "audio")
    video = _media_flow(manifest, "video")
    try:
        validate_output_contract(manifest)
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    capabilities = manifest_outputs(manifest)
    selected = None
    if device.output_policy is not None:
        try:
            selected = select_outputs(
                capabilities=capabilities,
                policy=device.output_policy,
                requested=device.output_policy.allowed,
                ceiling=OutputSelection(speech=True, dialogue_text=True, expression=True, audio_cue=True,
                                        motion=True),
                require_response=False,
            )
        except ValueError as exc:
            raise ContractError(str(exc)) from exc
        if not (selected.speech or selected.audio_cue):
            audio = MediaFlow.PUBLISH if audio.publishes else MediaFlow.NONE
    elif output_policy_required(capabilities, manifest=manifest):
        # New Companion clients require an explicit Owner policy. Missing or
        # corrupt policy must never silently restore legacy speech defaults.
        raise ContractError("OUTPUT_POLICY_REQUIRED")
    camera = None
    if device.output_policy is not None and device.output_policy.inputs is not None:
        if not device.output_policy.inputs.microphone:
            audio = MediaFlow.SUBSCRIBE if audio.subscribes else MediaFlow.NONE
        # None: the Owner has not decided, so the camera follows the Manifest as before.
        camera = device.output_policy.inputs.camera
        if camera == "off":
            video = MediaFlow.SUBSCRIBE if video.subscribes else MediaFlow.NONE
    serving = (
        ServingSpec(
            agent_name=agent_name,
            interaction_mode=_declared_interaction_mode(manifest),
            application=_declared_voice_application(manifest),
            audio_input=_declared_audio_input(manifest),
        )
        if _media_flow(manifest, "audio") is not MediaFlow.NONE or selected is not None
        else None
    )
    registry = installed_registry()
    # A device action is read only when it equals the installed contract: the
    # Host owns the schema, the device can only claim it. Version 1 because that
    # is what a session's invocation carries.
    installed = tuple(dict.fromkeys(
        a["name"] for a in manifest.get("actions", ())
        if isinstance(a, dict) and a.get("version") == 1 and a.get("name") in registry
        and registry.get(a["name"]).execution_location == "device"
        and a == registry.get(a["name"]).action()))
    permitted = registry.permitted(
        installed,
        outputs=selected.model_dump() if selected is not None else {},
        inputs={"camera": camera},
    )
    # What a session may use: what the selection permits, less what the Owner
    # restricted on this device. Only the two session tables read it; the
    # Owner's own controls and the declaration below do not.
    denied = frozenset(tool_restriction.denied) if tool_restriction is not None else frozenset()
    session = [name for name in installed if name in permitted and name not in denied]
    if serving is not None and serving.application == SESSION_APPLICATION_HOME_COMMAND:
        if not smarthome_panel or not audio.publishes:
            raise ContractError("home.command.v1 requires a smart-home panel and microphone")
    # The head profile the device states about itself; a session gets it only
    # when the Owner selected motion.
    declared_motion_profile = next((p["schema"]["const"] for p in manifest.get("properties", ())
        if p.get("name") == "motion.profile"
        and p.get("writable") is False and isinstance(p.get("schema"), dict)
        and p["schema"].get("const") in {"stackchan.head.v1", "eidolon.motion.v1", "eidolon.motion.v2"}), None)
    return ChannelSpec(
        device_id=device_instance_id,
        owner_id=str(device.owner_id),
        manifest_id=device.manifest_id,
        audio=audio,
        video=video,
        serving=serving,
        output_policy=device.output_policy,
        selected_outputs=selected,
        motion_profile=declared_motion_profile if selected and selected.motion else None,
        motion_capabilities=tuple(
            name for name in session if registry.get(name).required_output == "motion"),
        # Every other installed device contract the Owner's selection allows: a
        # still only when capture is allowed, a setting only with the output it
        # tunes. One rule, the registry's, rather than a branch per gate.
        device_capabilities=tuple(
            name for name in session if registry.get(name).required_output != "motion"),
        owner_capabilities=tuple(
            name for name in installed
            if registry.get(name).owner_control and device.output_policy is not None),
        observable_properties=tuple(declared_observable_properties(manifest)),
        declared_capabilities=installed,
        declared_motion_profile=declared_motion_profile,
        tool_restriction=tool_restriction,
        media_declared=_media_flow(manifest, "audio") is not MediaFlow.NONE
        or _media_flow(manifest, "video") is not MediaFlow.NONE,
        smarthome_panel=smarthome_panel,
    )
