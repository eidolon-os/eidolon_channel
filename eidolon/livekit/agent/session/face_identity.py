"""``vision.who``: who is in view, decided here from two independent kinds of evidence.

One request does four things, all inside this process's session:

1. asks the device for one still (the existing ``camera.capture``; the device
   knows nothing about faces),
2. asks Data which people of this Owner enabled face recognition, and for the
   template each of them agreed to,
3. hands the still and those templates to the Host's Vision service, which
   returns a score band per face and per enrolled person,
4. decides what may be said, together with the voice evidence of the turn that
   asked.

Voice and face are never added together. They agree, disagree, or only one of
them says anything, and the result states which: agreement makes a name safe to
use, disagreement withholds it. A face match is resemblance: it never unlocks
anything, there is no liveness check, and the result carries names only.

Nothing is kept. The still and the templates live in locals for one call; a face
that matches nobody leaves no trace beyond a count in the log. Names, person
ids, scores and features are never logged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from urllib.parse import quote

from eidolon_sdk.biz.capabilities.camera import MAX_FRAME_BYTES
from eidolon_sdk.biz.capabilities.vision import VISION_WHO
from eidolon_sdk.biz.person_face import FaceCandidates
from eidolon_sdk.biz.presentation import CAMERA_CAPTURE

from ..eidolon_agent_rpc.channel_tools import ChannelToolCall, ChannelToolResult

log = logging.getLogger(__name__)

# The whole request, inside the capability's declared 12 s.
BUDGET_S = 10.0
CAPTURE_TIMEOUT_S = 5.0
DATA_TIMEOUT_S = 1.5

_CAPTURE_REFUSED = {"camera_disabled", "UNAVAILABLE"}

_MULTIPLE = (
    " Several people are in view: faces say who is present, never which of them is speaking."
)


class RecognitionUnavailable(Exception):
    """Data or Vision could not be asked, or answered something unusable."""


@dataclass(frozen=True, slots=True)
class Voice:
    """What the turn's voice evidence allows for addressing: a person, or nobody."""

    person_id: str | None = None
    name: str | None = None


class FaceIdentityClient:
    """The two Host services a comparison needs. Holds nothing between calls."""

    def __init__(self, http, *, data_url, data_token, vision_url, vision_token, timeout_s):
        self._http = http
        self._data = data_url.rstrip("/")
        self._data_headers = {"Authorization": f"Bearer {data_token}"}
        self._vision = vision_url.rstrip("/")
        self._vision_headers = {"Authorization": f"Bearer {vision_token}"}
        self._timeout_s = timeout_s

    async def candidates(self, owner_id: str) -> FaceCandidates:
        try:
            response = await self._http.get(
                f"{self._data}/api/companion-authority/v1/owners/{quote(owner_id, safe='')}"
                "/faces/recognition-candidates",
                headers=self._data_headers,
                timeout=DATA_TIMEOUT_S,
            )
            response.raise_for_status()
            result = FaceCandidates.model_validate(response.json())
        except Exception as exc:
            raise RecognitionUnavailable(f"candidates:{type(exc).__name__}") from exc
        if result.owner_id != owner_id:
            raise RecognitionUnavailable("candidates:owner_mismatch")
        return result

    async def match(self, image: bytes, candidates: FaceCandidates, *, device_id, correlation_id):
        templates = [
            {
                "ref": person.person_id,
                "vector": template.features.vector,
                "model_id": template.features.model_id,
                "preprocessing_version": template.features.preprocessing_version,
            }
            for person in candidates.persons
            for template in person.templates
        ]
        try:
            response = await self._http.post(
                f"{self._vision}/v1/faces/match",
                headers=self._vision_headers,
                files={"image": ("frame.jpg", image, "image/jpeg")},
                data={
                    "templates": json.dumps(templates, separators=(",", ":")),
                    "owner_id": candidates.owner_id,
                    "device_id": device_id,
                    "correlation_id": correlation_id,
                },
                timeout=self._timeout_s,
            )
            response.raise_for_status()
            body = response.json()
            if not isinstance(body.get("faces"), list):
                raise ValueError("faces missing")
            return body
        except Exception as exc:
            # 409 here means the stored templates were made by another model
            # than Vision runs now: they need re-enrolling, not a retry.
            status = getattr(getattr(exc, "response", None), "status_code", None)
            raise RecognitionUnavailable(f"match:{status or type(exc).__name__}") from exc


def read_faces(evidence: dict, names: dict[str, str]) -> list[dict]:
    """Each face Vision reported as ``{"status", "person_id"?, "name"?}``.

    Only a strong match that is clear of the runner-up names someone. A weaker
    resemblance, a face too small to compare, and two faces claiming the same
    person are all ``uncertain``: present, not named.
    """
    faces = []
    for face in evidence.get("faces", []):
        comparisons = face.get("comparisons") or []
        leader = comparisons[0] if comparisons else None
        if face.get("quality") != "ok":
            faces.append({"status": "uncertain"})
        elif leader is None or leader.get("band") == "none":
            faces.append({"status": "unknown"})
        elif (
            leader.get("band") == "strong"
            and face.get("distinct", True)
            and leader.get("ref") in names
        ):
            faces.append(
                {"status": "identified", "person_id": leader["ref"], "name": names[leader["ref"]]}
            )
        else:
            faces.append({"status": "uncertain"})
    claimed = [f["person_id"] for f in faces if f["status"] == "identified"]
    for face in faces:
        if face["status"] == "identified" and claimed.count(face["person_id"]) > 1:
            face.pop("person_id")
            face.pop("name")
            face["status"] = "uncertain"
    return faces + [{"status": "uncertain"}] * int(evidence.get("faces_dropped", 0))


def decide(faces: list[dict], voice: Voice, *, enrolled: int) -> dict:
    """The capability result: what was seen, what was heard, and what may be said."""
    seen = [f for f in faces if f["status"] == "identified"]
    heard = voice.person_id is not None and bool(voice.name)
    address = None

    if len(faces) == 1 and seen:
        face = seen[0]
        if not heard:
            agreement, address = "face_only", face["name"]
            guidance = (
                f"The person in view looks like {face['name']}; the voice was not recognized. "
                f"You may say they look like {face['name']} and use the name, but keep it "
                "tentative: this is resemblance, not confirmation."
            )
        elif face["person_id"] == voice.person_id:
            agreement, address = "consistent", face["name"]
            guidance = (
                f"Voice and face both point to {face['name']}. You may address them by name; "
                "if asked how you know, say they sound and look like them."
            )
        else:
            agreement = "conflict"
            guidance = (
                f"The voice sounds like {voice.name} but the person in view looks like "
                f"{face['name']}. Do not address anyone by name. If asked, say exactly that "
                "you hear one and see the other, and do not share either person's private details."
            )
    elif heard:
        in_view = any(f["person_id"] == voice.person_id for f in seen)
        agreement, address = ("consistent" if in_view else "voice_only"), voice.name
        guidance = (
            f"Voice and face both point to {voice.name}; you may address them by name."
            if in_view
            else f"The voice sounds like {voice.name}; no face in view confirms it. "
            "Address them as you would from the voice alone."
        )
    else:
        agreement = "none"
        if not faces:
            guidance = "Nobody is in view and the voice was not recognized. You cannot tell who this is."
        elif enrolled == 0:
            guidance = (
                "Someone is in view, but nobody has enabled face recognition yet, so you "
                "cannot tell who it is by sight. The Owner can enable it for a person in "
                "the phone app. Do not guess a name."
            )
        elif seen:
            guidance = "You cannot tell who is speaking. Do not address anyone by name."
        else:
            guidance = (
                "You do not recognize the person in view: they may not be enrolled, or the "
                "view may be too far or unclear. Do not guess a name; you may ask them to "
                "come closer and face you."
            )

    if len(faces) > 1:
        guidance += _MULTIPLE
        present = [f["name"] for f in seen]
        if present:
            guidance += " Recognized in view: " + ", ".join(present) + "."

    result = {
        "faces": len(faces),
        "enrolled": enrolled,
        "visible": [
            {"status": f["status"], **({"name": f["name"]} if "name" in f else {})} for f in faces
        ],
        "voice": {"status": "identified", "name": voice.name} if heard else {"status": "unidentified"},
        "agreement": agreement,
        "guidance": guidance[:512],
    }
    if address:
        result["address_as"] = address
    return result


class SessionFaceIdentity:
    """Executes ``vision.who`` for one session."""

    capability_names = frozenset({VISION_WHO})

    def __init__(self, *, capture, client, owner_id, device_id, voice, max_edge=1280):
        self._capture, self._client = capture, client
        self._owner_id, self._device_id = owner_id, device_id
        self._voice, self._max_edge = voice, max_edge

    async def cancel_channel_tool(self, call) -> None:
        return None  # nothing outlives the call: cancelling the task is the whole cancel

    async def execute_channel_tool(self, call: ChannelToolCall) -> ChannelToolResult:
        started = time.monotonic()
        outcome = {"status": "failed", "faces": 0, "enrolled": 0, "image_bytes": 0}
        try:
            async with asyncio.timeout(BUDGET_S):
                content = await self._who(call, outcome)
            outcome["status"] = content["agreement"]
            return ChannelToolResult(
                call.turn_id, call.call_id, call.name, True, content, outcome_state="completed"
            )
        except _Refused as refused:
            outcome["status"] = refused.code
            return self._failed(call, refused.code)
        except TimeoutError:
            outcome["status"] = "timeout"
            return self._failed(call, "timeout")
        finally:
            # Counts and timings only: no name, person, score or feature.
            log.info(
                "vision_who turn=%s call=%s status=%s faces=%d enrolled=%d image_bytes=%d ms=%d",
                call.turn_id, call.call_id, outcome["status"], outcome["faces"],
                outcome["enrolled"], outcome["image_bytes"], int((time.monotonic() - started) * 1000),
            )

    async def _who(self, call, outcome) -> dict:
        # Who is enrolled is asked while the device takes the picture.
        candidates_task = asyncio.create_task(self._client.candidates(self._owner_id))
        try:
            try:
                async with asyncio.timeout(CAPTURE_TIMEOUT_S):
                    captured = await self._capture(
                        ChannelToolCall(
                            call.turn_id,
                            f"{call.call_id}:{CAMERA_CAPTURE}",
                            CAMERA_CAPTURE,
                            {"purpose": "look", "max_edge": self._max_edge},
                        )
                    )
            except TimeoutError:
                raise _Refused("camera_unavailable") from None
            if not captured.ok:
                raise _Refused(
                    "not_permitted" if captured.error_code in _CAPTURE_REFUSED else "camera_unavailable"
                )
            image = captured.attachment
            if not image or len(image) > MAX_FRAME_BYTES:
                raise _Refused("camera_unavailable")
            outcome["image_bytes"] = len(image)
            try:
                candidates = await candidates_task
                evidence = await self._client.match(
                    image, candidates, device_id=self._device_id, correlation_id=call.turn_id
                )
            except RecognitionUnavailable as exc:
                log.warning("vision_who recognition unavailable turn=%s cause=%s", call.turn_id, exc)
                raise _Refused("recognition_unavailable") from None
        finally:
            if not candidates_task.done():
                candidates_task.cancel()
            elif not candidates_task.cancelled():
                candidates_task.exception()  # retrieved: an abandoned failure is not an error

        names = {person.person_id: person.display_name for person in candidates.persons}
        faces = read_faces(evidence, names)
        outcome["faces"], outcome["enrolled"] = len(faces), len(names)
        return decide(faces, self._voice(), enrolled=len(names))

    @staticmethod
    def _failed(call, code) -> ChannelToolResult:
        return ChannelToolResult(
            call.turn_id, call.call_id, call.name, False, {}, error_code=code,
            error_message=code, outcome_state="failed",
        )


class _Refused(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def voice_of(observer) -> Voice:
    """What the session's speaker observer allows for addressing in this turn."""
    observation = observer.latest() if observer is not None else None
    if observation is None or observation.expires_at <= time.time():
        return Voice()
    evidence = observation.decided("addressing")
    if evidence is None:
        return Voice()
    return Voice(person_id=evidence.person_id, name=observation.display_name)


async def attach_face_identity(llm, cfg, *, owner_id, device_id, speaker_observer):
    """Offer ``vision.who`` in this session when everything it needs is present.

    Needs: a device that may take a still on request (the Owner's camera level
    is ``capture``), a configured Vision service, and its token. Returns the
    HTTP client to close with the session, or None when the capability is off.
    """
    device = getattr(llm, "device_capability_rpc", None)
    face = cfg.face_identity
    token = os.environ.get(face.vision_token_env, "")
    if device is None or CAMERA_CAPTURE not in device.plan.device_capabilities:
        return None
    if not face.vision_url or not token:
        log.info("vision.who off: Vision service is not configured")
        return None
    from eidolon_sdk.core.http import create_async_client

    http = create_async_client(timeout=face.timeout_sec)
    runtime = SessionFaceIdentity(
        capture=device.execute,
        client=FaceIdentityClient(
            http,
            data_url=cfg.runtime_authority.data_api_url,
            data_token=os.environ.get(cfg.runtime_authority.data_service_token_env, ""),
            vision_url=face.vision_url,
            vision_token=token,
            timeout_s=face.timeout_sec,
        ),
        owner_id=owner_id,
        device_id=device_id,
        voice=lambda: voice_of(speaker_observer),
        max_edge=face.max_edge,
    )
    llm.add_channel_tools(
        runtime.execute_channel_tool, runtime.cancel_channel_tool, runtime.capability_names
    )
    log.info("vision.who offered for this session")
    return http
