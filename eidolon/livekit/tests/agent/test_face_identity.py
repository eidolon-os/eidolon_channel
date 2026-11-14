"""vision.who: what may be said from a still, the enrolled faces and the turn's voice."""

import json
import logging
import time
from types import SimpleNamespace

import httpx
import pytest
from eidolon_sdk.biz.capabilities import installed_registry, validate
from eidolon_sdk.biz.person_face import FaceCandidates

from eidolon.livekit.agent.eidolon_agent_rpc.channel_tools import ChannelToolCall, ChannelToolResult
from eidolon.livekit.agent.session.capability_router import CapabilityRouter
from eidolon.livekit.agent.session.face_identity import (
    FaceIdentityClient,
    SessionFaceIdentity,
    Voice,
    attach_face_identity,
    decide,
    describe,
    read_faces,
    voice_of,
)

ADA, BOB = "a" * 32, "b" * 32
NAMES = {ADA: "Ada", BOB: "Bob"}
SCHEMA = installed_registry().get("vision.who").result_schema
IMAGE = b"\xff\xd8 still \xff\xd9"
CALL = ChannelToolCall("turn-1", "call-1", "vision.who", {})


def face(*comparisons, quality="ok", distinct=True):
    return {
        "quality": quality,
        "distinct": distinct,
        "comparisons": [{"ref": ref, "score": 0.0, "band": band} for ref, band in comparisons],
    }


def candidates(*people):
    features = {"model_id": "m", "weights_hash": "c" * 64, "preprocessing_version": "p",
                "vector": [0.25, 0.5, 0.75]}
    return FaceCandidates(
        owner_id="owner", revision="d" * 64,
        persons=[{"person_id": pid, "display_name": NAMES[pid],
                  "templates": [{"revision": 1, "features": features}]} for pid in people])


def result(faces, voice=Voice(), enrolled=2):
    content = decide(read_faces({"faces": faces}, NAMES), voice, enrolled=enrolled)
    validate(SCHEMA, content)
    return content


# -- reading Vision's evidence -------------------------------------------------


def test_only_a_strong_distinct_match_names_someone():
    faces = read_faces({"faces": [
        face((ADA, "strong"), (BOB, "none")),
        face((ADA, "weak")),
        face((ADA, "strong"), (BOB, "strong"), distinct=False),
        face((ADA, "none")),
        face(),
        face(quality="too_small"),
    ]}, NAMES)
    # Two faces cannot both be Ada, so the first is demoted with nothing to demote it: only one claims her.
    assert [f["status"] for f in faces] == [
        "identified", "uncertain", "uncertain", "unknown", "unknown", "uncertain"]


def test_two_faces_claiming_one_person_name_neither():
    faces = read_faces({"faces": [face((ADA, "strong")), face((ADA, "strong"))]}, NAMES)
    assert faces == [{"status": "uncertain"}, {"status": "uncertain"}]


def test_a_match_to_someone_no_longer_enrolled_is_not_named():
    assert read_faces({"faces": [face(("z" * 32, "strong"))]}, NAMES) == [{"status": "uncertain"}]


def test_faces_beyond_the_compared_ones_still_count():
    faces = read_faces({"faces": [face((ADA, "strong"))], "faces_dropped": 2}, NAMES)
    assert [f["status"] for f in faces] == ["identified", "uncertain", "uncertain"]


# -- deciding with the voice ---------------------------------------------------


def test_voice_and_face_agreeing_allow_the_name():
    content = result([face((ADA, "strong"))], Voice(ADA, "Ada"))
    assert (content["agreement"], content["address_as"]) == ("consistent", "Ada")
    assert content["voice"] == {"status": "identified", "name": "Ada"}


def test_voice_and_face_disagreeing_withhold_the_name_and_say_so():
    content = result([face((ADA, "strong"))], Voice(BOB, "Bob"))
    assert content["agreement"] == "conflict" and "address_as" not in content
    assert "Bob" in content["guidance"] and "Ada" in content["guidance"]
    assert "Do not use a name" in content["guidance"]


def test_a_face_alone_names_the_person_plainly():
    content = result([face((ADA, "strong"))])
    assert (content["agreement"], content["address_as"]) == ("face_only", "Ada")
    assert content["voice"] == {"status": "unidentified"}


@pytest.mark.parametrize(
    ("faces", "voice"),
    [
        ([face((ADA, "strong"))], Voice()),              # face alone
        ([face((ADA, "strong"))], Voice(ADA, "Ada")),    # both
        ([face((ADA, "none"))], Voice(ADA, "Ada")),      # voice alone
    ],
)
def test_someone_recognized_is_answered_by_name_without_hedging(faces, voice):
    guidance = result(faces, voice)["guidance"]
    assert guidance.startswith("This is Ada. Answer plainly, by name")
    assert "Do not hedge" in guidance
    # The model is told not to narrate the evidence, and is not handed hedges to repeat.
    for hedge in ("looks like", "sounds like", "tentative", "not confirmation", "resemblance, not"):
        assert hedge not in guidance


@pytest.mark.parametrize("faces", [[], [face((ADA, "weak"))], [face((ADA, "none"))]])
def test_a_voice_alone_is_used_as_before(faces):
    content = result(faces, Voice(BOB, "Bob"))
    assert (content["agreement"], content["address_as"]) == ("voice_only", "Bob")


@pytest.mark.parametrize(
    ("faces", "enrolled", "hint"),
    [
        ([], 2, "cannot see anyone"),
        ([face()], 0, "nobody has enabled face recognition"),
        ([face((ADA, "none"))], 2, "do not recognize"),
        ([face((ADA, "weak"))], 2, "do not recognize"),
    ],
)
def test_no_evidence_means_no_name(faces, enrolled, hint):
    content = result(faces, enrolled=enrolled)
    assert content["agreement"] == "none" and "address_as" not in content
    assert hint in content["guidance"]


def test_several_faces_say_who_is_present_not_who_is_speaking():
    both = [face((ADA, "strong")), face((BOB, "strong"))]
    silent = result(both)
    assert silent["agreement"] == "none" and "address_as" not in silent
    assert "In view you recognize: Ada, Bob." in silent["guidance"]
    heard = result(both, Voice(BOB, "Bob"))
    assert (heard["agreement"], heard["address_as"]) == ("consistent", "Bob")
    elsewhere = result([face((ADA, "strong")), face((ADA, "none"))], Voice(BOB, "Bob"))
    assert (elsewhere["agreement"], elsewhere["address_as"]) == ("voice_only", "Bob")


def test_the_result_carries_names_only():
    content = result([face((ADA, "strong")), face((BOB, "weak"))], Voice(ADA, "Ada"))
    text = json.dumps(content)
    assert ADA not in text and BOB not in text and "score" not in text and "band" not in text


def test_the_log_tells_a_small_face_from_a_weak_match_without_saying_who():
    evidence = {"faces": [
        {"quality": "ok", "face_px": 180, "distinct": True,
         "comparisons": [{"ref": ADA, "score": 0.71, "band": "strong"}]},
        {"quality": "ok", "face_px": 44, "distinct": False,
         "comparisons": [{"ref": BOB, "score": 0.41, "band": "weak"}]},
        {"quality": "too_small", "face_px": 21, "comparisons": []},
    ]}
    text = describe(evidence)
    assert text == "ok:strong:180px,ok:weak:close:44px,too_small:-:21px"
    assert ADA not in text and "0.71" not in text
    assert describe({"faces": []}) == "-"


# -- the turn's voice ----------------------------------------------------------


def observation(person=None, *, expires_in=10.0, name="Ada"):
    evidence = SimpleNamespace(person_id=person) if person else None
    return SimpleNamespace(
        expires_at=time.time() + expires_in, display_name=name,
        decided=lambda use: evidence if use == "addressing" else None)


def test_voice_is_the_addressing_decision_of_a_live_observation():
    assert voice_of(None) == Voice()
    assert voice_of(SimpleNamespace(latest=lambda: None)) == Voice()
    assert voice_of(SimpleNamespace(latest=lambda: observation())) == Voice()
    assert voice_of(SimpleNamespace(latest=lambda: observation(ADA))) == Voice(ADA, "Ada")
    assert voice_of(SimpleNamespace(latest=lambda: observation(ADA, expires_in=-1))) == Voice()


# -- one whole request ---------------------------------------------------------


class Services:
    """Data and Vision behind one transport; records what each was asked."""

    def __init__(self, people=(ADA, BOB), faces=None, *, data_status=200, vision_status=200):
        self.people, self.faces = people, faces if faces is not None else [face((ADA, "strong"))]
        self.data_status, self.vision_status = data_status, vision_status
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.host == "data":
            assert request.headers["authorization"] == "Bearer data-token"
            assert request.url.path.endswith("/owners/owner/faces/recognition-candidates")
            return httpx.Response(self.data_status, json=candidates(*self.people).model_dump(mode="json"))
        assert request.headers["authorization"] == "Bearer vision-token"
        assert request.url.path == "/v1/faces/match"
        return httpx.Response(self.vision_status, json={"faces": self.faces, "faces_dropped": 0})


def runtime(services, *, capture=None, voice=Voice()):
    async def take(call):
        assert (call.name, call.arguments) == ("camera.capture", {"purpose": "look", "max_edge": 1280})
        assert call.turn_id == "turn-1" and call.call_id == "call-1:camera.capture"
        return ChannelToolResult(call.turn_id, call.call_id, call.name, True, {},
                                 outcome_state="completed", attachment=IMAGE,
                                 attachment_media_type="image/jpeg")

    http = httpx.AsyncClient(transport=httpx.MockTransport(services))
    client = FaceIdentityClient(http, data_url="http://data", data_token="data-token",
                                vision_url="http://vision/", vision_token="vision-token", timeout_s=2)
    return SessionFaceIdentity(capture=capture or take, client=client, owner_id="owner",
                               device_id="device", voice=lambda: voice)


@pytest.mark.asyncio
async def test_a_request_takes_one_still_and_compares_it_with_the_enrolled(caplog):
    services = Services()
    router = CapabilityRouter()
    who = runtime(services, voice=Voice(ADA, "Ada"))
    router.register(who.capability_names, who.execute_channel_tool, who.cancel_channel_tool,
                    location="channel")
    with caplog.at_level(logging.INFO):
        outcome = await router.execute(CALL)

    assert outcome.ok and outcome.content["address_as"] == "Ada"
    assert outcome.attachment == b""  # the still stops here; the Agent gets text
    sent = next(r for r in services.requests if r.url.host == "vision").content
    assert IMAGE in sent and b"owner" in sent and b"device" in sent
    templates = json.loads(sent.split(b'name="templates"\r\n\r\n')[1].split(b"\r\n--")[0])
    assert [t["ref"] for t in templates] == [ADA, BOB] and templates[0]["vector"] == [0.25, 0.5, 0.75]
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("vision_who"))
    assert "status=consistent faces=1 enrolled=2 seen=ok:strong:0px" in line
    assert f"image_bytes={len(IMAGE)}" in line
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "Ada" not in logged and ADA not in logged and "0.25" not in logged


@pytest.mark.asyncio
async def test_nobody_enrolled_still_reports_what_is_in_view():
    outcome = await runtime(Services(people=(), faces=[face()])).execute_channel_tool(CALL)
    assert outcome.ok and outcome.content["enrolled"] == 0 and outcome.content["faces"] == 1
    assert outcome.content["visible"] == [{"status": "unknown"}]


@pytest.mark.parametrize(
    ("capture_error", "code"),
    [("camera_disabled", "not_permitted"), ("UNAVAILABLE", "not_permitted"),
     ("camera_unavailable", "camera_unavailable"), ("transfer_failed", "camera_unavailable")],
)
@pytest.mark.asyncio
async def test_a_refused_or_failed_still_is_reported_and_nothing_is_compared(capture_error, code):
    services = Services()

    async def refuse(call):
        return ChannelToolResult(call.turn_id, call.call_id, call.name, False, {},
                                 error_code=capture_error, outcome_state="failed")

    outcome = await runtime(services, capture=refuse).execute_channel_tool(CALL)
    assert (outcome.ok, outcome.error_code, outcome.content) == (False, code, {})
    assert not [r for r in services.requests if r.url.host == "vision"]


@pytest.mark.asyncio
async def test_a_still_without_bytes_is_no_still():
    async def empty(call):
        return ChannelToolResult(call.turn_id, call.call_id, call.name, True, {},
                                 outcome_state="completed")

    outcome = await runtime(Services(), capture=empty).execute_channel_tool(CALL)
    assert outcome.error_code == "camera_unavailable"


@pytest.mark.parametrize("broken", [{"data_status": 503}, {"vision_status": 503}, {"vision_status": 409}])
@pytest.mark.asyncio
async def test_data_or_vision_failing_is_said_not_guessed(broken):
    outcome = await runtime(Services(**broken), voice=Voice(ADA, "Ada")).execute_channel_tool(CALL)
    assert (outcome.ok, outcome.error_code) == (False, "recognition_unavailable")


@pytest.mark.asyncio
async def test_candidates_of_another_owner_are_refused():
    def other_owner(request):
        body = candidates(ADA).model_dump(mode="json") | {"owner_id": "someone-else"}
        return httpx.Response(200, json=body)

    outcome = await runtime(other_owner).execute_channel_tool(CALL)
    assert outcome.error_code == "recognition_unavailable"


# -- when it is offered --------------------------------------------------------


class Llm:
    def __init__(self, capabilities):
        self.device_capability_rpc = None if capabilities is None else SimpleNamespace(
            plan=SimpleNamespace(device_capabilities=capabilities), execute=None)
        self.added = []

    def add_channel_tools(self, execute, cancel, names):
        self.added.append(names)


def config(url="http://vision"):
    return SimpleNamespace(
        face_identity=SimpleNamespace(vision_url=url, vision_token_env="TEST_VISION_TOKEN",
                                      timeout_sec=2.0, max_edge=1280),
        runtime_authority=SimpleNamespace(data_api_url="http://data",
                                          data_service_token_env="TEST_DATA_TOKEN"))


@pytest.mark.parametrize(
    ("capabilities", "url", "token", "offered"),
    [
        (("camera.capture",), "http://vision", "t", True),
        ((), "http://vision", "t", False),       # camera level is not "capture"
        (None, "http://vision", "t", False),     # the device offers no capabilities
        (("camera.capture",), "", "t", False),   # Vision is not configured
        (("camera.capture",), "http://vision", "", False),
    ],
)
@pytest.mark.asyncio
async def test_offered_only_with_a_capturing_camera_and_a_configured_vision(
    monkeypatch, capabilities, url, token, offered
):
    monkeypatch.setenv("TEST_VISION_TOKEN", token)
    llm = Llm(capabilities)
    http = await attach_face_identity(llm, config(url), owner_id="owner", device_id="device",
                                      speaker_observer=None)
    assert (http is not None, llm.added) == (offered, [frozenset({"vision.who"})] if offered else [])
    if http is not None:
        await http.aclose()


# -- beside the session's other Channel tools ----------------------------------


def test_standing_tools_survive_the_music_runtime_coming_and_going():
    from eidolon.livekit.agent.eidolon_agent_rpc.grpc_llm import EidolonAgentGrpcLlm

    async def handler(call):
        raise AssertionError("not called")

    llm = EidolonAgentGrpcLlm(target="127.0.0.1:1", device_token=lambda: "x", conversation_id="c")
    llm.add_channel_tools(handler, None, frozenset({"vision.who"}))
    assert llm._channel_tool_capabilities == frozenset({"vision.who"})
    llm.configure_channel_tools(handler, None, frozenset({"music.status"}))
    assert llm._channel_tool_capabilities == frozenset({"vision.who", "music.status"})
    llm.configure_channel_tools(None, None, frozenset())
    assert llm._channel_tool_capabilities == frozenset({"vision.who"})


def test_the_speaker_observer_reports_a_result_that_arrived_after_the_turn_was_frozen():
    from eidolon.livekit.agent.session.conversation_speaker import ConversationSpeaker

    observer = ConversationSpeaker(recognizer=None, owner_id="o", session_id="s", secret="k",
                                   publish=lambda value: None)
    assert observer.latest() is None
    observer._frozen = "pending"
    assert observer.latest() == "pending"
    observer._task = SimpleNamespace(done=lambda: True, cancelled=lambda: False,
                                     result=lambda: "recognized")
    assert observer.latest() == "recognized"
