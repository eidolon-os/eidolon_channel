import asyncio
from types import SimpleNamespace

import pytest

from eidolon_sdk.biz.presentation import SessionOutputPlan, OutputSelection
from eidolon_sdk.biz.presentation.capability_rpc import ExecutionSnapshot
from eidolon.livekit.agent.session import capability_rpc as module
from eidolon.livekit.agent.eidolon_agent_rpc.channel_tools import ChannelToolCall


@pytest.mark.asyncio
async def test_rpc_waits_for_terminal_and_keeps_execution_identity(monkeypatch):
    calls = []
    async def peer(room):
        return "bound-device"
    async def perform_rpc(**kw):
        import json
        body = json.loads(kw["payload"])
        calls.append((kw["method"], body["execution_id"]))
        assert kw["destination_identity"] == "bound-device"
        return ExecutionSnapshot(execution_id=body["execution_id"],
            state="accepted" if len(calls) == 1 else "completed", sequence=len(calls),
            completion_basis="software_sequence" if len(calls) > 1 else "unconfirmed").model_dump_json()
    monkeypatch.setattr(module, "wait_for_runtime_participant_identity", peer)
    room = SimpleNamespace(local_participant=SimpleNamespace(perform_rpc=perform_rpc))
    plan = SessionOutputPlan(session_id="session", policy_revision=1,
        outputs=OutputSelection(motion=True), motion_profile="eidolon.motion.v1",
        motion_capabilities=("motion.nod",))
    runtime = module.SessionCapabilityRPC(room, plan)
    call = ChannelToolCall("turn", "call", "motion.nod", {"repeats": 1, "wait_for_completion": True})
    result = await runtime.execute(call)
    assert result.ok and result.outcome_state == "completed"
    assert calls[0][1] == calls[1][1]
    assert not runtime.pending


@pytest.mark.asyncio
async def test_undeclared_capability_never_calls_transport():
    runtime = module.SessionCapabilityRPC(None, SessionOutputPlan(session_id="s", policy_revision=1,
        outputs=OutputSelection(motion=False)))
    result = await runtime.execute(ChannelToolCall("t", "c", "motion.nod", {}))
    assert not result.ok


@pytest.mark.asyncio
async def test_background_acceptance_does_not_wait_and_close_cancels_before_worker_starts(monkeypatch):
    import json
    methods = []
    async def peer(room):
        return "device"
    async def rpc(**kwargs):
        body = json.loads(kwargs["payload"])
        methods.append(kwargs["method"])
        assert "wait_for_completion" not in body.get("arguments", {})
        state = "cancelled" if kwargs["method"] == module.CANCEL else "accepted"
        return ExecutionSnapshot(execution_id=body["execution_id"], state=state, sequence=1).model_dump_json()
    monkeypatch.setattr(module, "wait_for_runtime_participant_identity", peer)
    plan = SessionOutputPlan(session_id="s", policy_revision=1, outputs=OutputSelection(motion=True),
        motion_profile="eidolon.motion.v2", motion_capabilities=("motion.play",))
    runtime = module.SessionCapabilityRPC(SimpleNamespace(local_participant=SimpleNamespace(perform_rpc=rpc)), plan)
    result = await runtime.execute(ChannelToolCall("t", "c", "motion.play",
        {"clip_id": "emotion:laughing1", "wait_for_completion": False}))
    assert result.ok and result.outcome_state == "accepted"
    assert len(methods) == 1
    await runtime.close()
    assert module.CANCEL in methods
    assert not runtime.executions and not runtime.pending


CLIPS = ["emotion:cheerful1", "dance:simple_nod"]


class CatalogDevice:
    """A device that registers its handlers only once ``ready`` and, like the
    Reachy client, first reports the catalog as running, then completed."""

    def __init__(self, *, ready=True):
        self.ready = asyncio.Event()
        if ready:
            self.ready.set()
        self.methods = []

    async def perform_rpc(self, **kwargs):
        import json
        from livekit.rtc import RpcError
        self.methods.append(kwargs["method"])
        if not self.ready.is_set():
            raise RpcError(RpcError.ErrorCode.UNSUPPORTED_METHOD, "Method not supported at destination")
        body = json.loads(kwargs["payload"])
        if kwargs["method"] == module.STATUS:
            return ExecutionSnapshot(execution_id=body["execution_id"], state="completed", sequence=2,
                result={"clips": CLIPS}).model_dump_json()
        return ExecutionSnapshot(execution_id=body["execution_id"], state="running",
            sequence=1).model_dump_json()

    def runtime(self):
        plan = SessionOutputPlan(session_id="s", policy_revision=1, outputs=OutputSelection(motion=True),
            motion_profile="eidolon.motion.v2", motion_capabilities=("motion.catalog", "motion.play"))
        return module.SessionCapabilityRPC(SimpleNamespace(local_participant=self), plan)


@pytest.fixture
def device(monkeypatch):
    async def peer(room):
        return "device"
    monkeypatch.setattr(module, "wait_for_runtime_participant_identity", peer)
    monkeypatch.setattr(module, "RESOURCE_RETRY_S", 0.01)
    return CatalogDevice


async def settled(runtime):
    await asyncio.wait({runtime._resource_hints}, timeout=2)


@pytest.mark.asyncio
async def test_first_turn_reads_prefetched_catalog_without_waiting_on_the_device(device):
    loop = asyncio.get_running_loop()
    lazy = device().runtime()
    started = loop.time()
    assert await lazy.resource_hints() == {"motion.catalog": CLIPS}
    lazy_ms = 1000 * (loop.time() - started)

    warm = device()
    runtime = warm.runtime()
    runtime.prefetch_resource_hints()  # session warmup
    await settled(runtime)
    calls = len(warm.methods)
    started = loop.time()
    assert await runtime.resource_hints() == {"motion.catalog": CLIPS}
    first_turn_ms = 1000 * (loop.time() - started)
    assert len(warm.methods) == calls  # the turn itself sent nothing to the device
    # The status poll alone holds a lazy first turn for 150 ms.
    assert lazy_ms >= 150 and first_turn_ms < 20, (lazy_ms, first_turn_ms)
    await lazy.close()
    await runtime.close()


@pytest.mark.asyncio
async def test_discovery_keeps_asking_until_the_device_registers_its_handlers(device):
    waking = device(ready=False)
    runtime = waking.runtime()
    runtime.prefetch_resource_hints()
    await asyncio.sleep(0.05)
    assert waking.methods and not runtime._resource_hints.done()
    waking.ready.set()
    await settled(runtime)
    assert await runtime.resource_hints() == {"motion.catalog": CLIPS}
    await runtime.close()


@pytest.mark.asyncio
async def test_turn_shares_the_discovery_already_in_flight(device):
    catalog = device()
    runtime = catalog.runtime()
    runtime.prefetch_resource_hints()
    first, second = await asyncio.gather(runtime.resource_hints(), runtime.resource_hints())
    assert first == second == {"motion.catalog": CLIPS}
    assert catalog.methods.count(module.INVOKE_V2) == 1
    await runtime.close()


@pytest.mark.asyncio
async def test_device_that_never_answers_neither_blocks_a_turn_nor_poisons_the_next(device, monkeypatch):
    monkeypatch.setattr(module, "RESOURCE_DISCOVERY_S", 0.05)
    silent = device(ready=False)
    runtime = silent.runtime()
    assert await runtime.resource_hints() == {}
    silent.ready.set()
    assert await runtime.resource_hints() == {"motion.catalog": CLIPS}
    await runtime.close()


@pytest.mark.asyncio
async def test_close_stops_discovery_still_waiting_for_the_device(device):
    silent = device(ready=False)
    runtime = silent.runtime()
    runtime.prefetch_resource_hints()
    await asyncio.sleep(0.03)
    discovery = runtime._resource_hints
    await runtime.close()
    asked = len(silent.methods)
    await asyncio.sleep(0.05)
    assert discovery.cancelled() and len(silent.methods) == asked
    assert await runtime.resource_hints() == {}


@pytest.mark.asyncio
async def test_session_warmup_starts_device_discovery_without_waiting_for_it():
    from eidolon.livekit.agent.providers.llm import LivekitLlmStage
    from eidolon.livekit.agent.shared.pipeline import BasePipeline

    started = []
    stage = LivekitLlmStage(SimpleNamespace(prefetch_device_resources=lambda: started.append(True)))

    class Pipeline(BasePipeline):
        async def run(self, room):
            pass

    pipeline = Pipeline.__new__(Pipeline)
    pipeline._factory = SimpleNamespace(stt=None, tts=None, vad=None, llm=stage)
    assert pipeline._lifecycle_stages() == [stage]
    await pipeline._warmup_stages()
    assert started == [True]


class StillDevice:
    """A device that answers ``camera.capture`` and sends the frame as one byte stream."""

    def __init__(self, *, frame=b"\xff\xd8still\xff\xd9", claimed=None, sends=True, sender="device"):
        self.frame, self.claimed, self.sends, self.sender = frame, claimed or frame, sends, sender
        self.handlers = {}
        self.local_participant = SimpleNamespace(perform_rpc=self.perform_rpc)

    def register_byte_stream_handler(self, topic, handler):
        self.handlers[topic] = handler

    def unregister_byte_stream_handler(self, topic):
        self.handlers.pop(topic, None)

    async def perform_rpc(self, **kwargs):
        import hashlib
        import json

        body = json.loads(kwargs["payload"])
        if self.sends:
            self.handlers["eidolon.camera.frame.v1"](
                Stream(self.frame, {"execution_id": body["execution_id"],
                    "session_id": body["session_id"], "policy_revision": str(body["policy_revision"])}),
                self.sender)
            await asyncio.sleep(0)
        return ExecutionSnapshot(execution_id=body["execution_id"], state="completed", sequence=2,
            completion_basis="software_sequence", result={"media_type": "image/jpeg",
                "bytes": len(self.claimed), "sha256": hashlib.sha256(self.claimed).hexdigest(),
                "width": 768, "height": 432}).model_dump_json()


class Stream:
    def __init__(self, data, attributes):
        self.data, self.closed = data, False
        self.info = SimpleNamespace(attributes=attributes, size=len(data))

    async def __aiter__(self):
        yield self.data[:4]
        yield self.data[4:]

    async def aclose(self):
        self.closed = True


def still_plan(**changes):
    return SessionOutputPlan(**{"session_id": "s", "policy_revision": 3,
        "outputs": OutputSelection(), "device_capabilities": ("camera.capture",), **changes})


async def capture(monkeypatch, device, plan=None):
    async def peer(room):
        return "device"
    monkeypatch.setattr(module, "wait_for_runtime_participant_identity", peer)
    monkeypatch.setattr(module, "FRAME_WAIT_S", 0.05)
    runtime = module.SessionCapabilityRPC(device, plan or still_plan())
    result = await runtime.execute(ChannelToolCall("t", "c", "camera.capture", {"purpose": "look"}))
    assert not runtime.frames and not runtime.pending
    return runtime, result


@pytest.mark.asyncio
async def test_a_still_comes_back_attached_and_out_of_the_content(monkeypatch):
    device = StillDevice()
    runtime, result = await capture(monkeypatch, device)
    assert result.ok and result.attachment == device.frame
    assert result.attachment_media_type == "image/jpeg"
    assert "still" not in repr(result) and result.content["result"]["bytes"] == len(device.frame)
    await runtime.close()
    assert not device.handlers


@pytest.mark.asyncio
@pytest.mark.parametrize("device", [
    StillDevice(sends=False),
    StillDevice(claimed=b"another frame"),
    StillDevice(sender="someone-else"),
])
async def test_a_missing_altered_or_foreign_frame_fails_the_capture(monkeypatch, device):
    _, result = await capture(monkeypatch, device)
    assert not result.ok and result.error_code == "transfer_failed" and not result.attachment


@pytest.mark.asyncio
async def test_a_still_needs_the_plan_to_list_it_not_the_motion_output(monkeypatch):
    plan = SessionOutputPlan(session_id="s", policy_revision=3, outputs=OutputSelection(motion=True),
        motion_profile="eidolon.motion.v2", motion_capabilities=("motion.nod",))
    runtime = module.SessionCapabilityRPC(StillDevice(), plan)
    assert not (await runtime.execute(ChannelToolCall("t", "c", "camera.capture", {}))).ok
    assert module.SessionCapabilityRPC(StillDevice(), still_plan(
        outputs=OutputSelection(motion=True), motion_profile="eidolon.motion.v2",
        motion_capabilities=("motion.nod",))).names == ("motion.nod", "camera.capture")
