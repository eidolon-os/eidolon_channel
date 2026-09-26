import asyncio
import json
from types import SimpleNamespace
import pytest
from eidolon_sdk.biz.contracts import CONTROL_TOPIC
from eidolon_sdk.biz.control.audio_presentation import AudioPresentation
from eidolon.livekit.agent.session.audio_control import DevicePlaybackControl


class Room:
    def __init__(self):
        self.handlers = {}
        self.sent = asyncio.Queue()
        self.local_participant = SimpleNamespace(publish_data=self.publish)

    def on(self, event, callback):
        self.handlers[event] = callback

    def off(self, event, callback):
        assert self.handlers.pop(event) == callback

    async def publish(self, data, **options):
        assert options["destination_identities"] == ["device"]
        self.sent.put_nowait(json.loads(data))

    def ack(self, command, status, *, peer="device", **changes):
        body = dict(
            v=1,
            kind="ack",
            device_id="device",
            ref=command["id"],
            op=command["op"],
            status=status,
            **changes,
        )
        self.handlers["data_received"](
            SimpleNamespace(
                topic=CONTROL_TOPIC,
                participant=SimpleNamespace(identity=peer),
                data=json.dumps(body).encode(),
            )
        )


def setup():
    room = Room()
    control = DevicePlaybackControl(
        room, device_id="device", conversation_id="physical-session", policy_revision=3
    )
    request = AudioPresentation(session_id="scene", turn_id="turn", stream_id="stream", epoch=1)
    return room, control, request


async def test_prepare_acceptance_and_physical_completion_use_one_exchange():
    room, control, request = setup()
    phases = []
    prepared = asyncio.create_task(control.prepare(request, lambda: phases.append("speaking")))
    command = await room.sent.get()
    assert command["payload"]["session_id"] == "physical-session"
    assert command["payload"]["presentation"]["session_id"] == "scene"
    room.ack(command, "accepted", peer="outsider")
    await asyncio.sleep(0)
    assert not prepared.done()
    room.ack(command, "accepted")
    terminal = await prepared
    assert not terminal.done() and len(control.pending) == 1
    room.ack(command, "running")
    room.ack(command, "running")
    assert phases == ["speaking"]
    room.ack(
        command,
        "completed",
        result=dict(
            session_id="scene",
            turn_id="turn",
            stream_id="stream",
            epoch=1,
            rendered_bytes=640,
            drained=True,
        ),
    )
    assert (await terminal).rendered_bytes == 640
    await asyncio.sleep(0)
    assert not control.pending
    control.close()
    assert not room.handlers


async def test_stop_does_not_wait_for_pending_presentation():
    room, control, request = setup()
    prepared = asyncio.create_task(control.prepare(request, lambda: None))
    presentation = await room.sent.get()
    room.ack(presentation, "accepted")
    terminal = await prepared
    stopping = asyncio.create_task(control.stop())
    stop = await room.sent.get()
    assert stop["op"] == "playback.stop"
    room.ack(stop, "completed")
    assert await stopping is True
    assert not terminal.done()
    room.ack(presentation, "failed")
    with pytest.raises(RuntimeError):
        await terminal
    control.close()


@pytest.mark.parametrize("status", ["unsupported", "rejected", "completed"])
async def test_rejection_or_completion_before_acceptance_cannot_start_audio(status):
    room, control, request = setup()
    task = asyncio.create_task(control.prepare(request, lambda: None))
    command = await room.sent.get()
    room.ack(command, status)
    with pytest.raises(RuntimeError):
        await task
    await asyncio.sleep(0)
    assert not control.pending
    control.close()


async def test_disconnect_fails_pending_completion_and_removes_listener():
    room, control, request = setup()
    prepared = asyncio.create_task(control.prepare(request, lambda: None))
    command = await room.sent.get()
    room.ack(command, "accepted")
    terminal = await prepared
    room.handlers["participant_disconnected"](SimpleNamespace(identity="device"))
    with pytest.raises(ConnectionError):
        await terminal
    await asyncio.sleep(0)
    assert not control.pending
    control.close()


async def test_cancel_after_acceptance_does_not_leak_pending_exchange():
    room, control, request = setup()
    prepared = asyncio.create_task(control.prepare(request, lambda: None))
    command = await room.sent.get()
    room.ack(command, "accepted")
    terminal = await prepared
    terminal.cancel()
    await asyncio.gather(terminal, return_exceptions=True)
    await asyncio.sleep(0)
    assert not control.pending
    control.close()


async def test_control_and_tts_adapter_wait_for_device_after_normal_trailer():
    from livekit import rtc
    from eidolon_sdk.biz.control.coordination_stream import ReplyStart
    from eidolon.livekit.agent.coordination.presentation import GroupSpeechPresenter, SpeechEndpoint

    room, control, request = setup()
    sealed = asyncio.Event()
    input_ended = asyncio.Event()
    media = bytearray()

    class Writer:
        async def write(self, data):
            media.extend(data)

        async def aclose(self, *, reason=""):
            assert reason == ""
            sealed.set()

    async def stream_bytes(name, **options):
        assert options["destination_identities"] == ["device"]
        return Writer()

    room.local_participant.stream_bytes = stream_bytes

    class Tts:
        def push_text(self, value):
            assert value == "hello"

        def end_input(self):
            input_ended.set()

        async def __aiter__(self):
            await input_ended.wait()
            yield SimpleNamespace(frame=rtc.AudioFrame(bytes(640), 16000, 1, 320))

        async def aclose(self):
            pass

    async def words():
        yield "hello"

    presenter = GroupSpeechPresenter(
        {
            ("a", "device"): SpeechEndpoint(
                room.local_participant, "device", SimpleNamespace(stream=Tts), control.prepare
            )
        }
    )
    start = ReplyStart(
        type="reply_start",
        session_id="scene",
        stream_id="agent",
        request_id="turn",
        turn_id="turn",
        epoch=1,
        companion_id="a",
        device_id="device",
    )
    task = asyncio.create_task(presenter(start, words(), lambda: None))
    command = await room.sent.get()
    assert not media
    room.ack(command, "accepted")
    await asyncio.wait_for(sealed.wait(), 1)
    assert not task.done() and len(media) == 640
    descriptor = command["payload"]["presentation"]
    room.ack(
        command,
        "completed",
        result={key: descriptor[key] for key in ("session_id", "turn_id", "stream_id", "epoch")}
        | {"rendered_bytes": 640, "drained": True},
    )
    assert await task is True
    control.close()


def test_input_pipelines_do_not_import_conversation_scenarios():
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "agent"
    for directory in ("half_duplex", "full_duplex", "shared"):
        for path in (root / directory).glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    assert "coordination" not in (node.module or "").split("."), path
                elif isinstance(node, ast.Import):
                    assert all("coordination" not in item.name.split(".") for item in node.names), (
                        path
                    )
