import asyncio
from types import SimpleNamespace

import pytest
from livekit import rtc
from eidolon_sdk.biz.control.audio_presentation import AudioPresentationResult
from eidolon_sdk.biz.control.coordination_stream import ReplyStart
from eidolon.livekit.agent.coordination.presentation import GroupSpeechPresenter, SpeechEndpoint


def reply():
    return ReplyStart(
        type="reply_start",
        stream_id="agent",
        session_id="scene",
        request_id="turn",
        turn_id="turn",
        device_id="device",
        companion_id="a",
        epoch=1,
    )


async def text():
    yield "hello"


class Stream:
    def __init__(self, rate=16000):
        self.ended = asyncio.Event()
        self.words = []
        self.closed = False
        self.rate = rate

    def push_text(self, text):
        self.words.append(text)

    def end_input(self):
        self.ended.set()

    async def __aiter__(self):
        await self.ended.wait()
        yield SimpleNamespace(
            frame=rtc.AudioFrame(bytes(self.rate // 50 * 2), self.rate, 1, self.rate // 50)
        )

    async def aclose(self):
        self.closed = True


class Device:
    def __init__(self, rate=16000):
        self.stream = Stream(rate)
        self.receipt = asyncio.get_running_loop().create_future()
        self.sealed = asyncio.Event()
        self.data = bytearray()
        self.request = None
        self.reason = None
        self.reject = False

    async def prepare(self, request, speaking):
        if self.reject:
            raise RuntimeError("unsupported firmware")
        self.request = request
        self.speaking = speaking
        return self.receipt

    async def stream_bytes(self, name, **options):
        assert self.request is not None  # accepted before any media
        assert options["stream_id"] == self.request.stream_id
        assert options["destination_identities"] == ["physical-device"]
        return self

    async def write(self, data):
        assert len(data) <= 640
        self.data.extend(data)

    async def aclose(self, *, reason=""):
        self.reason = reason
        self.sealed.set()

    def confirm(self, **changes):
        result = dict(
            session_id="scene",
            turn_id="turn",
            stream_id=self.request.stream_id,
            epoch=1,
            rendered_bytes=len(self.data),
            drained=True,
        )
        result.update(changes)
        self.receipt.set_result(AudioPresentationResult(**result))

    def presenter(self):
        endpoint = SpeechEndpoint(
            self, "physical-device", SimpleNamespace(stream=lambda: self.stream), self.prepare
        )
        return GroupSpeechPresenter({("a", "device"): endpoint})


@pytest.mark.parametrize("rate", [16000, 24000])
async def test_streams_existing_tts_and_waits_for_physical_receipt(rate):
    device = Device(rate)
    speaking = []
    task = asyncio.create_task(device.presenter()(reply(), text(), lambda: speaking.append(True)))
    await asyncio.wait_for(device.sealed.wait(), 2)
    assert not task.done() and not speaking  # no fake playback state from producer EOF
    assert device.stream.words == ["hello"] and device.reason == ""
    device.speaking()
    device.confirm()
    assert await task is True
    assert device.stream.closed and speaking == [True]


@pytest.mark.parametrize(
    "changes", [dict(epoch=2), dict(turn_id="old"), dict(rendered_bytes=2), dict(stream_id="old")]
)
async def test_mismatched_physical_completion_cannot_advance_queue(changes):
    device = Device()
    task = asyncio.create_task(device.presenter()(reply(), text(), lambda: None))
    await asyncio.wait_for(device.sealed.wait(), 2)
    device.confirm(**changes)
    assert await task is False


async def test_old_firmware_rejection_sends_no_audio():
    device = Device()
    device.reject = True
    with pytest.raises(RuntimeError, match="unsupported"):
        await device.presenter()(reply(), text(), lambda: None)
    assert not device.data and device.request is None


async def test_cancel_closes_tts_and_never_sends_successful_trailer():
    device = Device()
    entered = asyncio.Event()

    async def unfinished_text():
        entered.set()
        await asyncio.Event().wait()
        yield "unreachable"

    task = asyncio.create_task(device.presenter()(reply(), unfinished_text(), lambda: None))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert device.stream.closed
    assert device.reason == "presentation_aborted"
    assert device.receipt.cancelled()


async def test_failed_text_source_cancels_waiting_synthesis():
    device = Device()

    async def failed_text():
        raise ValueError("failed model stream")
        yield "unreachable"

    with pytest.raises(ValueError, match="failed model"):
        await asyncio.wait_for(device.presenter()(reply(), failed_text(), lambda: None), 1)
    assert device.stream.closed and device.reason == "presentation_aborted"


async def test_early_device_completion_aborts_unfinished_generation():
    device = Device()
    entered = asyncio.Event()

    async def unfinished_text():
        entered.set()
        await asyncio.Event().wait()
        yield "unreachable"

    task = asyncio.create_task(device.presenter()(reply(), unfinished_text(), lambda: None))
    await entered.wait()
    device.confirm()
    with pytest.raises(ValueError, match="before stream end"):
        await asyncio.wait_for(task, 1)
    assert device.reason == "presentation_aborted" and device.stream.closed
