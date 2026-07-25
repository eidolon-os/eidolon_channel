"""Hermetic tests for the streaming ingestion path (no real Ditto service).

Covers the WS URL scheme mapping, the segment/end framing of
``DittoStreamSession``, and an end-to-end run of ``StreamingDHVideoGenerator``
against a fake client that replays a synthesized fragmented MP4.
"""

from __future__ import annotations

import asyncio
import io
from collections.abc import AsyncIterator

import aiohttp
import av
import numpy as np
from livekit import rtc
from livekit.agents.voice.avatar import AudioSegmentEnd

from eidolon.livekit.avatar.ditto_streaming_client import DittoStreamSession, _ws_url
from eidolon.livekit.avatar.streaming_video_generator import StreamingDHVideoGenerator

WIDTH, HEIGHT, FPS, N_FRAMES = 64, 48, 25, 10


def _make_fragmented_mp4() -> bytes:
    buf = io.BytesIO()
    container = av.open(
        buf, mode="w", format="mp4",
        options={"movflags": "frag_keyframe+empty_moov+default_base_moof"},
    )
    try:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        for i in range(N_FRAMES):
            arr = np.full((HEIGHT, WIDTH, 3), (i * 23) % 256, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    return buf.getvalue()


def test_ws_url_maps_scheme() -> None:
    assert _ws_url("https://host:61320/") == "wss://host:61320/ws/audio_stream"
    assert _ws_url("http://host:52320") == "ws://host:52320/ws/audio_stream"


class _Msg:
    def __init__(self, type_, data):
        self.type = type_
        self.data = data


class _FakeWS:
    def __init__(self, messages):
        self._messages = list(messages)
        self.sent_str: list[str] = []
        self.closed = False

    async def receive(self):
        if not self._messages:
            return _Msg(aiohttp.WSMsgType.CLOSED, None)
        return self._messages.pop(0)

    async def send_str(self, s):
        self.sent_str.append(s)

    async def close(self):
        self.closed = True


async def test_session_segments_yields_binary_until_end() -> None:
    ws = _FakeWS([
        _Msg(aiohttp.WSMsgType.BINARY, b"seg-1"),
        _Msg(aiohttp.WSMsgType.TEXT, '{"type":"ping","ts":1}'),
        _Msg(aiohttp.WSMsgType.BINARY, b"seg-2"),
        _Msg(aiohttp.WSMsgType.TEXT, '{"type":"end","total_segments":2}'),
        _Msg(aiohttp.WSMsgType.BINARY, b"after-end-ignored"),
    ])
    session = DittoStreamSession(ws, {})
    segs = [s async for s in session.segments()]
    assert segs == [b"seg-1", b"seg-2"]  # stops at end, ignores trailing
    assert any('"cmd": "pong"' in s or '"cmd":"pong"' in s for s in ws.sent_str)


class _FakeStreamSession:
    def __init__(self, fmp4: bytes):
        self._fmp4 = fmp4
        self.sent = 0
        self.audio_bytes = 0
        self.stopped = False
        self.closed = False
        self.input_complete = False

    async def send_audio(self, b: bytes) -> None:
        self.sent += 1
        self.audio_bytes += len(b)

    def mark_input_complete(self) -> None:
        self.input_complete = True

    async def request_stop(self) -> None:
        self.stopped = True

    async def segments(self) -> AsyncIterator[bytes]:
        for i in range(0, len(self._fmp4), 2048):
            yield self._fmp4[i : i + 2048]

    async def aclose(self) -> None:
        self.closed = True


class _FakeStreamClient:
    def __init__(self, fmp4: bytes):
        self._fmp4 = fmp4
        self.sessions: list[_FakeStreamSession] = []
        self.open_kwargs: list[dict] = []

    async def open(self, **kwargs) -> _FakeStreamSession:
        self.open_kwargs.append(kwargs)
        s = _FakeStreamSession(self._fmp4)
        self.sessions.append(s)
        return s

    async def aclose(self) -> None:
        pass


def _audio_frame(ms: int = 20, sr: int = 24000) -> rtc.AudioFrame:
    n = sr * ms // 1000
    data = np.zeros(n, dtype=np.int16).tobytes()
    return rtc.AudioFrame(data=data, sample_rate=sr, num_channels=1, samples_per_channel=n)


class _FakeAvSync:
    """Stands in for rtc.AVSynchronizer to observe the barge-in flush."""

    def __init__(self) -> None:
        self.cleared = 0

    async def clear_queue(self) -> None:
        self.cleared += 1


async def test_barge_in_flushes_generator_and_synchronizer() -> None:
    """A hard stop must clear the runner's synchronizer too — otherwise audio and
    video each drain their own residue and end at different moments (A/V desync)."""
    gen = StreamingDHVideoGenerator(
        _FakeStreamClient(_make_fragmented_mp4()),  # type: ignore[arg-type]
        width=WIDTH,
        height=HEIGHT,
        target_fps=FPS,
        output_sample_rate=24000,
    )
    av_sync = _FakeAvSync()
    gen.attach_av_sync(av_sync)  # type: ignore[arg-type]
    # Something already queued for publication when the barge-in lands.
    gen._out_queue.put_nowait(_audio_frame())

    await gen.clear_buffer()

    assert gen._out_queue.empty()  # our buffered output is dropped
    assert av_sync.cleared == 1  # and so is the synchronizer's
    await gen.aclose()


async def test_barge_in_without_av_sync_is_safe() -> None:
    """No synchronizer attached (e.g. before the runner starts) → still clean."""
    gen = StreamingDHVideoGenerator(
        _FakeStreamClient(b""),  # type: ignore[arg-type]
        width=WIDTH,
        height=HEIGHT,
        target_fps=FPS,
        output_sample_rate=24000,
    )
    gen._out_queue.put_nowait(_audio_frame())
    await gen.clear_buffer()
    assert gen._out_queue.empty()
    await gen.aclose()


async def test_streaming_generator_decodes_a_turn_end_to_end() -> None:
    client = _FakeStreamClient(_make_fragmented_mp4())
    gen = StreamingDHVideoGenerator(
        client,  # type: ignore[arg-type]
        width=WIDTH,
        height=HEIGHT,
        target_fps=FPS,
        output_sample_rate=24000,
        face_image=b"\xff\xd8jpeg",
    )
    await gen.warmup()  # no-op in streaming mode
    for _ in range(3):
        await gen.push_audio(_audio_frame())
    await gen.push_audio(AudioSegmentEnd())

    videos = 0
    saw_segment_end = False
    for _ in range(N_FRAMES + 50):  # bounded drain
        item = await asyncio.wait_for(gen._out_queue.get(), timeout=5.0)
        if isinstance(item, AudioSegmentEnd):
            saw_segment_end = True
            break
        if isinstance(item, rtc.VideoFrame):
            videos += 1
    assert saw_segment_end
    assert videos == N_FRAMES
    # the configured face was passed to the service as cond_image
    assert client.open_kwargs[0]["cond_image_b64"].startswith("data:image/jpeg;base64,")
    # End-of-audio must NOT send `stop`: that aborts generation and the service
    # returns end/0-segments, leaving nothing to decode and no audio published.
    assert client.sessions[0].stopped is False
    # Instead it feeds a silence tail and declares the input complete, so the
    # service renders the tail of the utterance instead of holding it.
    assert client.sessions[0].input_complete is True
    assert client.sessions[0].audio_bytes > 0
    await gen.aclose()


async def test_new_utterance_supersedes_a_draining_turn() -> None:
    """Audio arriving after a turn's input closed is a new utterance. The old
    turn's sender has stopped, so feeding it there would drop the audio."""
    client = _FakeStreamClient(_make_fragmented_mp4())
    gen = StreamingDHVideoGenerator(
        client,  # type: ignore[arg-type]
        width=WIDTH,
        height=HEIGHT,
        target_fps=FPS,
        output_sample_rate=24000,
    )
    await gen.push_audio(_audio_frame())
    await gen.push_audio(AudioSegmentEnd())
    await asyncio.sleep(0.05)
    await gen.push_audio(_audio_frame())  # next utterance
    await asyncio.sleep(0.05)
    assert len(client.sessions) == 2, "a second utterance must open its own session"
    await gen.aclose()


async def test_failed_turn_does_not_signal_playback_end() -> None:
    """A turn that produced no frames must not emit AudioSegmentEnd, or the
    runner reports playback finished for audio that was never captured."""

    class _DeadClient:
        async def open(self, **kwargs):
            raise RuntimeError("service unreachable")

        async def aclose(self):
            pass

    gen = StreamingDHVideoGenerator(
        _DeadClient(),  # type: ignore[arg-type]
        width=WIDTH,
        height=HEIGHT,
        target_fps=FPS,
        output_sample_rate=24000,
    )
    await gen.push_audio(_audio_frame())
    await gen.push_audio(AudioSegmentEnd())
    await asyncio.sleep(0.05)  # let the turn fail
    assert gen._out_queue.empty()
    await gen.aclose()


async def test_barge_in_stops_generation_upstream() -> None:
    """Barge-in *is* a cancel — it must tell the service to stop generating."""
    client = _FakeStreamClient(_make_fragmented_mp4())
    gen = StreamingDHVideoGenerator(
        client,  # type: ignore[arg-type]
        width=WIDTH,
        height=HEIGHT,
        target_fps=FPS,
        output_sample_rate=24000,
    )
    await gen.push_audio(_audio_frame())
    await asyncio.sleep(0.05)  # let the session open
    await gen.clear_buffer()
    assert client.sessions[0].stopped is True
    await gen.aclose()
