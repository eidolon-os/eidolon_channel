"""Exercise the installed LiveKit RoomIO queue, not an immediately-writing fake."""
import asyncio

import numpy as np
import pytest
from livekit import rtc
from livekit.agents.voice.room_io._output import _ParticipantAudioOutput

from eidolon.livekit.agent.output.controller import OutputController


class Source:
    def __init__(self, *args, **kwargs):
        self.frames = []
        self.queued_duration = 0.0

    async def capture_frame(self, frame):
        self.frames.append(frame)
        await asyncio.sleep(0)

    def clear_queue(self):
        pass

    async def wait_for_playout(self):
        await asyncio.sleep(0)

    async def aclose(self):
        pass


async def settle():
    for _ in range(30):
        await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_already_synthesized_roomio_tail_obeys_duck(monkeypatch, cancel):
    monkeypatch.setattr(rtc, "AudioSource", Source)
    sink = _ParticipantAudioOutput(None, sample_rate=16000, num_channels=1,
                                   track_publish_options=rtc.TrackPublishOptions())
    sink._subscribed_fut.set_result(None)
    output = OutputController(sink)
    frame = rtc.AudioFrame(np.full(16000, 1234, dtype=np.int16).tobytes(),
                           sample_rate=16000, num_channels=1, samples_per_channel=16000)
    await output.capture_frame(frame)
    output.flush()
    assert output.buffered_frames == 0  # the whole second belongs to RoomIO now
    output.duck()
    sink._forwarding_task = asyncio.create_task(sink._forward_audio())
    try:
        await settle()
        assert sink._audio_source.frames == [], "native queue must pause without new TTS"
        if cancel:
            output.cancel()
            await settle()
            output.reset()
            await settle()
            assert sink._audio_source.frames == [], "cancelled queued speech must not resume"
        else:
            # Repeated VAD and opt-in passthrough cannot release a native backlog.
            output.duck()
            assert not output.enable_suspended_passthrough(volume=0.25)
            assert sink._audio_source.frames == []
            output.unduck()
            await asyncio.wait_for(sink._flush_task, 1)
            played = np.concatenate([np.frombuffer(f.data, dtype=np.int16)
                                     for f in sink._audio_source.frames])
            assert len(played) == 16000
            assert (played == 1234).all(), "false interruption must retain queued content"
    finally:
        await sink.aclose()
