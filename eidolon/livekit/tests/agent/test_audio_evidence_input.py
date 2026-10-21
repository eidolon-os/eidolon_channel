"""Use public SDK input lifecycle; no room subscription or microphone."""

import asyncio

import pytest
from livekit import rtc
from livekit.agents.voice.io import AgentInput, AudioInput

from eidolon.livekit.agent.session.audio_evidence_input import EvidenceAudioInput
from eidolon.livekit.common.audio_evidence import AudioEvidenceCapture
from eidolon.livekit.common.audio_timeline import AudioSource, AudioSpan


class Input(AudioInput):
    def __init__(self):
        super().__init__(label="selected-synthetic-input")
        self.frames = asyncio.Queue()
        self.attaches = self.detaches = 0

    async def __anext__(self):
        return await self.frames.get()

    def on_attached(self):
        self.attaches += 1

    def on_detached(self):
        self.detaches += 1


def setup():
    capture = AudioEvidenceCapture(session_id="session", endpoint_id="endpoint")
    raw = Input()
    guarded = EvidenceAudioInput(source=raw, capture=capture)
    owner = AgentInput(video_changed=lambda: None, audio_changed=lambda: None)
    owner.audio = guarded
    capture.select_stream("asr")
    source = AudioSource("asr", 1, 16000)
    capture.begin_run(source)
    capture.append(source, start=0, pcm=b"\x01\x00" * 4)
    return capture, raw, guarded, owner


def test_sdk_pause_resume_cannot_revalidate_queued_audio():
    capture, raw, guarded, owner = setup()
    owner.set_audio_enabled(False)
    assert capture.active_source is None and capture.retained_samples == 0
    owner.set_audio_enabled(True)
    capture.begin_run(AudioSource("asr", 2, 16000))
    assert capture.active_source is None
    assert (raw.attaches, raw.detaches) == (2, 1)
    assert owner.audio is guarded and owner.audio_enabled


def test_sdk_input_replacement_immediately_releases_old_evidence():
    capture, raw, guarded, owner = setup()
    owner.audio = Input()
    assert capture.active_source is None and capture.retained_samples == 0
    assert raw.detaches == 1


@pytest.mark.asyncio
async def test_inflight_old_frame_is_forwarded_without_becoming_new_source_evidence():
    capture, raw, guarded, owner = setup()
    reading = asyncio.create_task(anext(guarded))
    await asyncio.sleep(0)
    owner.set_audio_enabled(False)
    owner.set_audio_enabled(True)
    frame = rtc.AudioFrame(b"\x05\x00" * 160, 16000, 1, 160)
    raw.frames.put_nowait(frame)
    assert await reading is frame
    assert capture.active_source is None
    assert not capture.snapshot(candidate="late", span=AudioSpan("asr", 1, 16000, 0, 4)).complete


@pytest.mark.asyncio
async def test_cancelled_input_read_releases_evidence():
    capture, raw, guarded, owner = setup()
    reading = asyncio.create_task(anext(guarded))
    await asyncio.sleep(0)
    reading.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reading
    assert capture.active_source is None and capture.retained_samples == 0
