import asyncio

import pytest
from livekit.agents.voice import AgentSession, Agent
from livekit.agents.voice.io import TextOutput
from livekit.agents.voice.transcription import TranscriptSynchronizer
from eidolon_sdk.biz.presentation import OutputSelection

from eidolon.livekit.agent.session.welcome import play_welcome
from eidolon.livekit.common.welcome import WelcomeAudio
from .._harness.headless import RecordingAudioOutput


class DeferredPlayback(RecordingAudioOutput):
    def on_playback_finished(self, **kwargs):
        # RTC playback completion is delivered asynchronously after sink flush.
        asyncio.get_running_loop().call_soon(
            lambda: super(DeferredPlayback, self).on_playback_finished(**kwargs))


class TextSink(TextOutput):
    def __init__(self):
        super().__init__(label="test", next_in_chain=None)
        self.parts = []
    async def capture_text(self, text):
        self.parts.append(text)
    def flush(self):
        pass


@pytest.mark.asyncio
async def test_audio_cue_finishes_empty_segment_then_normal_text_still_plays(caplog):
    audio, text = DeferredPlayback(), TextSink()
    sync = TranscriptSynchronizer(next_in_chain_audio=audio, next_in_chain_text=text)
    session = AgentSession(turn_handling={"turn_detection": "manual", "interruption": {"enabled": False}})
    session.output.audio = sync.audio_output
    session.output.transcription = sync.text_output
    pcm = bytes(16000 * 2 // 5)
    try:
        await session.start(Agent(instructions=""))
        cue = play_welcome(session, WelcomeAudio(), outputs=OutputSelection(audio_cue=True),
                           pcm=pcm, sample_rate=16000)
        await asyncio.wait_for(cue.wait_for_playout(), 2)
        assert audio.collected_pcm == pcm
        assert not ''.join(text.parts)
        assert not any(item.type == "message" for item in session.history.items)
        # A second real segment proves the empty cue has not poisoned rotation.
        from eidolon.livekit.agent.session.welcome import _audio_frames
        speech = session.say("正常字幕", audio=_audio_frames(pcm, 16000))
        await asyncio.wait_for(speech.wait_for_playout(), 2)
        assert '正常字幕' in ''.join(text.parts)
        assert not any("before text/audio input is done" in r.message for r in caplog.records)
    finally:
        await session.aclose()
        await sync.aclose()
