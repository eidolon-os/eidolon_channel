"""Button packets -> production recorder/transcriber -> real SDK -> PCM sink."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from eidolon_sdk.biz.contracts import CLIENT_AUDIO_STATE_TOPIC, WIRE_SCHEMA_VERSION

from .._harness.audio import synth_voiced, synth_silence, frames_from_pcm
from .._harness.human_profiles import HUMAN_PROFILES
from .._harness.mocks import MockLLM, MockTTS
from .._harness.production import production_ptt_session


def button(pipeline, held, seq):
    packet = SimpleNamespace(
        topic=CLIENT_AUDIO_STATE_TOPIC,
        participant=SimpleNamespace(identity='human-simulator'),
        data=json.dumps({
            'schema_v': WIRE_SCHEMA_VERSION, 'type': 'client.audio_state',
            'seq': seq, 'input_mode': 'ptt', 'playback_state': 'idle',
            'mic_muted': not held, 'ptt': held, 'rms': .1 if held else 0,
            'client_ts_ms': seq * 100,
        }).encode(),
    )
    pipeline._room_data.handle_packet(packet)
    pipeline._on_room_packet(packet)


@pytest.mark.asyncio
@pytest.mark.parametrize('profile', HUMAN_PROFILES, ids=lambda p: p.name)
async def test_release_is_only_commit_boundary(profile, caplog):
    classifier = SimpleNamespace(classify=AsyncMock(side_effect=AssertionError('PTT uses button ownership')))
    async with production_ptt_session(
        text=profile.text, llm=MockLLM.scripted([(profile.text, '已经收到。')]),
        tts=MockTTS(char_seconds=.01),
        interrupt_classifier=classifier,
    ) as (pipeline, h, stt):
        button(pipeline, True, 1)
        button(pipeline, True, 2)  # heartbeat while held must not open another segment
        pcm = synth_voiced(profile.duration_ms / 1000)
        if profile.pauses:
            pcm = synth_voiced(.35) + synth_silence(.7) + synth_voiced(.65)
        frames = list(frames_from_pcm(pcm))
        async def audio():
            for frame in frames:
                yield SimpleNamespace(frame=frame)
        await pipeline._consume_audio_stream(audio(), None)
        await asyncio.sleep(.02)
        assert stt.audio == []
        assert h.events.user_messages() == []
        button(pipeline, False, 3)
        button(pipeline, False, 4)  # duplicated release must not submit twice
        await h.events.wait_for_agent_messages(1, timeout=5)
        assert stt.audio == [b"".join(bytes(f.data) for f in frames)]
        assert h.events.user_messages() == [profile.text]
        assert h.events.agent_messages() == ['已经收到。']
        assert h.audio_out.collected_pcm
        assert not [r for r in caplog.records if r.levelname == 'ERROR']
        classifier.classify.assert_not_called()


@pytest.mark.asyncio
async def test_press_during_output_preempts_and_short_tap_does_not_submit_again():
    async with production_ptt_session(
        text='给我讲一个故事',
        llm=MockLLM.scripted([('给我讲一个故事', '从前有一座山，山里住着几个朋友，他们每天都一起出门探索。')]),
        tts=MockTTS(char_seconds=.05, chunk_delay_ms=20),
    ) as (pipeline, h, stt):
        button(pipeline, True, 1)
        for frame in frames_from_pcm(synth_voiced(.5)):
            pipeline._ptt_controller.push_frame(frame)
        button(pipeline, False, 2)
        await h.audio_out.wait_for_first_audio()
        speech = h.session.current_speech
        assert speech is not None
        button(pipeline, True, 3)
        button(pipeline, True, 4)
        for frame in frames_from_pcm(synth_voiced(.12)):
            pipeline._ptt_controller.push_frame(frame)
        button(pipeline, False, 5)
        async with asyncio.timeout(3):
            while pipeline._ptt_controller.state != 'idle':
                await asyncio.sleep(.01)
        assert speech.interrupted
        assert len(stt.audio) == 1
        assert h.events.user_messages() == ['给我讲一个故事']


@pytest.mark.asyncio
async def test_ptt_reply_plays_before_stt_cleanup_and_next_turn_stays_isolated():
    """Real recorder/transcriber/SDK must reply while the old transport closes."""
    from eidolon.livekit.agent.providers.stt import SttStage
    from .._harness.mocks import MockSTT, ScriptedTranscript

    plugin = MockSTT.scripted([ScriptedTranscript(text='讲个故事', trigger_after_ms=200)])
    original_stream = plugin.stream
    close_started, allow_close = asyncio.Event(), asyncio.Event()
    streams = []

    def stream(**kwargs):
        current = original_stream(**kwargs)
        streams.append(current)
        if len(streams) == 1:
            original_close = current.aclose
            async def close():
                close_started.set()
                await allow_close.wait()
                await original_close()
            current.aclose = close
        return current

    plugin.stream = stream
    stage = SttStage(plugin)
    async with production_ptt_session(
        text='', stt_stage=stage,
        llm=MockLLM.scripted([('讲个故事', '收到。')]), tts=MockTTS(char_seconds=.01),
    ) as (pipeline, h, _):
        try:
            button(pipeline, True, 1)
            for frame in frames_from_pcm(synth_voiced(.5)):
                pipeline._ptt_controller.push_frame(frame)
            assert not h.events.user_messages()
            button(pipeline, False, 2)
            await asyncio.wait_for(close_started.wait(), 2)
            await h.events.wait_for_agent_messages(1, timeout=2)
            assert h.audio_out.first_audible_at is not None
            assert not allow_close.is_set()
            assert h.events.user_messages() == ['讲个故事']
            button(pipeline, True, 3)
            for frame in frames_from_pcm(synth_voiced(1.1)):
                pipeline._ptt_controller.push_frame(frame)
            button(pipeline, False, 4)
            await asyncio.sleep(.03)
            assert len(streams) == 1, 'previous shared connection must be drained first'
            assert h.events.user_messages() == ['讲个故事']
            allow_close.set()
            await h.events.wait_for_agent_messages(2, timeout=3)
            assert h.events.user_messages() == ['讲个故事', '讲个故事']
            assert len(streams) == 2
        finally:
            allow_close.set()
