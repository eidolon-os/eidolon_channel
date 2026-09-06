"""Opt-in real-service audio replay. Never falls back to scripted providers.

Uses configured Bailian speech providers and the configured direct LLM endpoint;
this is a direct_llm variant, not the room-bound eidolon_agent RPC deployment.
Requires real env credentials, hydrated EOT/pVAD models and ffmpeg on PATH.
"""
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
import time
import json

import pytest

from eidolon.livekit.agent.factory import SharedStageFactory
from eidolon.livekit.common.config import load_effective_config
from .._harness.production import production_session, production_ptt_session
from .._harness.audio import frames_from_pcm
from .._harness.latency import ReplyLatencyProbe
from .test_human_ptt import button

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.mark.parametrize('mode,instructions', [
    pytest.param('full_duplex', '请用中文简短回复，每次不超过两句话。', id='full_duplex'),
    pytest.param('half_duplex', '请用中文简短回复，每次不超过两句话。', id='half_duplex'),
    pytest.param('ptt', '请用中文简短回复，每次不超过两句话。', id='ptt'),
    pytest.param('ptt', '', id='ptt_without_brief_instruction'),
])
@pytest.mark.parametrize('speed', [.7, 1.0, 1.4], ids=['slow', 'normal', 'fast'])
async def test_real_speech_to_real_reply(speed, mode, instructions, record_property):
    ffmpeg = shutil.which('ffmpeg')
    assert ffmpeg is not None, 'ffmpeg is required for pitch-preserving speed variation'
    clip = Path(__file__).resolve().parents[4] / 'benchmark/audio/generated/normal_ask_intro.wav'
    pcm = subprocess.run([
        ffmpeg, '-v', 'error', '-i', str(clip), '-filter:a', f'atempo={speed}',
        '-ar', '16000', '-ac', '1', '-f', 's16le', 'pipe:1',
    ], check=True, capture_output=True).stdout
    assert len(pcm) > 8000
    cfg = load_effective_config()
    cfg = replace(cfg, llm=replace(cfg.llm, max_completion_tokens=96, timeout=20))
    stages = SharedStageFactory.components_from_config(cfg)
    assert stages.vad is not None, 'real VAD failed to load; do not silently degrade'
    llm = SharedStageFactory._build_llm(cfg)
    assert llm is not None
    try:
        await stages.stt.warmup()
        await stages.tts.warmup()
        @asynccontextmanager
        async def conversation():
            if mode == 'ptt':
                async with production_ptt_session(
                    text='', llm=llm, tts=stages.tts.tts, stt_stage=stages.stt, instructions=instructions,
                ) as (pipeline, handle, _):
                    yield pipeline, handle
            else:
                async with production_session(
                    mode=mode, llm=llm, stt=stages.stt.stt, tts=stages.tts.tts,
                    vad=stages.vad.vad, instructions=instructions,
                    real_time_audio=True,
                ) as pair:
                    yield pair

        async with conversation() as (pipeline, h):
            probe = ReplyLatencyProbe(pipeline, h) if mode != 'ptt' else None
            started = time.monotonic()
            if mode == 'ptt':
                button(pipeline, True, 1)
                ptt_timeline = pipeline._timeline
                for frame in frames_from_pcm(pcm):
                    pipeline._ptt_controller.push_frame(frame)
                assert h.events.user_messages() == []
                released_at = time.monotonic()
                button(pipeline, False, 2)
            else:
                h.audio_in.feed_pcm(pcm)
                h.audio_in.feed_silence(2)
            try:
                await h.events.wait_for_agent_messages(1, timeout=45)
            finally:
                if mode == 'ptt':
                    # Keep first-audio evidence even if whole-reply completion times out.
                    first_audible = h.audio_out.first_audible_at
                    record_property('release_to_reply_audio_ms',
                        round((first_audible - released_at) * 1000, 1) if first_audible is not None else None)
                    record_property('committed_user_turns', len(h.events.user_messages()))
                    record_property('captured_audio_bytes', h.audio_out.captured_bytes)
                    record_property('ptt_timing', json.dumps({
                        'from_release_ms': {name: round((at - released_at) * 1000, 1)
                            for name, at in ptt_timeline.timestamps.items()},
                        'llm_metrics': ptt_timeline.attrs.get('llm_metrics'),
                        'session_states': h.events.agent_state_history(),
                    }, ensure_ascii=False))
            assert len(h.events.user_messages()) == 1
            assert '方案' in h.events.user_messages()[0]
            assert h.audio_out.captured_bytes > 8000
            record_property('speed', speed)
            record_property('mode', mode)
            record_property('input_seconds', len(pcm) / 32000)
            record_property('stt_text', h.events.user_messages()[0])
            record_property('reply_chars', len(h.events.agent_messages()[0]))
            record_property('output_bytes', h.audio_out.captured_bytes)
            record_property('elapsed_seconds', time.monotonic() - started)
            record_property('providers', f'{cfg.providers.stt_provider}/direct_llm/{cfg.providers.tts_provider}')
            record_property('turn_boundary', 'button_release' if mode == 'ptt' else 'firered_pvad/real_eot')
            if mode == 'ptt':
                assert first_audible is not None and first_audible >= released_at
                metrics = ptt_timeline.attrs.get('llm_metrics')
                assert metrics is not None, 'require actual usage to validate the configured reply budget'
                assert 0 < metrics['completion_tokens'] <= cfg.llm.max_completion_tokens
            if probe is not None:
                record_property('reply_latency', json.dumps(probe.report(pcm), ensure_ascii=False))
                probe.close()
    finally:
        await stages.stt.shutdown()
        await stages.tts.shutdown()
        await llm.aclose()


@pytest.mark.parametrize('clip_name,action', [
    ('backchannel_ok', 'resume'), ('correction', 'reply'), ('hard_stop_stop', 'stop'),
])
async def test_real_audio_overlap_to_playback_effects(clip_name, action, record_property):
    """Synthetic benchmark WAV -> real VAD/STT/intent/LLM/TTS -> memory sink.

    Provenance: scripts/generate_voice_benchmark_audio.py DEFAULT_CLIPS and
    benchmark/audio/generated/manifest.yaml. These are generic TTS fixtures,
    not recordings of real users. This does not test device AEC or RTC.
    """
    import asyncio

    ffmpeg = shutil.which('ffmpeg')
    assert ffmpeg is not None
    clip = Path(__file__).resolve().parents[4] / f'benchmark/audio/generated/{clip_name}.wav'
    pcm = subprocess.run([
        ffmpeg, '-v', 'error', '-i', str(clip),
        '-ar', '16000', '-ac', '1', '-f', 's16le', 'pipe:1',
    ], check=True, capture_output=True).stdout
    cfg = load_effective_config()
    # This experiment opts in without enabling the policy in shared settings.
    cfg = replace(cfg,
        turn_policy=replace(cfg.turn_policy, interrupt=replace(
            cfg.turn_policy.interrupt, intent_provider='llm',
        )),
        llm=replace(cfg.llm, max_completion_tokens=96, timeout=20),
    )
    stages = SharedStageFactory.components_from_config(cfg)
    assert stages.vad is not None
    llm = SharedStageFactory._build_llm(cfg)
    classifier = SharedStageFactory.build_interrupt_classifier(cfg)
    welcome = '我们继续讨论这个方案。我先把背景说明清楚，然后再介绍各个部分的分工。接下来会逐一解释输入、处理和输出之间的关系，最后再一起看看需要注意的细节。'
    try:
        startup = time.monotonic()
        # The lightweight harness wraps raw plugins; retain real STT/TTS stage
        # startup as well as the production classifier lifecycle warmup below.
        await stages.stt.warmup()
        await stages.tts.warmup()
        async with production_session(
            welcome=welcome, llm=llm, stt=stages.stt.stt, tts=stages.tts.tts,
            vad=stages.vad.vad, turn_policy=cfg.turn_policy, interrupt_classifier=classifier,
            instructions='请用中文简短回复，每次不超过两句话。', real_time_audio=True,
            warmup=True,
        ) as (pipeline, h):
            record_property('session_ready_ms', round((time.monotonic() - startup) * 1000, 1))
            await h.audio_out.wait_for_first_audio(timeout=15)
            speech = h.session.current_speech
            assert speech is not None
            probe = ReplyLatencyProbe(pipeline, h) if action == 'reply' else None
            h.audio_in.feed_pcm(pcm)
            h.audio_in.feed_silence(3)
            async with asyncio.timeout(12):
                while pipeline._semantic_interrupts._intent_result is None:
                    await asyncio.sleep(.02)
            evidence = pipeline._semantic_interrupts._intent_result
            record_property('clip', clip_name)
            record_property('intent', evidence.intent.value)
            record_property('source', evidence.source)
            record_property('stt_text', pipeline._semantic_interrupts._intent_key[1])
            assert evidence.source == 'llm'
            await asyncio.sleep(1)
            assert speech.interrupted is (action != 'resume')
            if action == 'resume':
                assert pipeline._ducking.mixer.state == 'NORMAL'
                await h.events.wait_for_agent_messages(1, timeout=35)
                assert h.events.agent_messages() == [welcome]
                assert not h.events.user_messages()
                assert not any(s.cleared for s in h.audio_out.segments)
            elif action == 'reply':
                user_event = await h.events.wait_for(lambda e: e.type == 'conversation_item_added'
                    and getattr(e.payload.item, 'role', None) == 'user', timeout=15)
                reply_event = await h.events.wait_for(lambda e: e.type == 'conversation_item_added'
                    and getattr(e.payload.item, 'role', None) == 'assistant' and e.timestamp > user_event.timestamp,
                    timeout=35)
                assert len(h.events.user_messages()) == 1
                assert reply_event.payload.item.text_content
                record_property('reply_chars', len(reply_event.payload.item.text_content))
                reply_audio_bytes = sum(s.duration_bytes for s in h.audio_out.segments
                    if s.started_at >= user_event.timestamp and not s.cleared)
                record_property('reply_audio_bytes', reply_audio_bytes)
                assert reply_audio_bytes > 0
                record_property('reply_latency', json.dumps(probe.report(pcm), ensure_ascii=False))
                probe.close()
            else:
                assert not h.events.user_messages()
    finally:
        await stages.stt.shutdown()
        await stages.tts.shutdown()
        await llm.aclose()
        await classifier.aclose()
