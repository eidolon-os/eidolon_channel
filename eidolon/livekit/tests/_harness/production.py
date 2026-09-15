
"""Reuse production Agent and callback wiring with scripted external providers.

This exercises channel orchestration through real LiveKit AgentSession and real
EOT inference. It does not measure ASR accuracy, RTC transport or device playback.
"""

from eidolon_sdk.biz.presentation import OutputSelection
from contextlib import asynccontextmanager
from types import SimpleNamespace

from eidolon.livekit.agent.full_duplex.pipeline import StreamingPipeline

from .headless import headless_session


def stub_factory(*, llm, stt, tts, vad, interrupt_classifier=None, runtime_session_id='sess-harness'):
    """Stand in for SharedStageFactory with the attributes a pipeline reads.

    One builder for both duplex modes on purpose. Written as two doubles, they
    drift: an attribute the pipelines start reading lands on the one whose
    tests were in front of you, and every test on the other path dies on it.

    ``outputs`` is derived rather than declared, because the real factory
    derives it too. It rejects any plan whose ``speech`` disagrees with whether
    it holds a TTS stage, falls back to speech plus dialogue text when the
    dispatch carries no output plan, and refuses an expression plan unless the
    brain is ``eidolon_agent`` — which a scripted LLM is not. So a double is
    entitled to exactly one output shape per TTS it is handed.

    ``stt`` and ``vad`` arrive already shaped. The real ``SttStage`` answers
    both the segment API half duplex calls and the ``.stt`` full duplex hands
    to AgentSession; each caller supplies the half its own pipeline reads.
    """
    return SimpleNamespace(
        outputs=OutputSelection(speech=tts is not None, dialogue_text=True),
        llm=SimpleNamespace(llm=llm), stt=stt,
        tts=SimpleNamespace(tts=tts) if tts is not None else None, vad=vad,
        interrupt_classifier=interrupt_classifier,
        # The real factory always carries the named dispatch's conversation_id.
        # A session-scoped observer keys its file by it, so a double without one
        # would silently observe nothing.
        runtime_session_id=runtime_session_id,
    )


@asynccontextmanager
async def production_session(*, llm, stt, tts, vad, mode='full_duplex', welcome='', real_time_audio=False, interrupt_classifier=None, warmup=False, **kwargs):
    factory = stub_factory(
        llm=llm, stt=SimpleNamespace(stt=stt), tts=tts, vad=SimpleNamespace(vad=vad),
        interrupt_classifier=interrupt_classifier,
    )
    pipeline = StreamingPipeline(
        factory, interaction_mode=mode, allow_interruptions=mode == 'full_duplex',
        welcome_message=welcome, aec_warmup_duration=None, **kwargs,
    )

    def configure(session):
        pipeline._lifecycle.bind_session(session)
        if pipeline._barge_in_enabled:
            pipeline._ensure_output_flow().install_duck_mixer(session)

    try:
        if warmup:
            await pipeline._warmup_stages()
        async with headless_session(
            llm=llm, stt=stt, tts=tts, vad=vad, agent=pipeline._build_agent(),
            configure_session=configure, strict_cleanup=True, real_time_audio=real_time_audio,
            extra_session_kwargs={
                'turn_handling': pipeline._build_turn_handling(),
                'transcription_timeout': pipeline._stt_commit_transcript_timeout,
            },
        ) as handle:
            yield pipeline, handle
    finally:
        await pipeline.shutdown()


class ScriptedSegmentSTT:
    """Scripted external ASR boundary for the production PTT recorder/transcriber."""
    def __init__(self, text):
        self.text = text
        self.audio = []

    async def recognize_streaming(self, audio):
        self.audio.append(audio)
        return self.text

    async def recognize(self, audio):
        return await self.recognize_streaming(audio)


@asynccontextmanager
async def production_ptt_session(*, text, llm, tts, stt_stage=None, interrupt_classifier=None, instructions=''):
    from eidolon.livekit.agent.half_duplex.pipeline import HalfDuplexPttPipeline
    from .mocks import MockSTT, MockVAD

    stage = stt_stage if stt_stage is not None else ScriptedSegmentSTT(text)
    pipeline = HalfDuplexPttPipeline(stub_factory(
        llm=llm, stt=stage, tts=tts, vad=None,
        interrupt_classifier=interrupt_classifier,
    ), instructions=instructions)
    try:
        # Match run(): reuse production provider timing/usage observers.
        pipeline._provider_events.install_all()
        async with headless_session(
            llm=llm, tts=tts, stt=MockSTT.scripted([]), vad=MockVAD.silent(),
            agent=pipeline._build_agent(), configure_session=pipeline._bind_session, strict_cleanup=True,
            extra_session_kwargs={'turn_handling': pipeline._build_turn_handling()},
        ) as handle:
            yield pipeline, handle, stage
    finally:
        await pipeline.shutdown()
