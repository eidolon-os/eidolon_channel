"""Bailian TTS shutdown lifecycle tests."""

from __future__ import annotations

import pytest
from livekit.agents.types import APIConnectOptions

from eidolon.livekit.plugins.tts.bailian.config import BailianTTSConfig
from eidolon.livekit.plugins.tts.bailian.tts import BailianTTS
from eidolon.livekit.plugins.tts.bailian.tts_client import BailianTTSError


class _FakeEmitter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def initialize(self, **_kwargs) -> None:
        self.calls.append("initialize")

    def start_segment(self, **_kwargs) -> None:
        self.calls.append("start_segment")

    def end_segment(self) -> None:
        self.calls.append("end_segment")

    def end_input(self) -> None:
        self.calls.append("end_input")

    def push(self, _data: bytes) -> None:
        self.calls.append("push")


class _FakeBailianClient:
    def __init__(self) -> None:
        self.connected = False
        self.started = 0
        self.disconnected = 0

    async def connect(self) -> bool:
        self.connected = True
        return True

    async def start_task(self) -> None:
        self.started += 1

    async def disconnect(self) -> None:
        self.disconnected += 1


def _tts() -> BailianTTS:
    return BailianTTS(
        BailianTTSConfig(
            api_key="test",
            pool_size=1,
            pool_size_bootstrap=1,
        )
    )


@pytest.mark.asyncio
async def test_acquire_after_shutdown_is_not_recoverable() -> None:
    tts = _tts()
    await tts.shutdown()

    with pytest.raises(BailianTTSError) as exc_info:
        await tts._acquire_conn()

    assert exc_info.value.recoverable is False
    assert "shutting down" in str(exc_info.value)


@pytest.mark.asyncio
async def test_stream_started_during_shutdown_ends_without_pool_acquire() -> None:
    tts = _tts()
    await tts.shutdown()
    stream = tts.stream(conn_options=APIConnectOptions())
    emitter = _FakeEmitter()

    await stream._run(emitter)  # type: ignore[arg-type]

    assert emitter.calls == [
        "initialize",
        "start_segment",
        "end_segment",
        "end_input",
    ]


@pytest.mark.asyncio
async def test_warmup_preconnects_without_starting_provider_task() -> None:
    tts = _tts()
    fake = _FakeBailianClient()
    tts._create_connection = lambda: fake  # type: ignore[method-assign]

    await tts.warmup()
    await tts.shutdown()

    assert fake.connected is True
    assert fake.started == 0
    assert fake.disconnected == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['full_duplex', 'half_duplex', 'ptt'])
@pytest.mark.parametrize('streaming_reply', [False, True], ids=['short', 'streaming'])
async def test_reply_reaches_audio_after_failed_tts_warmup(mode, streaming_reply) -> None:
    """Real pipeline/SDK/TTS pool; only network services are simulated."""
    from eidolon.livekit.tests._harness.audio import frames_from_pcm, synth_voiced
    from eidolon.livekit.tests._harness.mocks import MockLLM, MockSTT, MockVAD, MockVADEvent, ScriptedTranscript
    from eidolon.livekit.tests._harness.production import production_ptt_session, production_session
    from eidolon.livekit.tests.scenarios.test_human_ptt import button

    unavailable = True
    clients = []
    sent_text = []

    class Client(_FakeBailianClient):
        async def connect(self):
            if unavailable:
                return False
            return await super().connect()

        async def send_continue(self, text):
            sent_text.append(text)
            await self._on_binary_callback(synth_voiced(.12))

        async def send_finish(self):
            await self._on_message_callback({'header': {'event': 'task-finished'}})

    def create_connection():
        client = Client()
        clients.append(client)
        return client

    tts = BailianTTS(BailianTTSConfig(
        api_key='test', pool_size=1, pool_size_bootstrap=1,
        pool_acquire_timeout=.2, pool_refill_backoff=.01,
    ))
    tts._create_connection = create_connection
    question, reply = '帮我详细介绍一下这个方案。', '现在介绍方案。'
    if streaming_reply:
        reply = '先说明费用。然后介绍服务，最后还有一些细节需要慢慢说明。'
    llm = MockLLM.scripted([(question, reply)], chunk_delay_ms=15 if streaming_reply else 0)
    llm_metrics = []
    llm.on('metrics_collected', llm_metrics.append)
    conversation = production_ptt_session(text=question, llm=llm, tts=tts) if mode == 'ptt' else production_session(
        mode=mode, llm=llm, tts=tts,
        stt=MockSTT.scripted([ScriptedTranscript(text=question, trigger_after_ms=300)]),
        vad=MockVAD.scripted([MockVADEvent('start', 100), MockVADEvent('end', 500, .1)]),
    )
    try:
        with pytest.raises(RuntimeError, match='warmup failed'):
            await tts.warmup()
        unavailable = False
        async with conversation as result:
            pipeline, h = result[:2]
            if mode == 'ptt':
                button(pipeline, True, 1)
                for frame in frames_from_pcm(synth_voiced(.5)):
                    pipeline._ptt_controller.push_frame(frame)
                assert not h.events.user_messages()
                button(pipeline, False, 2)
            else:
                h.audio_in.feed_pcm(synth_voiced(.5))
                h.audio_in.feed_silence(.7)
            if streaming_reply:
                await h.audio_out.wait_for_first_audio(timeout=3)
                assert llm_metrics == [], 'speech must start while the LLM is still generating'
            await h.events.wait_for_agent_messages(1, timeout=3)
            assert h.events.user_messages() == [question]
            assert h.events.agent_messages() == [reply]
            assert h.audio_out.first_audible_at is not None
            assert ''.join(sent_text) == reply
    finally:
        await tts.shutdown()
    assert all(client.disconnected == 1 for client in clients if client.connected)
