"""LiveKit TTS over this Host's own synthesis service.

Small, and the reason it is small is the point: everything this plugin does not
contain is something only a provider needs. There are no credentials, no
regions, no retry budget against someone else's rate limit, no billing gate,
and no mp3 to decode — the service hands over raw PCM at the rate the contract
states, on loopback, from a process on the same board.

What is here that the cloud providers do not have is sentence batching. They
take a token stream and synthesize continuously; this engine says one whole
utterance at a time and holds it in a fixed context, so tokens are gathered
into sentences first. `SentenceAggregator` beside this package already does
exactly that, and it is what makes the first sentence leave early instead of
the reply arriving as one block.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

import aiohttp
from eidolon_sdk.biz.contracts import local_tts as contract
from livekit.agents import APIConnectOptions
from livekit.agents.tts import AudioEmitter, SynthesizeStream, TTS, TTSCapabilities

from .._aggregator import SentenceAggregator
from .config import LocalTtsConfig
from .endpoint import resolve_stream_url

logger = logging.getLogger(__name__)

PROVIDER = contract.LOCAL_TTS_CAPABILITY


class LocalTtsError(RuntimeError):
    """The Host's own synthesis refused or stopped answering."""


class LocalTTS(TTS):
    """This Host's voice, as a LiveKit TTS."""

    def __init__(
        self,
        *,
        config: LocalTtsConfig | None = None,
        stream_url: str | None = None,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=TTSCapabilities(streaming=True, aligned_transcript=False),
            sample_rate=contract.AUDIO_SAMPLE_RATE,
            num_channels=contract.AUDIO_CHANNELS,
        )
        self._config = config or LocalTtsConfig()
        #: Resolved once, when the worker builds the provider, so a Host that
        #: does not declare the capability fails at construction with the
        #: registry's own message rather than on the first spoken turn.
        self._stream_url = stream_url or resolve_stream_url()
        self._session = session
        self._owns_session = session is None

    @property
    def stream_url(self) -> str:
        return self._stream_url

    async def _http(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    def synthesize(self, text: str, *, conn_options: APIConnectOptions | None = None):
        raise NotImplementedError(
            f"{PROVIDER} is a streaming provider; use stream()"
        )

    def stream(
        self, *, conn_options: APIConnectOptions | None = None
    ) -> "LocalSynthesizeStream":
        return LocalSynthesizeStream(tts=self, conn_options=conn_options)

    async def aclose(self) -> None:
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None


class LocalSynthesizeStream(SynthesizeStream):
    def __init__(self, *, tts: LocalTTS, conn_options: APIConnectOptions | None) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts = tts
        self._config = tts._config

    async def _run(self, output_emitter: AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=uuid.uuid4().hex[:16],
            sample_rate=contract.AUDIO_SAMPLE_RATE,
            num_channels=contract.AUDIO_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )
        output_emitter.start_segment(segment_id=uuid.uuid4().hex[:16])

        session = await self._tts._http()
        try:
            async with session.ws_connect(self._tts.stream_url) as socket:
                await self._expect_greeting(socket)
                await self._speak(socket, output_emitter)
        finally:
            output_emitter.end_segment()
            output_emitter.end_input()

    async def _expect_greeting(self, socket: aiohttp.ClientWebSocketResponse) -> None:
        """Read the version off the connection about to be used.

        Not from readiness: that is a different request and could describe a
        different process. A version this build does not understand is a
        refusal here rather than mis-parsed audio later.
        """

        message = await socket.receive(timeout=self._config.first_frame_timeout_s)
        if message.type is not aiohttp.WSMsgType.TEXT:
            raise LocalTtsError(f"expected a greeting, got {message.type}")
        greeting = json.loads(message.data)
        if greeting.get("type") != contract.CONNECTED:
            raise LocalTtsError(f"expected {contract.CONNECTED}, got {greeting.get('type')!r}")
        served = greeting.get(contract.PROTOCOL_VERSION_FIELD)
        if served != contract.LOCAL_TTS_PROTOCOL_VERSION:
            raise LocalTtsError(
                f"this Host serves local TTS protocol {served!r}; this build "
                f"speaks {contract.LOCAL_TTS_PROTOCOL_VERSION}"
            )
        if greeting.get("sample_rate") != contract.AUDIO_SAMPLE_RATE:
            raise LocalTtsError(
                f"this Host sends {greeting.get('sample_rate')!r} Hz audio; the "
                f"contract says {contract.AUDIO_SAMPLE_RATE}"
            )

    async def _speak(
        self, socket: aiohttp.ClientWebSocketResponse, output_emitter: AudioEmitter
    ) -> None:
        async def say(sentence: str) -> None:
            await self._say_one(socket, sentence, output_emitter)

        aggregator = SentenceAggregator(
            say,
            soft_min_chars=self._config.soft_min_chars,
            hard_max_chars=self._config.hard_max_chars,
            idle_ms=self._config.idle_ms,
            first_sentence_soft_min_chars=self._config.first_sentence_soft_min_chars,
            first_sentence_flush_any_punct=self._config.first_sentence_flush_any_punct,
        )

        timeout = self._config.first_token_timeout_s
        while True:
            try:
                token = await asyncio.wait_for(self._input_ch.__anext__(), timeout=timeout)
            except TimeoutError:
                logger.warning("[%s] no token within %.1fs; ending the turn", PROVIDER, timeout)
                break
            except StopAsyncIteration:
                break
            timeout = self._config.inter_token_timeout_s
            if isinstance(token, SynthesizeStream._FlushSentinel):
                await aggregator.flush()
                continue
            if isinstance(token, str):
                await aggregator.feed(token)
        await aggregator.flush()

    def _report(self, event: dict[str, Any]) -> None:
        """Say something only when the listener would have heard something.

        Whether the Host kept ahead of playback is the one quality fact this
        side cannot observe for itself, so it is worth a line in the log —
        but only from the field that answers that question.

        That field is `minimum_buffer_ms`: how much unplayed audio was left
        at the worst moment, below zero meaning a gap was audible. It is
        *not* `late_chunks`, which counts chunks that missed a deadline
        measured from the first one being ready, i.e. as though playback
        began with an empty buffer. For a producer near real time that is
        almost every chunk by construction, so it tracks the length of the
        audio and not the listener's experience — 17 seconds of untroubled
        speech reports about 17 late chunks.

        This read that field under its older name, `underruns`, and reported
        "the Host underran N times" for utterances whose real gap count was
        zero. The rename to `late_chunks` then left the branch permanently
        dead, and nothing noticed because both ends spelled the name out
        separately. Both halves of that are why the names now live in the
        contract and are read from it here.

        Absent is not zero: a Host that did not measure the floor, or an
        older one that did not have it, omits the field, and its absence is
        not a claim that nothing was heard.
        """

        floor = event.get(contract.MINIMUM_BUFFER_MS_FIELD)
        if not isinstance(floor, (int, float)) or isinstance(floor, bool) or floor >= 0:
            return
        logger.warning(
            "[%s] audio broke up: the Host's buffer went %.0f ms below empty on "
            "%.2fs of audio (late_chunks=%s steady_rtf=%s)",
            PROVIDER,
            floor,
            event.get(contract.AUDIO_SECONDS_FIELD, 0.0),
            event.get(contract.LATE_CHUNKS_FIELD),
            event.get(contract.STEADY_RTF_FIELD),
        )

    async def _say_one(
        self,
        socket: aiohttp.ClientWebSocketResponse,
        sentence: str,
        output_emitter: AudioEmitter,
    ) -> None:
        request_id = uuid.uuid4().hex[:16]
        await socket.send_json(
            {"type": contract.SYNTHESIZE, contract.REQUEST_ID_FIELD: request_id, "text": sentence}
        )
        timeout = self._config.first_frame_timeout_s
        while True:
            message = await socket.receive(timeout=timeout)
            if message.type is aiohttp.WSMsgType.BINARY:
                timeout = self._config.inter_frame_timeout_s
                output_emitter.push(message.data)
                continue
            if message.type is not aiohttp.WSMsgType.TEXT:
                raise LocalTtsError(f"the Host closed the stream: {message.type}")
            event = json.loads(message.data)
            kind = event.get("type")
            if kind == contract.SYNTHESIS_STARTED:
                continue
            if kind == contract.SYNTHESIS_FINISHED:
                self._report(event)
                return
            if kind == contract.SYNTHESIS_CANCELLED:
                return
            if kind == contract.ERROR:
                raise LocalTtsError(
                    f"{event.get('code')}: {event.get('message')}"
                    + (" (retryable)" if event.get("retryable") else "")
                )
            raise LocalTtsError(f"unexpected message from the Host: {kind!r}")
