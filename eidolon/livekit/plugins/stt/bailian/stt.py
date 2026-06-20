"""Bailian FunASR STT plugin for LiveKit Agents.

Connects to the Alibaba Bailian FunASR Realtime WebSocket API
(https://dashscope.aliyuncs.com/api-ws/v1/inference) and provides both
streaming (SpeechStream) and batch (recognize) modes.
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any

import livekit
from livekit.agents import stt
from livekit.agents.stt import (
    SpeechData,
    SpeechEvent,
    SpeechEventType,
    STTCapabilities,
)
from livekit.agents.types import APIConnectOptions

from .config import BailianSTTConfig
from .connection_manager import BailianConnectionManager
from .models import FunASREventType, FunASRResultGenerated, FunASRSentence, parse_funasr_message
from .speech_stream import BailianFunASRSpeechStream

logger = logging.getLogger("bailian.stt")

if TYPE_CHECKING:
    from livekit import rtc

#: Default API key used when no explicit key or config is provided.
#: Can be overridden via the DASHSCOPE_API_KEY environment variable.
DEFAULT_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")


class BailianFunASRSTT(stt.STT):
    """Bailian FunASR STT provider for LiveKit Agents.

    Can be instantiated with either a :class:`BailianSTTConfig` object or
    individual keyword arguments (which are used when ``config`` is ``None``).

    Parameters
    ----------
    config : BailianSTTConfig | None
        All configuration options in one object. When provided, all other
        keyword arguments are ignored.
    model : str
        Model name for the FunASR task (e.g. ``"fun-asr-realtime-2026-02-28"``).
        Only used when ``config`` is ``None``.
    language : str
        Language hint passed to the FunASR server. Default: ``"zh"``.
    api_key : str | None
        Bailian / DashScope API key. Falls back to the ``DASHSCOPE_API_KEY``
        environment variable if not provided.
    api_url : str
        Full WebSocket URL for the FunASR inference endpoint.
        Default: ``"wss://dashscope.aliyuncs.com/api-ws/v1/inference"``.
    sample_rate : int
        Expected audio sample rate in Hz. Must be 16 000 for the default model.
        Default: ``16000``.
    itn : bool
        Enable inverse text normalization (ITN). When ``True`` the server returns
        numbers as Arabic numerals (e.g. "2024" instead of "二千零二十四").
        Default: ``True``.
    conn_options : APIConnectOptions
        Connection options controlling retry behaviour. Passed to each
        :class:`BailianFunASRSpeechStream` created by :meth:`stream`.
    """

    def __init__(
        self,
        config: BailianSTTConfig | None = None,
        *,
        model: str = "fun-asr-realtime-2026-02-28",
        language: str = "zh",
        api_key: str | None = None,
        api_url: str = "wss://dashscope.aliyuncs.com/api-ws/v1/inference",
        sample_rate: int = 16000,
        itn: bool = True,
        conn_options: livekit.agents.types.APIConnectOptions | None = None,
    ):
        super().__init__(
            capabilities=STTCapabilities(
                streaming=True,
                interim_results=True,
                offline_recognize=True,
                aligned_transcript=False,
                diarization=False,
            )
        )

        if config is not None:
            self._config = config
            self._model = config.model
            self._language = config.language
            self._api_key = config.api_key
            self._api_url = config.api_url
            self._sample_rate = config.sample_rate
            self._itn = config.itn
        else:
            # G16 (2026-05-17): build a default config so gate / future
            # features can access typed fields uniformly. Honors env vars.
            self._config = BailianSTTConfig(
                model=model,
                language=language,
                api_key=api_key or os.environ.get("DASHSCOPE_API_KEY", ""),
                api_url=api_url,
                sample_rate=sample_rate,
                itn=itn,
            )
            self._model = model
            self._language = language
            self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
            self._api_url = api_url
            self._sample_rate = sample_rate
            self._itn = itn

        self._conn_options = conn_options or livekit.agents.types.DEFAULT_API_CONNECT_OPTIONS
        # G16 (2026-05-17): track the most-recently-created stream so the
        # caller (streaming.py's VAD inference callback) can route per-frame
        # VAD signal into the gate inside the active stream. weakref-style
        # — only one stream is "current" at a time per session.
        self._current_stream: "BailianFunASRSpeechStream | None" = None

        if not self._api_key:
            logger.warning(
                "No DASHSCOPE_API_KEY provided to BailianFunASRSTT. "
                "Set the DASHSCOPE_API_KEY environment variable or pass api_key explicitly."
            )

    def emit_provider_event(self, name: str, **payload: Any) -> None:
        """Emit provider-level STT timing events for Channel observability."""

        self.emit(
            "provider_event",
            {
                "provider": self.provider,
                "event": name,
                "timestamp": time.monotonic(),
                "model": self.model,
                **payload,
            },
        )

    def observe_next_audio_for_turn(
        self,
        *,
        turn_id: str,
        speech_started_at: float,
    ) -> bool:
        """Ask the active stream to mark the next provider audio chunk.

        FunASR runs as a long-lived streaming websocket, so the stream's first
        audio packet is often silence or pre-roll before VAD opens a user turn.
        This hook lets the Channel mark the first packet sent after the turn is
        known, which is the useful per-turn latency anchor.
        """

        stream = self._current_stream
        if stream is None:
            return False
        stream.observe_next_audio_for_turn(
            turn_id=turn_id,
            speech_started_at=speech_started_at,
        )
        return True

    # ------------------------------------------------------------------
    # Properties (LiveKit STT contract)
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "bailian"

    @property
    def label(self) -> str:
        return f"Bailian FunASR ({self._model})"

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def api_url(self) -> str:
        return self._api_url

    @property
    def itn(self) -> bool:
        return self._itn

    @property
    def max_sentence_silence_ms(self) -> int:
        return self._config.max_sentence_silence_ms

    @property
    def keepalive_interval_sec(self) -> float:
        return self._config.keepalive_interval_sec

    @property
    def language(self) -> str:
        return self._language

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def conn_options(self) -> livekit.agents.types.APIConnectOptions:
        return self._conn_options

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def stream(
        self,
        *,
        language: str | None = None,
        conn_options: livekit.agents.types.APIConnectOptions | None = None,
    ) -> BailianFunASRSpeechStream:
        """Create a streaming transcription session.

        Returns an async iterator of :class:`SpeechEvent` objects. Push audio
        frames via ``push_frame()`` (sync). The framework calls ``flush()`` when
        VAD detects end of speech and ``end_input()`` when the user turn ends.

        Usage::

            stream = stt.stream()
            stream.push_frame(audio_frame)
            async for ev in stream:
                print(ev.alternatives[0].text)
        """
        s = BailianFunASRSpeechStream(
            stt=self,
            conn_options=conn_options or self._conn_options,
            sample_rate=self._sample_rate,
            language=language or self._language,
        )
        # G16: register as current so external VAD-signal callers route here.
        self._current_stream = s
        return s

    # G16 (2026-05-17): VAD signal bridge. streaming.py invokes this once
    # per VAD inference frame; we delegate to the active stream's gate.
    def notify_vad_state(self, probability: float, rms: float) -> None:
        """Forward per-frame VAD probability + RMS energy to the current
        SpeechStream's gate. No-op if no active stream or gate disabled."""
        s = self._current_stream
        if s is not None:
            try:
                s.notify_vad_state(probability, rms)
            except Exception as e:
                logger.debug("[BailianSTT] notify_vad_state failed: %s", e)

    # ------------------------------------------------------------------
    # Batch recognition
    # ------------------------------------------------------------------

    async def _recognize_impl(
        self,
        buffer: "list[rtc.AudioFrame] | rtc.AudioFrame",
        *,
        language: "Any",
        conn_options: "APIConnectOptions",
    ) -> SpeechEvent:
        """Transcribe a single audio buffer using a fresh WebSocket session.

        For batch transcription of larger audio, consider using :meth:`stream`
        and waiting for ``FINAL_TRANSCRIPT`` instead.
        """
        import asyncio
        import time
        from livekit import rtc

        lang = language
        # Extract raw bytes from the buffer
        if isinstance(buffer, rtc.AudioFrame):
            audio_bytes = buffer.data.tobytes()
        else:
            audio_bytes = b"".join(f.data.tobytes() for f in buffer)

        all_sentences: list[FunASRSentence] = []
        conn = BailianConnectionManager(
            api_url=self._api_url,
            api_key=self._api_key,
            model=self._model,
            sample_rate=self._sample_rate,
            itn=self._itn,
            language_hints=lang,
        )

        async def collector(data: dict[str, Any]) -> None:
            try:
                event_name, parsed = parse_funasr_message(data)
                if event_name == FunASREventType.RESULT_GENERATED.value:
                    result: FunASRResultGenerated = parsed
                    for sentence in result.sentences:
                        if sentence.sentence_end:
                            all_sentences.append(sentence)
            except ValueError:
                pass

        start_time = time.monotonic()
        recv_task: asyncio.Task | None = None
        try:
            await conn.connect(message_callback=collector)

            # Spawn a receive loop so the collector processes responses and
            # _connected is set to False when the server closes the connection.
            recv_task = asyncio.create_task(conn.receive_loop(message_cb=collector))

            await conn.send_audio(audio_bytes)
            await conn.finish()

            # Drain results with a timeout — the receive loop exits when the server closes
            deadline = start_time + 30.0
            while time.monotonic() < deadline:
                await asyncio.sleep(0.1)
                if not conn._connected:
                    break
        finally:
            if recv_task is not None and not recv_task.done():
                recv_task.cancel()
            if recv_task is not None:
                try:
                    await recv_task
                except asyncio.CancelledError:
                    pass
            await conn.close()

        # Build final transcript
        if not all_sentences:
            return SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    SpeechData(
                        language=lang,
                        text="",
                        start_time=0.0,
                        end_time=0.0,
                        confidence=1.0,
                        words=[],
                    )
                ],
            )

        text = "".join(s.text for s in all_sentences)
        s_start = all_sentences[0].begin_time / 1000.0
        s_end = all_sentences[-1].end_time / 1000.0

        words: list[SpeechData] = []
        for sentence in all_sentences:
            for word in sentence.words:
                words.append(
                    SpeechData(
                        language=lang,
                        text=word.text,
                        start_time=word.begin_time / 1000.0,
                        end_time=word.end_time / 1000.0,
                        confidence=1.0,
                    )
                )

        return SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                SpeechData(
                    language=lang,
                    text=text,
                    start_time=s_start,
                    end_time=s_end,
                    confidence=1.0,
                    words=words,
                )
            ],
        )
