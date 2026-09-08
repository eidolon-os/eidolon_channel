"""Bridge between LiveKit's RecognizeStream and this Host's ASR service.

One utterance per VAD flush. The framework hands audio frames on the input
channel and a `_FlushSentinel` when the speaker stops; this opens an utterance
on the first audio after a flush, streams frames, closes it on the sentinel,
and pushes the answers out as speech events.

Two things here are deliberately absent, and both are why this file is a
fraction of the size of the provider plugins beside it:

* **No VAD-gated forwarding.** Its whole purpose next door is to stop paying a
  provider for silence — that plugin's own header puts the number at 55-65% of
  billing wasted. This Host bills nothing.
* **No reconnect budget.** The service is on this machine, reached over
  loopback. A remote endpoint drops because a network is between you and it;
  this one is not answering because it is not running, and reconnecting on a
  backoff schedule would only delay saying so.

The two-pass shape is what a local Host can do that a provider cannot: interim
answers arrive from a streaming model while the words are still being spoken,
and the final one is re-decoded by an offline model and punctuated. A client
renders the interims and replaces them with the final — which is exactly the
INTERIM_TRANSCRIPT / FINAL_TRANSCRIPT contract LiveKit already has.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

import aiohttp
from eidolon_sdk.biz.contracts import local_asr as contract
from livekit.agents import stt as lk_stt
from livekit.agents.utils.audio import AudioByteStream

if TYPE_CHECKING:
    from .config import LocalAsrSTTConfig

logger = logging.getLogger("eidolon.livekit.plugins.stt.local_asr")


class LocalAsrSpeechStream(lk_stt.RecognizeStream):
    def __init__(
        self,
        *,
        stt: lk_stt.STT,
        config: "LocalAsrSTTConfig",
        stream_url: str,
        session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(stt=stt, conn_options=stt._conn_options)
        self._config = config
        self._stream_url = stream_url
        self._session = session
        self._stream_id = uuid.uuid4().hex
        self._utterance_index = 0

    async def _run(self) -> None:
        flush_sentinel: type = lk_stt.RecognizeStream._FlushSentinel  # type: ignore[attr-defined]
        samples = int(self._config.sample_rate * self._config.frame_ms / 1000)
        buffer = AudioByteStream(
            sample_rate=self._config.sample_rate,
            num_channels=self._config.channels,
            samples_per_channel=samples,
        )
        try:
            async with self._session.ws_connect(
                self._stream_url,
                timeout=aiohttp.ClientWSTimeout(ws_close=self._config.connect_timeout_s),
                heartbeat=None,
            ) as socket:
                await self._read_greeting(socket)
                await self._stream(socket, buffer, flush_sentinel)
        except aiohttp.ClientError as exc:
            # Said plainly rather than retried. This service is on this
            # machine: if it is not answering, it is not running, and the Host
            # is the thing to look at.
            self._emit_error(
                RuntimeError(
                    f"this Host's speech recognition did not answer at "
                    f"{self._stream_url}: {exc}"
                ),
                recoverable=False,
            )

    async def _read_greeting(self, socket: aiohttp.ClientWebSocketResponse) -> None:
        """Take the version the service states on the connection being used.

        The service sends this unprompted the moment a stream opens. Reading it
        here rather than trusting the readiness document is the stronger check:
        readiness is a separate request that could have been answered by a
        different process than the one this socket is attached to.

        A version mismatch is fatal, and deliberately: the alternative to
        refusing is parsing a shape this build does not know, and being wrong
        about a transcript is worse than being absent from one.
        """

        greeting = await self._receive(socket, timeout=self._config.connect_timeout_s)
        if greeting is None or greeting.get("type") != contract.CONNECTED:
            raise RuntimeError(
                f"expected {contract.CONNECTED} from this Host's recognition, "
                f"got {greeting!r}"
            )
        served = greeting.get(contract.PROTOCOL_VERSION_FIELD)
        if served != contract.LOCAL_ASR_PROTOCOL_VERSION:
            raise RuntimeError(
                f"this Host serves local recognition protocol {served} and this "
                f"build reads {contract.LOCAL_ASR_PROTOCOL_VERSION}"
            )
        logger.info(
            "local_asr connected: backend=%s streaming=%s offline=%s",
            greeting.get("backend"),
            greeting.get("model_id"),
            greeting.get("offline_model_id"),
        )

    async def _stream(
        self,
        socket: aiohttp.ClientWebSocketResponse,
        buffer: AudioByteStream,
        flush_sentinel: type,
    ) -> None:
        open_utterance = False
        async for item in self._input_ch:  # type: ignore[attr-defined]
            if isinstance(item, flush_sentinel):
                if not open_utterance:
                    continue
                for frame in buffer.flush():
                    await socket.send_bytes(frame.data.tobytes())
                await socket.send_str(json.dumps({"type": contract.END_UTTERANCE}))
                await self._await_final(socket)
                open_utterance = False
                continue
            if not open_utterance:
                await self._open_utterance(socket)
                open_utterance = True
            for frame in buffer.push(item.data.tobytes()):
                await socket.send_bytes(frame.data.tobytes())
                await self._drain(socket)
        if open_utterance:
            # The framework closed the input mid-utterance: ask for what the
            # service has rather than dropping a half-spoken sentence.
            for frame in buffer.flush():
                await socket.send_bytes(frame.data.tobytes())
            await socket.send_str(json.dumps({"type": contract.END_UTTERANCE}))
            await self._await_final(socket)
        await socket.send_str(json.dumps({"type": contract.CLOSE_STREAM}))

    async def _open_utterance(self, socket: aiohttp.ClientWebSocketResponse) -> None:
        self._utterance_index += 1
        message = contract.start_message(
            self._stream_id, f"{self._stream_id}-{self._utterance_index}"
        )
        await socket.send_str(json.dumps(message))
        answer = await self._receive(socket, timeout=self._config.connect_timeout_s)
        if answer is None or answer.get("type") != contract.UTTERANCE_STARTED:
            raise RuntimeError(
                f"expected {contract.UTTERANCE_STARTED} from this Host's recognition, "
                f"got {answer!r}"
            )

    async def _drain(self, socket: aiohttp.ClientWebSocketResponse) -> None:
        """Take whatever interim answers have arrived, without waiting for one.

        Interims are a courtesy: they make words appear while they are still
        being spoken. Blocking the send loop for one would trade the thing they
        are for the thing they help.
        """

        while True:
            try:
                message = await asyncio.wait_for(socket.receive(), timeout=0)
            except (TimeoutError, asyncio.TimeoutError):
                return
            payload = self._decode(message)
            if payload is None:
                return
            self._handle(payload)

    async def _await_final(self, socket: aiohttp.ClientWebSocketResponse) -> None:
        deadline = asyncio.get_running_loop().time() + self._config.final_timeout_s
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                self._emit_error(
                    RuntimeError(
                        "this Host's recognition did not finish the utterance within "
                        f"{self._config.final_timeout_s}s"
                    ),
                    recoverable=True,
                )
                return
            payload = await self._receive(socket, timeout=remaining)
            if payload is None:
                return
            if self._handle(payload):
                return

    async def _receive(
        self, socket: aiohttp.ClientWebSocketResponse, *, timeout: float
    ) -> dict[str, Any] | None:
        try:
            message = await asyncio.wait_for(socket.receive(), timeout=timeout)
        except (TimeoutError, asyncio.TimeoutError):
            return None
        return self._decode(message)

    def _decode(self, message: aiohttp.WSMessage) -> dict[str, Any] | None:
        if message.type is not aiohttp.WSMsgType.TEXT:
            return None
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            logger.warning("local_asr sent a message that is not JSON")
            return None
        return payload if isinstance(payload, dict) else None

    def _handle(self, payload: dict[str, Any]) -> bool:
        """Push one answer out. Returns whether it was the final one."""

        kind = payload.get("type")
        if kind == contract.ERROR:
            code = str(payload.get("code", ""))
            self._emit_error(
                RuntimeError(f"this Host's recognition refused: {code}"),
                recoverable=code in contract.RETRYABLE_ERROR_CODES,
            )
            return True
        if kind != contract.TRANSCRIPT:
            return False
        text = str(payload.get("text", ""))
        is_final = bool(payload.get(contract.IS_FINAL_FIELD))
        if not text and not is_final:
            return False
        self._push_event(
            lk_stt.SpeechEvent(
                type=(
                    lk_stt.SpeechEventType.FINAL_TRANSCRIPT
                    if is_final
                    else lk_stt.SpeechEventType.INTERIM_TRANSCRIPT
                ),
                alternatives=[
                    lk_stt.SpeechData(
                        language=self._config.language,
                        text=text,
                        start_time=0.0,
                        end_time=0.0,
                        confidence=1.0,
                    )
                ],
            )
        )
        return is_final

    def _push_event(self, event: lk_stt.SpeechEvent) -> None:
        try:
            self._event_ch.send_nowait(event)  # type: ignore[attr-defined]
        except asyncio.QueueFull:
            logger.warning("local_asr event queue full, dropping event")

    def _emit_error(self, error: Exception, *, recoverable: bool) -> None:
        logger.error("local_asr stream error (recoverable=%s): %s", recoverable, error)
        self._push_event(
            lk_stt.SpeechEvent(type=lk_stt.SpeechEventType.END_OF_SPEECH, alternatives=[])
        )
        raise error
