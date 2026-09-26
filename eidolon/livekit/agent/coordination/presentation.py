"""Stream existing TTS into a prepared device's correlated PCM presentation.

This adapter owns synthesis and byte transport, not device admission or playback
completion. prepare() must send playback.present through the existing control
path, await its accepted ACK, and return the matching terminal-result awaitable.
It must reject firmware without this capability; there is no idle/timer fallback.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from uuid import uuid4

from livekit import rtc
from eidolon_sdk.biz.control.audio_presentation import (
    AUDIO_PRESENTATION_TOPIC,
    MAX_PRESENTATION_BYTES,
    PCM_SAMPLE_RATE,
    AudioPresentation,
    AudioPresentationResult,
)
from eidolon_sdk.biz.control.coordination_stream import ReplyStart


@dataclass(frozen=True)
class SpeechEndpoint:
    participant: object
    destination_identity: str
    tts: object  # Existing TtsStage, separately configured for this Companion.
    prepare: Callable[
        [AudioPresentation, Callable[[], None]], Awaitable[Awaitable[AudioPresentationResult]]
    ]


class GroupSpeechPresenter:
    def __init__(self, endpoints: dict[tuple[str, str], SpeechEndpoint]):
        # Exact Companion/device pair; no default voice or unselected endpoint.
        self.endpoints = dict(endpoints)

    async def __call__(self, start: ReplyStart, text: AsyncIterator[str], speaking):
        endpoint = self.endpoints[(start.companion_id, start.device_id)]
        request = AudioPresentation(
            session_id=start.session_id,
            turn_id=start.turn_id,
            stream_id=uuid4().hex,
            epoch=start.epoch,
        )
        result = await endpoint.prepare(request, speaking)
        terminal = asyncio.ensure_future(result)
        writer = None
        feeder = None
        stream = None
        normal_end = False
        sent_bytes = 0
        try:
            writer = await endpoint.participant.stream_bytes(
                "speech.pcm",
                stream_id=request.stream_id,
                topic=AUDIO_PRESENTATION_TOPIC,
                mime_type="application/octet-stream",
                destination_identities=[endpoint.destination_identity],
            )
            stream = endpoint.tts.stream()

            async def feed():
                count = 0
                async for chunk in text:
                    count += len(chunk)
                    if count > 32768:
                        raise ValueError("role-group speech text limit exceeded")
                    stream.push_text(chunk)
                stream.end_input()

            feeder = asyncio.create_task(feed())

            # Consume synthesis independently from token delivery. Failed input
            # must stop a provider even if its output iterator is waiting forever.
            async def pcm():
                nonlocal sent_bytes
                resampler = None
                source_rate = None
                began = None
                loop = asyncio.get_running_loop()

                async def send(frame):
                    nonlocal sent_bytes, began
                    data = bytes(frame.data)
                    if frame.num_channels != 1 or frame.sample_rate != PCM_SAMPLE_RATE:
                        raise ValueError("unexpected PCM output format")
                    for offset in range(0, len(data), 640):  # <=20 ms per reliable packet
                        chunk = data[offset : offset + 640]
                        if len(chunk) % 2 or sent_bytes + len(chunk) > MAX_PRESENTATION_BYTES:
                            raise ValueError("invalid or oversized PCM presentation")
                        if began is None:
                            began = loop.time()
                        # <=100 ms ahead of realtime, keeping reliable control
                        # traffic responsive and preventing unbounded device audio.
                        delay = began + sent_bytes / 32000 - loop.time() - 0.1
                        if delay > 0:
                            await asyncio.sleep(delay)
                        await writer.write(chunk)
                        sent_bytes += len(chunk)

                async for audio in stream:
                    frame = audio.frame
                    if frame.num_channels != 1:
                        raise ValueError("role-group presentation requires mono TTS")
                    if source_rate is None:
                        source_rate = frame.sample_rate
                        if source_rate != PCM_SAMPLE_RATE:
                            resampler = rtc.AudioResampler(
                                source_rate, PCM_SAMPLE_RATE, num_channels=1
                            )
                    if frame.sample_rate != source_rate:
                        raise ValueError("TTS changed sample rate during a reply")
                    for converted in resampler.push(frame) if resampler else [frame]:
                        await send(converted)
                if resampler:
                    for converted in resampler.flush():
                        await send(converted)

            producer = asyncio.create_task(pcm())
            try:
                pending = {feeder, producer}
                while pending:
                    done, _ = await asyncio.wait(
                        pending | {terminal}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if terminal in done:
                        raise ValueError("device completed before stream end")
                    for task in done:
                        task.result()
                    if producer in done and not feeder.done():
                        raise ValueError("TTS ended before input completed")
                    pending -= done
                if not sent_bytes:
                    raise ValueError("TTS produced no speech audio")
                if terminal.done():
                    raise ValueError("device completed before normal trailer")
                await writer.aclose()
                normal_end = True
                confirmed = await terminal
                return isinstance(confirmed, AudioPresentationResult) and confirmed.confirms(
                    request, sent_bytes
                )
            finally:
                for task in (feeder, producer):
                    task.cancel()
                await asyncio.gather(feeder, producer, return_exceptions=True)
        finally:
            terminal.cancel()
            await asyncio.gather(terminal, return_exceptions=True)
            if stream is not None:
                try:
                    async with asyncio.timeout(2):
                        await stream.aclose()
                except Exception:
                    pass
            if writer is not None and not normal_end:
                # Abnormal trailer cannot cause a successful playback receipt.
                try:
                    async with asyncio.timeout(2):
                        await writer.aclose(reason="presentation_aborted")
                except Exception:
                    pass  # RoleGroupClient still independently stops the endpoint.
