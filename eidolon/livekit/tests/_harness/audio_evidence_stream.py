"""Test-only tap at LiveKit's public Agent.stt_node boundary.

No room subscription, background reader, resampling, inference or turn guessing.
The caller must supply explicit boundaries in this input's sample coordinate
system. Provider timestamps and delayed speech-state callbacks are not such
boundaries. Production evidence uses the normalized provider-input coordinate,
not this native-rate proof harness. An input track replacement calls reset_source;
the public node receives frames, not track IDs. Keep this tap after SDK audio
suppression, so evidence cannot include real audio when ASR received silence.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import AsyncIterable
from uuid import uuid4

from livekit import rtc

from eidolon.livekit.common.audio_evidence import AudioEvidence, AudioEvidenceBuffer


@dataclass(frozen=True)
class AudioInputPosition:
    source: str
    generation: int
    sample_rate: int
    sample: int


class AudioEvidenceStream:
    """A bounded, in-memory tap; forwards the exact original frame to the SDK.

    Offsets count frames delivered at this boundary, not transport packets or
    wall time. Completeness cannot prove absence of upstream packet loss, nor
    that a recognizer subsequently consumed every delivered frame.
    """

    def __init__(
        self,
        *,
        source: AsyncIterable[rtc.AudioFrame],
        session_id: str,
        endpoint_id: str,
        sample_rate: int,
        max_seconds: float = 30,
    ) -> None:
        if not session_id or not endpoint_id or sample_rate <= 0 or not 0 < max_seconds <= 30:
            raise ValueError("scoped input, sample rate and bounded duration required")
        self._iterator = aiter(source)
        self._session_id = session_id
        self._endpoint_id = endpoint_id
        self._source_id = uuid4().hex
        self._sample_rate = sample_rate
        self._max_samples = max(1, int(sample_rate * max_seconds))
        self._generation = 0
        self._closed = False
        self._offset = 0
        self._fault: str | None = None
        self._buffer = self._new_buffer()

    def _new_buffer(self) -> AudioEvidenceBuffer:
        return AudioEvidenceBuffer(
            source=self._source_id,
            generation=self._generation,
            sample_rate=self._sample_rate,
            max_samples=self._max_samples,
        )

    @property
    def position(self) -> AudioInputPosition:
        return AudioInputPosition(
            self._source_id,
            self._generation,
            self._sample_rate,
            self._offset,
        )

    @property
    def retained_samples(self) -> int:
        return self._buffer.retained_samples

    def reset_source(self) -> None:
        """Invalidate previous coordinates when the input owner switches tracks."""
        self._buffer.clear()
        self._generation += 1
        self._offset = 0
        self._fault = None
        self._buffer = self._new_buffer()

    def close(self) -> None:
        """Release evidence only; the SDK owns and closes the upstream iterator."""
        self._closed = True
        self.reset_source()

    def __aiter__(self):
        return self

    async def __anext__(self) -> rtc.AudioFrame:
        if self._closed:
            raise StopAsyncIteration
        generation = self._generation
        try:
            frame = await anext(self._iterator)
        except BaseException:
            self.close()
            raise
        if self._closed:
            return frame
        if generation != self._generation:
            self._fault = "input_changed_during_read"
            self._buffer.clear()
            return frame
        if (
            frame.sample_rate != self._sample_rate
            or frame.num_channels != 1
            or len(frame.data) * frame.data.itemsize != frame.samples_per_channel * 2
            or frame.samples_per_channel <= 0
        ):
            self._fault = "audio_format_changed"
            self._buffer.clear()
            return frame
        if self._fault is None:
            self._buffer.append(start=self._offset, pcm=bytes(frame.data))
        self._offset += frame.samples_per_channel
        return frame

    def snapshot(
        self,
        *,
        candidate: str,
        start: AudioInputPosition,
        end: AudioInputPosition,
    ) -> AudioEvidence:
        """Freeze only the explicit range; never substitute a retained suffix."""
        if not candidate:
            raise ValueError("candidate required")
        reason = None
        if start.source != self._source_id or end.source != self._source_id:
            reason = "input_source_changed"
        elif start.generation != self._generation or end.generation != self._generation:
            reason = "source_generation_changed"
        elif self._closed:
            reason = "input_closed"
        elif self._fault:
            reason = self._fault
        elif start.sample_rate != self._sample_rate or end.sample_rate != self._sample_rate:
            reason = "audio_format_changed"
        elif start.sample < 0 or end.sample <= start.sample:
            reason = "audio_range_missing"
        if reason:
            return AudioEvidence(
                source=start.source,
                generation=start.generation,
                candidate=candidate,
                sample_rate=start.sample_rate,
                start=start.sample,
                end=end.sample,
                complete=False,
                reason=reason,
                pcm=b"",
                session_id=self._session_id,
                endpoint_id=self._endpoint_id,
            )
        return replace(
            self._buffer.snapshot(
                candidate=candidate,
                start=start.sample,
                end=end.sample,
                generation=start.generation,
            ),
            session_id=self._session_id,
            endpoint_id=self._endpoint_id,
        )
