"""Speculate on sample-addressed audio; ASR still validates turn ownership.

The normalized Provider input is the only coordinate. A second *state* of the
existing VAD consumes it without emitting semantic/interrupt pipeline callbacks.
There is no second microphone reader, person history, vote, or unbounded queue.
"""

import asyncio
import logging
from dataclasses import dataclass

from livekit import rtc
from livekit.agents.vad import VADEventType
from eidolon.livekit.common.audio_timeline import AudioSpan


@dataclass
class PrefetchedAudio:
    candidate_id: str
    generation: int
    evidence: object
    validated_through: int
    voice_end: int


class SpeakerAudioPrefetch:
    quiet_samples = 1600  # 100ms: speculative only, never an endpoint decision.
    interval_samples = 8000  # At most two launches per second.
    max_lag_samples = 16000

    def __init__(self, *, capture, observer, coordinator, stream_factory, on_voice=None):
        self.capture, self.observer, self.coordinator = capture, observer, coordinator
        self.stream_factory = stream_factory
        # Told when this VAD state starts or stops hearing speech. It reads the
        # audio STT reads, with state of its own, so it can hear quiet speech
        # the session's VAD misses (interaction facts use it for that).
        self.on_voice = on_voice
        self.voice = False
        self.source = self.stream = self.task = None
        self.record = None
        self._closing = set()
        self._received = self._processed = 0
        self._start = self._voice_end = None
        self._last_launch = -self.interval_samples
        self._candidate = None
        self._candidate_start = None

    def source_started(self, source):
        self.source_ended()
        if source.sample_rate != 16000:
            return
        self.source = source
        self.stream = self.stream_factory()
        self.task = asyncio.create_task(self._consume(self.stream, source))

    def source_ended(self):
        self._set_voice(False)
        self.record = None
        if self.source is not None:
            self.observer.cancel()
        self.source = None
        self._start = self._voice_end = self._candidate = self._candidate_start = None
        self._received = self._processed = 0
        self._last_launch = -self.interval_samples
        if self.task is not None:
            self.task.cancel()
        if self.stream is not None:
            task = asyncio.create_task(self.stream.aclose())
            self._closing.add(task)
            task.add_done_callback(self._closing.discard)
        self.stream = self.task = None

    def append(self, source, *, start, pcm):
        if source != self.source or self.stream is None:
            return
        end = start + len(pcm) // 2
        if start != self._received or end - self._processed > self.max_lag_samples:
            # Never drop frames and then pretend the remaining coordinates match.
            self.source_ended()
            return
        self._received = end
        self.stream.push_frame(
            rtc.AudioFrame(
                data=pcm, sample_rate=16000, num_channels=1, samples_per_channel=len(pcm) // 2
            )
        )

    async def _consume(self, stream, source):
        try:
            async for event in stream:
                if source != self.source:
                    return
                self.consume(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            logging.getLogger(__name__).warning("Speaker audio prefetch unavailable")
            if source == self.source:
                self.source_ended()

    def consume(self, event):
        index = event.samples_index
        self._processed = index
        if event.type == VADEventType.START_OF_SPEECH:
            # START carries actual buffered prefix in input sample coordinates.
            count = sum(f.samples_per_channel for f in event.frames)
            self._start = max(0, index - count)
            self._voice_end = index
            self._set_voice(True)
        elif event.type == VADEventType.INFERENCE_DONE:
            if event.probability >= 0.5:
                self._voice_end = index
                # An old prefix loses adoption eligibility immediately, even if
                # the rate limiter postpones its replacement computation.
                if self.record is not None and index > self.record.voice_end:
                    self.record = None
                    self.observer.cancel()
            else:
                if self.record is not None:
                    self.record.validated_through = index
                if (
                    self._start is not None
                    and self._voice_end is not None
                    and index - self._voice_end >= self.quiet_samples
                ):
                    self._prefetch(index)
        elif event.type == VADEventType.END_OF_SPEECH:
            self._start = None
            self._set_voice(False)

    def _set_voice(self, speaking):
        if speaking == self.voice:
            return
        self.voice = speaking
        if self.on_voice is not None:
            self.on_voice(speaking)

    def _prefetch(self, index):
        candidate = self.coordinator.active
        if candidate is None or candidate.state != "open" or candidate.timeline is None:
            return
        key = (candidate.candidate_id, candidate.latest_generation_id)
        if self.record is not None and (self.record.candidate_id, self.record.generation) == key:
            self.record.validated_through = index
            return
        if index - self._last_launch < self.interval_samples:
            return
        if self._candidate != candidate.candidate_id:
            self._candidate = candidate.candidate_id
            self._candidate_start = self._start
        else:
            self._candidate_start = min(self._candidate_start, self._start)
        span = AudioSpan(
            self.source.source,
            self.source.generation,
            16000,
            self._candidate_start,
            self._voice_end + self.quiet_samples,
        )
        evidence = self.capture.snapshot(candidate=candidate.candidate_id, span=span)
        if not evidence.complete:
            return
        self.record = PrefetchedAudio(*key, evidence, index, self._voice_end)
        self._last_launch = index
        self.observer.begin_evidence(evidence, candidate.timeline.turn_id)

    def resolve(self, candidate, span):
        """Only reuse when every ASR sample is covered or verified trailing silence."""
        if self.mismatch(candidate, span) is not None:
            return None
        return self.record.evidence

    def mismatch(self, candidate, span):
        """Why ``resolve`` would not reuse the prefetched audio for ``span``; None if it would.

        A short reason with the offset in ms where one applies, for the log
        only: the turn then pays a whole recognition after it is committed.
        """
        record = self.record
        if record is None or candidate is None or span is None:
            return "no_prefetch"
        evidence = record.evidence

        def ms(samples):
            return int(samples * 1000 // 16000)

        if record.candidate_id != candidate.candidate_id:
            return "other_candidate"
        if record.generation != candidate.latest_generation_id:
            return "speech_continued"
        if self.source != self.capture.active_source or not span.same_source(self.source):
            return "source_changed"
        if span.start < evidence.start:
            return f"asr_starts_{ms(evidence.start - span.start)}ms_before_prefetch"
        if span.end > record.validated_through:
            return f"asr_ends_{ms(span.end - record.validated_through)}ms_after_verified_silence"
        if span.end <= evidence.start or span.start >= record.voice_end:
            return "asr_outside_prefetch"
        if span.end > self._processed:
            return f"vad_{ms(span.end - self._processed)}ms_behind_asr"
        return None

    async def aclose(self):
        task = self.task
        self.source_ended()
        pending = list(self._closing)
        if task is not None:
            pending.append(task)
        await asyncio.gather(*pending, return_exceptions=True)
