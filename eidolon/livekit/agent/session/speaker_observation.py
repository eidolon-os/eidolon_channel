"""Bounded, post-commit numerical observation. Never controls conversation.

There is one in-flight extraction and no audio queue, disk storage, identity
decision, score threshold, device trust or cross-turn acceptance cache.
"""

import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace

from eidolon_sdk.biz.voiceprint import ExtractedSample
from eidolon_sdk.core.http.errors import ServiceUpstreamError

from eidolon.livekit.common.audio_evidence import AudioEvidenceCapture
from .user_turn_coordinator import UserTurnCoordinator, UserTurnDecision

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeakerObservation:
    candidate_id: str
    reason: str
    session_id: str = ""
    endpoint_id: str = ""  # diagnostic source, never an enrolled identity
    source: str = ""
    generation: int = 0
    start: int = 0
    end: int = 0
    audio_ms: float = 0
    elapsed_ms: float = 0
    speech_ms: int = 0
    model_id: str = ""
    weights_hash: str = ""
    preprocessing_version: str = ""


class SpeakerTurnObservation:
    def __init__(self, *, extract: Callable[[bytes], Awaitable[ExtractedSample]],
                 record: Callable[[SpeakerObservation], None], timeout_sec: float = 8,
                 lifetime_sec: float | None = None):
        if not math.isfinite(timeout_sec) or not 0 < timeout_sec <= 8:
            raise ValueError("bounded extraction timeout required")
        self._extract, self._record, self._timeout = extract, record, timeout_sec
        self._capture: AudioEvidenceCapture | None = None
        self._pending: asyncio.Task | None = None
        self._closed = False
        if lifetime_sec is not None and (not math.isfinite(lifetime_sec) or not 0 < lifetime_sec <= 86400):
            raise ValueError("bounded observation lifetime required")
        self._deadline = (asyncio.get_running_loop().call_later(lifetime_sec, self._expire)
                          if lifetime_sec is not None else None)

    @property
    def pending(self) -> asyncio.Task | None:
        return self._pending

    def bind_capture(self, capture: AudioEvidenceCapture | None) -> None:
        if self._closed and capture is not None:
            capture.invalidate("observation_closed")
        self._capture = None if self._closed else capture

    def _expire(self) -> None:
        self._closed = True
        if self._deadline is not None:
            self._deadline.cancel()
        if self._capture is not None:
            self._capture.invalidate("observation_closed")
        self._capture = None
        if self._pending is not None:
            self._pending.cancel()

    def _emit(self, observation: SpeakerObservation) -> None:
        try:
            self._record(observation)
        except Exception:
            # Reporting is not a dependency of reply or recovery; never log
            # arbitrary callback exceptions, which may contain private data.
            logger.warning("Speaker observation reporter failed")

    def submit(self, decision: UserTurnDecision, coordinator: UserTurnCoordinator) -> None:
        """Called synchronously only after the existing owner commits a turn.

        Freeze terminal coverage here; a later candidate may start while this
        historical observation finishes. Results have no admission consumer.
        """
        candidate = coordinator.active
        if (self._closed or decision.action != "commit" or candidate is None
                or candidate.state != "committed" or candidate.candidate_id != decision.candidate_id):
            return
        result = SpeakerObservation(candidate_id=candidate.candidate_id, reason="")
        capture = self._capture
        source = capture.active_source if capture else None
        if source is None:
            self._emit(replace(result, reason="audio_source_unavailable"))
            return
        reference = coordinator.capture_committed_audio_reference(
            candidate_id=candidate.candidate_id, source=source)
        if reference.span is None:
            self._emit(replace(result, reason=reference.reason))
            return
        if self._pending is not None:
            self._emit(replace(result, reason="busy"))
            return
        evidence = capture.snapshot(candidate=candidate.candidate_id, span=reference.span)
        result = replace(result, session_id=evidence.session_id, endpoint_id=evidence.endpoint_id,
            source=evidence.source, generation=evidence.generation, start=evidence.start,
            end=evidence.end, audio_ms=(evidence.end - evidence.start) * 1000 / evidence.sample_rate)
        reason = evidence.reason
        if evidence.sample_rate != 16000:
            reason = "unsupported_sample_rate"
        elif evidence.complete and result.audio_ms < 1500:
            reason = "audio_too_short"
        if reason:
            self._emit(replace(result, reason=reason))
            return

        async def observe():
            started = time.monotonic()
            outcome = result
            try:
                async with asyncio.timeout(self._timeout):
                    sample = await self._extract(evidence.pcm)
                if self._capture is not capture or capture.active_source != source:
                    outcome = replace(result, reason="audio_source_changed")
                else:
                    outcome = replace(result, reason="extracted", speech_ms=sample.speech_ms,
                        model_id=sample.model_id, weights_hash=sample.weights_hash,
                        preprocessing_version=sample.preprocessing_version)
            except TimeoutError:
                outcome = replace(result, reason="timeout")
            except asyncio.CancelledError:
                raise
            except ServiceUpstreamError as exc:
                # Status is enough for operational diagnosis. Never publish an
                # arbitrary response body, embedding, request or exception text.
                outcome = replace(result, reason={
                    401: "provider_unauthorized", 422: "audio_rejected",
                    429: "provider_busy", 503: "provider_unavailable", 504: "provider_timeout",
                }.get(exc.status_code, "extraction_failed"))
            except Exception:
                outcome = replace(result, reason="extraction_failed")
            finally:
                self._pending = None
            if not self._closed:
                self._emit(replace(outcome, elapsed_ms=(time.monotonic() - started) * 1000))

        self._pending = asyncio.create_task(observe())

    async def aclose(self) -> None:
        self._expire()
        task, self._pending = self._pending, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
