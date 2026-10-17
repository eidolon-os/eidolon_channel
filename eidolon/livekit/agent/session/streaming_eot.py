"""Native audio EOT lifecycle with Channel's existing pre-commit barrier.

The SDK owns audio buffering, VAD prediction triggers and resampling. Channel
owns only the validity of a prediction and settling its interruption candidate.
No text model, second audio queue or reply scheduler runs behind this adapter.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.voice.turn import TurnDetectionEvent

from eidolon.livekit.agent.turn_policy import eot_kwargs_from_turn_policy
from eidolon.livekit.common.config import TurnPolicyConfig
from eidolon.livekit.plugins.eot.config import EidolonEOTConfig

logger = logging.getLogger(__name__)


class MiniEotSession:
    """Session-local audio evidence; interrupt decisions remain in TurnPolicyRuntime.

The legacy pipeline reads its playback policy from ``_config``. Reuse that
configuration mapping without constructing FireRed's model or text scorer.
Text revisions cannot manufacture or refresh an audio probability.
"""

    model = "turn-detector-v1-mini"
    provider = "livekit"

    def __init__(self, policy: TurnPolicyConfig) -> None:
        if not policy.uses_channel_model_intent:
            raise ValueError("livekit_mini requires Channel-owned model intent")
        self._config = EidolonEOTConfig(**eot_kwargs_from_turn_policy(policy))
        self.policy = policy
        self.current_eot_score = 0.0

    def update_vad(self, is_active: bool) -> None:
        if is_active:
            self.reset()

    def update_vad_probability(self, probability: float) -> None:
        # Native SDK receives the original VAD events, not this legacy text hook.
        pass

    def update_asr(self, text: str, is_final: bool) -> float:
        return self.current_eot_score

    def start_session(self, session_id: str) -> None:
        self.reset()

    def end_session(self, session_id: str | None = None) -> None:
        self.reset()

    def reset(self) -> None:
        self.current_eot_score = 0.0

    def detector(self, *, settle: Callable[[], Awaitable[None]],
                 begin_observation: Callable[[], Any],
                 bind_scope: Callable[[], Callable[[], None]] | None = None) -> NativeAudioTurnDetector:
        from livekit.agents.inference import TurnDetector

        options = {}
        if self.policy.eot.mini_unlikely_threshold is not None:
            options["unlikely_threshold"] = self.policy.eot.mini_unlikely_threshold
        return NativeAudioTurnDetector(
            TurnDetector(version="v1-mini", **options),
            settle=settle, begin_observation=begin_observation,
            bind_scope=bind_scope,
            accept_score=lambda value: setattr(self, "current_eot_score", value),
            reset_score=self.reset,
            # The SDK's one-second inference timeout must not bypass the existing
            # Channel evidence barrier. This is a ceiling, never an added sleep.
            settlement_budget=self.policy.interrupt.post_speech_evidence_timeout_ms / 1000,
        )


class NativeAudioTurnDetector:
    """Preserve the SDK streaming protocol instead of disguising it as text EOT."""

    def __init__(self, detector: Any, *, settle: Callable[[], Awaitable[None]],
                 begin_observation: Callable[[], Any], accept_score: Callable[[float], None],
                 reset_score: Callable[[], None], settlement_budget: float,
                 bind_scope: Callable[[], Callable[[], None]] | None = None) -> None:
        self._detector = detector
        self._options = dict(settle=settle, begin_observation=begin_observation,
                             accept_score=accept_score, reset_score=reset_score,
                             settlement_budget=settlement_budget, bind_scope=bind_scope)

    @property
    def model(self) -> str:
        return self._detector.model

    @property
    def provider(self) -> str:
        return self._detector.provider

    def stream(self, *, conn_options=DEFAULT_API_CONNECT_OPTIONS) -> NativeAudioTurnStream:
        return NativeAudioTurnStream(self._detector.stream(conn_options=conn_options), **self._options)


class NativeAudioTurnStream:
    def __init__(self, stream: Any, *, settle: Callable[[], Awaitable[None]],
                 begin_observation: Callable[[], Any], accept_score: Callable[[float], None],
                 reset_score: Callable[[], None], settlement_budget: float,
                 bind_scope: Callable[[], Callable[[], None]] | None = None) -> None:
        self._stream = stream
        self._settle = settle
        self._begin_observation = begin_observation
        self._accept_score = accept_score
        self._reset_score = reset_score
        self._settlement_budget = settlement_budget
        self._bind_scope = bind_scope
        self._task: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()
        self._epoch = 0
        self._closed = False

    @property
    def model(self) -> str:
        return self._stream.model

    @property
    def provider(self) -> str:
        return self._stream.provider

    @property
    def is_fallback(self) -> bool:
        return self._stream.is_fallback

    @property
    def prediction_timeout(self) -> float:
        return self._stream.prediction_timeout + self._settlement_budget

    async def supports_language(self, language: str | None) -> bool:
        return await self._stream.supports_language(language)

    async def unlikely_threshold(self, language: str | None) -> float | None:
        return await self._stream.unlikely_threshold(language)

    async def backchannel_threshold(self, language: str | None) -> float | None:
        return await self._stream.backchannel_threshold(language)

    def push_audio(self, frame: Any) -> None:
        if not self._closed:
            self._stream.push_audio(frame)

    def predict(self) -> asyncio.Future[TurnDetectionEvent]:
        self.cancel_inference()
        epoch = self._epoch
        check_scope = self._bind_scope() if self._bind_scope is not None else lambda: None
        native_future = None if self._closed else self._stream.predict()
        task = asyncio.create_task(self._predict(native_future, epoch, check_scope))
        self._task = task
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _predict(self, future: Any, epoch: int, check_scope: Callable[[], None]) -> TurnDetectionEvent:
        finish = None
        try:
            finish = self._begin_observation()
        except Exception:
            logger.debug("endpointing observation unavailable", exc_info=True)
        outcome, score = "unavailable", None
        event = TurnDetectionEvent(type="eot_prediction", end_of_turn_probability=0.0,
                                   last_speaking_time=time.time())
        try:
            if future is not None:
                try:
                    async with asyncio.timeout(self._stream.prediction_timeout):
                        result = await asyncio.shield(future)
                    value = result.end_of_turn_probability
                    # Closed/superseded SDK streams return default probabilities
                    # without inference_duration; never label these model evidence.
                    if result.inference_duration is not None and math.isfinite(value) and 0 <= value <= 1:
                        event, score, outcome = result, value, "returned"
                except TimeoutError:
                    self._stream.cancel_inference(timed_out=True)
                    outcome = "timeout"
                except Exception:
                    logger.exception("native audio EOT unavailable")
            if epoch != self._epoch or self._closed:
                raise asyncio.CancelledError("audio prediction invalidated")
            check_scope()
            # Native inference may begin at the first VAD pause. Its result cannot
            # let the SDK skip an uninterruptible old SpeechHandle before Channel
            # consumes final intent and completes that handle's cancellation.
            await self._settle()
            if epoch != self._epoch or self._closed:
                raise asyncio.CancelledError("audio prediction invalidated")
            check_scope()
            self._accept_score(score if score is not None else 0.0)
            return event if score is not None else replace(event, end_of_turn_probability=0.0)
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        finally:
            if finish is not None:
                try:
                    finish(outcome, score)
                except Exception:
                    logger.debug("endpointing observation unavailable", exc_info=True)

    def cancel_inference(self, *, timed_out: bool = False) -> None:
        self._epoch += 1
        self._reset_score()
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._stream.cancel_inference(timed_out=timed_out)

    def flush(self, reason: str | None = None) -> None:
        self.cancel_inference()
        self._stream.flush(reason=reason)

    def end_input(self) -> None:
        self._closed = True
        self.cancel_inference()
        self._stream.end_input()

    async def aclose(self) -> None:
        self.end_input()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self._stream.aclose()
