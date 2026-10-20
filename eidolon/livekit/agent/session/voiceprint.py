"""Observe-only voiceprint verification for completed user turns."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any

from eidolon_sdk.biz.persona import ResolvedRuntimeIdentity as ResolvedContext

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.runtime.resolver import DeviceTokenResolverError
from eidolon.livekit.agent.session.voiceprint_reasons import (
    VOICEPRINT_CONTEXT_ERROR_PREFIX,
    VOICEPRINT_REASON_CACHED_OWNER_CONTEXT,
    VOICEPRINT_REASON_OWNER_HIGH_CONFIDENCE,
    VOICEPRINT_REASON_OWNER_KNOWN_SHORT_AUDIO,
    VOICEPRINT_REASON_SCORE_MISSING,
    VOICEPRINT_REASON_SPEAKER_NOT_OWNER,
    VOICEPRINT_REASON_TRUSTED_PAIRED_DEVICE,
    voiceprint_failopen_reason,
    voiceprint_owner_above_provider_threshold_reason,
)
from eidolon.livekit.agent.speaker_verification import SpeakerVerificationService
from eidolon.livekit.common.speaker_verification import SpeakerSignal
from eidolon.livekit.common.config import VoiceprintConfig

logger = logging.getLogger("agent.session.voiceprint")

ContextResolver = Callable[[Any], Awaitable[ResolvedContext]]
_DEFAULT_VOICEPRINT_TENANT_ID = "default"


@dataclass
class _ActiveVoiceprintTurn:
    turn_id: str
    timeline: TurnTimeline
    started_at: float = field(default_factory=time.monotonic)
    chunks: list[bytes] = field(default_factory=list)
    sample_rate: int = 16000
    samples: int = 0
    overflow: bool = False

    @property
    def audio_ms(self) -> int:
        if self.sample_rate <= 0:
            return 0
        return int(self.samples * 1000 / self.sample_rate)

    def append(self, frame: Any, *, max_audio_ms: int) -> None:
        sample_rate = int(getattr(frame, "sample_rate", self.sample_rate) or self.sample_rate)
        if sample_rate != self.sample_rate and not self.chunks:
            self.sample_rate = sample_rate
        if sample_rate != self.sample_rate:
            return
        samples = int(getattr(frame, "samples_per_channel", 0) or 0)
        if samples <= 0:
            data = bytes(getattr(frame, "data", b""))
            samples = len(data) // 2
        if (self.samples + samples) * 1000 > max_audio_ms * self.sample_rate:
            self.overflow = True
            return
        data = bytes(getattr(frame, "data", b""))
        if not data:
            return
        self.chunks.append(data)
        self.samples += samples

    def audio_bytes(self) -> bytes:
        return b"".join(self.chunks)


@dataclass(frozen=True)
class VoiceprintTurnResult:
    """Voiceprint outcome plus conservative MVP commit-gate decision."""

    signal: SpeakerSignal
    cached: bool = False
    commit_allowed: bool = False
    commit_reason: str = ""


class VoiceprintTurnObserver:
    """Capture per-turn user audio and verify it without blocking the pipeline."""

    def __init__(
        self,
        *,
        service: SpeakerVerificationService | None,
        sample_rate: int = 16000,
        max_audio_ms: int | None = None,
        accept_cache_ttl_sec: float | None = None,
        accept_cache_short_audio_max_ms: int | None = None,
        commit_threshold: float | None = None,
        owner_short_audio_bypass_ms: int | None = None,
        trust_paired_devices: bool = True,
        context_resolver: ContextResolver | None = None,
        admission=None,
    ) -> None:
        voiceprint_defaults = VoiceprintConfig()
        self._admission = admission
        self._service = service or admission
        self._sample_rate = sample_rate
        self._max_audio_ms = max_audio_ms or voiceprint_defaults.turn_max_audio_ms
        self._accept_cache_ttl_sec = (
            accept_cache_ttl_sec
            if accept_cache_ttl_sec is not None
            else voiceprint_defaults.accept_cache_ttl_ms / 1000.0
        )
        self._accept_cache_short_audio_max_ms = (
            accept_cache_short_audio_max_ms
            if accept_cache_short_audio_max_ms is not None
            else voiceprint_defaults.accept_cache_short_audio_max_ms
        )
        self._commit_threshold = (
            commit_threshold
            if commit_threshold is not None
            else voiceprint_defaults.owner_commit_threshold
        )
        self._owner_short_audio_bypass_ms = (
            owner_short_audio_bypass_ms
            if owner_short_audio_bypass_ms is not None
            else voiceprint_defaults.owner_short_audio_bypass_ms
        )
        self._trust_paired_devices = trust_paired_devices
        self._context_resolver = context_resolver
        self._room: Any | None = None
        self._active: _ActiveVoiceprintTurn | None = None
        self._stream_tasks: set[asyncio.Task] = set()
        self._verify_tasks: set[asyncio.Task] = set()
        self._context_cache: ResolvedContext | None = None
        self._accept_cache: dict[tuple[str, str], tuple[float, SpeakerSignal]] = {}
        self._participant_identity: str | None = None
        self._observed_tracks: set[str] = set()
        self._pre_roll = bytearray()

    async def authorize(self, result) -> bool:
        if self._admission is None:
            return True
        try:
            ctx = await self._resolve_context()
            return await self._admission.authorize(ctx.companion_id, result)
        except Exception:
            return False

    async def authorize_interruption(self, completed=None) -> bool:
        if self._admission is None:
            return True
        active = self._active
        if active is None:
            if completed is not None and getattr(completed, "task", None) is not None:
                try:
                    result = await asyncio.shield(completed.task)
                except Exception:
                    return False
                return await self.authorize(result)
            return await self.authorize(getattr(completed, "result", None))
        try:
            ctx = await self._resolve_context()
            result = await self._admission.verify(ctx.companion_id, b"" if active.overflow else active.audio_bytes(), active.sample_rate)
            if self._active is not active:
                return False
            allowed = await self.authorize(result)
            return allowed and self._active is active
        except Exception:
            return False

    def install(self, room: Any, *, participant_identity: str | None = None) -> None:
        """Subscribe to remote audio tracks for observe-only capture."""
        if self._service is None:
            return
        if self._room is room:
            return
        self._room = room
        if participant_identity is not None:
            self._participant_identity = participant_identity

        @room.on("track_subscribed")
        def _on_track_subscribed(track: Any, publication: Any, participant: Any) -> None:
            del publication
            self._maybe_start_audio_stream(track, participant)

        for participant in getattr(room, "remote_participants", {}).values():
            for publication in getattr(participant, "track_publications", {}).values():
                track = getattr(publication, "track", None)
                if track is not None:
                    self._maybe_start_audio_stream(track, participant)

    def start_turn(self, *, timeline: TurnTimeline) -> None:
        if self._service is None:
            timeline.set_attr("voiceprint", {"status": "disabled"})
            return
        self._active = _ActiveVoiceprintTurn(
            turn_id=timeline.turn_id,
            timeline=timeline,
            sample_rate=self._sample_rate,
        )
        if self._admission is not None and self._pre_roll:
            self._active.chunks.append(bytes(self._pre_roll))
            self._active.samples = len(self._pre_roll) // 2
            self._pre_roll.clear()
        timeline.set_attr(
            "voiceprint",
            {
                "status": "capturing",
                "sample_rate": self._sample_rate,
                "max_audio_ms": self._max_audio_ms,
            },
        )

    def append_frame(self, frame: Any) -> None:
        active = self._active
        if active is None:
            if self._admission is not None:
                if len(self._stream_tasks) > 1 or getattr(frame, "sample_rate", self._sample_rate) != self._sample_rate:
                    self._pre_roll.clear()
                else:
                    # 200 ms of the selected input preserves the VAD onset;
                    # never retain a previous completed turn as identity proof.
                    self._pre_roll.extend(bytes(getattr(frame, "data", b"")))
                    del self._pre_roll[:-self._sample_rate * 2 // 5]
            return
        if self._admission is not None and len(self._stream_tasks) > 1:
            # Never use one track's identity to authorize another track's text.
            # Ambiguous multi-track turns require fresh, unambiguous evidence.
            active.overflow = True
            return
        active.append(frame, max_audio_ms=self._max_audio_ms)

    def finish_turn(self) -> asyncio.Task[VoiceprintTurnResult] | None:
        active = self._active
        self._active = None
        self._pre_roll.clear()
        if active is None or self._service is None:
            return None
        task = asyncio.create_task(self._verify_turn(active))
        self._verify_tasks.add(task)
        task.add_done_callback(self._verify_tasks.discard)
        return task

    async def aclose(self) -> None:
        self._pre_roll.clear()
        for task in list(self._stream_tasks):
            task.cancel()
        for task in list(self._verify_tasks):
            task.cancel()
        if self._stream_tasks:
            await asyncio.gather(*self._stream_tasks, return_exceptions=True)
        if self._verify_tasks:
            await asyncio.gather(*self._verify_tasks, return_exceptions=True)
        self._stream_tasks.clear()
        self._verify_tasks.clear()

    def _maybe_start_audio_stream(self, track: Any, participant: Any) -> None:
        identity = str(getattr(participant, "identity", ""))
        if self._participant_identity is not None and identity != self._participant_identity:
            return
        track_id = str(getattr(track, "sid", None) or id(track))
        if track_id in self._observed_tracks:
            return
        try:
            from livekit import rtc

            audio_kind = getattr(rtc.TrackKind, "KIND_AUDIO", 1)
            if getattr(track, "kind", None) != audio_kind:
                return
            stream = rtc.AudioStream(
                track,
                sample_rate=self._sample_rate,
                num_channels=1,
                capacity=32,
            )
            self._participant_identity = identity
        except Exception:
            logger.exception(
                "[VoiceprintTurnObserver] failed to start audio stream participant=%s",
                getattr(participant, "identity", ""),
            )
            return
        task = asyncio.create_task(self._consume_audio_stream(stream, participant))
        self._observed_tracks.add(track_id)
        self._stream_tasks.add(task)
        task.add_done_callback(self._stream_tasks.discard)
        task.add_done_callback(lambda _: self._observed_tracks.discard(track_id))
        logger.info(
            "[VoiceprintTurnObserver] observing audio track participant=%s",
            getattr(participant, "identity", ""),
        )

    async def _consume_audio_stream(self, stream: Any, participant: Any) -> None:
        try:
            async for event in stream:
                frame = getattr(event, "frame", None)
                if frame is not None:
                    self.append_frame(frame)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "[VoiceprintTurnObserver] audio stream failed participant=%s",
                getattr(participant, "identity", ""),
            )
        finally:
            close = getattr(stream, "aclose", None)
            if callable(close):
                try:
                    await close()
                except Exception:
                    logger.debug(
                        "[VoiceprintTurnObserver] audio stream close failed",
                        exc_info=True,
                    )

    async def _verify_turn(self, turn: _ActiveVoiceprintTurn) -> VoiceprintTurnResult:
        timeline = turn.timeline
        audio = turn.audio_bytes()
        audio_ms = turn.audio_ms
        try:
            ctx = await self._resolve_context()
        except Exception as exc:  # noqa: BLE001 - observe-only failure
            signal = SpeakerSignal.error_signal(
                provider=self._provider_name(),
                model=self._model_name(),
                error=f"context_error:{type(exc).__name__}",
                audio_ms=audio_ms,
            )
            return self._record_result(
                timeline,
                signal,
                cached=False,
                trusted_paired_device=False,
            )

        if self._admission is not None:
            result = await self._admission.verify(ctx.companion_id, b"" if turn.overflow else audio, turn.sample_rate)
            timeline.set_attr("voiceprint", {
                **result.signal.as_timeline_attrs(), "commit_allowed": result.commit_allowed,
                "commit_reason": result.commit_reason, "policy_revision": result.policy_revision,
            })
            return result

        trusted_paired_device = self._is_trusted_paired_device(ctx)
        tenant_id = self._voiceprint_tenant_id(ctx)
        user_id = self._voiceprint_user_id(ctx)
        if not user_id:
            signal = SpeakerSignal.error_signal(
                provider=self._provider_name(),
                model=self._model_name(),
                error="context_error:missing_owner_id",
                audio_ms=audio_ms,
            )
            return self._record_result(
                timeline,
                signal,
                cached=False,
                trusted_paired_device=trusted_paired_device,
            )
        if not audio:
            signal = SpeakerSignal.error_signal(
                provider=self._provider_name(),
                model=self._model_name(),
                error="audio_not_captured",
                audio_ms=audio_ms,
            )
            return self._record_result(
                timeline,
                signal,
                cached=False,
                trusted_paired_device=trusted_paired_device,
            )

        cache_key = (tenant_id, user_id)
        cached = self._cached_accept(cache_key)
        if cached is not None and audio_ms < self._accept_cache_short_audio_max_ms:
            return self._record_result(
                timeline,
                replace(cached, audio_ms=audio_ms, latency_ms=0.0),
                cached=True,
                trusted_paired_device=trusted_paired_device,
            )

        signal = await self._service.verify_turn(
            tenant_id=tenant_id,
            user_id=user_id,
            audio=audio,
            sample_rate=turn.sample_rate,
            audio_ms=audio_ms,
        )
        if signal.known:
            self._accept_cache[cache_key] = (
                time.monotonic() + self._accept_cache_ttl_sec,
                signal,
            )
        return self._record_result(
            timeline,
            signal,
            cached=False,
            trusted_paired_device=trusted_paired_device,
        )

    async def _resolve_context(self) -> ResolvedContext:
        if self._context_cache is not None:
            return self._context_cache
        if self._context_resolver is not None:
            self._context_cache = await self._context_resolver(self._room)
            return self._context_cache
        raise DeviceTokenResolverError("runtime context resolver missing")

    def _cached_accept(self, key: tuple[str, str]) -> SpeakerSignal | None:
        cached = self._accept_cache.get(key)
        if cached is None:
            return None
        expires_at, signal = cached
        if time.monotonic() >= expires_at:
            self._accept_cache.pop(key, None)
            return None
        return signal

    def _record_result(
        self,
        timeline: TurnTimeline,
        signal: SpeakerSignal,
        *,
        cached: bool,
        trusted_paired_device: bool = False,
    ) -> VoiceprintTurnResult:
        commit_allowed, commit_reason = self._commit_decision(
            signal,
            cached=cached,
            trusted_paired_device=trusted_paired_device,
        )
        payload = signal.as_timeline_attrs()
        payload["status"] = "verified" if signal.error is None else "error"
        payload["cached"] = cached
        payload["trusted_paired_device"] = trusted_paired_device
        payload["commit_allowed"] = commit_allowed
        payload["commit_reason"] = commit_reason
        timeline.set_attr("voiceprint", payload)
        logger.info(
            "[VoiceprintTurnObserver] turn=%s known=%s score=%s audio_ms=%s "
            "latency_ms=%s cached=%s commit_allowed=%s commit_reason=%s error=%s",
            timeline.turn_id,
            signal.known,
            signal.score,
            signal.audio_ms,
            signal.latency_ms,
            cached,
            commit_allowed,
            commit_reason,
            signal.error,
        )
        if commit_allowed and signal.error and not trusted_paired_device:
            logger.warning(
                "[VoiceprintTurnObserver] fail-open: committing turn=%s despite "
                "verify error=%s (speaker unverified — our-side capture/verify "
                "failure, not a wrong-speaker signal)",
                timeline.turn_id,
                signal.error,
            )
        return VoiceprintTurnResult(
            signal=signal,
            cached=cached,
            commit_allowed=commit_allowed,
            commit_reason=commit_reason,
        )

    def _commit_decision(
        self,
        signal: SpeakerSignal,
        *,
        cached: bool,
        trusted_paired_device: bool = False,
    ) -> tuple[bool, str]:
        if trusted_paired_device:
            return True, VOICEPRINT_REASON_TRUSTED_PAIRED_DEVICE
        if signal.error:
            # ``context_error`` means we couldn't resolve the agent context — the
            # turn literally cannot run, so it must still block (task #9 surfaces
            # it to the user). Any OTHER verify error (audio_not_captured,
            # provider/timeout, …) is an OUR-SIDE failure to verify the speaker,
            # NOT evidence of a wrong speaker — fail OPEN (allow + warn) instead
            # of silently dropping the user's turn.
            if signal.error.startswith(VOICEPRINT_CONTEXT_ERROR_PREFIX):
                return False, signal.error
            return True, voiceprint_failopen_reason(signal.error)
        if not signal.known:
            return False, VOICEPRINT_REASON_SPEAKER_NOT_OWNER
        score = signal.score if signal.score is not None else signal.owner_confidence
        if score is None:
            return False, VOICEPRINT_REASON_SCORE_MISSING
        if cached:
            return True, VOICEPRINT_REASON_CACHED_OWNER_CONTEXT
        if signal.audio_ms is not None and signal.audio_ms < self._owner_short_audio_bypass_ms:
            return True, VOICEPRINT_REASON_OWNER_KNOWN_SHORT_AUDIO
        if score < self._commit_threshold:
            return True, voiceprint_owner_above_provider_threshold_reason(
                score=score,
                threshold=self._commit_threshold,
            )
        return True, VOICEPRINT_REASON_OWNER_HIGH_CONFIDENCE

    def _is_trusted_paired_device(self, ctx: ResolvedContext) -> bool:
        if not self._trust_paired_devices:
            return False
        return bool(
            ctx.device_id and self._voiceprint_user_id(ctx) and self._voiceprint_tenant_id(ctx)
        )

    @staticmethod
    def _voiceprint_tenant_id(ctx: ResolvedContext) -> str:
        return str(getattr(ctx, "tenant_id", None) or _DEFAULT_VOICEPRINT_TENANT_ID)

    @staticmethod
    def _voiceprint_user_id(ctx: ResolvedContext) -> str:
        return str(getattr(ctx, "user_id", None) or getattr(ctx, "owner_id", None) or "")

    def _provider_name(self) -> str:
        provider = getattr(self._service, "_provider", None)
        return str(getattr(provider, "provider", "unknown"))

    def _model_name(self) -> str:
        provider = getattr(self._service, "_provider", None)
        return str(getattr(provider, "model", "unknown"))
