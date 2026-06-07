"""Real LiveKit room benchmark runner.

This runner exercises the actual room boundary: a benchmark participant joins a
LiveKit room with agent dispatch, publishes microphone audio, subscribes to the
agent's audio, and records room-level latency. It assumes the LiveKit server and
the Eidolon agent worker are already running.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from livekit import api as lk_api
from livekit import rtc

from eidolon.livekit.agent.client_audio_state import CLIENT_AUDIO_STATE_TOPIC
from eidolon.livekit.common.config import load_effective_config
from eidolon.livekit.tests._harness.audio import frames_from_pcm, synth_silence

from .audio_assets import load_clip_pcm
from .realcall import provider_config_from_cfg
from .schema import BenchmarkCase, BenchmarkSuite, CaseResult, RunResult


@dataclass(frozen=True)
class LiveKitRoomOptions:
    timeout_sec: float = 45.0
    settle_after_first_audio_sec: float = 2.0
    agent_ready_timeout_sec: float = 12.0
    agent_quiet_ms: int = 800
    agent_speaking_wait_sec: float = 8.0
    agent_name: str = "eidolon"
    participant_prefix: str = "voice-bench"
    participant_identity: str | None = None
    participant_kind: str = "user"
    participant_metadata: dict[str, Any] | None = None
    room_prefix: str = "voice-bench"
    agent_missing_retry_count: int = 1


_AGENT_MISSING_ERROR = "timed out waiting for agent participant before user audio"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


async def run_livekit_room_suite(
    suites: list[BenchmarkSuite],
    *,
    root: Path,
    run_id: str | None = None,
    options: LiveKitRoomOptions | None = None,
) -> RunResult:
    cfg = load_effective_config()
    options = options or LiveKitRoomOptions()
    results: list[CaseResult] = []
    for suite in suites:
        for case in suite.cases:
            results.append(
                await _run_room_case_with_retries(
                    case,
                    root=root,
                    options=options,
                    livekit_url=cfg.core.livekit_url,
                    api_key=cfg.core.api_key,
                    api_secret=cfg.core.api_secret,
                )
            )

    return RunResult(
        run_id=run_id or time.strftime("%Y%m%d-%H%M%S"),
        git_sha=_git_sha(),
        runner="livekit_room",
        profile=(
            "real_livekit_room:"
            f"url={_redact_url(cfg.core.livekit_url)},"
            f"agent={options.agent_name}"
        ),
        cases=results,
        provider_config=provider_config_from_cfg(cfg),
    )


def write_livekit_room_outputs(run: RunResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run.write_jsonl(output_dir / "livekit_room_results.jsonl")


async def _run_room_case_with_retries(
    case: BenchmarkCase,
    *,
    root: Path,
    options: LiveKitRoomOptions,
    livekit_url: str,
    api_key: str,
    api_secret: str,
) -> CaseResult:
    result = await _run_room_case(
        case,
        root=root,
        options=options,
        livekit_url=livekit_url,
        api_key=api_key,
        api_secret=api_secret,
    )
    for attempt in range(max(0, options.agent_missing_retry_count)):
        if not _should_retry_room_case(result):
            return result
        retry_event = {
            "type": "case_retry",
            "attempt": attempt + 1,
            "reason": "agent_missing",
            "previous_room_name": result.metrics.get("room_name"),
            "previous_errors": list(result.errors),
        }
        result = await _run_room_case(
            case,
            root=root,
            options=options,
            livekit_url=livekit_url,
            api_key=api_key,
            api_secret=api_secret,
        )
        result.events.insert(0, retry_event)
        result.metrics["retry_attempts"] = attempt + 1
    return result


def _should_retry_room_case(result: CaseResult) -> bool:
    return _AGENT_MISSING_ERROR in result.errors


async def _run_room_case(
    case: BenchmarkCase,
    *,
    root: Path,
    options: LiveKitRoomOptions,
    livekit_url: str,
    api_key: str,
    api_secret: str,
) -> CaseResult:
    started = time.monotonic()
    metrics: dict[str, float | int | str | bool | None] = {}
    events: list[dict[str, Any]] = []
    errors: list[str] = []
    room_name = f"{options.room_prefix}-{case.case_id}-{uuid.uuid4().hex[:8]}"
    participant = (
        options.participant_identity
        or f"{options.participant_prefix}-{uuid.uuid4().hex[:8]}"
    )
    token = _make_dispatch_token(
        api_key=api_key,
        api_secret=api_secret,
        room_name=room_name,
        participant=participant,
        agent_name=options.agent_name,
        metadata=_participant_metadata(options),
    )

    room = rtc.Room()
    state = _RoomCaseState(started=started, events=events)
    audio_tasks: list[asyncio.Task] = []

    @room.on("participant_connected")
    def _on_participant_connected(participant_obj) -> None:
        state.mark("participant_connected_at")
        events.append(
            {
                "type": "participant_connected",
                "timestamp_ms": _elapsed_ms(started),
                "identity": getattr(participant_obj, "identity", ""),
            }
        )

    @room.on("track_subscribed")
    def _on_track_subscribed(track, publication, participant_obj) -> None:
        events.append(
            {
                "type": "track_subscribed",
                "timestamp_ms": _elapsed_ms(started),
                "participant": getattr(participant_obj, "identity", ""),
                "kind": str(getattr(track, "kind", "")),
                "source": str(getattr(publication, "source", "")),
            }
        )
        if getattr(track, "kind", None) == rtc.TrackKind.KIND_AUDIO:
            state.mark("agent_track_subscribed_at")
            audio_tasks.append(asyncio.create_task(_consume_agent_audio(track, state)))

    @room.on("transcription_received")
    def _on_transcription_received(segments, participant_obj, publication) -> None:
        state.mark("transcript_first_at")
        for segment in segments:
            text = getattr(segment, "text", "")
            is_final = bool(getattr(segment, "final", False))
            if is_final:
                state.mark("transcript_final_at")
            events.append(
                {
                    "type": "transcription",
                    "timestamp_ms": _elapsed_ms(started),
                    "participant": getattr(participant_obj, "identity", ""),
                    "text": text,
                    "final": is_final,
                    "track_sid": getattr(publication, "sid", ""),
                }
            )

    try:
        await asyncio.wait_for(room.connect(livekit_url, token), timeout=10.0)
        state.mark("room_connected_at")

        source = rtc.AudioSource(sample_rate=16_000, num_channels=1, queue_size_ms=1000)
        track = rtc.LocalAudioTrack.create_audio_track("voice-benchmark", source)
        publish_options = rtc.TrackPublishOptions()
        publish_options.source = rtc.TrackSource.SOURCE_MICROPHONE
        await room.local_participant.publish_track(track, publish_options)
        state.mark("local_track_published_at")
        await _publish_client_audio_state(
            room.local_participant,
            events=events,
            started=started,
            playback_state="idle",
        )

        try:
            await asyncio.wait_for(
                state.agent_connected.wait(),
                timeout=options.agent_ready_timeout_sec,
            )
            events.append(
                {
                    "type": "agent_ready_for_user_audio",
                    "timestamp_ms": _elapsed_ms(started),
                }
            )
        except asyncio.TimeoutError:
            errors.append("timed out waiting for agent participant before user audio")

        await _feed_case_audio(
            source,
            case=case,
            root=root,
            events=events,
            started=started,
            state=state,
            options=options,
            local_participant=room.local_participant,
        )
        state.mark("user_audio_done_at")

        try:
            response_event = (
                state.agent_audio_after_user_done
                if case.expectations.min_agent_messages > 0
                else state.first_agent_audio
            )
            await asyncio.wait_for(
                response_event.wait(),
                timeout=min(options.timeout_sec, case.timeout_sec + 20.0),
            )
            await asyncio.sleep(options.settle_after_first_audio_sec)
        except asyncio.TimeoutError:
            errors.append("timed out waiting for agent audio in LiveKit room")

        metrics.update(state.metrics())
        if state.agent_audio_frames <= 0:
            errors.append("no agent audio frames captured")
    except Exception as exc:
        metrics.update(state.metrics())
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        await room.disconnect()
        for task in audio_tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    metrics.setdefault("elapsed_ms", _elapsed_ms(started))
    metrics["room_name"] = room_name
    return CaseResult(
        case_id=case.case_id,
        suite=case.suite,
        runner="livekit_room",
        passed=not errors,
        metrics=metrics,
        events=events,
        errors=errors,
    )


async def _feed_case_audio(
    source: rtc.AudioSource,
    *,
    case: BenchmarkCase,
    root: Path,
    events: list[dict[str, Any]],
    started: float,
    state: "_RoomCaseState",
    options: LiveKitRoomOptions,
    local_participant: Any | None = None,
) -> None:
    cursor_ms = 0
    clip_paths = {clip.id: clip.path for clip in case.audio_clips}
    for step in sorted(case.user_steps, key=lambda s: s.start_ms):
        if step.start_ms > cursor_ms:
            cursor_ms += await _capture_pcm(
                source,
                synth_silence((step.start_ms - cursor_ms) / 1000),
            )
        publish_client_state = step.client_playback_state != "none"
        if step.agent_speaking:
            await _wait_for_agent_speaking(
                state,
                timeout_sec=options.agent_speaking_wait_sec,
            )
            playback_state = _step_playback_state(step, default="agent_speaking")
            if publish_client_state:
                await _publish_client_audio_state(
                    local_participant,
                    events=events,
                    started=started,
                    playback_state=playback_state,
                    ptt=step.client_ptt,
                    manual_interrupt=step.client_manual_interrupt,
                    mic_muted=step.client_mic_muted,
                )
        else:
            await _wait_for_agent_quiet(
                state,
                quiet_ms=options.agent_quiet_ms,
                timeout_sec=options.agent_speaking_wait_sec,
            )
            playback_state = _step_playback_state(step, default="idle")
            if publish_client_state:
                await _publish_client_audio_state(
                    local_participant,
                    events=events,
                    started=started,
                    playback_state=playback_state,
                    ptt=step.client_ptt,
                    manual_interrupt=step.client_manual_interrupt,
                    mic_muted=step.client_mic_muted,
                )
        rel = clip_paths.get(step.audio)
        if rel is None:
            raise ValueError(f"{case.case_id}: missing audio clip id {step.audio!r}")
        pcm, sample_rate = load_clip_pcm(root / rel)
        events.append(
            {
                "type": "user_audio_started",
                "timestamp_ms": _elapsed_ms(started),
                "text": step.text,
                "clip": step.audio,
            }
        )
        refresh_task = (
            asyncio.create_task(
                _refresh_client_audio_state(
                    local_participant,
                    events=events,
                    started=started,
                    playback_state=playback_state,
                    ptt=step.client_ptt,
                    manual_interrupt=step.client_manual_interrupt,
                    mic_muted=step.client_mic_muted,
                )
            )
            if publish_client_state
            else None
        )
        try:
            clip_ms = await _capture_pcm(source, pcm, sample_rate=sample_rate)
        finally:
            if refresh_task is not None:
                refresh_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await refresh_task
        events.append(
            {
                "type": "user_audio_finished",
                "timestamp_ms": _elapsed_ms(started),
                "text": step.text,
                "clip": step.audio,
            }
        )
        cursor_ms = max(cursor_ms, step.start_ms) + clip_ms
    await _capture_pcm(source, synth_silence(0.8))


def _step_playback_state(step: Any, *, default: str) -> str:
    if step.client_playback_state in ("idle", "agent_speaking"):
        return step.client_playback_state
    return default


async def _publish_client_audio_state(
    local_participant: Any | None,
    *,
    events: list[dict[str, Any]],
    started: float,
    playback_state: str,
    input_mode: str = "auto",
    ptt: bool = False,
    manual_interrupt: bool = False,
    mic_muted: bool = False,
) -> None:
    """Publish benchmark client audio hints through the real data channel."""
    if local_participant is None:
        return
    payload = {
        "type": "client.audio_state",
        "input_mode": input_mode,
        "ptt": ptt,
        "manual_interrupt": manual_interrupt,
        "playback_state": playback_state,
        "mic_muted": mic_muted,
        "client_ts_ms": int(time.time() * 1000),
    }
    await local_participant.publish_data(
        json.dumps(payload).encode("utf-8"),
        reliable=False,
        topic=CLIENT_AUDIO_STATE_TOPIC,
    )
    events.append(
        {
            "type": "client_audio_state_published",
            "timestamp_ms": _elapsed_ms(started),
            "playback_state": playback_state,
            "ptt": ptt,
            "manual_interrupt": manual_interrupt,
            "mic_muted": mic_muted,
            "topic": CLIENT_AUDIO_STATE_TOPIC,
        }
    )


async def _refresh_client_audio_state(
    local_participant: Any | None,
    *,
    events: list[dict[str, Any]],
    started: float,
    playback_state: str,
    ptt: bool = False,
    manual_interrupt: bool = False,
    mic_muted: bool = False,
    interval_sec: float = 0.5,
) -> None:
    while True:
        await asyncio.sleep(interval_sec)
        await _publish_client_audio_state(
            local_participant,
            events=events,
            started=started,
            playback_state=playback_state,
            ptt=ptt,
            manual_interrupt=manual_interrupt,
            mic_muted=mic_muted,
        )


async def _wait_for_agent_quiet(
    state: "_RoomCaseState",
    *,
    quiet_ms: int,
    timeout_sec: float,
) -> None:
    """Wait until the room has observed a quiet window in agent audio.

    Real room cases use ``agent_speaking: false`` to model a normal user turn,
    not an interruption of the greeting. Waiting here keeps that scenario honest
    without changing the Channel runtime.
    """
    deadline = time.monotonic() + timeout_sec
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(
            state.first_agent_audio.wait(),
            timeout=min(2.0, timeout_sec),
        )
    quiet_sec = quiet_ms / 1000
    while time.monotonic() < deadline:
        last_audio = state.last_agent_audio_monotonic
        if last_audio is None or time.monotonic() - last_audio >= quiet_sec:
            return
        await asyncio.sleep(0.05)


async def _wait_for_agent_speaking(
    state: "_RoomCaseState",
    *,
    timeout_sec: float,
) -> None:
    """Wait for currently flowing agent audio before injecting an interrupt."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if state.agent_audio_recent(window_ms=250):
            return
        await asyncio.sleep(0.02)


async def _capture_pcm(
    source: rtc.AudioSource,
    pcm: bytes,
    *,
    sample_rate: int = 16_000,
) -> int:
    frame_count = 0
    for frame in frames_from_pcm(pcm, sample_rate=sample_rate, frame_ms=20):
        await source.capture_frame(frame)
        frame_count += 1
        await asyncio.sleep(0.02)
    return frame_count * 20


async def _consume_agent_audio(track: rtc.Track, state: "_RoomCaseState") -> None:
    stream = rtc.AudioStream(track, sample_rate=16_000, num_channels=1, frame_size_ms=20)
    async for event in stream:
        payload = bytes(event.frame.data)
        state.agent_audio_frames += 1
        state.agent_audio_bytes += len(payload)
        if _pcm16_rms(payload) >= 120.0:
            state.mark("agent_audio_first_at")
            state.agent_audio_frame_timestamps.append(_elapsed_ms(state.started))
            if not state.first_agent_audio.is_set():
                state.first_agent_audio.set()


def _pcm16_rms(payload: bytes) -> float:
    if len(payload) < 2:
        return 0.0
    samples = memoryview(payload).cast("h")
    if not samples:
        return 0.0
    return math.sqrt(sum(int(sample) * int(sample) for sample in samples) / len(samples))


def _make_dispatch_token(
    *,
    api_key: str,
    api_secret: str,
    room_name: str,
    participant: str,
    agent_name: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    if not api_key or not api_secret:
        raise ValueError("LIVEKIT api_key/api_secret are required")
    token = (
        lk_api.AccessToken(api_key, api_secret)
        .with_identity(participant)
        .with_name(participant)
        .with_grants(
            lk_api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .with_room_config(
            lk_api.RoomConfiguration(
                agents=[lk_api.RoomAgentDispatch(agent_name=agent_name)],
            )
        )
    )
    if metadata:
        token = token.with_metadata(json.dumps(metadata, ensure_ascii=False))
    return token.to_jwt()


def _participant_metadata(options: LiveKitRoomOptions) -> dict[str, Any]:
    metadata = dict(options.participant_metadata or {})
    kind = str(metadata.get("kind") or options.participant_kind or "").strip().lower()
    if kind:
        metadata["kind"] = kind
    return metadata


@dataclass
class _RoomCaseState:
    started: float
    events: list[dict[str, Any]]
    agent_audio_frames: int = 0
    agent_audio_bytes: int = 0

    def __post_init__(self) -> None:
        self.timestamps: dict[str, int] = {}
        self.agent_audio_frame_timestamps: list[int] = []
        self.last_agent_audio_monotonic: float | None = None
        self.agent_connected = asyncio.Event()
        self.first_agent_audio = asyncio.Event()
        self.agent_audio_after_user_done = asyncio.Event()

    def mark(self, key: str) -> None:
        self.timestamps.setdefault(key, _elapsed_ms(self.started))
        if key == "agent_audio_first_at":
            self.last_agent_audio_monotonic = time.monotonic()
            if (
                "user_audio_done_at" in self.timestamps
                and not self.agent_audio_after_user_done.is_set()
            ):
                self.agent_audio_after_user_done.set()
        if key == "participant_connected_at" and not self.agent_connected.is_set():
            self.agent_connected.set()

    def agent_audio_recent(self, *, window_ms: int) -> bool:
        last_audio = self.last_agent_audio_monotonic
        if last_audio is None:
            return False
        return time.monotonic() - last_audio <= window_ms / 1000

    def metrics(self) -> dict[str, float | int | str | bool | None]:
        published = self.timestamps.get("local_track_published_at")
        first_audio = self.timestamps.get("agent_audio_first_at")
        user_done = self.timestamps.get("user_audio_done_at")
        first_audio_after_user_done = None
        if user_done is not None:
            first_audio_after_user_done = next(
                (
                    timestamp
                    for timestamp in self.agent_audio_frame_timestamps
                    if timestamp >= user_done
                ),
                None,
            )
        return {
            "elapsed_ms": _elapsed_ms(self.started),
            "room_connected_ms": self.timestamps.get("room_connected_at"),
            "local_track_published_ms": published,
            "participant_connected_ms": self.timestamps.get("participant_connected_at"),
            "agent_track_subscribed_ms": self.timestamps.get("agent_track_subscribed_at"),
            "transcript_first_ms": self.timestamps.get("transcript_first_at"),
            "transcript_final_ms": self.timestamps.get("transcript_final_at"),
            "user_audio_done_ms": user_done,
            "agent_audio_first_ms": first_audio,
            "agent_audio_first_after_user_done_ms": first_audio_after_user_done,
            "publish_to_agent_audio_first_ms": (
                first_audio - published
                if first_audio is not None and published is not None
                else None
            ),
            "user_done_to_agent_audio_first_ms": (
                first_audio - user_done
                if first_audio is not None and user_done is not None
                else None
            ),
            "user_done_to_agent_audio_after_user_done_ms": (
                first_audio_after_user_done - user_done
                if first_audio_after_user_done is not None and user_done is not None
                else None
            ),
            "agent_audio_frames": self.agent_audio_frames,
            "agent_audio_bytes": self.agent_audio_bytes,
        }


def _elapsed_ms(started: float) -> int:
    return round((time.monotonic() - started) * 1000)


def _redact_url(url: str) -> str:
    return url.split("?", 1)[0]
