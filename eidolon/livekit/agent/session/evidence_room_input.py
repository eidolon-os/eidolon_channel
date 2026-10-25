"""Own a recoverable, audio-only SDK RoomIO for one selected participant.

No parallel recorder or custom RTC stream. On discontinuity, detach/close the
old RoomIO, clear the SDK user input via its public API, then create a fresh
RoomIO and capture. Session/output remain alive. Opt-in; not installed by the
production factory. The session must already be running without a RoomIO audio
input, and its existing turn owner must retire the candidate on discontinuity.
"""

import asyncio
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from livekit import rtc
from livekit.agents import AgentSession
from livekit.agents.voice.io import AudioInput
from livekit.agents.voice.room_io import AudioInputOptions, RoomIO, RoomOptions

from eidolon.livekit.common.audio_evidence import AudioEvidenceCapture
from .audio_evidence_input import EvidenceAudioInput


class _PendingInput(AudioInput):
    """Keep public pause/resume callbacks alive while no RTC reader is attached."""

    def __init__(self, owner: "EvidenceRoomInput"):
        super().__init__(label="input-recovery")
        self._owner = owner
        self._paused = False

    async def __anext__(self):
        await asyncio.Future()  # SDK cancels this read when fresh input is installed
        raise StopAsyncIteration

    def on_detached(self):
        if not self._owner._session.input.audio_enabled:
            self._paused = True
            self._owner._request("input_paused")

    def on_attached(self):
        if self._paused:
            self._paused = False
            self._owner._request("input_resumed")


class EvidenceRoomInput:
    def __init__(
        self,
        *,
        session: AgentSession,
        room: rtc.Room,
        participant_identity: str,
        session_id: str,
        endpoint_id: str,
        bind_capture: Callable[[AudioEvidenceCapture | None], None],
        retire_candidate: Callable[[str], None],
        audio_options: AudioInputOptions | None = None,
        capture_max_seconds: float = 30,
    ):
        if not participant_identity or not session_id or not endpoint_id:
            raise ValueError("explicit selected input and session scope required")
        self._session, self._room = session, room
        self._participant = participant_identity
        self._session_id, self._endpoint_id = session_id, endpoint_id
        self._capture_max_seconds = capture_max_seconds
        self._bind_capture, self._retire_candidate = bind_capture, retire_candidate
        # A new queue must not replay a pre-connect recording from an old epoch.
        self._audio_options = replace(audio_options or AudioInputOptions(), pre_connect_audio=False)
        self._io: RoomIO | None = None
        self._io_close_task: asyncio.Task | None = None
        self._guard: EvidenceAudioInput | None = None
        self._pending = _PendingInput(self)
        self._capture: AudioEvidenceCapture | None = None
        self._task: asyncio.Task | None = None
        self._closing_task: asyncio.Task | None = None
        self._revision = 0
        self._closed = False
        self._started = False
        self._rebinding = False
        self._detached = False
        self._ready = asyncio.Event()
        self._listeners: list[tuple[str, Callable[..., None]]] = []
        self.state = "idle"
        self.failure: str | None = None

    @property
    def capture(self) -> AudioEvidenceCapture | None:
        return self._capture

    async def start(self) -> None:
        if self._started or self._closed or self._session.input.audio is not None:
            raise RuntimeError("requires one running session with an unowned audio input")
        self._started = True
        for event in (
            "track_subscribed",
            "track_unsubscribed",
            "track_unpublished",
            "track_muted",
            "track_unmuted",
        ):
            self._listeners.append((event, self._on_track))
        for event in ("participant_connected", "participant_disconnected"):
            self._listeners.append((event, self._on_participant))
        for event in ("reconnecting", "reconnected", "disconnected"):
            self._listeners.append((event, self._on_connection))
        for event, callback in self._listeners:
            self._room.on(event, callback)
        self._session.on("close", self._on_session_close)
        self._request("input_started")
        try:
            await asyncio.wait_for(self.wait_for_ready(), 10)
        except BaseException:
            await self.aclose()
            raise

    async def wait_for_ready(self) -> None:
        await self._ready.wait()
        if self.state != "ready":
            raise RuntimeError(f"audio input {self.state}")

    def _on_track(self, *args: Any) -> None:
        publication, participant = args[-2:]
        if (
            participant.identity == self._participant
            and publication.source == rtc.TrackSource.SOURCE_MICROPHONE
        ):
            self._request("input_track_changed")

    def _on_participant(self, participant: Any) -> None:
        if participant.identity == self._participant:
            self._request("input_participant_changed")

    def _on_connection(self, *_: Any) -> None:
        self._request("input_connection_changed")

    def _request(self, reason: str) -> None:
        if self._closed:
            return
        self._revision += 1
        self._ready.clear()
        if self._capture is not None:
            self._capture.invalidate(reason)
            self._retire_candidate(reason)
        if reason in ("input_detached", "input_ended", "input_paused"):
            # SDK shutdown detaches/cancels before session.close. Only a real
            # attach can authorize resuming; detach alone never reopens audio.
            self._detached = True
            self.state = "paused" if not self._session.input.audio_enabled else "waiting"
            asyncio.get_running_loop().call_soon(self._audit_ownership)
            return
        if reason == "input_resumed":
            self._detached = False
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._recover())

    def _on_guard_invalidated(self, reason: str) -> None:
        if not self._rebinding:
            self._request(reason)

    def _set_input(self, value: AudioInput | None) -> None:
        self._rebinding = True
        try:
            self._session.input.audio = value
        finally:
            self._rebinding = False

    def _audit_ownership(self) -> None:
        # Run after AgentInput's synchronous setter has assigned its new value.
        if not self._closed and not self._owns_input():
            self._stop("closed" if self._session.input.audio is None else "replaced")
            self._closing_task = asyncio.create_task(self.aclose())

    def _available(self) -> bool:
        participant = self._room.remote_participants.get(self._participant)
        return bool(
            self._room.isconnected()
            and participant
            and any(
                p.source == rtc.TrackSource.SOURCE_MICROPHONE
                and p.track is not None
                and not p.muted
                for p in participant.track_publications.values()
            )
        )

    async def _close_io(self) -> None:
        if self._io is None:
            return
        if self._io_close_task is None:
            self._io_close_task = asyncio.create_task(self._io.aclose())
        task = self._io_close_task
        try:
            # Closing the owner during recovery must finish the same SDK
            # cleanup, not orphan it when the recovery task is cancelled.
            await asyncio.shield(task)
        finally:
            if task.done() and not task.cancelled() and task.exception() is None:
                self._io = None
                self._io_close_task = None

    def _owns_input(self) -> bool:
        current = self._session.input.audio
        return current is not None and (
            current in (self._guard, self._pending)
            or (self._io is not None and current is self._io.audio_input)
        )

    async def _recover(self) -> None:
        try:
            while not self._closed:
                revision = self._revision
                if self._detached or not self._session.input.audio_enabled:
                    self.state = "paused" if not self._session.input.audio_enabled else "waiting"
                    return  # retain the invalid guard solely to observe attach
                if self._session.input.audio not in (None, self._guard, self._pending):
                    self._stop("replaced")  # another owner took over; never steal input
                    await self._close_io()
                    return
                self.state = "starting"
                self.failure = None
                self._set_input(self._pending)
                await self._close_io()
                if self._session.input.audio is not self._pending:
                    self._audit_ownership()
                    return
                if revision != self._revision:
                    continue
                if not self._session.input.audio_enabled:
                    self.state = "paused"
                    return
                if not self._available():
                    self.state = "waiting"
                    return
                capture = AudioEvidenceCapture(
                    session_id=self._session_id, endpoint_id=self._endpoint_id,
                    max_seconds=self._capture_max_seconds,
                )
                self._capture = capture
                self._bind_capture(capture)
                # Old source can no longer deliver frames. Public SDK reset
                # discards queued STT input/resampler, not assistant playback.
                self._session.clear_user_turn()
                self._io = RoomIO(
                    self._session,
                    self._room,
                    options=RoomOptions(
                        participant_identity=self._participant,
                        audio_input=self._audio_options,
                        video_input=False,
                        text_input=False,
                        audio_output=False,
                        text_output=False,
                        close_on_disconnect=False,
                        delete_room_on_close=False,
                    ),
                )
                await self._io.start()
                assert self._io.audio_input is not None
                if self._session.input.audio is not self._io.audio_input:
                    self._audit_ownership()
                    return
                self._guard = EvidenceAudioInput(
                    source=self._io.audio_input,
                    capture=capture,
                    on_invalidated=self._on_guard_invalidated,
                    on_resumed=lambda: self._request("input_resumed"),
                )
                self._set_input(self._guard)
                async with asyncio.timeout(10):
                    await self._io.wait_for_ready()
                if revision != self._revision:
                    continue
                self.state = "ready"
                self._ready.set()
                return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._capture is not None:
                self._capture.invalidate("input_rebuild_failed")
            self._bind_capture(None)
            if self._owns_input():
                self._set_input(None)
            try:
                await self._close_io()
            except Exception as cleanup_error:
                self.failure = f"cleanup:{type(cleanup_error).__name__}"
            else:
                self.failure = type(exc).__name__
            finally:
                self.state = "failed"
                self._ready.set()

    def _stop(self, state: str = "closed") -> None:
        if self._closed:
            return
        self._closed = True
        self.state = state
        self._revision += 1
        if self._capture is not None:
            self._capture.invalidate("input_closed")
        self._capture = None
        self._bind_capture(None)
        for event, callback in self._listeners:
            self._room.off(event, callback)
        self._listeners.clear()
        self._session.off("close", self._on_session_close)
        if self._owns_input():
            self._set_input(None)
        self._ready.set()

    def _on_session_close(self, *_: Any) -> None:
        self._stop()
        self._closing_task = asyncio.create_task(self.aclose())

    async def aclose(self) -> None:
        self._stop()
        if self._closing_task is not None and self._closing_task is not asyncio.current_task():
            await self._closing_task
            return
        if self._task is not None and self._task is not asyncio.current_task():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        if (
            self._io_close_task is not None
            and self._io_close_task.done()
            and (self._io_close_task.cancelled() or self._io_close_task.exception() is not None)
        ):
            # Retry failed cleanup only during terminal shutdown. Recovery must
            # never open another reader without proof the previous one closed.
            self._io_close_task = None
        await self._close_io()
