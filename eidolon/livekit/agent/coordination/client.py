"""One scene's authenticated Agent stream and revocable media delivery.

The Provider owns admission and device reservations. ``present`` must use the
existing TTS/output path and return True only after native playout or an
additional device confirmation. Reply receipts explicitly report native playout. ``stop``
returns True only after the device confirms playback.stop. Failed cleanup must
retain the Provider's reservations. No automatic reconnect can replay a scene.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

import aiohttp

from eidolon_sdk.biz.control.coordination_stream import (
    MAX_FRAME_BYTES,
    STOP_EXECUTION_TIMEOUT,
    ROLE_GROUP_STREAM_PATH,
    SERVER_FRAME,
    Capturing,
    Close,
    OpenScene,
    Prepared,
    Press,
    Receipt,
    Release,
    ReplyDelta,
    ReplyEnd,
    ReplyStart,
    SceneState,
    Speaking,
    Stop,
    Transcript,
)


@dataclass(frozen=True)
class StopResult:
    error_code: str = ""

    def __bool__(self):
        return not self.error_code


@dataclass
class Playback:
    start: ReplyStart
    text: asyncio.Queue
    ended: bool = False
    task: asyncio.Task | None = None


class RoleGroupClient:
    def __init__(
        self,
        opened: OpenScene,
        *,
        present: Callable[[ReplyStart, AsyncIterator[str], Callable[[], None]], Awaitable[bool]],
        stop: Callable[[str], Awaitable[bool]],
        stop_timeout: float = STOP_EXECUTION_TIMEOUT,
        on_state: Callable[[SceneState], Awaitable[None]] | None = None,
    ):
        self.opened = opened
        self.present = present
        self.stop = stop
        self.stop_timeout = stop_timeout
        self.on_state = on_state
        self.members = {
            m.output_device.device_instance_id: m.companion_id for m in opened.selection.members
        }
        self.ready = asyncio.Event()  # Agent prepared, NOT physical device readiness.
        self.closed = asyncio.Event()
        self.cleanup_ok = False
        self.state: SceneState | None = None
        self._stream_id: str | None = None
        self._capture: str | None = None
        self._released = False
        self._epoch: int | None = None
        self._last_epoch = -1
        self._playback: Playback | None = None
        self._outbound = asyncio.Queue(maxsize=128)
        self._tasks: set[asyncio.Task] = set()
        self._failure = asyncio.Event()
        self._failure_cause: BaseException | None = None
        self._capture_stops: dict[str, dict[str, asyncio.Task]] = {}
        self._running = False
        self._closing = False
        self._close_requested = False
        self._close_requested_event = asyncio.Event()
        self._seen_turns: set[str] = set()
        self._discarded_turns: set[str] = set()
        self._stop_locks = {device: asyncio.Lock() for device in self.members}

    def _send(self, frame):
        if not self._running or self._closing:
            raise ConnectionError("role-group stream unavailable")
        try:
            self._outbound.put_nowait(frame.model_dump(mode="json"))
        except asyncio.QueueFull:
            self._failure.set()
            raise ConnectionError("role-group control queue full") from None

    def _spawn(self, coroutine):
        if len(self._tasks) >= 64:
            coroutine.close()
            self._failure.set()
            raise ConnectionError("too many pending media controls")
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)

        def done(task):
            self._tasks.discard(task)
            if not task.cancelled() and task.exception() is not None:
                self._failure_cause = self._failure_cause or task.exception()
                self._failure.set()
                self._epoch = None
                self._revoke()

        task.add_done_callback(done)
        return task

    def _revoke(self):
        active, self._playback = self._playback, None
        if active is not None and active.task is not None:
            active.task.cancel()

    def press(self, capture_id: str):
        """Synchronous local invalidation; never await a stop ACK before recording."""
        frame = Press(type="press", capture_id=capture_id)
        if (
            not self.ready.is_set()
            or self._closing
            or self._close_requested
            or self._failure.is_set()
        ):
            raise ConnectionError("role-group not prepared")
        if capture_id == self._capture:
            if self._released:
                raise ValueError("capture replayed after release")
            return
        self._revoke()
        self._capture, self._released, self._epoch = capture_id, False, None
        self.state = None
        # Warm device controls start locally, without a Channel -> Agent round trip.
        if len(self._capture_stops) >= 256:
            raise ValueError("session capture budget exhausted")
        self._capture_stops[capture_id] = {
            device: self._spawn(self._stop_device(device, capture_id=capture_id))
            for device in self.members
        }
        self._send(frame)

    def release(self, capture_id: str):
        if self._close_requested:
            return
        if capture_id == self._capture and not self._released:
            self._send(Release(type="release", capture_id=capture_id))
            self._released = True

    def transcript(self, capture_id: str, text: str, commitment: dict | None = None):
        if self._close_requested or capture_id != self._capture:
            return  # A superseded ASR task has no authority.
        if not self._released:
            raise ValueError("transcript before PTT release")
        self._send(
            Transcript(type="transcript", capture_id=capture_id, text=text, commitment=commitment)
        )

    async def _stop_device(self, device, *, capture_id=None, epoch=None):
        # Serialize commands, but start the physical execution budget only when
        # this operation owns the endpoint. Superseded queued controls never run.
        async with self._stop_locks[device]:
            if ((capture_id is not None and capture_id != self._capture)
                    or (epoch is not None and epoch < self._last_epoch)):
                return StopResult("TEAM_STOP_SUPERSEDED")
            try:
                async with asyncio.timeout(self.stop_timeout):
                    confirmed = await self.stop(device)
                result = StopResult() if confirmed is True else StopResult("TEAM_STOP_UNCONFIRMED")
            except TimeoutError:
                result = StopResult("TEAM_STOP_TIMEOUT")
            except Exception as exc:
                result = StopResult(getattr(exc, "code", "TEAM_STOP_TRANSPORT_FAILED"))
                logging.getLogger(__name__).warning(
                    "team stop failed device=%s capture=%s epoch=%s cause=%r",
                    device, capture_id, epoch, exc)
            if not result:
                logging.getLogger(__name__).warning(
                    "team stop unconfirmed device=%s capture=%s epoch=%s code=%s",
                    device, capture_id, epoch, result.error_code)
            return result

    async def _remote_stop(self, frame: Stop):
        if frame.capture_id is not None:
            operation = self._capture_stops.get(frame.capture_id, {}).get(frame.device_id)
            # Missing correlation is never permission to issue a late physical stop.
            stopped = (await asyncio.shield(operation) if operation is not None
                       else StopResult("TEAM_STOP_UNKNOWN_CAPTURE"))
        else:
            stopped = await self._stop_device(frame.device_id, epoch=frame.epoch)
        self._send(Receipt(
            type="receipt", request_id=frame.request_id, device_id=frame.device_id,
            result="completed" if stopped else "failed", error_code=stopped.error_code,
        ))

    async def _present(self, playback: Playback):
        async def text():
            while True:
                chunk = await playback.text.get()
                if chunk is None:
                    return
                yield chunk

        def speaking():
            if self._playback is playback:
                self._send(
                    Speaking(
                        type="speaking",
                        turn_id=playback.start.turn_id,
                        device_id=playback.start.device_id,
                    )
                )

        try:
            barriers = self._capture_stops.get(self._capture, {})
            silence = await asyncio.gather(*(asyncio.shield(t) for t in barriers.values()))
            played = (await self.present(playback.start, text(), speaking)
                      if len(silence) == len(self.members) and all(silence) else False)
        except asyncio.CancelledError:
            raise
        except Exception:
            logging.getLogger(__name__).exception(
                "team presentation failed turn=%s device=%s", playback.start.turn_id,
                playback.start.device_id)
            played = False
        if self._playback is not playback:
            return
        # The physical adapter cannot complete before the producer closed the text.
        completed = played is True and playback.ended and playback.text.empty()
        self._playback = None
        if not completed:
            self._discarded_turns.add(playback.start.turn_id)
        logging.getLogger(__name__).info(
            "team playout turn=%s device=%s completed=%s text_ended=%s",
            playback.start.turn_id, playback.start.device_id, completed, playback.ended)
        self._send(
            Receipt(
                type="receipt",
                request_id=playback.start.request_id,
                device_id=playback.start.device_id,
                result="completed" if completed else "failed",
                completion_basis="native_playout",
                error_code="" if completed else "TEAM_PLAYBACK_UNCONFIRMED",
            )
        )

    def accept(self, data: str):
        frame = SERVER_FRAME.validate_json(data)
        if frame.session_id != self.opened.selection.session_id:
            raise ValueError("scene mismatch")
        if isinstance(frame, Prepared):
            if self._stream_id is not None:
                raise ValueError("duplicate prepared frame")
            self._stream_id = frame.stream_id
            self.ready.set()
            return
        if self._stream_id is None or frame.stream_id != self._stream_id:
            raise ValueError("stream mismatch")
        if isinstance(frame, Capturing):
            if frame.capture_id == self._capture and not self._close_requested:
                if frame.epoch <= self._last_epoch:
                    raise ValueError("capture epoch did not advance")
                self._last_epoch = self._epoch = frame.epoch
            return
        if isinstance(frame, SceneState):
            if not set(frame.members) <= set(self.members.values()):
                raise ValueError("unknown scene state member")
            if (frame.capture_id == self._capture and self._released
                    and frame.epoch >= self._last_epoch):
                self.state = frame
                if self.on_state is not None:
                    self._spawn(self._notify_state(frame))
            return
        if frame.device_id not in self.members:
            raise ValueError("output is not a scene member")
        if isinstance(frame, Stop):
            if self._epoch is not None and frame.epoch >= self._epoch:
                if self._playback is not None or frame.epoch > self._epoch:
                    self._epoch = None
                self._revoke()
            self._spawn(self._remote_stop(frame))
            return
        if self._close_requested or self._failure.is_set():
            return
        if self._epoch is None or frame.epoch < self._epoch:
            return  # Late output after local PTT is never handed to TTS.
        if frame.epoch != self._epoch or not self._released:
            raise ValueError("reply outside committed capture epoch")
        if isinstance(frame, ReplyStart):
            logging.getLogger(__name__).info("team reply start turn=%s companion=%s device=%s",
                frame.turn_id, frame.companion_id, frame.device_id)
            if self._playback is not None:
                raise ValueError("overlapping role-group replies")
            if frame.turn_id in self._seen_turns or len(self._seen_turns) >= 8192:
                raise ValueError("replayed reply or scene reply limit")
            self._seen_turns.add(frame.turn_id)
            if frame.companion_id != self.members[frame.device_id]:
                raise ValueError("Companion endpoint mismatch")
            playback = Playback(frame, asyncio.Queue(maxsize=64))
            self._playback = playback
            playback.task = self._spawn(self._present(playback))
            return
        if frame.turn_id in self._discarded_turns:
            return  # A failed presentation may still have text in flight.
        playback = self._playback
        if (
            playback is None
            or (frame.turn_id, frame.device_id)
            != (playback.start.turn_id, playback.start.device_id)
            or playback.ended
        ):
            raise ValueError("reply stream correlation mismatch")
        if isinstance(frame, ReplyEnd):
            logging.getLogger(__name__).info("team text ended turn=%s device=%s",
                frame.turn_id, frame.device_id)
            playback.ended = True
            playback.text.put_nowait(None)
        elif isinstance(frame, ReplyDelta):
            playback.text.put_nowait(frame.text)

    async def _notify_state(self, frame: SceneState):
        # A queued completion must not clear a newer PTT capture's UI.
        if (self.state is frame and frame.capture_id == self._capture
                and self._released and self.on_state is not None):
            await self.on_state(frame)

    async def run(self, http: aiohttp.ClientSession, *, base_url: str, token: str):
        """Use the Host's configured service credential; never a Mobile token."""
        if self._running or self.closed.is_set():
            raise RuntimeError("scene stream cannot be reused")
        if not token or not base_url.startswith(("http://", "https://")):
            raise ValueError("role-group service configuration missing")
        self._running = True
        workers = []
        try:
            async with http.ws_connect(
                base_url.rstrip("/") + ROLE_GROUP_STREAM_PATH,
                headers={"Authorization": f"Bearer {token}"},
                max_msg_size=MAX_FRAME_BYTES,
                heartbeat=10,
            ) as socket:
                await socket.send_json(self.opened.model_dump(mode="json"))

                async def read():
                    async for message in socket:
                        if message.type != aiohttp.WSMsgType.TEXT:
                            raise ConnectionError("role-group stream closed")
                        self.accept(message.data)
                    if socket.close_code not in (None, 1000):
                        raise ConnectionError("role-group peer rejected scene")

                async def write():
                    while True:
                        await socket.send_json(await self._outbound.get())

                async def closing_deadline():
                    await self._close_requested_event.wait()
                    await asyncio.sleep(self.stop_timeout + 3)
                    raise TimeoutError("role-group close timed out")

                workers = [
                    asyncio.create_task(read()),
                    asyncio.create_task(write()),
                    asyncio.create_task(self._failure.wait()),
                    asyncio.create_task(closing_deadline()),
                ]
                async with asyncio.timeout(10):
                    prepared = asyncio.create_task(self.ready.wait())
                    try:
                        done, _ = await asyncio.wait(
                            [*workers, prepared], return_when=asyncio.FIRST_COMPLETED
                        )
                        if prepared not in done:
                            raise ConnectionError("role-group prepare failed")
                    finally:
                        prepared.cancel()
                        await asyncio.gather(prepared, return_exceptions=True)
                done, _ = await asyncio.wait(workers, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
                if not self._close_requested or self._failure.is_set():
                    raise ConnectionError("role-group stream lost") from self._failure_cause
        finally:
            self._closing = True
            self._revoke()
            for task in [*workers, *self._tasks]:
                task.cancel()
            pending_tasks = set(workers) | self._tasks
            pending = set()
            if pending_tasks:
                _, pending = await asyncio.wait(pending_tasks, timeout=self.stop_timeout)
                for task in pending_tasks - pending:
                    if not task.cancelled():
                        task.exception()
            # Independent of Agent/connection availability. Never claim silence
            # or release reservations when a physical endpoint did not confirm.
            stopped = await asyncio.gather(*(self._stop_device(d) for d in self.members))
            self.cleanup_ok = all(stopped) and not pending
            self._running = False
            self.closed.set()

    def close(self):
        # Reader continues collecting stop requests until Agent closes the socket.
        if not self._close_requested:
            self._send(Close(type="close"))
            self._close_requested = True
            self._close_requested_event.set()
            self._epoch = None
            self._revoke()
