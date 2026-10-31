"""Per-turn memory authority within a continuous room conversation."""
from __future__ import annotations

import asyncio
import time

from eidolon_sdk.biz.runtime import sign_runtime_token
from eidolon_sdk.biz.speaker_attestation import verify_observation


class AutomaticMemorySession:
    def __init__(self, *, services, room, source_session_id, secret, llm, stop_outputs,
                 ttl_seconds=600, publish=None):
        self.services, self.room = services, room
        self.source_session_id, self.secret = source_session_id, secret
        self.llm, self.stop_outputs, self.ttl_seconds = llm, stop_outputs, ttl_seconds
        self.publish = publish
        self._sequence = 0
        self._notices = set()
        self._lock = asyncio.Lock()
        self._person = None
        self._memory_mode = "personal"
        self._epoch = 0
        self._blocked = getattr(services, "personal_memory", None) is not None
        self._closed = False
        self._watcher = None
        self._prepared = None
        self._context = None
        self._session_id = source_session_id
        self._renew_at = 0.0
        self._admission_task = None
        self._admission_observation = None

    def observe(self, observation):
        """Prepare authority from trusted Channel evidence, never grant it here.

        Consumption still requires the signed, text-bound observation for this
        exact turn. A completed lookup is not reusable for another observation.
        """
        if self._closed:
            return
        if self._admission_observation == observation:
            return
        if self._admission_task is not None:
            self._admission_task.cancel()
        self._admission_task = None
        observation = observation.model_copy(deep=True)
        self._admission_observation = observation
        # Personal memory follows only its own use's decision (design §3.3).
        person = observation.decided("personal_memory")
        if (person is None
                or observation.session_id != self.source_session_id
                or observation.expires_at <= time.time()):
            return

        async def resolve():
            try:
                async with asyncio.timeout(2.0):
                    context = await self.services.resolve_room(self.room)
                    if context.owner_id != observation.owner_id:
                        return None
                    mode = "provisional" if person.maturity == "candidate" else "personal"
                    realm = await self._resolve_memory(context, (
                        person.person_id, person.person_revision), mode)
                    if not realm or observation.expires_at <= time.time():
                        return None
                    return ((person.person_id, person.person_revision), mode)
            except Exception:
                return None

        self._admission_task = asyncio.create_task(resolve(), name="person-memory-admission")

    async def prepare(self, *, observation_token=None, trace_id=None, text=""):
        async with self._lock:
            if self._closed:
                raise RuntimeError("memory session closed")
            self._prepared = None
            context = await self.services.resolve_room(self.room)
            self._context = context
            person = None
            mode = "personal"
            observation = None
            if observation_token and trace_id:
                try:
                    observation = verify_observation(observation_token, secret=self.secret,
                        owner_id=context.owner_id, session_id=self.source_session_id,
                        turn_id=trace_id, text=text)
                    task = self._admission_task
                    if (observation.decided("personal_memory") is not None
                            and observation == self._admission_observation
                            and task is not None and task.done() and not task.cancelled()):
                        admission = task.result()
                        if admission is not None:
                            person, mode = admission
                except Exception:
                    # Unbound/disabled/missing/stale/failed recognition all keep
                    # conversation available without any default-Realm access.
                    person = None
            if (self._blocked or person != self._person or mode != self._memory_mode
                    or time.time() >= self._renew_at):
                await self._bind_turn(context, person, mode)
            await self._publish_state("active" if person and mode == "personal" else "disabled")
            self._prepared = (trace_id, text)
            if self._watcher is None:
                self._watcher = asyncio.create_task(self._watch(), name="automatic-memory-authority")
            return False  # A speaker change never resets the room conversation.

    def consume(self, *, trace_id, text):
        if self._blocked or self._closed or self._prepared != (trace_id, text):
            raise RuntimeError("turn has no current memory admission")
        self._prepared = None

    async def _bind_turn(self, context, person, mode="personal"):
        self._blocked = True
        self._prepared = None
        # Replace the authenticated RPC at the committed turn boundary. Normal
        # turn scheduling owns interruptions; identity changes do not flush Body
        # playback, public history, or the interrupted-conversation ledger.
        await self.llm.close_runtime()
        self._epoch += 1
        self._install(context, person, mode)
        self._person = person
        self._memory_mode = mode
        self._blocked = False

    def _install(self, context, person, mode):
        session_id = self._session_id
        token, exp = sign_runtime_token(secret=self.secret, owner_id=context.owner_id,
            companion_id=context.companion_id, device_id=context.device_id,
            session_id=session_id, source_session_id=self.source_session_id,
            ttl_seconds=self.ttl_seconds, memory_mode=mode,
            **({"person_id": person[0], "person_revision": person[1], "identity_epoch": self._epoch}
               if person else {}))
        self.llm.install_runtime(token, f"room-session:{session_id}")
        self._renew_at = exp.timestamp() - min(30, self.ttl_seconds / 2)

    async def _publish_state(self, state):
        if self.publish is None:
            return
        self._sequence += 1
        now = time.time()
        payload = {"schema_version": 1, "session_id": self.source_session_id,
            "sequence": self._sequence, "state": state,
            "person_id": self._person[0] if state == "active" and self._person else None,
            "issued_at": now, "expires_at": now + 10}
        async def send():
            try:
                async with asyncio.timeout(1.0):
                    await self.publish(payload)
            except Exception:
                pass  # Display delivery cannot grant or retain a runtime scope.
        task = asyncio.create_task(send())
        self._notices.add(task)
        task.add_done_callback(self._notices.discard)

    async def _resolve_memory(self, context, person, mode):
        if mode == "provisional":
            return await self.services.runtime.resolve_provisional_memory(
                owner_id=context.owner_id, person_id=person[0], person_revision=person[1],
                session_id=self.source_session_id)
        return await self.services.runtime.resolve_person_memory(
            owner_id=context.owner_id, person_id=person[0], person_revision=person[1])

    async def _watch(self):
        while True:
            await asyncio.sleep(1.0)
            await self._check_authority()

    async def _check_authority(self):
        # Do not hold the turn-transition lock across network IO. A stale
        # response may only revoke the exact binding it checked.
        async with self._lock:
            if self._closed or (self._person is None and not self._blocked):
                return
            context, person, epoch = self._context, self._person, self._epoch
            mode = self._memory_mode
            blocked = self._blocked
        try:
            if blocked:
                raise RuntimeError("cleanup pending")
            async with asyncio.timeout(2.0):
                await self._resolve_memory(context, person, mode)
        except Exception:
            async with self._lock:
                if (self._closed or self._context is not context
                        or self._person != person or self._epoch != epoch
                        or self._memory_mode != mode):
                    return
                try:
                    self._blocked = True
                    self._prepared = None
                    if self._admission_task is not None:
                        self._admission_task.cancel()
                    self._admission_task = None
                    self._admission_observation = None
                    await self.llm.close_runtime()
                    await self.stop_outputs()
                    await self._bind_turn(context, None)
                    await self._publish_state("disabled")
                except Exception:
                    self._blocked = True

    async def aclose(self):
        self._closed = True
        self._prepared = None
        admission = self._admission_task
        self._admission_task = None
        self._admission_observation = None
        if admission is not None:
            admission.cancel()
            await asyncio.gather(admission, return_exceptions=True)
        for task in tuple(self._notices):
            task.cancel()
        await asyncio.gather(*tuple(self._notices), return_exceptions=True)
        if self._watcher is not None:
            self._watcher.cancel()
            await asyncio.gather(self._watcher, return_exceptions=True)
            self._watcher = None
