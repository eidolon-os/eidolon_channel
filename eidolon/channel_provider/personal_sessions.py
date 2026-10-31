"""Provider-owned private session lifecycle; dispatch existence is not a cleanup receipt."""

from __future__ import annotations

import asyncio
import json
import logging

from .contracts import BackendUnavailable, InvalidTransition

_log = logging.getLogger(__name__)


class PersonalSessions:
    """Uses the Provider's existing DB, lifecycle lock, adapter and Body control path."""

    def __init__(self, store, registry, lock):
        self.store, self.registry, self.lock = store, registry, lock
        self._observers = {}
        self._tasks = {}
        self._stopping = False
        self._authority_task = None
        # Why each pending cleanup last failed. Retries run every second while
        # a Body is away; the log says each reason once, not once a second.
        self._pending_reasons = {}

    def watch_authority(self, resolve):
        """Recheck open bindings independently of speech; retry durable cleanup."""
        if self._authority_task is not None:
            raise RuntimeError("personal authority watcher already started")
        async def run():
            while True:
                await self.reconcile_authority(resolve)
                await asyncio.sleep(1.0)
        self._authority_task = asyncio.create_task(run(), name="personal-authority")

    async def reconcile_authority(self, resolve):
        async def check(record):
            invalid = record["state"] == "closing"
            if not invalid:
                binding = record["binding"]
                person = binding["personal_memory"]
                if person is None:
                    return  # Ordinary sessions have no private-memory grant to monitor.
                try:
                    async with asyncio.timeout(2.0):
                        realm = await resolve(owner_id=binding["owner_id"],
                            person_id=person["person_id"], person_revision=person["person_revision"])
                        if not realm:
                            raise ValueError("personal memory unavailable")
                except Exception:
                    invalid = True
            if invalid:
                # The same lifecycle handles worker end, explicit close and
                # authority loss. Failure retains the durable reservation.
                self.request_close(record["session_id"])
        await asyncio.gather(*(check(row) for row in self.store.pending_personal_sessions()))

    def require_available(self, device_id):
        if self.store.pending_personal_sessions(device_id):
            raise InvalidTransition("device has pending personal session cleanup")

    def claim(self, provision, session_id, *, personal_memory, target_companion_id, session_intent, device_requested=False):
        adapter = self.registry.get(provision.adapter_name)
        if not all(
            callable(getattr(adapter, name, None))
            for name in (
                "observe_session_end",
                "end_prepared_session",
                "close_session",
                "resume_prepared_channels",
            )
        ):
            raise InvalidTransition("transport lacks acknowledged personal session cleanup")
        binding = {
            "owner_id": provision.owner_id,
            "device_ref": provision.device_ref.model_dump(mode="json"),
            "operation_kind": provision.operation_kind,
            "operation_id": provision.operation_id,
            "adapter": provision.adapter_name,
            "handle": json.loads(provision.handle_json),
            "personal_memory": personal_memory,
            "target_companion_id": target_companion_id,
            "session_intent": session_intent,
        }
        if personal_memory is None:
            binding["device_requested"] = device_requested
        previous = self.store.personal_session(session_id)
        if previous is not None:
            # Credential refresh alone does not change this session's identity
            # or transport. Keep the original provision for cleanup/replay.
            ignored = {"operation_kind", "operation_id"}
            if {k: v for k, v in binding.items() if k not in ignored} == {
                k: v for k, v in previous["binding"].items() if k not in ignored
            }:
                binding = previous["binding"]
        record = self.store.claim_personal_session(session_id, provision.device_id, binding)
        if session_id not in self._observers:
            try:
                self._observers[session_id] = adapter.observe_session_end(
                    binding["handle"], session_id, lambda *_reason: self.request_close(session_id)
                )
            except BaseException:
                self.store.transition_personal_session(session_id, expected="open", state="closing")
                self.request_close(session_id)
                raise
        return record

    def request_close(self, session_id):
        if self._stopping or session_id in self._tasks:
            return

        async def run():
            try:
                async with self.lock:
                    await self.close(session_id)
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                if self._pending_reasons.get(session_id) != reason:
                    self._pending_reasons[session_id] = reason
                    _log.warning("personal cleanup pending session=%s reason=%s",
                                 session_id, reason, exc_info=not isinstance(exc, BackendUnavailable))

        task = asyncio.create_task(run(), name=f"personal-close:{session_id}")
        self._tasks[session_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(session_id, None))

    async def close(self, session_id, *, body_confirmed=False):
        """Caller holds the Provider lock. Every failure keeps the durable reservation.

        ``body_confirmed`` is the Body itself having said, on its own channel and
        for this conversation, that it ended it. A Body says that only after
        closing its own microphone and playback — the firmware's leave and this
        phone's both do — which is everything a room.leave receipt would
        confirm. Asking it again would wait on a Body that may already be gone:
        a phone drops its channel the moment it has said so.
        """
        record = self.store.personal_session(session_id)
        if record is None:
            raise InvalidTransition("unknown personal session")
        if record["state"] == "closed":
            return
        if record["state"] == "open":
            self.store.transition_personal_session(session_id, expected="open", state="closing")
        remove = self._observers.pop(session_id, None)
        if remove is not None:
            remove()
        binding = record["binding"]
        adapter = self.registry.get(binding["adapter"])
        # Stop producers first, then clear the Body. Missing dispatch is normal
        # after worker idle/shutdown; it never substitutes for the second step.
        await adapter.close_session(binding["handle"], session_id)
        if not body_confirmed:
            await adapter.resume_prepared_channels((binding["handle"],))
            await adapter.end_prepared_session(binding["handle"], session_id)
        self.store.transition_personal_session(session_id, expected="closing", state="closed")
        self._pending_reasons.pop(session_id, None)

    def settle_revoked(self, device_id):
        """Caller holds the Provider lock, after the device's channel was deleted.

        Revocation deleted the room, so the Body has no IO left in it and no
        channel to confirm anything on; waiting for its receipt would hold the
        reservation — and the Companion it names — forever.
        """
        for record in self.store.pending_personal_sessions(device_id):
            session_id = record["session_id"]
            remove = self._observers.pop(session_id, None)
            if remove is not None:
                remove()
            if record["state"] == "open":
                self.store.transition_personal_session(session_id, expected="open", state="closing")
            self.store.transition_personal_session(session_id, expected="closing", state="closed")
            self._pending_reasons.pop(session_id, None)

    async def recover(self):
        for record in self.store.pending_personal_sessions():
            try:
                async with self.lock:
                    await self.close(record["session_id"])
            except Exception:
                _log.exception("personal recovery cleanup pending session=%s", record["session_id"])

    async def shutdown(self):
        self._stopping = True
        if self._authority_task is not None:
            self._authority_task.cancel()
            await asyncio.gather(self._authority_task, return_exceptions=True)
            self._authority_task = None
        for remove in self._observers.values():
            remove()
        self._observers.clear()
        for task in self._tasks.values():
            task.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        await self.recover()
