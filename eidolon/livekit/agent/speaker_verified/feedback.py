"""Bounded presentation on the existing session-control bus; no turn ownership."""

import asyncio
from eidolon_sdk.biz.contracts import SESSION_CONTROL_TOPIC
from eidolon_sdk.biz.voiceprint import VoiceAdmissionFeedback


class VoiceAdmissionPublisher:
    def __init__(self, room, participant, conversation_id):
        self._room, self._participant, self._conversation = room, participant, conversation_id
        self._queue = asyncio.Queue(maxsize=1)
        self._sequence = 0
        self._task = None
        self._closed = False

    def update(self, state):
        if self._closed:
            return
        self._sequence += 1
        event = VoiceAdmissionFeedback(
            conversation_id=self._conversation, sequence=self._sequence, state=state
        )
        if self._queue.full():
            self._queue.get_nowait()
        self._queue.put_nowait(event)

    def start(self):
        if self._task is None and not self._closed:
            self._task = asyncio.create_task(self._run())
            self.update("ready")

    async def _run(self):
        while True:
            event = await self._queue.get()
            try:
                async with asyncio.timeout(2):
                    await self._room.local_participant.publish_data(
                        event.model_dump_json().encode(),
                        reliable=True,
                        topic=SESSION_CONTROL_TOPIC,
                        destination_identities=[self._participant],
                    )
            except Exception:
                # Presentation failure must not authorize, cancel or retry a voice turn.
                continue

    async def aclose(self):
        self._closed = True
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        if not self._queue.empty():
            self._queue.get_nowait()
