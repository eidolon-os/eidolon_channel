import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from eidolon.livekit.agent.speaker_verified.feedback import VoiceAdmissionPublisher


async def test_feedback_is_bounded_ordered_and_contains_no_speech_or_permission():
    entered, release = asyncio.Event(), asyncio.Event()
    sent = []

    async def publish(data, **kwargs):
        sent.append(json.loads(data))
        assert kwargs["destination_identities"] == ["pad"]
        entered.set()
        await release.wait()

    room = SimpleNamespace(local_participant=SimpleNamespace(publish_data=publish))
    publisher = VoiceAdmissionPublisher(room, "pad", "conversation")
    publisher.start()
    await entered.wait()
    for _ in range(100):
        publisher.update("checking")
    publisher.update("non_match")
    assert publisher._queue.qsize() == 1
    release.set()
    async with asyncio.timeout(1):
        while len(sent) < 2:
            await asyncio.sleep(0)
    await publisher.aclose()
    assert [e["state"] for e in sent] == ["ready", "non_match"]
    assert sent[-1]["sequence"] > sent[0]["sequence"]
    assert all(set(e) == {"schema_v", "type", "conversation_id", "sequence", "state"} for e in sent)
    publisher.update("accepted")
    assert publisher._queue.empty()


async def test_failed_publication_does_not_prevent_next_feedback_or_shutdown():
    publish = AsyncMock(side_effect=[RuntimeError("disconnected"), None])
    publisher = VoiceAdmissionPublisher(
        SimpleNamespace(local_participant=SimpleNamespace(publish_data=publish)),
        "pad",
        "conversation",
    )
    publisher.start()
    async with asyncio.timeout(1):
        while publish.await_count < 1:
            await asyncio.sleep(0)
    publisher.update("unavailable")
    async with asyncio.timeout(1):
        while publish.await_count < 2:
            await asyncio.sleep(0)
    await publisher.aclose()
