"""A room must not close on old audio before its canonical reply exists."""

import asyncio
import json

import pytest

from benchmark.livekit_room_runner import _wait_for_canonical_reply
from benchmark.timeline import TimelineCapture


@pytest.mark.asyncio
async def test_wait_ignores_old_room_and_incomplete_reply(tmp_path):
    path = tmp_path / "timeline.jsonl"
    record = {
        "turn_id": "t",
        "attrs": {
            "room_name": "room",
            "canonical_user_text": {
                "text_preview": "前半句后半句",
            },
        },
        "timestamps": {
            "speech_stopped_at": 1,
            "turn_committed_at": 2,
            "brain_request_sent_at": 3,
            "brain_first_answer_delta_at": 4,
            "tts_first_audio_at": 5,
        },
    }
    path.write_text(json.dumps(record) + "\n")
    capture = TimelineCapture.start(str(path))
    task = asyncio.create_task(_wait_for_canonical_reply(capture, "room", ("前半句", "后半句")))
    try:
        incomplete = {**record, "timestamps": {"speech_stopped_at": 1}}
        other = {**record, "attrs": {**record["attrs"], "room_name": "other"}}
        with path.open("a") as f:
            f.write(json.dumps(incomplete) + "\n" + json.dumps(other) + "\n")
        await asyncio.sleep(0.08)
        assert not task.done()
        with path.open("a") as f:
            f.write(json.dumps(record))
        await asyncio.sleep(0.08)
        assert not task.done()  # A partially flushed JSONL row is not evidence.
        with path.open("a") as f:
            f.write("\n")
        await asyncio.wait_for(task, 1)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_wait_requires_timeline_and_is_cancellable(tmp_path):
    with pytest.raises(ValueError, match="timeline path"):
        await _wait_for_canonical_reply(TimelineCapture.start(""), "room", ("x",))
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(
            _wait_for_canonical_reply(
                TimelineCapture.start(str(tmp_path / "not-created")),
                "room",
                ("x",),
            ),
            0.02,
        )
