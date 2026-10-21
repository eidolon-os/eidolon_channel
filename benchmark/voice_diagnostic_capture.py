"""Explicit, bounded diagnostic capture; never part of the conversation pipeline.

Run only during a consented test window. Publishes nothing and dispatches no agent.
Capture PCM is private and temporary; only anonymous analysis metrics may survive.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import signal
import tempfile
import time
import uuid
import wave


async def capture_room(*, room, rtc, url, token, participant, wall_seconds, audio_seconds):
    if not participant or not (math.isfinite(wall_seconds) and 0 < wall_seconds <= 60):
        raise ValueError("explicit participant and wall limit in (0, 60] required")
    if not (math.isfinite(audio_seconds) and 0 < audio_seconds <= 30):
        raise ValueError("audio limit must be in (0, 30]")
    max_bytes = int(audio_seconds * 16000) * 2
    if max_bytes == 0:
        raise ValueError("audio limit must contain at least one sample")
    pcm = bytearray()
    frames = []
    tasks = []
    finished = asyncio.Event()
    selected_sid = None
    started = time.monotonic()
    stop_reason = "wall_timeout"

    def matches(publication, remote):
        return (remote.identity == participant
                and publication.kind == rtc.TrackKind.KIND_AUDIO
                and publication.source == rtc.TrackSource.SOURCE_MICROPHONE)

    @room.on("track_published")
    def published(publication, remote):
        nonlocal selected_sid
        if matches(publication, remote) and selected_sid is None:
            selected_sid = publication.sid
            publication.set_subscribed(True)

    async def consume(track):
        nonlocal stop_reason
        stream = rtc.AudioStream(track, sample_rate=16000, num_channels=1, frame_size_ms=20)
        try:
            async for event in stream:
                payload = bytes(event.frame.data)
                if not payload or len(payload) % 2:
                    stop_reason = "invalid_frame"
                    break
                payload = payload[:max_bytes - len(pcm)]
                frames.append(dict(sample_offset=len(pcm) // 2, samples=len(payload) // 2,
                                   received_ms=round((time.monotonic() - started) * 1000, 3)))
                pcm.extend(payload)
                if len(pcm) == max_bytes:
                    stop_reason = "audio_limit"
                    break
            else:
                stop_reason = "track_ended"
        except asyncio.CancelledError:
            raise
        except Exception:
            stop_reason = "capture_error"
        finally:
            finished.set()
            await stream.aclose()

    @room.on("track_subscribed")
    def subscribed(track, publication, remote):
        if matches(publication, remote) and publication.sid == selected_sid and not tasks:
            tasks.append(asyncio.create_task(consume(track)))

    @room.on("disconnected")
    def disconnected(*_):
        nonlocal stop_reason
        if not finished.is_set():
            stop_reason = "disconnected"
            finished.set()

    @room.on("track_unsubscribed")
    def unsubscribed(track, publication, remote):
        nonlocal stop_reason
        if publication.sid == selected_sid and matches(publication, remote):
            stop_reason = "track_ended"
            finished.set()

    try:
        async with asyncio.timeout(wall_seconds):
            await room.connect(url, token, options=rtc.RoomOptions(auto_subscribe=False))
            for remote in room.remote_participants.values():
                for publication in remote.track_publications.values():
                    published(publication, remote)
            print("CAPTURE_WINDOW_OPEN", flush=True)
            await finished.wait()
    except TimeoutError:
        stop_reason = "wall_timeout"
    finally:
        finished.set()  # Our own disconnect must not overwrite the original stop reason.
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await room.disconnect()
    return dict(pcm=bytes(pcm), frames=frames, stop_reason=stop_reason,
                sample_rate=16000, channels=1, duration_ms=len(pcm) / 32,
                reference="Decoded RTC microphone PCM; arrival times are not turn boundaries",
                window_started_unix=time.time() - (time.monotonic() - started))


@contextmanager
def private_capture(parent, result):
    """Own exactly one temporary directory; never touch enrolled recordings."""
    with tempfile.TemporaryDirectory(prefix="voice-test-", dir=parent) as name:
        directory = Path(name)
        fd = os.open(directory / "audio.wav", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as file:
            with wave.open(file, "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(result["pcm"])
        fd = os.open(directory / "capture.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as output:
            json.dump({k: v for k, v in result.items() if k != "pcm"}, output, allow_nan=False)
        yield directory


async def run(args):
    from eidolon_sdk.integrations.livekit import build_livekit_token
    from livekit import rtc

    token = build_livekit_token(
        api_key=os.environ["LIVEKIT_API_KEY"], api_secret=os.environ["LIVEKIT_API_SECRET"],
        room_name=args.room, identity=f"voice-diagnostic-{uuid.uuid4().hex}",
        dispatch_agent=False, can_publish=False, can_publish_data=False,
    )
    result = await capture_room(room=rtc.Room(), rtc=rtc, url=os.environ["LIVEKIT_URL"],
                                token=token, participant=args.participant,
                                wall_seconds=args.wall_seconds, audio_seconds=args.audio_seconds)
    if not result["pcm"]:
        print(json.dumps(dict(event="NO_AUDIO", reason=result["stop_reason"])), flush=True)
        return
    with private_capture(args.private_parent, result) as directory:
        # Ready for a separate local analysis process. No audio or vectors on stdout.
        result.pop("pcm")
        print(json.dumps(dict(event="CAPTURE_SAVED", directory=str(directory),
                              duration_ms=result["duration_ms"], reason=result["stop_reason"],
                              delete_by_unix=time.time() + args.retain_seconds)), flush=True)
        try:
            # Local analysis may request early deletion by creating this marker.
            async with asyncio.timeout(args.retain_seconds):
                while not (directory / "analysis.done").exists():
                    await asyncio.sleep(.25)
        except TimeoutError:
            pass
    print("CAPTURE_DELETED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-authorized-window", action="store_true", required=True)
    parser.add_argument("--room", required=True)
    parser.add_argument("--participant", required=True)
    parser.add_argument("--private-parent", type=Path, required=True,
                        help="existing private directory on diagnostic host")
    parser.add_argument("--wall-seconds", type=float, default=45)
    parser.add_argument("--audio-seconds", type=float, default=30)
    parser.add_argument("--retain-seconds", type=int, default=600)
    args = parser.parse_args()
    if not 1 <= args.retain_seconds <= 900:
        parser.error("retain-seconds must be in [1, 900]")
    parent = args.private_parent
    if parent.is_symlink() or not parent.is_dir() or parent.stat().st_mode & 0o077:
        parser.error("private-parent must be an existing private directory, not a symlink")
    # SIGINT/SIGTERM cancel the task so context managers remove only this run's files.
    async def supervised():
        task = asyncio.current_task()
        loop = asyncio.get_running_loop()
        for signum in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(signum, task.cancel)
        await run(args)
    try:
        asyncio.run(supervised())
    except asyncio.CancelledError:
        print("CAPTURE_CANCELLED_AND_CLEANED", flush=True)


if __name__ == "__main__":
    main()
