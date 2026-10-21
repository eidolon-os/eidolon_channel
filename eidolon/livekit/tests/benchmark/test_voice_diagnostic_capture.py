import asyncio
import json
from types import SimpleNamespace as NS
import wave

import pytest

from benchmark.voice_diagnostic_capture import capture_room, private_capture, run


class Publication:
    kind = 1
    source = 2

    def __init__(self, sid, source=2):
        self.sid = sid
        self.source = source
        self.subscriptions = []

    def set_subscribed(self, value):
        self.subscriptions.append(value)


class Room:
    def __init__(self, publications, frames=None):
        self.handlers = {}
        self.remote_participants = {
            identity: NS(identity=identity, track_publications={p.sid: p for p in pubs})
            for identity, pubs in publications.items()
        }
        self.frames = frames
        self.disconnected = False
        self.streams = []

    def on(self, event):
        def register(fn):
            self.handlers[event] = fn
            return fn
        return register

    async def connect(self, url, token, options):
        assert options.auto_subscribe is False
        if self.frames is not None:
            async def emit():
                await asyncio.sleep(0)
                for participant in self.remote_participants.values():
                    for pub in participant.track_publications.values():
                        if pub.subscriptions == [True]:
                            self.handlers['track_subscribed'](self.frames, pub, participant)
            asyncio.create_task(emit())

    async def disconnect(self):
        self.disconnected = True
        self.handlers['disconnected']()

    def stream(self, frames, **kwargs):
        assert kwargs == dict(sample_rate=16000, num_channels=1, frame_size_ms=20)
        room = self

        class Stream:
            closed = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if frames:
                    return NS(frame=NS(data=frames.pop(0)))
                await asyncio.Event().wait()

            async def aclose(self):
                self.closed = True

        stream = Stream()
        room.streams.append(stream)
        return stream


def rtc(room):
    return NS(TrackKind=NS(KIND_AUDIO=1), TrackSource=NS(SOURCE_MICROPHONE=2),
              RoomOptions=NS, AudioStream=room.stream)


async def test_capture_only_target_microphone_and_bound_pcm():
    target, other, screen = Publication('mic'), Publication('other'), Publication('screen', 3)
    room = Room({'box': [target, screen], 'agent': [other]}, [b'\x01\x00' * 32000])
    result = await capture_room(room=room, rtc=rtc(room), url='local', token='secret',
                                participant='box', wall_seconds=1, audio_seconds=1)
    assert target.subscriptions == [True]
    assert other.subscriptions == screen.subscriptions == []
    assert len(result['pcm']) == 32000
    assert result['stop_reason'] == 'audio_limit'
    assert room.disconnected and room.streams[0].closed
    assert result['frames'][0]['sample_offset'] == 0


async def test_missing_track_expires_without_audio():
    room = Room({'agent': [Publication('agent')]})
    result = await capture_room(room=room, rtc=rtc(room), url='local', token='secret',
                                participant='box', wall_seconds=.01, audio_seconds=1)
    assert result['pcm'] == b''
    assert result['stop_reason'] == 'wall_timeout'
    assert room.disconnected


async def test_cancel_closes_connection_and_stream():
    room = Room({'box': [Publication('mic')]}, [])
    task = asyncio.create_task(capture_room(room=room, rtc=rtc(room), url='local',
                               token='secret', participant='box', wall_seconds=1, audio_seconds=1))
    while not room.streams:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert room.disconnected and room.streams[0].closed


@pytest.mark.parametrize('wall,audio', [(61, 1), (1, 31), (0, 1), (float('nan'), 1)])
async def test_invalid_limits_never_connect(wall, audio):
    room = Room({})
    with pytest.raises(ValueError):
        await capture_room(room=room, rtc=rtc(room), url='local', token='secret',
                           participant='box', wall_seconds=wall, audio_seconds=audio)
    assert not room.disconnected


def test_private_files_are_deleted_even_on_analysis_error(tmp_path):
    result = dict(pcm=b'\x01\x00' * 320, stop_reason='wall_timeout', frames=[])
    with pytest.raises(RuntimeError):
        with private_capture(tmp_path, result) as directory:
            assert directory.stat().st_mode & 0o777 == 0o700
            assert (directory / 'audio.wav').stat().st_mode & 0o777 == 0o600
            assert (directory / 'capture.json').stat().st_mode & 0o777 == 0o600
            with wave.open(str(directory / 'audio.wav')) as audio:
                assert audio.getnframes() == 320
            assert 'pcm' not in json.loads((directory / 'capture.json').read_text())
            raise RuntimeError('analysis failed')
    assert not directory.exists()


async def test_expiry_deletes_capture_and_grants_cannot_publish_or_dispatch(tmp_path, monkeypatch):
    from eidolon_sdk.integrations import livekit
    from livekit import rtc as livekit_rtc
    import benchmark.voice_diagnostic_capture as module

    grants = {}

    def token(**kwargs):
        grants.update(kwargs)
        return 'test-token'

    async def capture(**kwargs):
        return dict(pcm=b'\x01\x00' * 320, frames=[], duration_ms=20, stop_reason='audio_limit')

    monkeypatch.setattr(livekit, 'build_livekit_token', token)
    monkeypatch.setattr(livekit_rtc, 'Room', lambda: None)
    monkeypatch.setattr(module, 'capture_room', capture)
    for key in ('LIVEKIT_API_KEY', 'LIVEKIT_API_SECRET', 'LIVEKIT_URL'):
        monkeypatch.setenv(key, 'test-value')
    await run(NS(room='room', participant='box', wall_seconds=1, audio_seconds=1,
                 private_parent=tmp_path, retain_seconds=.01))
    assert list(tmp_path.iterdir()) == []
    assert grants['dispatch_agent'] is False
    assert grants['can_publish'] is False
    assert grants['can_publish_data'] is False
    assert grants['room_name'] == 'room'
