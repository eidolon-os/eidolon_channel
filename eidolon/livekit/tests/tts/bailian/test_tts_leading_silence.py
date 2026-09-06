"""Keep quiet speech and internal pauses while removing provider zero padding."""
from types import SimpleNamespace

import pytest
from livekit.agents.utils.audio import AudioByteStream

from eidolon.livekit.plugins.tts.bailian.config import BailianTTSConfig
from eidolon.livekit.plugins.tts.bailian.tts import BailianSynthesizeStream


@pytest.mark.asyncio
@pytest.mark.parametrize("sample_rate", [16000, 24000])
@pytest.mark.parametrize("packet_bytes", [317, 640, 8192])
async def test_zero_prefix_only_is_skipped(sample_rate, packet_bytes):
    stream = BailianSynthesizeStream.__new__(BailianSynthesizeStream)
    stream._config = BailianTTSConfig(api_key="test", audio_format="pcm", sample_rate=sample_rate)
    stream._first_provider_audio_emitted = False
    stream._pcm_total_bytes = 0
    stream._audio_byte_stream = AudioByteStream(
        sample_rate, 1, samples_per_channel=sample_rate * 60 // 1000,
    )
    events = []
    stream._tts = SimpleNamespace(emit_provider_event=lambda name, **kw: events.append(name))
    output = []
    emitter = SimpleNamespace(push=output.append)
    silence = b"\0\0" * (sample_rate * 300 // 1000)
    # Even +/-1 PCM is retained. No RMS threshold may trim a quiet onset.
    onset = b"\1\0\xff\xff" * (sample_rate * 60 // 1000 // 2)
    interior = b"\0\0" * (sample_rate * 120 // 1000)
    tail = b"\2\0" * (sample_rate * 60 // 1000)
    pcm = silence + onset + interior + tail
    for start in range(0, len(pcm), packet_bytes):
        await stream._handle_audio_chunk(pcm[start:start + packet_bytes], emitter)
    assert b"".join(output) == onset + interior + tail
    assert stream._pcm_total_bytes == len(onset + interior + tail)
    # Provider timing must still describe first received bytes, including zeros.
    assert events == ["tts_provider_first_audio"]


def test_partial_flush_preserves_onset_frame_and_post_speech_zeros():
    stream = BailianSynthesizeStream.__new__(BailianSynthesizeStream)
    stream._pcm_total_bytes = 0
    byte_stream = AudioByteStream(16000, 1, samples_per_channel=960)
    output = []
    emitter = SimpleNamespace(push=output.append)
    # A silence-only final partial frame must not become fake output.
    assert byte_stream.push(b"\0\0" * 80) == []
    stream._emit_pcm_frames(byte_stream.flush(), emitter)
    assert output == []
    # Keep all zeros inside the first frame that contains any speech.
    onset = b"\0\0" * 79 + b"\1\0"
    assert byte_stream.push(onset) == []
    stream._emit_pcm_frames(byte_stream.flush(), emitter)
    assert b"".join(output) == onset
    tail = b"\0\0" * 80
    assert byte_stream.push(tail) == []
    stream._emit_pcm_frames(byte_stream.flush(), emitter)
    assert b"".join(output) == onset + tail
    assert stream._pcm_total_bytes == len(onset + tail)
