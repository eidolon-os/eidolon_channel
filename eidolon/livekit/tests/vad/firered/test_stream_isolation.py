"""No devices or recordings: interleaved streams equal isolated inference."""
import numpy as np
import pytest
from livekit import rtc
from livekit.agents.vad import VADEventType
from eidolon.livekit.plugins.vad.firered.processor import PvadProcessor
from eidolon.livekit.plugins.vad.firered.vad import VAD
from eidolon.livekit.plugins.vad.firered.config import FireredPvadConfig


def test_interleaved_recurrence_and_reset_are_stream_local():
    resources = PvadProcessor()
    left, right = resources.new_stream(), resources.new_stream()
    control_left, control_right = resources.new_stream(), resources.new_stream()
    assert left.onnx_session is right.onnx_session is resources.onnx_session
    rng = np.random.default_rng(42)
    a = rng.normal(0, .08, (30, 160)).astype('float32')
    b = np.zeros((30, 160), 'float32')
    actual_a, actual_b = [], []
    for x, y in zip(a, b, strict=True):
        actual_a.append(left(x))
        actual_b.append(right(y))
    np.testing.assert_array_equal(actual_a, [control_left(x) for x in a])
    np.testing.assert_array_equal(actual_b, [control_right(y) for y in b])
    left.spkemb[:] = 1
    left.reset()
    np.testing.assert_array_equal(right.gru_buffer, control_right.gru_buffer)
    assert not np.shares_memory(left.mel_buffer, right.mel_buffer)
    assert not np.shares_memory(left.spkemb, right.spkemb)
    right.spkemb[:] = 1
    fresh = resources.new_stream()
    assert not fresh.spkemb.any()
    assert not fresh.gru_buffer.any()


@pytest.mark.asyncio
async def test_sdk_streams_do_not_inherit_other_speaker_or_history():
    vad = VAD(processor=PvadProcessor(), opts=FireredPvadConfig())
    first, second = vad.stream(), vad.stream()
    try:
        first._processor.spkemb[:] = 1
        first._processor.gru_buffer[:] = 1
        first._processor.reset()
        assert not second._processor.spkemb.any()
        assert first._processor is not second._processor
        assert first._processor.onnx_session is second._processor.onnx_session
        for stream in (first, second):
            stream.push_frame(rtc.AudioFrame(data=bytes(3200), sample_rate=16000,
                                            num_channels=1, samples_per_channel=1600))
            stream.end_input()
        events = [[e async for e in stream] for stream in (first, second)]
        probabilities = [[e.probability for e in es if e.type == VADEventType.INFERENCE_DONE]
                         for es in events]
        assert len(probabilities[0]) == 10
        assert probabilities[0] == probabilities[1]
    finally:
        await first.aclose()
        await second.aclose()
