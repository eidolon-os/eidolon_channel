"""The latency clock must include endpointing and exclude earlier assistant audio."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from livekit import rtc

from .._harness.audio import synth_voiced
from .._harness.headless import RecordingAudioOutput
from .._harness.latency import ReplyLatencyProbe
from eidolon.livekit.common.config import TurnPolicyConfig


@pytest.mark.parametrize('streaming', [False, True])
def test_latency_uses_acoustic_end_not_late_vad_or_old_reply(streaming):
    detector = SimpleNamespace(unlikely_threshold=AsyncMock())
    if not streaming:
        detector.predict_end_of_turn = AsyncMock()
    model = SimpleNamespace(update_asr=MagicMock())
    pipeline = SimpleNamespace(_timeline=SimpleNamespace(timestamps={'speech_stopped_at': 10.6}),
        _get_eot_model=lambda: model, _turn_policy=TurnPolicyConfig())
    input_ = SimpleNamespace(frame_delivered_at=[])
    handle = SimpleNamespace(
        audio_in=input_, agent=SimpleNamespace(turn_detection=detector), session=MagicMock(),
        audio_out=SimpleNamespace(segments=[
            SimpleNamespace(started_at=9, first_audible_at=9.1, cleared=False),
            SimpleNamespace(started_at=10.85, first_audible_at=10.9, cleared=False),
        ]),
        events=SimpleNamespace(of_type=lambda _: [SimpleNamespace(
            payload=SimpleNamespace(item=SimpleNamespace(role='user', text_content='测试')), timestamp=10.8)]),
    )
    probe = ReplyLatencyProbe(pipeline, handle)
    # 40 ms speech followed by 200 ms trailing silence. VAD end is much later.
    input_.frame_delivered_at.extend([9.98, 10.0] + [10.02 + .02*i for i in range(10)])
    probe.predictions = [(10.7, .9, .5)]
    probe.final_predictions = [(10.5, .9, '测试')]
    result = probe.report(synth_voiced(.04) + bytes(6400))
    assert result['stop_to_reply_audio_ms'] == 900
    assert result['complete_to_reply_audio_ms'] == 400
    assert result['sdk_complete_to_reply_audio_ms'] == 200
    assert result['from_acoustic_stop_ms']['speech_stopped_at'] == 600
    probe.close()
    assert getattr(detector, 'predict_end_of_turn', None) is probe.predict


@pytest.mark.asyncio
async def test_silent_tts_prefix_does_not_count_as_audible_reply():
    sink = RecordingAudioOutput()
    await sink.capture_frame(rtc.AudioFrame(data=bytes(640), sample_rate=16000, num_channels=1, samples_per_channel=320))
    assert sink._current.first_audible_at is None
    await sink.capture_frame(rtc.AudioFrame(data=synth_voiced(.02), sample_rate=16000, num_channels=1, samples_per_channel=320))
    assert sink._current.first_audible_at is not None
    sink.flush()
    assert sink.segments[0].first_audible_at >= sink.segments[0].started_at
