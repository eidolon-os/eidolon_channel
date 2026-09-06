"""Concurrent conversations share weights, never mutable turn state."""
from eidolon.livekit.agent.full_duplex.pipeline import StreamingPipeline
from eidolon.livekit.common.config import TurnPolicyConfig


def test_sessions_do_not_share_transcripts_scores_or_reset():
    first = StreamingPipeline.__new__(StreamingPipeline)
    second = StreamingPipeline.__new__(StreamingPipeline)
    first._turn_policy = second._turn_policy = TurnPolicyConfig()
    a, b = first._get_eot_model(), second._get_eot_model()
    assert a is not b
    assert first._get_eot_model() is a
    assert a._eot_manager is b._eot_manager, 'reuse model weights via existing EotManager'
    a.update_asr('我想了解一下明天的天气', is_final=True)
    score = a.current_eot_score
    b.update_asr('好', is_final=True)
    b.reset()
    assert a.current_eot_score == score > 0
