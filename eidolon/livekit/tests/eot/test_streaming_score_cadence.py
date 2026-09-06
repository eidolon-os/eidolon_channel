"""Provider cadence must not starve inference; short finals are valid speech."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from eidolon.livekit.plugins.eot.models.base import EidolonEOTModel


def model():
    m = EidolonEOTModel.__new__(EidolonEOTModel)
    m._state = SimpleNamespace(update_asr=Mock(), update_eot_score=Mock())
    m._context_eot = SimpleNamespace(p_complete_score=Mock(return_value=.4))
    m._current_eot_score = 0.0
    m._current_eot_text = ""
    m._last_asr_inference_time = None
    return m


def test_fast_continuous_interims_do_not_starve_inference(monkeypatch):
    m = model()
    now = [10.0]
    monkeypatch.setattr('eidolon.livekit.plugins.eot.models.base.time.monotonic', lambda: now[0])
    monkeypatch.setattr('eidolon.livekit.plugins.eot.models.base.time.time', lambda: now[0])
    for i in range(12):
        now[0] = 10.0 + i * .09
        m.update_asr('我想了解' + '一下' * i, False)
    assert 4 <= m._context_eot.p_complete_score.call_count <= 6


@pytest.mark.parametrize('text', ['好', '7', '停', '嗯', 'no', '对'])
def test_short_final_is_scored_without_phrase_whitelist(text):
    m = model()
    assert m.update_asr(text, True) == .4
    m._context_eot.p_complete_score.assert_called_once_with(text)
    m._state.update_eot_score.assert_called_with(.4)


def test_short_interim_clears_both_score_views():
    m = model()
    m._current_eot_score = .99
    assert m.update_asr('我', False) == 0
    m._state.update_eot_score.assert_called_with(0)


def test_throttled_revision_cannot_reuse_previous_high_score(monkeypatch):
    m = model()
    now = [10.0]
    monkeypatch.setattr('eidolon.livekit.plugins.eot.models.base.time.monotonic', lambda: now[0])
    m._context_eot.p_complete_score.side_effect = [.95, .1]
    assert m.update_asr('明天就这样安排。', False) == .95
    now[0] += .05
    assert m.update_asr('明天就这样安排。', False) == .95
    assert m.update_asr('不是，我还没有说完', False) == 0
    m._state.update_eot_score.assert_called_with(0)
    assert m._context_eot.p_complete_score.call_count == 1
    # Final evidence bypasses the throttle and acquires its own score.
    assert m.update_asr('不是，我还没有说完', True) == .1
    assert m._context_eot.p_complete_score.call_count == 2
