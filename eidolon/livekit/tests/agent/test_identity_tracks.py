"""A conversation's voice tracks (design §3.2)."""

import itertools

import numpy as np
import pytest

from eidolon.livekit.plugins.speaker_verification import tracks as tr


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


_turns = itertools.count()


def turn():
    return f"turn-{next(_turns)}"


def voice(i):
    v = np.zeros(4)
    v[i] = 1.0
    return v


def same_voice(vector):
    """LLR 10 for a track of exactly this voice, -10 otherwise."""
    return lambda track: 10.0 if np.allclose(track.probe(8)[0], vector) else -10.0


def test_a_segment_continues_the_track_of_its_voice_or_opens_one():
    tracks = tr.Tracks(clock=Clock())
    a1 = tracks.assign(voice(0), 1500, same_voice(voice(0)), 3.0, turn_id=turn())
    b1 = tracks.assign(voice(1), 1200, same_voice(voice(1)), 3.0, turn_id=turn())
    a2 = tracks.assign(voice(0), 1800, same_voice(voice(0)), 3.0, turn_id=turn())
    assert a2 is a1 and b1 is not a1
    assert a1.probe(8)[1:] == (3300, 2) and b1.probe(8)[1:] == (1200, 1)


def test_the_best_matching_track_is_continued():
    tracks = tr.Tracks(clock=Clock())
    first = tracks.assign(voice(0), 1000, lambda t: 0.0, 3.0, turn_id=turn())
    second = tracks.assign(voice(1), 1000, lambda t: 0.0, 3.0, turn_id=turn())
    chosen = tracks.assign(voice(1), 1000, lambda t: {first.track_id: 4.0, second.track_id: 6.0}[t.track_id], 3.0, turn_id=turn())
    assert chosen is second


def test_a_track_scores_its_latest_segments():
    tracks = tr.Tracks(clock=Clock())
    track = None
    for i in range(tr.MAX_SEGMENTS + 2):
        track = tracks.assign(voice(i % 4), 1000 + i, lambda t: 10.0, 3.0, turn_id=turn())
    assert len(track.segments) == tr.MAX_SEGMENTS
    vector, speech_ms, n = track.probe(3)
    assert n == 3 and speech_ms == sum(1000 + i for i in range(tr.MAX_SEGMENTS - 1, tr.MAX_SEGMENTS + 2))
    assert np.allclose(vector, np.mean([voice(i % 4) for i in range(7, 10)], axis=0))


def test_idle_tracks_expire_and_the_count_is_bounded():
    clock = Clock()
    tracks = tr.Tracks(clock=clock)
    old = tracks.assign(voice(0), 1000, lambda t: -10.0, 3.0, turn_id=turn())
    clock.now = tr.IDLE_S + 1
    again = tracks.assign(voice(0), 1000, lambda t: 10.0, 3.0, turn_id=turn())
    assert again is not old
    opened = [tracks.assign(voice(1), 1000, lambda t: -10.0, 3.0, turn_id=turn()) for _ in range(tr.MAX_TRACKS + 2)]
    assert len(tracks._tracks) == tr.MAX_TRACKS and opened[-1] in tracks._tracks
    assert again not in tracks._tracks


@pytest.mark.parametrize("llr,continued", [(3.0, True), (2.999, False)])
def test_continuing_needs_the_threshold(llr, continued):
    tracks = tr.Tracks(clock=Clock())
    first = tracks.assign(voice(0), 1000, lambda t: llr, 3.0, turn_id=turn())
    assert (tracks.assign(voice(0), 1000, lambda t: llr, 3.0, turn_id=turn()) is first) is continued


def test_a_turn_heard_again_replaces_its_segment():
    """A speculative prefix, then the whole turn: one segment, not two."""
    tracks = tr.Tracks(clock=Clock())
    prefix = tracks.assign(voice(0), 900, same_voice(voice(0)), 3.0, turn_id="turn-a")
    whole = tracks.assign(voice(0), 2400, same_voice(voice(0)), 3.0, turn_id="turn-a")
    assert whole is not prefix or len(whole.segments) == 1
    assert [t.probe(8)[1:] for t in tracks._tracks] == [(2400, 1)]
    # The next turn continues it as usual.
    tracks.assign(voice(0), 1000, same_voice(voice(0)), 3.0, turn_id="turn-b")
    assert [t.probe(8)[1:] for t in tracks._tracks] == [(3400, 2)]
