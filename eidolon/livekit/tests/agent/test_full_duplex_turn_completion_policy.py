from types import SimpleNamespace

from eidolon.livekit.agent.full_duplex.turn_completion_policy import (
    select_combined_voiceprint_result,
    voiceprint_result_is_inconclusive,
)


def test_select_combined_voiceprint_result_prefers_definitive_block() -> None:
    allowed = SimpleNamespace(commit_allowed=True, commit_reason="owner_high_confidence")
    blocked = SimpleNamespace(commit_allowed=False, commit_reason="speaker_not_owner")
    inconclusive = SimpleNamespace(
        commit_allowed=False,
        commit_reason="audio_too_short",
    )

    assert select_combined_voiceprint_result([allowed, blocked]) is blocked
    assert select_combined_voiceprint_result([inconclusive, blocked]) is blocked
    assert select_combined_voiceprint_result([blocked, allowed]) is blocked


def test_select_combined_voiceprint_result_allows_when_no_definitive_block() -> None:
    allowed = SimpleNamespace(commit_allowed=True, commit_reason="owner_high_confidence")
    inconclusive = SimpleNamespace(
        commit_allowed=False,
        commit_reason="audio_too_short",
    )

    assert select_combined_voiceprint_result([allowed, inconclusive]) is inconclusive
    assert select_combined_voiceprint_result([inconclusive, allowed]) is allowed
    assert select_combined_voiceprint_result([allowed, allowed]) is allowed
    assert select_combined_voiceprint_result([]) is None


def test_voiceprint_inconclusive_reason_contract() -> None:
    assert voiceprint_result_is_inconclusive(SimpleNamespace(commit_reason="audio_too_short"))
    assert not voiceprint_result_is_inconclusive(SimpleNamespace(commit_reason="speaker_not_owner"))
