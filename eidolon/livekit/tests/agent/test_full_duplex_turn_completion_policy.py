from types import SimpleNamespace

from eidolon.livekit.agent.full_duplex.turn_completion_policy import (
    decide_low_eot_commit_deferral,
    eot_score_from_model,
    eot_thinks_turn_complete,
    looks_like_short_statement_continuation,
    select_combined_voiceprint_result,
    should_wait_for_inconclusive_voiceprint_merge,
    voiceprint_result_is_inconclusive,
)


def test_eot_score_reads_public_and_legacy_fields() -> None:
    assert eot_score_from_model(SimpleNamespace(current_eot_score=0.42), default=1.0) == 0.42
    assert eot_score_from_model(SimpleNamespace(_current_eot_score=0.31), default=1.0) == 0.31
    assert eot_score_from_model(None, default=1.0) == 1.0


def test_framework_completed_eot_contract_treats_missing_model_as_complete() -> None:
    assert eot_thinks_turn_complete(None, unlikely_threshold=0.5) is True
    assert (
        eot_thinks_turn_complete(
            SimpleNamespace(current_eot_score=0.49),
            unlikely_threshold=0.5,
        )
        is False
    )
    assert (
        eot_thinks_turn_complete(
            SimpleNamespace(current_eot_score=0.5),
            unlikely_threshold=0.5,
        )
        is True
    )


def test_low_eot_deferral_contract_keeps_short_statement_wait_path() -> None:
    low_score = decide_low_eot_commit_deferral(
        "我们继续",
        eot_score=0.2,
        unlikely_threshold=0.5,
        short_statement_defer_max_cjk_chars=8,
    )
    assert low_score.should_defer is True
    assert low_score.reason == "eot_score_unlikely"

    short_statement = decide_low_eot_commit_deferral(
        "还有一个，",
        eot_score=0.9,
        unlikely_threshold=0.5,
        short_statement_defer_max_cjk_chars=8,
    )
    assert short_statement.should_defer is True
    assert short_statement.reason == "short_statement_continuation"

    command = decide_low_eot_commit_deferral(
        "帮我总结，",
        eot_score=0.9,
        unlikely_threshold=0.5,
        short_statement_defer_max_cjk_chars=8,
    )
    assert command.should_defer is False
    assert command.reason == "turn_complete"


def test_short_statement_continuation_contract() -> None:
    assert (
        looks_like_short_statement_continuation(
            "还有一个，",
            max_cjk_chars=8,
        )
        is True
    )
    assert (
        looks_like_short_statement_continuation(
            "还有一个?",
            max_cjk_chars=8,
        )
        is False
    )
    assert (
        looks_like_short_statement_continuation(
            "please,",
            max_cjk_chars=8,
        )
        is False
    )
    assert (
        looks_like_short_statement_continuation(
            "这是一个比较长的补充，",
            max_cjk_chars=4,
        )
        is False
    )
    assert (
        looks_like_short_statement_continuation(
            "换个话题，",
            max_cjk_chars=8,
        )
        is False
    )


def test_select_combined_voiceprint_result_prefers_blocking_failure() -> None:
    allowed = SimpleNamespace(commit_allowed=True, commit_reason="owner_high_confidence")
    blocked = SimpleNamespace(commit_allowed=False, commit_reason="speaker_not_owner")
    inconclusive = SimpleNamespace(
        commit_allowed=False,
        commit_reason="audio_too_short",
    )

    assert select_combined_voiceprint_result([allowed, blocked]) is blocked
    assert select_combined_voiceprint_result([allowed, inconclusive]) is inconclusive
    assert select_combined_voiceprint_result([inconclusive, allowed]) is allowed
    assert select_combined_voiceprint_result([allowed, allowed]) is allowed
    assert select_combined_voiceprint_result([]) is None


def test_inconclusive_voiceprint_merge_wait_contract() -> None:
    assert (
        voiceprint_result_is_inconclusive(
            SimpleNamespace(commit_reason="audio_too_short")
        )
        is True
    )
    assert (
        should_wait_for_inconclusive_voiceprint_merge(
            commit_reason="audio_too_short",
            candidate_state="waiting_merge",
            selected_text="",
            transcript="",
            short_statement_defer_max_cjk_chars=8,
        )
        is True
    )
    assert (
        should_wait_for_inconclusive_voiceprint_merge(
            commit_reason="audio_too_short",
            candidate_state="collecting",
            selected_text="还有一个，",
            transcript="",
            short_statement_defer_max_cjk_chars=8,
        )
        is True
    )
    assert (
        should_wait_for_inconclusive_voiceprint_merge(
            commit_reason="audio_too_short",
            candidate_state="committed",
            selected_text="还有一个，",
            transcript="",
            short_statement_defer_max_cjk_chars=8,
        )
        is False
    )
    assert (
        should_wait_for_inconclusive_voiceprint_merge(
            commit_reason="speaker_not_owner",
            candidate_state="collecting",
            selected_text="还有一个，",
            transcript="",
            short_statement_defer_max_cjk_chars=8,
        )
        is False
    )
