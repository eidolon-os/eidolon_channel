# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Eidolon EOT plugin."""

import time

import pytest
from pathlib import Path


class TestEidolonEOTConfig:
    """Test EidolonEOTConfig."""

    def test_default_values(self):
        from eidolon.livekit.plugins.eot import EidolonEOTConfig

        config = EidolonEOTConfig()
        assert config.min_threshold == 0.4
        assert config.max_threshold == 2.2
        assert config.urgent_threshold == 0.18
        assert config.enable_semantic_tail_hang is True
        assert config.prefer_multilingual is False
        assert config.model_dir is None

    def test_custom_values(self):
        from eidolon.livekit.plugins.eot import EidolonEOTConfig

        config = EidolonEOTConfig(
            min_threshold=0.5,
            max_threshold=3.0,
            prefer_multilingual=True,
        )
        assert config.min_threshold == 0.5
        assert config.max_threshold == 3.0
        assert config.prefer_multilingual is True

    def test_new_config_fields(self):
        from eidolon.livekit.plugins.eot import EidolonEOTConfig

        config = EidolonEOTConfig()
        # Newly added config fields
        assert config.tail_hang_silence_sec == 2.5
        assert config.streaming_eot_base_threshold == 0.7
        assert config.streaming_eot_weak_threshold == 0.5
        assert config.semantic_threshold_vad_active_delta == 0.1
        assert config.is_final_threshold_reduction == 0.2
        assert config.context_eot_cooldown_sec == 0.8
        assert config.vad_flip_window_sec == 1.0
        assert config.vad_flip_count_threshold == 3
        assert config.asr_stability_short_sec == 0.10
        assert config.asr_stability_long_sec == 0.40
        assert config.asr_stability_long_char_threshold == 10
        # Dead fields removed
        assert not hasattr(config, "playing_eot_threshold")
        assert not hasattr(config, "playing_min_chars")


class TestConstants:
    """Test that constants are correctly defined."""

    def test_filler_words(self):
        from eidolon.livekit.plugins.eot.impl.constants import FILLER_WORDS

        assert "嗯" in FILLER_WORDS
        assert "啊" in FILLER_WORDS
        assert "hello" in FILLER_WORDS

    def test_command_words(self):
        from eidolon.livekit.plugins.eot.impl.constants import COMMAND_WORDS

        assert "停" in COMMAND_WORDS
        assert "闭嘴" in COMMAND_WORDS

    def test_strong_interrupt_words(self):
        from eidolon.livekit.plugins.eot.impl.constants import (
            STRONG_INTERRUPT_INTENT_WORDS,
        )

        assert "停" in STRONG_INTERRUPT_INTENT_WORDS
        assert "闭嘴" in STRONG_INTERRUPT_INTENT_WORDS

    def test_terminal_punctuation(self):
        from eidolon.livekit.plugins.eot.impl.constants import (
            TERMINAL_PUNCTUATION,
        )

        assert "。" in TERMINAL_PUNCTUATION
        assert "？" in TERMINAL_PUNCTUATION
        assert "！" in TERMINAL_PUNCTUATION

    def test_continuation_intent_patterns(self):
        from eidolon.livekit.plugins.eot.impl.constants import (
            CONTINUATION_INTENT_PATTERNS,
        )

        # Replacement / swap requests
        assert "换一个" in CONTINUATION_INTENT_PATTERNS
        assert "再换一个" in CONTINUATION_INTENT_PATTERNS
        assert "换别的" in CONTINUATION_INTENT_PATTERNS
        # Continuation requests
        assert "继续" in CONTINUATION_INTENT_PATTERNS
        assert "继续说" in CONTINUATION_INTENT_PATTERNS
        assert "接着" in CONTINUATION_INTENT_PATTERNS
        # Negative-form corrections
        assert "不是，" in CONTINUATION_INTENT_PATTERNS
        # Demand-based requests
        assert "不好笑" in CONTINUATION_INTENT_PATTERNS
        assert "来一个" in CONTINUATION_INTENT_PATTERNS


class TestTurnEndPolicy:
    """Test TurnEndPolicy."""

    def test_strong_interrupt_intent(self):
        from eidolon.livekit.plugins.eot import (
            TurnEndPolicy,
        )

        policy = TurnEndPolicy()
        assert policy.is_strong_interrupt_intent("停") is True
        assert policy.is_strong_interrupt_intent("闭嘴") is True
        assert policy.is_strong_interrupt_intent("你好") is False

    def test_weak_interrupt_intent(self):
        from eidolon.livekit.plugins.eot import (
            TurnEndPolicy,
        )

        policy = TurnEndPolicy()
        assert policy.is_weak_interrupt_intent("好的") is True
        assert policy.is_weak_interrupt_intent("那个") is True
        assert policy.is_weak_interrupt_intent("北京") is False

    def test_continuation_intent(self):
        from eidolon.livekit.plugins.eot import (
            TurnEndPolicy,
        )

        policy = TurnEndPolicy()
        # Replacement / swap requests
        assert policy.is_continuation_intent("再换一个") is True
        assert policy.is_continuation_intent("给我换一个笑话") is True
        assert policy.is_continuation_intent("换一个吧") is True
        assert policy.is_continuation_intent("不好笑，换一个") is True
        assert policy.is_continuation_intent("这个不好笑") is True
        # Continuation requests
        assert policy.is_continuation_intent("继续说") is True
        assert policy.is_continuation_intent("接着讲") is True
        assert policy.is_continuation_intent("然后呢") is True
        # Negative-form corrections (overall intent is to continue, not stop)
        assert policy.is_continuation_intent("不是，你要换一个新的") is True
        # Substring match: "继续" inside a longer phrase
        assert policy.is_continuation_intent("我想继续听笑话") is True
        # Non-matching
        assert policy.is_continuation_intent("停") is False
        assert policy.is_continuation_intent("闭嘴") is False
        assert policy.is_continuation_intent("你好") is False
        assert policy.is_continuation_intent("") is False
        assert policy.is_continuation_intent("   ") is False

    def test_continuation_vs_strong_interrupt_priority(self):
        # Even if "不对" is a strong interrupt word, when paired with a continuation
        # request the overall intent is to continue. The continuation check runs first.
        from eidolon.livekit.plugins.eot import (
            TurnEndPolicy,
        )

        policy = TurnEndPolicy()
        # "不是，你要换一个新的" — continuation intent takes priority
        assert policy.is_continuation_intent("不是，你要换一个新的") is True
        # Standalone "不对" — strong interrupt (not a continuation pattern)
        assert policy.is_strong_interrupt_intent("不对") is True
        assert policy.is_continuation_intent("不对") is False

    def test_is_valid_speech(self):
        from eidolon.livekit.plugins.eot import (
            TurnEndPolicy,
        )

        policy = TurnEndPolicy()
        assert policy.is_valid_speech("你好") is True
        assert policy.is_valid_speech("我想问一下") is True
        assert policy.is_valid_speech("") is False
        assert policy.is_valid_speech("    ") is False

    def test_dynamic_threshold_command(self):
        from eidolon.livekit.plugins.eot import (
            TurnEndPolicy,
        )

        policy = TurnEndPolicy()
        # Command words should use T_URGENT
        threshold = policy.get_dynamic_threshold("停", p_complete=0.5, is_final=False)
        assert threshold == pytest.approx(policy.T_URGENT)

    def test_dynamic_threshold_high_score(self):
        from eidolon.livekit.plugins.eot import (
            TurnEndPolicy,
        )

        policy = TurnEndPolicy()
        # High EOT score should use T_FAST
        threshold = policy.get_dynamic_threshold("你好，请问天气怎么样？", p_complete=0.9, is_final=True)
        assert threshold == pytest.approx(policy.T_URGENT)

    def test_dynamic_threshold_medium_score(self):
        from eidolon.livekit.plugins.eot import (
            TurnEndPolicy,
        )

        policy = TurnEndPolicy()
        # Medium EOT score should use T_MID
        threshold = policy.get_dynamic_threshold("你好", p_complete=0.5, is_final=False)
        assert threshold == pytest.approx(policy.T_MID)

    # -----------------------------------------------------------------
    # Round 7 G3: question-ending detection
    # -----------------------------------------------------------------

    def test_is_question_ending_simple_particles(self):
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        policy = TurnEndPolicy()
        # Strong question particles at end → True
        for txt in ["你好吗", "你看呢", "可以么", "他来嘛"]:
            assert policy.is_question_ending(txt), f"{txt!r} should be a question"

    def test_is_question_ending_with_punctuation(self):
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        policy = TurnEndPolicy()
        # Particle followed by trailing punctuation still detected
        for txt in ["你好吗?", "你好吗。", "你好吗,"]:
            assert policy.is_question_ending(txt), f"{txt!r} should be a question"

    def test_is_question_ending_bound_forms(self):
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        policy = TurnEndPolicy()
        for txt in ["这样行不行", "你看好不好", "他来不来是不是", "可不可以"]:
            assert policy.is_question_ending(txt), f"{txt!r} should be a question"

    def test_is_question_ending_declarative_negative(self):
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        policy = TurnEndPolicy()
        # Declarative sentences that DON'T end with question particles
        for txt in ["今天天气真不错", "我要点一杯咖啡", "好的我知道了"]:
            assert not policy.is_question_ending(txt), (
                f"{txt!r} should NOT be a question"
            )

    def test_is_question_ending_ambiguous_particles_not_matched(self):
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        policy = TurnEndPolicy()
        # 吧 / 啊 / 呀 are ambiguous (suggestion / exclamation / emphasis),
        # NOT matched as question to avoid false positives.
        for txt in ["走吧", "好啊", "你来呀"]:
            assert not policy.is_question_ending(txt), (
                f"{txt!r} should NOT be matched (ambiguous particle)"
            )

    def test_is_question_ending_empty(self):
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        policy = TurnEndPolicy()
        for txt in ["", "   ", "。。。"]:
            assert not policy.is_question_ending(txt)

    def test_dynamic_threshold_question_ending_is_final(self):
        """G3: implicit question (no "?") at end of final transcript →
        T_URGENT, same as terminal_punctuation."""
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        policy = TurnEndPolicy()
        threshold = policy.get_dynamic_threshold(
            "你好吗", p_complete=0.5, is_final=True,
        )
        assert threshold == pytest.approx(policy.T_URGENT)

    def test_dynamic_threshold_question_ending_not_final(self):
        """G3: implicit question, not yet final → T_MIN (still short)."""
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        policy = TurnEndPolicy()
        threshold = policy.get_dynamic_threshold(
            "你好吗", p_complete=0.5, is_final=False,
        )
        assert threshold == pytest.approx(policy.T_MIN)

    def test_dynamic_threshold_no_question_falls_through(self):
        """Declarative without punctuation → falls to EOT-tier logic."""
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        policy = TurnEndPolicy()
        threshold = policy.get_dynamic_threshold(
            "我要去北京", p_complete=0.5, is_final=False,
        )
        # Falls through to EOT tier (medium score → T_MID), NOT T_URGENT/T_MIN.
        assert threshold == pytest.approx(policy.T_MID)

    def test_dynamic_threshold_priority_command_beats_question(self):
        """COMMAND_WORDS still wins over question detection."""
        from eidolon.livekit.plugins.eot import TurnEndPolicy
        from eidolon.livekit.plugins.eot.impl.constants import COMMAND_WORDS

        policy = TurnEndPolicy()
        # Pick a command word that doesn't end in a question particle
        cmd = next(c for c in COMMAND_WORDS if not c.endswith(("吗", "呢", "么", "嘛")))
        threshold = policy.get_dynamic_threshold(
            cmd, p_complete=0.5, is_final=False,
        )
        assert threshold == pytest.approx(policy.T_URGENT)


class TestTurnDetectionStateManager:
    """Test TurnDetectionStateManager."""

    def test_initialization(self):
        from eidolon.livekit.plugins.eot import (
            TurnDetectionStateManager,
        )

        manager = TurnDetectionStateManager()
        assert manager.vad_active is False
        assert manager.current_text == ""
        assert manager.eot_score == 0.0

    def test_vad_state(self):
        from eidolon.livekit.plugins.eot import (
            TurnDetectionStateManager,
        )

        manager = TurnDetectionStateManager()
        manager.vad_active = True
        assert manager.vad_active is True

    def test_asr_state(self):
        from eidolon.livekit.plugins.eot import (
            TurnDetectionStateManager,
        )

        manager = TurnDetectionStateManager()
        manager.current_text = "你好"
        assert manager.current_text == "你好"

    def test_update_vad_sets_active(self):
        from eidolon.livekit.plugins.eot import (
            TurnDetectionStateManager,
        )

        manager = TurnDetectionStateManager()
        manager.update_vad(True)
        assert manager.vad_active is True
        assert manager._vad.active_since is not None

    def test_update_vad_sets_silence_time(self):
        from eidolon.livekit.plugins.eot import (
            TurnDetectionStateManager,
        )

        manager = TurnDetectionStateManager()
        manager._vad.last_active_time = time.time() - 1.0
        manager.update_vad(False)
        assert manager.vad_active is False
        assert manager._vad.last_silence_time > 0

    def test_update_vad_records_transitions(self):
        from eidolon.livekit.plugins.eot import (
            TurnDetectionStateManager,
        )

        manager = TurnDetectionStateManager()
        manager.update_vad(True)   # 1 transition: False -> True
        manager.update_vad(False)  # 2 transitions: False -> True -> False
        manager.update_vad(True)   # 3 transitions: False -> True -> False -> True
        assert len(manager._vad.transition_times) == 3

    def test_update_asr_sets_final(self):
        from eidolon.livekit.plugins.eot import (
            TurnDetectionStateManager,
        )

        manager = TurnDetectionStateManager()
        manager.update_asr("你好", is_final=True)
        assert manager._asr.is_final is True
        assert manager._asr.stable_since is not None

    def test_update_asr_clears_stable_on_partial(self):
        from eidolon.livekit.plugins.eot import (
            TurnDetectionStateManager,
        )

        manager = TurnDetectionStateManager()
        manager._asr.stable_since = time.time() - 1.0
        manager.update_asr("你好", is_final=False)
        assert manager._asr.stable_since is None

    def test_get_speech_duration(self):
        from eidolon.livekit.plugins.eot import (
            TurnDetectionStateManager,
        )

        manager = TurnDetectionStateManager()
        manager._vad.active = True
        manager._vad.active_since = time.time() - 2.0
        duration = manager.get_speech_duration()
        assert 1.9 < duration < 2.1

    def test_get_speech_duration_inactive(self):
        from eidolon.livekit.plugins.eot import (
            TurnDetectionStateManager,
        )

        manager = TurnDetectionStateManager()
        manager._vad.active = False
        assert manager.get_speech_duration() == 0.0

    def test_get_silence_duration(self):
        from eidolon.livekit.plugins.eot import (
            TurnDetectionStateManager,
        )

        manager = TurnDetectionStateManager()
        manager._vad.active = False
        manager._vad.last_active_time = time.time() - 1.0
        duration = manager.get_silence_duration()
        assert 0.9 < duration < 1.1

    def test_get_silence_duration_active(self):
        from eidolon.livekit.plugins.eot import (
            TurnDetectionStateManager,
        )

        manager = TurnDetectionStateManager()
        manager._vad.active = True
        assert manager.get_silence_duration() == 0.0

    def test_check_vad_stale_timeout(self):
        from eidolon.livekit.plugins.eot import (
            TurnDetectionStateManager,
        )

        manager = TurnDetectionStateManager(vad_stale_timeout=0.3)
        manager._vad.active = True
        manager._vad.last_active_time = time.time() - 0.5
        assert manager.check_vad_stale() is True

    def test_check_vad_stale_active(self):
        from eidolon.livekit.plugins.eot import (
            TurnDetectionStateManager,
        )

        manager = TurnDetectionStateManager(vad_stale_timeout=0.3)
        manager._vad.active = True
        manager._vad.last_active_time = time.time()
        assert manager.check_vad_stale() is False

    def test_check_max_duration_exceeded(self):
        from eidolon.livekit.plugins.eot import (
            TurnDetectionStateManager,
        )

        manager = TurnDetectionStateManager(max_sentence_duration=10.0)
        manager._sentence.start_time = time.time() - 15.0
        assert manager.check_max_duration() is True

    def test_check_min_interval_met(self):
        from eidolon.livekit.plugins.eot import (
            TurnDetectionStateManager,
        )

        manager = TurnDetectionStateManager(min_cut_interval=0.5)
        manager._sentence.last_cut_time = time.time() - 1.0
        assert manager.check_min_interval() is True

    def test_check_min_interval_not_met(self):
        from eidolon.livekit.plugins.eot import (
            TurnDetectionStateManager,
        )

        manager = TurnDetectionStateManager(min_cut_interval=0.5)
        manager._sentence.last_cut_time = time.time()
        assert manager.check_min_interval() is False

    def test_record_cut_stores_text(self):
        from eidolon.livekit.plugins.eot import (
            TurnDetectionStateManager,
        )

        manager = TurnDetectionStateManager()
        manager.record_cut("abc123", "你好")
        assert manager._sentence.last_cut_time > 0
        assert manager._sentence.last_cut_text_hash == "abc123"
        assert manager._sentence.last_cut_text == "你好"

    def test_reset_session(self):
        from eidolon.livekit.plugins.eot import (
            TurnDetectionStateManager,
        )

        manager = TurnDetectionStateManager()
        manager.update_vad(True)
        manager.current_text = "test"
        manager.reset_session("session1", 1)
        assert manager.vad_active is False
        assert manager.current_text == ""


class TestUtils:
    """Test utility functions."""

    def test_longest_common_prefix_len(self):
        from eidolon.livekit.plugins.eot import (
            longest_common_prefix_len,
        )

        assert longest_common_prefix_len("你好世界", "你好吗") == 2
        assert longest_common_prefix_len("你好", "你好") == 2
        assert longest_common_prefix_len("abc", "xyz") == 0

    def test_compute_text_hash(self):
        from eidolon.livekit.plugins.eot import compute_text_hash

        h1 = compute_text_hash("hello")
        h2 = compute_text_hash("hello")
        h3 = compute_text_hash("world")
        assert h1 == h2
        assert h1 != h3

    def test_is_similar_text(self):
        from eidolon.livekit.plugins.eot import is_similar_text

        assert is_similar_text("你好世界", "你好吗", threshold=0.5) is True
        assert is_similar_text("abc", "xyz", threshold=0.5) is False


class TestPolicyChain:
    """Test PolicyChain."""

    def test_for_normal_turn_end_chain(self):
        from eidolon.livekit.plugins.eot import PolicyChain

        chain = PolicyChain.for_normal_turn_end()
        assert len(chain._policies) > 0

    def test_for_semantic_interruption_chain(self):
        from eidolon.livekit.plugins.eot import PolicyChain

        chain = PolicyChain.for_semantic_interruption()
        assert len(chain._policies) > 0

    def test_semantic_chain_has_eot_score_semantic_policy(self):
        from eidolon.livekit.plugins.eot import (
            PolicyChain,
            EOTScoreSemanticPolicy,
        )

        chain = PolicyChain.for_semantic_interruption()
        policy_types = [type(p).__name__ for p in chain._policies]
        assert "EOTScoreSemanticPolicy" in policy_types

    def test_normal_chain_has_eot_score_policy(self):
        from eidolon.livekit.plugins.eot import (
            PolicyChain,
            EOTScorePolicy,
        )

        chain = PolicyChain.for_normal_turn_end()
        policy_types = [type(p).__name__ for p in chain._policies]
        assert "EOTScorePolicy" in policy_types

    def test_min_interval_policy_blocks_when_not_met(self):
        from eidolon.livekit.plugins.eot import MinIntervalPolicy
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        state = TurnDetectionStateManager(min_cut_interval=0.5)
        state._sentence.last_cut_time = time.time()  # just now
        policy = MinIntervalPolicy()
        result = policy.check(state, TurnEndPolicy())
        # Returns CutDecision(should_cut=False) to block rapid cuts
        assert result is not None
        assert result.should_cut is False

    def test_interrupt_intent_policy_strong_interrupt(self):
        from eidolon.livekit.plugins.eot import (
            InterruptIntentPolicy,
        )
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        policy = InterruptIntentPolicy()
        state = TurnDetectionStateManager()
        state.current_text = "停"
        result = policy.check(state, TurnEndPolicy())
        assert result is not None
        assert result.should_cut is True
        assert "strong interrupt" in result.reason

    def test_interrupt_intent_policy_continuation(self):
        from eidolon.livekit.plugins.eot import (
            InterruptIntentPolicy,
        )
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        policy = InterruptIntentPolicy()
        state = TurnDetectionStateManager()
        state.current_text = "再换一个笑话"
        result = policy.check(state, TurnEndPolicy())
        assert result is not None
        assert result.should_cut is False
        assert "continuation" in result.reason

    def test_interrupt_intent_policy_no_override(self):
        from eidolon.livekit.plugins.eot import (
            InterruptIntentPolicy,
        )
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        policy = InterruptIntentPolicy()
        state = TurnDetectionStateManager()
        state.current_text = "北京天气怎么样"
        result = policy.check(state, TurnEndPolicy())
        # Returns None when no override applies
        assert result is None


class TestBackchannelSuppressionPolicy:
    """Round 7 G1 — block 'uh-huh' style acks while agent speaks."""

    def _state_with_text(self, text):
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager

        state = TurnDetectionStateManager()
        state.current_text = text
        return state

    def test_simple_chinese_ack_blocked(self):
        from eidolon.livekit.plugins.eot import (
            BackchannelSuppressionPolicy, TurnEndPolicy,
        )
        policy = BackchannelSuppressionPolicy()
        for txt in ["嗯", "嗯嗯", "好的", "对", "是的", "可以"]:
            state = self._state_with_text(txt)
            result = policy.check(state, TurnEndPolicy())
            assert result is not None and result.should_cut is False, (
                f"text={txt!r} should be blocked as backchannel"
            )

    def test_english_ack_blocked(self):
        from eidolon.livekit.plugins.eot import (
            BackchannelSuppressionPolicy, TurnEndPolicy,
        )
        policy = BackchannelSuppressionPolicy()
        for txt in ["OK", "okay", "yes", "yeah", "uh-huh", "right"]:
            state = self._state_with_text(txt)
            result = policy.check(state, TurnEndPolicy())
            assert result is not None and result.should_cut is False, (
                f"text={txt!r} should be blocked as backchannel"
            )

    def test_punctuation_stripped(self):
        from eidolon.livekit.plugins.eot import (
            BackchannelSuppressionPolicy, TurnEndPolicy,
        )
        policy = BackchannelSuppressionPolicy()
        # ASR sometimes appends punctuation; backchannel should still match.
        for txt in ["嗯。", "好的!", "OK!", "嗯嗯,"]:
            state = self._state_with_text(txt)
            result = policy.check(state, TurnEndPolicy())
            assert result is not None and result.should_cut is False, (
                f"text={txt!r} should be blocked despite punctuation"
            )

    def test_compound_backchannel_blocked(self):
        from eidolon.livekit.plugins.eot import (
            BackchannelSuppressionPolicy, TurnEndPolicy,
        )
        policy = BackchannelSuppressionPolicy()
        # Compound vocalisations made of single-char backchannel atoms.
        for txt in ["嗯好", "嗯对", "好对", "嗯嗯好"]:
            state = self._state_with_text(txt)
            result = policy.check(state, TurnEndPolicy())
            assert result is not None and result.should_cut is False, (
                f"text={txt!r} should be blocked as compound backchannel"
            )

    def test_real_speech_passes_through(self):
        from eidolon.livekit.plugins.eot import (
            BackchannelSuppressionPolicy, TurnEndPolicy,
        )
        policy = BackchannelSuppressionPolicy()
        for txt in ["北京天气怎么样", "好的我知道了", "停", "我想想看"]:
            state = self._state_with_text(txt)
            result = policy.check(state, TurnEndPolicy())
            assert result is None, (
                f"text={txt!r} is real speech, policy must abstain (got {result})"
            )

    def test_empty_text_abstains(self):
        from eidolon.livekit.plugins.eot import (
            BackchannelSuppressionPolicy, TurnEndPolicy,
        )
        policy = BackchannelSuppressionPolicy()
        result = policy.check(self._state_with_text(""), TurnEndPolicy())
        assert result is None

    def test_in_semantic_chain_only(self):
        """Backchannel suppression must be in semantic_interruption chain
        (agent speaking) but NOT in normal_turn_end chain."""
        from eidolon.livekit.plugins.eot import PolicyChain

        sem_types = [type(p).__name__ for p in
                     PolicyChain.for_semantic_interruption()._policies]
        normal_types = [type(p).__name__ for p in
                        PolicyChain.for_normal_turn_end()._policies]
        assert "BackchannelSuppressionPolicy" in sem_types
        assert "BackchannelSuppressionPolicy" not in normal_types


class TestNoiseLikeTranscriptPolicy:
    """Round 7 G2a — block cough / sigh / non-lexical noise transcripts."""

    def _state_with_text(self, text):
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager

        state = TurnDetectionStateManager()
        state.current_text = text
        return state

    def test_single_cough_token_blocked(self):
        from eidolon.livekit.plugins.eot import (
            NoiseLikeTranscriptPolicy, TurnEndPolicy,
        )
        policy = NoiseLikeTranscriptPolicy()
        for txt in ["啊", "咳", "咳咳", "嗯哼", "哎", "哎呀"]:
            state = self._state_with_text(txt)
            result = policy.check(state, TurnEndPolicy())
            assert result is not None and result.should_cut is False, (
                f"text={txt!r} should be blocked as noise"
            )

    def test_repeated_noise_chars_blocked(self):
        from eidolon.livekit.plugins.eot import (
            NoiseLikeTranscriptPolicy, TurnEndPolicy,
        )
        policy = NoiseLikeTranscriptPolicy()
        for txt in ["啊啊", "啊啊啊", "咳咳咳", "嗯嗯嗯嗯"]:
            state = self._state_with_text(txt)
            result = policy.check(state, TurnEndPolicy())
            assert result is not None and result.should_cut is False, (
                f"text={txt!r} should be blocked as repeated-noise"
            )

    def test_real_speech_passes_through(self):
        from eidolon.livekit.plugins.eot import (
            NoiseLikeTranscriptPolicy, TurnEndPolicy,
        )
        policy = NoiseLikeTranscriptPolicy()
        for txt in ["北京天气怎么样", "我要点一杯咖啡", "你好啊朋友"]:
            state = self._state_with_text(txt)
            result = policy.check(state, TurnEndPolicy())
            assert result is None, (
                f"text={txt!r} is real speech, policy must abstain"
            )

    def test_in_both_chains(self):
        """NoiseLike should be in BOTH chains — coughs aren't intentional in
        either turn-end nor interrupt scenarios."""
        from eidolon.livekit.plugins.eot import PolicyChain

        sem_types = [type(p).__name__ for p in
                     PolicyChain.for_semantic_interruption()._policies]
        normal_types = [type(p).__name__ for p in
                        PolicyChain.for_normal_turn_end()._policies]
        assert "NoiseLikeTranscriptPolicy" in sem_types
        assert "NoiseLikeTranscriptPolicy" in normal_types


class TestG6VadProbabilityState:
    """Round 7 G6 — per-frame VAD probability tracking in EOT state.

    Verifies that ``update_vad_probability`` records samples and that
    ``recent_avg_vad_confidence`` returns the time-windowed average.
    """

    def test_no_samples_returns_zero(self):
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager

        sm = TurnDetectionStateManager()
        assert sm.recent_avg_vad_confidence() == 0.0

    def test_single_sample(self):
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager

        sm = TurnDetectionStateManager()
        sm.update_vad_probability(0.7)
        assert sm.recent_avg_vad_confidence() == pytest.approx(0.7)

    def test_average_across_samples(self):
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager

        sm = TurnDetectionStateManager()
        for p in [0.6, 0.7, 0.8, 0.9, 1.0]:
            sm.update_vad_probability(p)
        # Average of [0.6, 0.7, 0.8, 0.9, 1.0] = 0.8
        assert sm.recent_avg_vad_confidence() == pytest.approx(0.8)

    def test_clamping_invalid_values(self):
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager

        sm = TurnDetectionStateManager()
        sm.update_vad_probability(1.5)   # > 1
        sm.update_vad_probability(-0.5)  # < 0
        sm.update_vad_probability(0.5)
        # 1.5 → 1.0, -0.5 → 0.0, 0.5 → 0.5; avg = 0.5
        assert sm.recent_avg_vad_confidence() == pytest.approx(0.5)

    def test_none_probability_ignored(self):
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager

        sm = TurnDetectionStateManager()
        sm.update_vad_probability(0.7)
        sm.update_vad_probability(None)  # ignored
        sm.update_vad_probability(0.9)
        assert sm.recent_avg_vad_confidence() == pytest.approx(0.8)

    def test_window_filters_old_samples(self):
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager

        sm = TurnDetectionStateManager()
        # Inject an old sample manually (simulate ~10s ago).
        old_t = time.time() - 10.0
        sm._vad.probability_samples.append((old_t, 0.1))
        # Add fresh samples
        sm.update_vad_probability(0.9)
        sm.update_vad_probability(0.9)
        # 2s window should exclude the old 0.1, only see 0.9 / 0.9
        assert sm.recent_avg_vad_confidence(window_sec=2.0) == pytest.approx(0.9)
        # Larger 30s window includes the old 0.1: avg = (0.1 + 0.9 + 0.9) / 3
        assert sm.recent_avg_vad_confidence(window_sec=30.0) == pytest.approx(
            (0.1 + 0.9 + 0.9) / 3, abs=0.01
        )

    def test_eot_model_bridge(self):
        """EidolonEOTModel.update_vad_probability forwards to state."""
        from eidolon.livekit.plugins.eot import ChineseModel

        m = ChineseModel()
        assert m._state.recent_avg_vad_confidence() == 0.0
        m.update_vad_probability(0.7)
        m.update_vad_probability(0.9)
        assert m._state.recent_avg_vad_confidence() == pytest.approx(0.8)

    def test_firered_vad_callback_hook(self):
        """FireredPvadVAD exposes register_inference_callback."""
        from eidolon.livekit.plugins.vad.firered import FireredPvadVAD

        vad = FireredPvadVAD.load()
        try:
            assert hasattr(vad, "register_inference_callback")
            assert vad._inference_callback is None

            calls = []

            def cb(probability, speaking):
                calls.append((probability, speaking))

            vad.register_inference_callback(cb)
            assert vad._inference_callback is cb

            # Clear with None
            vad.register_inference_callback(None)
            assert vad._inference_callback is None
        finally:
            FireredPvadVAD._processor = None


class TestG5ConversationPhase:
    """Round 7 G5 — phase detection + threshold scaling."""

    def test_detect_greeting_first_turn(self):
        from eidolon.livekit.plugins.eot import (
            ConversationPhase, ConversationPhaseDetector,
        )
        d = ConversationPhaseDetector()
        for txt in ["你好", "您好", "嗨"]:
            assert d.detect(txt, history_length=0) == ConversationPhase.GREETING

    def test_detect_greeting_only_in_early_turns(self):
        from eidolon.livekit.plugins.eot import (
            ConversationPhase, ConversationPhaseDetector,
        )
        d = ConversationPhaseDetector()
        # 5th turn: "你好" no longer GREETING (out of greeting window)
        assert d.detect("你好", history_length=5) == ConversationPhase.GATHERING

    def test_detect_thinking(self):
        from eidolon.livekit.plugins.eot import (
            ConversationPhase, ConversationPhaseDetector,
        )
        d = ConversationPhaseDetector()
        # Words actually in THINKING_WORDS (substring match works on these).
        for txt in ["让我想想", "我想想看", "我感觉这个", "我觉得吧"]:
            assert d.detect(txt, history_length=2) == ConversationPhase.THINKING

    def test_detect_closing_summary_words(self):
        from eidolon.livekit.plugins.eot import (
            ConversationPhase, ConversationPhaseDetector,
        )
        d = ConversationPhaseDetector()
        for txt in ["好的，就这样吧", "就这些", "讲完了"]:
            assert d.detect(txt, history_length=3) == ConversationPhase.CLOSING

    def test_detect_closing_full_word(self):
        from eidolon.livekit.plugins.eot import (
            ConversationPhase, ConversationPhaseDetector,
        )
        d = ConversationPhaseDetector()
        # Single word strong-ending words → CLOSING when whole utterance
        for txt in ["再见", "拜拜", "谢谢"]:
            assert d.detect(txt, history_length=2) == ConversationPhase.CLOSING

    def test_detect_gathering_default(self):
        from eidolon.livekit.plugins.eot import (
            ConversationPhase, ConversationPhaseDetector,
        )
        d = ConversationPhaseDetector()
        for txt in ["北京天气怎么样", "我要点一杯咖啡", ""]:
            assert d.detect(txt, history_length=2) == ConversationPhase.GATHERING

    def test_priority_closing_beats_thinking(self):
        """CLOSING has higher priority than THINKING."""
        from eidolon.livekit.plugins.eot import (
            ConversationPhase, ConversationPhaseDetector,
        )
        d = ConversationPhaseDetector()
        # Contains both "让我想想" (THINKING) and "就这样吧" (CLOSING-summary);
        # CLOSING wins per priority order.
        assert d.detect("好的让我想想就这样吧", history_length=2) == ConversationPhase.CLOSING

    def test_threshold_multiplier_loosens_thinking(self):
        from eidolon.livekit.plugins.eot import (
            ConversationPhase, TurnEndPolicy,
        )
        policy = TurnEndPolicy()
        base = policy.get_dynamic_threshold(
            "我想想", p_complete=0.5, is_final=False, phase=None,
        )
        thinking = policy.get_dynamic_threshold(
            "我想想", p_complete=0.5, is_final=False, phase=ConversationPhase.THINKING,
        )
        # THINKING multiplier (1.5) → larger threshold (allow longer pause).
        # But _ends_with_tail also returns T_TAIL_HANG for "我想想",
        # so test a non-tail text instead.
        base = policy.get_dynamic_threshold(
            "今天天气", p_complete=0.5, is_final=False, phase=None,
        )
        thinking = policy.get_dynamic_threshold(
            "今天天气", p_complete=0.5, is_final=False, phase=ConversationPhase.THINKING,
        )
        assert thinking > base

    def test_threshold_multiplier_tightens_greeting(self):
        from eidolon.livekit.plugins.eot import (
            ConversationPhase, TurnEndPolicy,
        )
        policy = TurnEndPolicy()
        base = policy.get_dynamic_threshold(
            "今天天气", p_complete=0.5, is_final=False, phase=None,
        )
        greeting = policy.get_dynamic_threshold(
            "今天天气", p_complete=0.5, is_final=False, phase=ConversationPhase.GREETING,
        )
        assert greeting <= base

    def test_threshold_clamped_to_T_URGENT_minimum(self):
        """Phase scaling never produces a threshold below T_URGENT."""
        from eidolon.livekit.plugins.eot import (
            ConversationPhase, TurnEndPolicy,
        )
        policy = TurnEndPolicy()
        # Even with most-aggressive phase + high score, can't dip below T_URGENT
        threshold = policy.get_dynamic_threshold(
            "好的", p_complete=0.95, is_final=True, phase=ConversationPhase.GREETING,
        )
        assert threshold >= policy.T_URGENT - 1e-6

    def test_eot_model_updates_phase_on_update_asr(self):
        """EidolonEOTModel.update_asr should refresh state.conversation_phase."""
        from eidolon.livekit.plugins.eot import (
            ChineseModel, ConversationPhase,
        )
        m = ChineseModel()
        m.start_session("test-session")
        m.update_asr("你好", is_final=True)
        # First turn (history empty) + greeting word → GREETING
        assert m._state.conversation_phase == ConversationPhase.GREETING

        # Simulate a recorded turn so history grows
        m.record_turn("你好", is_complete=True, eot_score=0.9)
        m.update_asr("让我想想这个问题", is_final=True)
        assert m._state.conversation_phase == ConversationPhase.THINKING

        m.update_asr("好的就这样吧", is_final=True)
        assert m._state.conversation_phase == ConversationPhase.CLOSING


class TestG2bMinSpeakingDurationVadGate:
    """G18c (2026-05-18) — VAD probability gate fully retired.

    The original Round 7 G2b gate (lowered to default 0.0 in Phase 1)
    is now a no-op even when constructor arguments request it: the
    ``check()`` method no longer reads ``min_avg_vad_confidence``.

    The constructor still accepts the parameters (backward compat) but
    nothing inside the gate enforces them. These tests document the
    new contract and guard against accidental re-introduction of the
    latency-inducing ramp-wait logic.
    """

    def _setup_state(self, vad_active=True, speech_duration=1.0):
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager

        sm = TurnDetectionStateManager()
        if vad_active:
            sm.update_vad(True)
            sm._vad.active_since = time.time() - speech_duration
        return sm

    def test_default_disabled(self):
        """Default behaviour unchanged — abstains."""
        from eidolon.livekit.plugins.eot import (
            MinSpeakingDurationPolicy, TurnEndPolicy,
        )

        policy = MinSpeakingDurationPolicy()
        sm = self._setup_state(vad_active=True, speech_duration=1.0)
        for _ in range(5):
            sm.update_vad_probability(0.3)
        assert policy.check(sm, TurnEndPolicy()) is None

    def test_low_confidence_no_longer_blocks(self):
        """G18c regression: even configured high, the gate doesn't fire.

        Pre-G18c, ``min_avg_vad_confidence=0.6`` + avg=0.3 would have
        returned a blocking CutDecision. Post-G18c, the gate abstains
        (returns None) so the chain can advance to score-based policies.
        """
        from eidolon.livekit.plugins.eot import (
            MinSpeakingDurationPolicy, TurnEndPolicy,
        )

        policy = MinSpeakingDurationPolicy(
            min_speech_duration_sec=0.15,
            min_avg_vad_confidence=0.6,
        )
        sm = self._setup_state(vad_active=True, speech_duration=1.0)
        for _ in range(5):
            sm.update_vad_probability(0.3)

        result = policy.check(sm, TurnEndPolicy())
        assert result is None, (
            "G18c: VAD confidence gate must be no-op (was: blocked cuts)"
        )

    def test_high_confidence_still_abstains(self):
        """Symmetric: high confidence also returns None (same as low)."""
        from eidolon.livekit.plugins.eot import (
            MinSpeakingDurationPolicy, TurnEndPolicy,
        )

        policy = MinSpeakingDurationPolicy(
            min_speech_duration_sec=0.15,
            min_avg_vad_confidence=0.6,
        )
        sm = self._setup_state(vad_active=True, speech_duration=1.0)
        for _ in range(5):
            sm.update_vad_probability(0.9)
        assert policy.check(sm, TurnEndPolicy()) is None

    def test_min_speech_duration_still_enforced(self):
        """The OTHER half of MinSpeakingDurationPolicy (short-utterance
        filter) is unchanged — must still block sub-threshold speech."""
        from eidolon.livekit.plugins.eot import (
            MinSpeakingDurationPolicy, TurnEndPolicy,
        )

        policy = MinSpeakingDurationPolicy(min_speech_duration_sec=0.15)
        sm = self._setup_state(vad_active=True, speech_duration=0.05)
        result = policy.check(sm, TurnEndPolicy())
        assert result is not None
        assert result.should_cut is False
        assert "too short" in result.reason.lower()

    def test_speech_duration_check_still_runs_first(self):
        """Speech-too-short check has higher priority than VAD confidence gate."""
        from eidolon.livekit.plugins.eot import (
            MinSpeakingDurationPolicy, TurnEndPolicy,
        )

        policy = MinSpeakingDurationPolicy(
            min_speech_duration_sec=0.5,
            min_avg_vad_confidence=0.6,
        )
        # Short speech (0.1s) but high confidence
        sm = self._setup_state(vad_active=True, speech_duration=0.1)
        sm.update_vad_probability(0.95)

        result = policy.check(sm, TurnEndPolicy())
        assert result is not None
        assert result.should_cut is False
        # Speech-too-short reason wins
        assert "too short" in result.reason

    def test_chain_factory_propagates_gate_config(self):
        """PolicyChain factory methods must accept and propagate the new
        VAD-confidence args to MinSpeakingDurationPolicy."""
        from eidolon.livekit.plugins.eot import (
            MinSpeakingDurationPolicy, PolicyChain,
        )

        chain = PolicyChain.for_semantic_interruption(
            min_avg_vad_confidence=0.5,
            confidence_window_sec=1.5,
        )
        for p in chain._policies:
            if isinstance(p, MinSpeakingDurationPolicy):
                assert p.min_avg_vad_confidence == 0.5
                assert p.confidence_window_sec == 1.5
                break
        else:
            pytest.fail("MinSpeakingDurationPolicy not in chain")


class TestG0aDualPathArchitecture:
    """Round 7 G0a — verify dual-path roles + record_interrupt API."""

    def test_record_interrupt_sets_internal_state(self):
        """record_interrupt() must update _last_interrupt_time and _last_text
        (the public replacement for direct private-field writes)."""
        from eidolon.livekit.plugins.eot import ChineseModel

        model = ChineseModel()
        before_t = model._context_eot._last_interrupt_time
        assert before_t is None or before_t == 0.0

        model._context_eot.record_interrupt("测试中断")

        assert model._context_eot._last_interrupt_time is not None
        assert model._context_eot._last_interrupt_time > 0
        assert model._context_eot._last_text == "测试中断"

    def test_max_history_respects_config(self):
        """Round 7 G0b — dialogue history maxlen must equal the config
        utterance_end_max_history (NOT the previously hard-coded 10)."""
        from eidolon.livekit.plugins.eot import ChineseModel
        from eidolon.livekit.plugins.eot import EidolonEOTConfig

        # Default config (utterance_end_max_history = 3)
        m = ChineseModel()
        assert m._context_eot._dialogue_history.maxlen == m._config.utterance_end_max_history

        # Custom config
        cfg = EidolonEOTConfig(utterance_end_max_history=7)
        from eidolon.livekit.plugins.eot.impl.context_enhanced_eot import ContextEnhancedEot
        ctx = ContextEnhancedEot(max_history=7)
        assert ctx._dialogue_history.maxlen == 7


class TestOnnxEotBackend:
    """Test OnnxEotBackend with bundled model."""

    def test_model_dir_resolution(self):
        from eidolon.livekit.plugins.eot.impl.eot_manager import (
            _resolve_default_model_dir,
        )

        default_dir = _resolve_default_model_dir()
        assert default_dir.exists()
        assert (default_dir / "chinese_best_model_q8.onnx").exists()
        assert (default_dir / "tokenizer").exists()

    def test_backend_loads(self):
        from eidolon.livekit.plugins.eot.impl.eot_backend import OnnxEotBackend

        # Model: eidolon/livekit/plugins/eot/data/model/firered_chat_turn_detector/
        # Test:   eidolon/livekit/tests/eot/test_eot_plugin.py
        # parent.parent.parent = eidolon/livekit/
        default_dir = (
            Path(__file__).parent.parent.parent
            / "plugins" / "eot" / "data" / "model" / "firered_chat_turn_detector"
        )
        backend = OnnxEotBackend(default_dir, prefer_multilingual=False)
        assert backend._session is not None
        assert backend._tokenizer is not None

    def test_score_chinese(self):
        from eidolon.livekit.plugins.eot.impl.eot_backend import OnnxEotBackend

        default_dir = (
            Path(__file__).parent.parent.parent
            / "plugins" / "eot" / "data" / "model" / "firered_chat_turn_detector"
        )
        backend = OnnxEotBackend(default_dir, prefer_multilingual=False)

        score = backend.score("你好")
        assert 0.0 <= score <= 1.0

        score2 = backend.score("你好，我想问一下今天的天气怎么样")
        assert 0.0 <= score2 <= 1.0

    def test_score_empty(self):
        from eidolon.livekit.plugins.eot.impl.eot_backend import OnnxEotBackend

        default_dir = (
            Path(__file__).parent.parent.parent
            / "plugins" / "eot" / "data" / "model" / "firered_chat_turn_detector"
        )
        backend = OnnxEotBackend(default_dir, prefer_multilingual=False)

        assert backend.score("") == 0.0
        assert backend.score("   ") == 0.0


class TestEidolonEOTModel:
    """Test EidolonEOTModel subclasses."""

    def test_chinese_model_init(self):
        from eidolon.livekit.plugins.eot import ChineseModel

        model = ChineseModel()
        assert model._config.prefer_multilingual is False

    def test_multilingual_model_init(self):
        from eidolon.livekit.plugins.eot import MultilingualModel

        model = MultilingualModel()
        assert model._config.prefer_multilingual is True

    def test_should_interrupt(self):
        from eidolon.livekit.plugins.eot import ChineseModel

        model = ChineseModel()
        # Should not interrupt for short empty text
        result = model.should_interrupt("你好", vad_active=True)
        assert isinstance(result, bool)

    def test_should_interrupt_continuation_intent(self):
        from eidolon.livekit.plugins.eot import ChineseModel

        model = ChineseModel()
        # Continuation requests should NOT trigger interruption
        assert model.should_interrupt("再换一个", vad_active=True) is False
        assert model.should_interrupt("不好笑，换一个笑话", vad_active=True) is False
        assert model.should_interrupt("继续说", vad_active=True) is False
        # Full phrase from the bug report
        assert (
            model.should_interrupt(
                "嗯，那你再换一个。我想听呃，人的冷笑话，不是带动物的。",
                vad_active=True,
            )
            is False
        )

    def test_should_interrupt_strong_intent_still_works(self):
        from eidolon.livekit.plugins.eot import ChineseModel

        model = ChineseModel()
        # Strong interrupt should still trigger interruption
        assert model.should_interrupt("停", vad_active=True) is True
        assert model.should_interrupt("闭嘴", vad_active=True) is True

    def test_dynamic_silence_threshold(self):
        from eidolon.livekit.plugins.eot import ChineseModel

        model = ChineseModel()
        threshold = model.get_dynamic_silence_threshold("你好", p_complete=0.5, is_final=False)
        assert threshold > 0

    def test_reset(self):
        from eidolon.livekit.plugins.eot import ChineseModel

        model = ChineseModel()
        model.update_asr("你好", is_final=False)
        model.reset()
        assert model._current_eot_score == 0.0


class TestContextEnhancedEot:
    """Test ContextEnhancedEot."""

    def test_compute_score_returns_zero_for_continuation_intent(self):
        from eidolon.livekit.plugins.eot import (
            ContextEnhancedEot,
        )

        ctx = ContextEnhancedEot()
        # Continuation phrases should return a low score (not triggering cut)
        assert ctx.compute_score("再换一个") == pytest.approx(0.0)
        assert ctx.compute_score("继续说") == pytest.approx(0.0)
        assert ctx.compute_score("不好笑，换一个") == pytest.approx(0.0)

    def test_compute_score_normal_text(self):
        from eidolon.livekit.plugins.eot import (
            ContextEnhancedEot,
        )

        ctx = ContextEnhancedEot()
        # Normal speech should return a score (not forced to 0.0)
        score = ctx.compute_score("你好，我想问一下今天的天气")
        assert 0.0 <= score <= 1.0


@pytest.mark.integration
class TestPluginRegistration:
    """Test plugin registration. Requires livekit.agents — run separately."""

    @pytest.mark.integration
    def test_plugin_import(self):
        from eidolon.livekit.plugins.eot import EidolonEOTPlugin

        plugin = EidolonEOTPlugin()
        # When livekit.agents is installed, it will have a name attribute
        # Otherwise it gracefully degrades
        if hasattr(plugin, "name"):
            assert plugin.name == "eidolon-eot"

    @pytest.mark.integration
    def test_all_exports(self):
        from eidolon.livekit.plugins.eot import (
            EidolonEOTConfig,
            ChineseModel,
            MultilingualModel,
            EidolonEOTPlugin,
        )

        assert EidolonEOTConfig is not None
        assert ChineseModel is not None
        assert MultilingualModel is not None
        assert EidolonEOTPlugin is not None




# =============================================================================
# New comprehensive tests — added in batch
# =============================================================================


class TestNewPolicies:
    """Comprehensive tests for the 5 new policy classes."""

    # -------------------------------------------------------------------------
    # InterruptIntentPolicy
    # -------------------------------------------------------------------------

    def test_interrupt_intent_policy_strong_interrupt(self):
        from eidolon.livekit.plugins.eot import (
            InterruptIntentPolicy,
        )
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        policy = InterruptIntentPolicy()
        state = TurnDetectionStateManager()
        state.current_text = "停"
        result = policy.check(state, TurnEndPolicy())
        assert result is not None
        assert result.should_cut is True
        assert "strong interrupt" in result.reason

    def test_interrupt_intent_policy_multiple_strong_words(self):
        from eidolon.livekit.plugins.eot import (
            InterruptIntentPolicy,
        )
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        policy = InterruptIntentPolicy()
        tp = TurnEndPolicy()
        for word in ["停", "闭嘴", "不对", "错了"]:
            state = TurnDetectionStateManager()
            state.current_text = word
            result = policy.check(state, tp)
            assert result is not None and result.should_cut is True, f"Failed for {word}"

    def test_interrupt_intent_policy_continuation(self):
        from eidolon.livekit.plugins.eot import (
            InterruptIntentPolicy,
        )
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        policy = InterruptIntentPolicy()
        state = TurnDetectionStateManager()
        state.current_text = "再换一个笑话"
        result = policy.check(state, TurnEndPolicy())
        assert result is not None
        assert result.should_cut is False
        assert "continuation" in result.reason

    def test_interrupt_intent_policy_no_override(self):
        from eidolon.livekit.plugins.eot import (
            InterruptIntentPolicy,
        )
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        policy = InterruptIntentPolicy()
        state = TurnDetectionStateManager()
        state.current_text = "北京天气怎么样"
        result = policy.check(state, TurnEndPolicy())
        assert result is None

    def test_interrupt_intent_policy_empty_text(self):
        from eidolon.livekit.plugins.eot import (
            InterruptIntentPolicy,
        )
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        policy = InterruptIntentPolicy()
        state = TurnDetectionStateManager()
        state.current_text = ""
        result = policy.check(state, TurnEndPolicy())
        assert result is None

    # -------------------------------------------------------------------------
    # EOTScoreSemanticPolicy
    # -------------------------------------------------------------------------

    def test_eot_score_semantic_high_triggers_cut(self):
        from eidolon.livekit.plugins.eot import (
            EOTScoreSemanticPolicy,
        )
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        policy = EOTScoreSemanticPolicy(base_threshold=0.7, weak_intent_threshold=0.5, vad_active_delta=0.1)
        state = TurnDetectionStateManager()
        state._vad.active = True
        state._vad.active_since = time.time() - 2.0
        state._vad.last_active_time = time.time() - 0.05
        state._sentence.start_time = time.time() - 2.0
        state.current_text = "北京天气"
        state.update_eot_score(0.8)
        result = policy.check(state, TurnEndPolicy())
        assert result is not None
        assert result.should_cut is True

    def test_eot_score_semantic_low_no_cut(self):
        from eidolon.livekit.plugins.eot import (
            EOTScoreSemanticPolicy,
        )
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        policy = EOTScoreSemanticPolicy()
        state = TurnDetectionStateManager()
        state._vad.active = True
        state._vad.active_since = time.time() - 2.0
        state._vad.last_active_time = time.time() - 0.05
        state._sentence.start_time = time.time() - 2.0
        state.current_text = "北京"
        state.update_eot_score(0.3)
        result = policy.check(state, TurnEndPolicy())
        assert result is None

    def test_eot_score_semantic_weak_intent_lowers_threshold(self):
        from eidolon.livekit.plugins.eot import (
            EOTScoreSemanticPolicy,
        )
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        policy = EOTScoreSemanticPolicy(base_threshold=0.7, weak_intent_threshold=0.5, vad_active_delta=0.1)
        state = TurnDetectionStateManager()
        state._vad.active = True
        state._vad.active_since = time.time() - 2.0
        state._vad.last_active_time = time.time() - 0.05
        state._sentence.start_time = time.time() - 2.0
        state.current_text = "好的"
        state.update_eot_score(0.5)
        # score=0.5 >= threshold(weak, vad_active)=0.5-0.1=0.4
        result = policy.check(state, TurnEndPolicy())
        assert result is not None and result.should_cut is True

    def test_eot_score_semantic_vad_inactive_raises_threshold(self):
        from eidolon.livekit.plugins.eot import (
            EOTScoreSemanticPolicy,
        )
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        policy = EOTScoreSemanticPolicy(base_threshold=0.7, weak_intent_threshold=0.5, vad_active_delta=0.1)
        state = TurnDetectionStateManager()
        state._vad.active = False
        state.current_text = "北京天气"
        state.update_eot_score(0.75)
        # score=0.75 < threshold(base, vad_inactive)=0.7+0.1=0.8
        result = policy.check(state, TurnEndPolicy())
        assert result is None

    def test_eot_score_semantic_empty_text(self):
        from eidolon.livekit.plugins.eot import (
            EOTScoreSemanticPolicy,
        )
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        policy = EOTScoreSemanticPolicy()
        state = TurnDetectionStateManager()
        state.current_text = ""
        state.update_eot_score(0.9)
        result = policy.check(state, TurnEndPolicy())
        assert result is None

    # -------------------------------------------------------------------------
    # VADStabilityPolicy
    # -------------------------------------------------------------------------

    def test_vad_stability_rapid_toggles_blocked(self):
        from eidolon.livekit.plugins.eot import VADStabilityPolicy
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        policy = VADStabilityPolicy(flip_window_sec=1.0, flip_count_threshold=3)
        state = TurnDetectionStateManager()
        state._vad.active = True
        now = time.time()
        for i in range(3):
            state._vad.transition_times.append(now - 0.2 * (i + 1))
        result = policy.check(state, TurnEndPolicy())
        assert result is not None
        assert result.should_cut is False
        assert "unstable" in result.reason

    def test_vad_stability_slow_toggles_passed(self):
        from eidolon.livekit.plugins.eot import VADStabilityPolicy
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        policy = VADStabilityPolicy(flip_window_sec=1.0, flip_count_threshold=3)
        state = TurnDetectionStateManager()
        state._vad.active = True
        now = time.time()
        state._vad.transition_times.append(now - 0.8)
        state._vad.transition_times.append(now - 0.4)
        result = policy.check(state, TurnEndPolicy())
        assert result is None

    def test_vad_stability_vad_inactive(self):
        from eidolon.livekit.plugins.eot import VADStabilityPolicy
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        policy = VADStabilityPolicy()
        state = TurnDetectionStateManager()
        state._vad.active = False
        result = policy.check(state, TurnEndPolicy())
        assert result is None

    def test_vad_stability_no_transitions(self):
        from eidolon.livekit.plugins.eot import VADStabilityPolicy
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        policy = VADStabilityPolicy()
        state = TurnDetectionStateManager()
        state._vad.active = True
        result = policy.check(state, TurnEndPolicy())
        assert result is None

    # -------------------------------------------------------------------------
    # MinSpeakingDurationPolicy
    # -------------------------------------------------------------------------

    def test_min_speaking_short_blocked(self):
        from eidolon.livekit.plugins.eot import (
            MinSpeakingDurationPolicy,
        )
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        policy = MinSpeakingDurationPolicy(min_speech_duration_sec=0.15)
        state = TurnDetectionStateManager()
        state._vad.active = True
        state._vad.active_since = time.time() - 0.05
        state._sentence.start_time = time.time() - 0.05
        result = policy.check(state, TurnEndPolicy())
        assert result is not None
        assert result.should_cut is False
        assert "too short" in result.reason

    def test_min_speaking_long_passed(self):
        from eidolon.livekit.plugins.eot import (
            MinSpeakingDurationPolicy,
        )
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        policy = MinSpeakingDurationPolicy(min_speech_duration_sec=0.15)
        state = TurnDetectionStateManager()
        state._vad.active = True
        state._vad.active_since = time.time() - 1.0
        state._sentence.start_time = time.time() - 1.0
        result = policy.check(state, TurnEndPolicy())
        assert result is None

    def test_min_speaking_vad_inactive(self):
        from eidolon.livekit.plugins.eot import (
            MinSpeakingDurationPolicy,
        )
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        policy = MinSpeakingDurationPolicy()
        state = TurnDetectionStateManager()
        state._vad.active = False
        result = policy.check(state, TurnEndPolicy())
        assert result is None

    # -------------------------------------------------------------------------
    # ASRStabilityPolicy
    # -------------------------------------------------------------------------

    def test_asr_stability_short_text_fast_stabilizes(self):
        from eidolon.livekit.plugins.eot import ASRStabilityPolicy
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        policy = ASRStabilityPolicy(stability_short_sec=0.10, stability_long_sec=0.40, long_char_threshold=10)
        state = TurnDetectionStateManager()
        state._asr.is_final = True
        state._asr.stable_since = time.time() - 0.2  # past short threshold
        state.current_text = "你好"  # < 10 chars
        result = policy.check(state, TurnEndPolicy())
        assert result is None

    def test_asr_stability_long_text_slow_stabilizes(self):
        from eidolon.livekit.plugins.eot import ASRStabilityPolicy
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        policy = ASRStabilityPolicy(stability_short_sec=0.10, stability_long_sec=0.40, long_char_threshold=10)
        state = TurnDetectionStateManager()
        state._asr.is_final = True
        state._asr.stable_since = time.time() - 0.2  # not past long threshold (0.4)
        state.current_text = "北京今天的天气怎么样"  # >= 10 chars
        result = policy.check(state, TurnEndPolicy())
        assert result is not None
        assert result.should_cut is False
        assert "not stable" in result.reason

    def test_asr_stability_not_final(self):
        from eidolon.livekit.plugins.eot import ASRStabilityPolicy
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        policy = ASRStabilityPolicy()
        state = TurnDetectionStateManager()
        state._asr.is_final = False
        result = policy.check(state, TurnEndPolicy())
        assert result is None

    def test_asr_stability_no_stable_since(self):
        from eidolon.livekit.plugins.eot import ASRStabilityPolicy
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        policy = ASRStabilityPolicy()
        state = TurnDetectionStateManager()
        state._asr.is_final = True
        state._asr.stable_since = None
        result = policy.check(state, TurnEndPolicy())
        assert result is None

    # -------------------------------------------------------------------------
    # DuplicateTextPolicy
    # -------------------------------------------------------------------------

    def test_duplicate_similar_blocked(self):
        from eidolon.livekit.plugins.eot import DuplicateTextPolicy
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        policy = DuplicateTextPolicy(similarity_threshold=0.85)
        state = TurnDetectionStateManager()
        state.current_text = "你好世界"
        state._sentence.last_cut_text = "你好世界"
        result = policy.check(state, TurnEndPolicy())
        assert result is not None
        assert result.should_cut is False
        assert "similarity" in result.reason

    def test_duplicate_dissimilar_passed(self):
        from eidolon.livekit.plugins.eot import DuplicateTextPolicy
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        policy = DuplicateTextPolicy(similarity_threshold=0.85)
        state = TurnDetectionStateManager()
        state.current_text = "你好"
        state._sentence.last_cut_text = "再见"
        result = policy.check(state, TurnEndPolicy())
        assert result is None

    def test_duplicate_fallback_to_hash(self):
        from eidolon.livekit.plugins.eot import DuplicateTextPolicy
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy
        from eidolon.livekit.plugins.eot import compute_text_hash

        policy = DuplicateTextPolicy()
        state = TurnDetectionStateManager()
        state.current_text = "test"
        state._sentence.last_cut_text = None
        state._sentence.last_cut_text_hash = compute_text_hash("test")
        result = policy.check(state, TurnEndPolicy())
        assert result is not None
        assert result.should_cut is False
        assert "hash" in result.reason

    def test_duplicate_no_previous(self):
        from eidolon.livekit.plugins.eot import DuplicateTextPolicy
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        policy = DuplicateTextPolicy()
        state = TurnDetectionStateManager()
        state.current_text = "test"
        state._sentence.last_cut_text = None
        state._sentence.last_cut_text_hash = None
        result = policy.check(state, TurnEndPolicy())
        assert result is None


class TestPolicyChainIntegration:
    """Integration tests for policy chain composition and interaction."""

    def test_chain_stops_at_first_decision(self):
        from eidolon.livekit.plugins.eot import (
            PolicyChain,
            MinSpeakingDurationPolicy,
            EOTScoreSemanticPolicy,
        )
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        chain = PolicyChain([
            MinSpeakingDurationPolicy(0.15),  # short speech → should_cut=False
            EOTScoreSemanticPolicy(0.9),  # high score → should_cut=True
        ])
        state = TurnDetectionStateManager()
        state._vad.active = True
        state._vad.active_since = time.time() - 0.05
        state._sentence.start_time = time.time() - 0.05
        state.current_text = "short"
        state.update_eot_score(0.95)
        result = chain.check(state, TurnEndPolicy())
        # MinSpeakingDurationPolicy returns CutDecision(should_cut=False)
        # but it's still a decision → chain stops
        assert result is not None
        assert result.should_cut is False

    def test_chain_continues_on_none(self):
        from eidolon.livekit.plugins.eot import (
            PolicyChain,
            MinSpeakingDurationPolicy,
            EOTScoreSemanticPolicy,
        )
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        chain = PolicyChain([
            MinSpeakingDurationPolicy(0.15),  # 1s speech → None (let chain continue)
            EOTScoreSemanticPolicy(0.9),  # high score → should_cut=True
        ])
        state = TurnDetectionStateManager()
        state._vad.active = True
        state._vad.active_since = time.time() - 1.0
        state._sentence.start_time = time.time() - 1.0
        state.current_text = "long enough speech"
        state.update_eot_score(0.95)
        result = chain.check(state, TurnEndPolicy())
        # MinSpeakingDurationPolicy → None, chain continues to EOTScoreSemanticPolicy
        assert result is not None
        assert result.should_cut is True

    def test_strong_interrupt_wins_in_chain(self):
        from eidolon.livekit.plugins.eot import (
            PolicyChain,
            InterruptIntentPolicy,
            EOTScoreSemanticPolicy,
        )
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        chain = PolicyChain([
            InterruptIntentPolicy(),
            EOTScoreSemanticPolicy(0.9),
        ])
        state = TurnDetectionStateManager()
        state._vad.active = True
        state._vad.active_since = time.time() - 1.0
        state._sentence.start_time = time.time() - 1.0
        state.current_text = "停"
        state.update_eot_score(0.95)
        result = chain.check(state, TurnEndPolicy())
        assert result is not None
        assert result.should_cut is True
        assert "strong interrupt" in result.reason

    def test_continuation_blocks_in_chain(self):
        from eidolon.livekit.plugins.eot import (
            PolicyChain,
            InterruptIntentPolicy,
            EOTScoreSemanticPolicy,
        )
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        chain = PolicyChain([
            InterruptIntentPolicy(),
            EOTScoreSemanticPolicy(0.1),
        ])
        state = TurnDetectionStateManager()
        state._vad.active = True
        state._vad.active_since = time.time() - 1.0
        state._sentence.start_time = time.time() - 1.0
        state.current_text = "再换一个笑话"
        state.update_eot_score(0.99)
        result = chain.check(state, TurnEndPolicy())
        assert result is not None
        assert result.should_cut is False
        assert "continuation" in result.reason

    def test_semantic_chain_excludes_asr_stability(self):
        from eidolon.livekit.plugins.eot import PolicyChain

        chain = PolicyChain.for_semantic_interruption()
        types = [type(p).__name__ for p in chain._policies]
        assert "ASRStabilityPolicy" not in types

    def test_normal_chain_includes_asr_stability(self):
        from eidolon.livekit.plugins.eot import PolicyChain

        chain = PolicyChain.for_normal_turn_end()
        types = [type(p).__name__ for p in chain._policies]
        assert "ASRStabilityPolicy" in types

    def test_normal_chain_excludes_eot_score_semantic(self):
        from eidolon.livekit.plugins.eot import PolicyChain

        chain = PolicyChain.for_normal_turn_end()
        types = [type(p).__name__ for p in chain._policies]
        assert "EOTScoreSemanticPolicy" not in types

    def test_semantic_chain_includes_interrupt_intent(self):
        from eidolon.livekit.plugins.eot import PolicyChain

        chain = PolicyChain.for_semantic_interruption()
        types = [type(p).__name__ for p in chain._policies]
        assert "InterruptIntentPolicy" in types

    def test_min_interval_policy_blocks_chain_when_not_met(self):
        from eidolon.livekit.plugins.eot import (
            PolicyChain,
            MinIntervalPolicy,
            EOTScoreSemanticPolicy,
        )
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        chain = PolicyChain([
            MinIntervalPolicy(),
            EOTScoreSemanticPolicy(0.1),
        ])
        state = TurnDetectionStateManager()
        state._vad.active = True
        state._vad.active_since = time.time() - 1.0
        state._sentence.start_time = time.time() - 1.0
        state._sentence.last_cut_time = time.time()  # just now → interval not met
        state.current_text = "test"
        state.update_eot_score(0.99)
        result = chain.check(state, TurnEndPolicy())
        # MinIntervalPolicy → CutDecision(should_cut=False) → blocks chain
        assert result is not None
        assert result.should_cut is False


class TestTurnEndPolicyPaths:
    """Comprehensive tests for TurnEndPolicy threshold computation paths."""

    def test_tail_hang_threshold(self):
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        tp = TurnEndPolicy(t_tail_hang=2.5)
        for word in ["然后", "因为"]:
            t = tp.get_dynamic_threshold(f"说完{word}", p_complete=0.5, is_final=False)
            assert t == 2.5, f"Failed for '{word}'"

    def test_wait_punctuation_threshold(self):
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        tp = TurnEndPolicy()
        t = tp.get_dynamic_threshold("你好，", p_complete=0.5, is_final=False)
        assert t == tp.T_MAX

    def test_terminal_punctuation_final(self):
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        tp = TurnEndPolicy()
        t = tp.get_dynamic_threshold("你好。", p_complete=0.5, is_final=True)
        assert t == tp.T_URGENT

    def test_terminal_punctuation_not_final(self):
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        tp = TurnEndPolicy()
        t = tp.get_dynamic_threshold("你好。", p_complete=0.5, is_final=False)
        assert t == tp.T_MIN

    def test_low_score_deep_wait(self):
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        tp = TurnEndPolicy()
        # No terminal punctuation, no wait punctuation → uses p_complete tier
        t = tp.get_dynamic_threshold("你好呃啊啊啊", p_complete=0.2, is_final=False)
        assert t == tp.T_DEEP

    def test_is_valid_speech_filler_only(self):
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        tp = TurnEndPolicy()
        assert tp.is_valid_speech("嗯") is False
        assert tp.is_valid_speech("啊") is False
        assert tp.is_valid_speech("嗯嗯啊啊") is False

    def test_is_valid_speech_mixed(self):
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        tp = TurnEndPolicy()
        assert tp.is_valid_speech("嗯，你好") is True
        assert tp.is_valid_speech("你好啊") is True

    def test_t_tail_hang_from_config(self):
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        tp = TurnEndPolicy(t_tail_hang=3.5)
        t = tp.get_dynamic_threshold("说完然后", p_complete=0.5, is_final=False)
        assert t == 3.5


class TestContextEnhancedEotPaths:
    """Comprehensive tests for ContextEnhancedEot scoring paths."""

    def test_cooldown_blocks_cut(self):
        from eidolon.livekit.plugins.eot import (
            ContextEnhancedEot,
        )

        ctx = ContextEnhancedEot(cooldown_period=0.5)
        ctx._last_interrupt_time = time.time() - 0.1  # 100ms ago, inside cooldown
        score = ctx.compute_score("你好天气")
        assert score == 0.0

    def test_cooldown_allows_after_elapsed(self):
        from eidolon.livekit.plugins.eot import (
            ContextEnhancedEot,
        )

        ctx = ContextEnhancedEot(cooldown_period=0.1)
        ctx._last_interrupt_time = time.time() - 0.5  # 500ms ago, past cooldown
        score = ctx.compute_score("你好天气")
        assert score > 0.0

    def test_similarity_blocks_cut(self):
        from eidolon.livekit.plugins.eot import (
            ContextEnhancedEot,
        )

        ctx = ContextEnhancedEot(similarity_threshold=0.85)
        ctx._last_interrupt_time = None
        ctx._last_text = "你好天气"
        score = ctx.compute_score("你好天气")  # identical
        assert score == 0.0

    def test_greeting_adjustment(self):
        from eidolon.livekit.plugins.eot import (
            ContextEnhancedEot,
        )

        ctx = ContextEnhancedEot()
        score = ctx.compute_score("你好，请问")
        base_score = ctx._base_eot.p_complete_score("你好，请问")
        # Greeting adjustment should boost score
        assert ctx.compute_score("你好，请问") >= base_score

    def test_strong_ending_adjustment(self):
        from eidolon.livekit.plugins.eot import (
            ContextEnhancedEot,
        )

        ctx = ContextEnhancedEot()
        score = ctx.compute_score("好的再见")
        base_score = ctx._base_eot.p_complete_score("好的再见")
        # Strong ending should boost score
        assert score >= base_score

    def test_followup_adjustment(self):
        """A user-finished follow-up question (ending with FOLLOWUP_INDICATORS
        like 呢 / 那 / 还有) signals "I'm done asking, please answer" — so
        the context adjustment RAISES the EOT score (more likely the user
        is done speaking).

        Round 8 R8.5.b note: the original test asserted ``score <= base``
        on text "为什么", but "为什么" isn't a follow-up by the plugin's
        contract (no FOLLOWUP_INDICATOR match, no short-text pronoun) and
        the test was simply checking the wrong direction. Updated to match
        the actual semantics (and plugin docstring: "Follow-up question,
        +0.15").
        """
        from eidolon.livekit.plugins.eot import (
            ContextEnhancedEot,
        )

        ctx = ContextEnhancedEot()
        # "好的呢" ends with 呢 (in FOLLOWUP_INDICATORS) → triggers +0.15.
        text = "好的呢"
        score = ctx.compute_score(text)
        base_score = ctx._base_eot.p_complete_score(text)
        assert ctx._is_follow_up(text) is True, (
            f"sanity: '{text}' should match _is_follow_up (it ends with 呢)"
        )
        assert score >= base_score, (
            f"follow-up should raise score; got score={score:.3f} "
            f"vs base={base_score:.3f}"
        )


class TestEidolonEOTModelEndToEnd:
    """End-to-end tests for EidolonEOTModel using the real ONNX model."""

    def test_should_interrupt_delegates_to_chain(self):
        from eidolon.livekit.plugins.eot import ChineseModel

        model = ChineseModel()
        model._state.update_vad(True)
        model._state._vad.active_since = time.time() - 1.0
        model._state._sentence.start_time = time.time() - 1.0
        model._state.current_text = "停"
        model._state.update_eot_score(0.5)
        # Strong interrupt intent handled by chain → should_cut=True
        result = model.should_interrupt("停", vad_active=True)
        assert result is True

    def test_should_interrupt_side_effects(self):
        from eidolon.livekit.plugins.eot import ChineseModel

        model = ChineseModel()
        model._state.update_vad(True)
        model._state._vad.active_since = time.time() - 2.0
        model._state._sentence.start_time = time.time() - 2.0
        model._state.current_text = "好的"
        model._state.update_eot_score(0.95)
        before = model._context_eot._last_interrupt_time
        result = model.should_interrupt("好的", vad_active=True)
        if result is True:
            assert model._context_eot._last_interrupt_time != before

    def test_reset_clears_state(self):
        from eidolon.livekit.plugins.eot import ChineseModel

        model = ChineseModel()
        model.update_asr("test", is_final=False)
        model.reset()
        assert model._current_eot_score == 0.0
        assert model._state.current_text == ""

    def test_predict_end_of_turn_returns_score(self):
        from eidolon.livekit.plugins.eot import ChineseModel

        model = ChineseModel()
        # predict_end_of_turn is async — test the underlying p_complete_score directly
        score = model._context_eot.p_complete_score("你好天气")
        assert 0.0 <= score <= 1.0


class TestL8UserProfileLifecycle:
    """Round 8 P2.L8 — UserProfile cache must not grow unbounded.

    Without explicit lifecycle, long-running daemons accumulate
    profiles forever as rooms come and go. Now there's an LRU cap +
    explicit ``end_session`` cleanup hook.
    """

    def test_end_session_removes_profile(self):
        from eidolon.livekit.plugins.eot import ChineseModel

        model = ChineseModel()
        model.start_session("room-1")
        assert "room-1" in model._context_eot._user_profiles

        model.end_session("room-1")
        assert "room-1" not in model._context_eot._user_profiles

    def test_end_session_with_none_clears_active(self):
        from eidolon.livekit.plugins.eot import ChineseModel

        model = ChineseModel()
        model.start_session("active-room")
        assert model._context_eot._current_session_id == "active-room"

        model.end_session()  # no arg → clears the active session
        assert model._context_eot._current_session_id is None
        assert "active-room" not in model._context_eot._user_profiles

    def test_end_session_idempotent(self):
        """Calling end_session for a non-existent / already-removed
        session is a no-op (no exception)."""
        from eidolon.livekit.plugins.eot import ChineseModel

        model = ChineseModel()
        model.end_session("never-started")
        model.end_session("never-started")  # idempotent

    def test_lru_eviction_on_cap_exceeded(self):
        """If 1000 profiles already cached and a new session starts,
        the least-recently-used one is evicted."""
        from eidolon.livekit.plugins.eot.impl.context_enhanced_eot import (
            ContextEnhancedEot,
        )

        # Use a small cap to keep the test fast
        ctx_eot = ContextEnhancedEot(max_profiles=3)

        ctx_eot.start_session("a")
        ctx_eot.start_session("b")
        ctx_eot.start_session("c")
        assert set(ctx_eot._user_profiles.keys()) == {"a", "b", "c"}

        # Adding a 4th evicts the LRU (which is "a", the first one)
        ctx_eot.start_session("d")
        assert set(ctx_eot._user_profiles.keys()) == {"b", "c", "d"}

        # Touching "b" makes it most recent; "c" is now LRU
        ctx_eot.start_session("b")
        ctx_eot.start_session("e")
        assert set(ctx_eot._user_profiles.keys()) == {"b", "d", "e"}

    def test_default_cap_is_1000(self):
        """Default cap is 1000 — large enough that production traffic
        rarely hits it, but small enough to bound memory."""
        from eidolon.livekit.plugins.eot.impl.context_enhanced_eot import (
            ContextEnhancedEot,
        )

        ctx_eot = ContextEnhancedEot()
        assert ctx_eot._max_profiles == 1000



class TestMultiTurnScenario:
    """Multi-turn scenario tests: simulate 5+ rounds of realistic dialogue.

    Each test represents a real user journey and verifies the policy chain
    produces the correct cut decisions across consecutive turns.
    """

    def _make_speech_state(self, state, text, eot_score, speaking_seconds=2.0):
        """Helper: set up a realistic speech state for the given duration."""
        state._vad.active = True
        state._vad.active_since = time.time() - speaking_seconds
        state._vad.last_active_time = time.time() - 0.05
        state._sentence.start_time = time.time() - speaking_seconds
        state.current_text = text
        state.update_eot_score(eot_score)

    # -------------------------------------------------------------------------
    # 5-turn normal end detection
    # -------------------------------------------------------------------------

    def test_5_turns_normal_end_detection(self):
        from eidolon.livekit.plugins.eot import PolicyChain
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        chain = PolicyChain.for_semantic_interruption()
        state = TurnDetectionStateManager()
        tp = TurnEndPolicy()

        # for_semantic_interruption uses EOT score + intent to trigger cuts
        turns = [
            ("好的", 0.75),
            ("北京天气", 0.80),
            ("天气怎么样", 0.85),
            ("明天怎么样", 0.90),
            ("谢谢", 0.95),
        ]

        cut_count = 0
        for i, (text, score) in enumerate(turns):
            state.reset_session("session", i)
            state._sentence.last_cut_time = time.time() - 1.0  # interval met
            state._interrupt.last_interrupt_time = 0.0  # no cooldown
            self._make_speech_state(state, text, score)
            decision = chain.check(state, tp)
            if decision is not None and decision.should_cut:
                cut_count += 1
        # High EOT scores should trigger cuts via EOTScoreSemanticPolicy
        assert cut_count >= 4, f"Expected at least 4 cuts, got {cut_count}"

    # -------------------------------------------------------------------------
    # 5-turn semantic interruption
    # -------------------------------------------------------------------------

    def test_5_turns_semantic_interrupt(self):
        from eidolon.livekit.plugins.eot import PolicyChain
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        chain = PolicyChain.for_semantic_interruption()
        state = TurnDetectionStateManager()
        tp = TurnEndPolicy()

        turns = [
            ("停", 0.5),
            ("不对", 0.5),
            ("等一下", 0.5),
            ("等会再说", 0.5),
            ("先不要说了", 0.5),
        ]

        cut_count = 0
        for i, (text, score) in enumerate(turns):
            state.reset_session("session", i)
            state._sentence.last_cut_time = time.time() - 1.0  # interval met
            self._make_speech_state(state, text, score)
            decision = chain.check(state, tp)
            if decision is not None and decision.should_cut:
                cut_count += 1
        assert cut_count >= 4

    # -------------------------------------------------------------------------
    # 5-turn mixed scenario
    # -------------------------------------------------------------------------

    def test_5_turns_mixed_scenario(self):
        from eidolon.livekit.plugins.eot import PolicyChain
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        chain = PolicyChain.for_semantic_interruption()
        state = TurnDetectionStateManager()
        tp = TurnEndPolicy()

        # Round 7 G1: "好的" is a backchannel and should NOT cut while agent
        # is speaking. To exercise the weak-interrupt path here, replaced
        # the original "好的" case with "不要这样" (not a backchannel, weak
        # intent via "不要"). Other turns unchanged.
        turns = [
            ("不要这样", 0.5),    # weak interrupt (not backchannel) → cut
            ("再换一个", 0.5),     # continuation → no cut
            ("停", 0.5),          # strong interrupt → cut
            ("北京天气", 0.3),    # low score < 0.4 → no cut
            ("不对错了", 0.8),    # strong interrupt + high score → cut
        ]
        expected_cuts = [True, False, True, False, True]

        results = []
        for i, (text, score) in enumerate(turns):
            state.reset_session("session", i)
            state._sentence.last_cut_time = time.time() - 1.0
            self._make_speech_state(state, text, score)
            decision = chain.check(state, tp)
            cut = decision is not None and decision.should_cut
            results.append(cut)
            assert cut == expected_cuts[i], (
                f"Turn {i}: text={text!r} expected_cut={expected_cuts[i]} got={cut}"
                + (f" reason={decision.reason}" if decision else "")
            )
        assert results == expected_cuts

    def test_backchannel_blocks_weak_interrupt_in_agent_speaking_chain(self):
        """Round 7 G1 — "好的" while agent speaking is a backchannel (NOT a
        weak interrupt). Document the policy interaction explicitly."""
        from eidolon.livekit.plugins.eot import PolicyChain
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        chain = PolicyChain.for_semantic_interruption()
        state = TurnDetectionStateManager()
        tp = TurnEndPolicy()
        state._sentence.last_cut_time = time.time() - 1.0
        self._make_speech_state(state, "好的", 0.7)

        decision = chain.check(state, tp)
        assert decision is not None
        assert decision.should_cut is False
        assert "ackchannel" in decision.reason.lower()

    # -------------------------------------------------------------------------
    # Cooldown persists across turns
    # -------------------------------------------------------------------------

    def test_cooldown_persists_across_turns(self):
        from eidolon.livekit.plugins.eot import PolicyChain
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        chain = PolicyChain.for_semantic_interruption()
        state = TurnDetectionStateManager()
        tp = TurnEndPolicy()

        # Turn 1: cut → record interrupt time so next turn hits cooldown
        state._sentence.last_cut_time = time.time() - 1.0
        state._interrupt.last_interrupt_time = 0.0  # no prior interrupt
        self._make_speech_state(state, "停", 0.5)
        decision = chain.check(state, tp)
        assert decision is not None and decision.should_cut
        state.update_interrupt()  # record so next turn is in cooldown

        # Immediately after: cooldown should block
        state._sentence.last_cut_time = time.time()  # just now
        self._make_speech_state(state, "好的", 0.99)
        decision2 = chain.check(state, tp)
        assert decision2 is not None
        assert decision2.should_cut is False
        assert "cooldown" in decision2.reason

    # -------------------------------------------------------------------------
    # Duplicate text filtered
    # -------------------------------------------------------------------------

    def test_duplicate_text_filtered_across_turns(self):
        from eidolon.livekit.plugins.eot import PolicyChain
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        chain = PolicyChain.for_semantic_interruption()
        state = TurnDetectionStateManager()
        tp = TurnEndPolicy()

        # Turn 1: high score → cut. Use a different text so last_cut_text != current.
        state._sentence.last_cut_time = time.time() - 1.0
        state._sentence.last_cut_text = "再见"  # different from "你好"
        state._sentence.last_cut_text_hash = None
        state._interrupt.last_interrupt_time = 0.0  # no cooldown
        self._make_speech_state(state, "你好", 0.99)
        decision1 = chain.check(state, tp)
        assert decision1 is not None and decision1.should_cut

        # Turn 2: same text as turn 1 → duplicate. Don't reset_session (it would
        # reset _interrupt too). Instead, just set up the duplicate scenario.
        state._sentence.last_cut_time = time.time() - 1.0
        state._sentence.last_cut_text = "你好"  # same as current text
        state._sentence.last_cut_text_hash = None
        state._interrupt.last_interrupt_time = 0.0  # clear cooldown from turn 1
        self._make_speech_state(state, "你好", 0.99)
        decision2 = chain.check(state, tp)
        assert decision2 is not None
        assert decision2.should_cut is False
        assert "similarity" in decision2.reason or "hash" in decision2.reason

    # -------------------------------------------------------------------------
    # Continuation intent across 5 turns → 0 cuts
    # -------------------------------------------------------------------------

    def test_continuation_intent_not_cut_across_5_turns(self):
        from eidolon.livekit.plugins.eot import PolicyChain
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        chain = PolicyChain.for_semantic_interruption()
        state = TurnDetectionStateManager()
        tp = TurnEndPolicy()

        continuation_turns = [
            "再换一个",
            "继续说",
            "接着讲",
            "然后呢",
            "不好笑换一个",
        ]

        for i, text in enumerate(continuation_turns):
            state.reset_session("session", i)
            state._sentence.last_cut_time = time.time() - 1.0
            self._make_speech_state(state, text, 0.99)
            decision = chain.check(state, tp)
            assert decision is not None
            assert decision.should_cut is False, (
                f"Turn {i}: continuation '{text}' should NOT cut, got cut={decision.should_cut}"
            )

    # -------------------------------------------------------------------------
    # Strong intent across 5 turns → 5 cuts
    # -------------------------------------------------------------------------

    def test_strong_intent_always_cuts_across_5_turns(self):
        from eidolon.livekit.plugins.eot import PolicyChain
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        chain = PolicyChain.for_semantic_interruption()
        state = TurnDetectionStateManager()
        tp = TurnEndPolicy()

        strong_turns = [
            ("停", 0.3),
            ("闭嘴", 0.3),
            ("不对", 0.3),
            ("错了", 0.3),
            ("重说", 0.3),
        ]

        for i, (text, score) in enumerate(strong_turns):
            state.reset_session("session", i)
            state._sentence.last_cut_time = time.time() - 1.0
            self._make_speech_state(state, text, score)
            decision = chain.check(state, tp)
            assert decision is not None and decision.should_cut, (
                f"Turn {i}: strong '{text}' should cut, got={decision}"
            )

    # -------------------------------------------------------------------------
    # Min interval blocks rapid cuts
    # -------------------------------------------------------------------------

    def test_min_interval_blocks_rapid_cuts(self):
        from eidolon.livekit.plugins.eot import PolicyChain
        from eidolon.livekit.plugins.eot import TurnDetectionStateManager
        from eidolon.livekit.plugins.eot import TurnEndPolicy

        chain = PolicyChain.for_semantic_interruption()
        state = TurnDetectionStateManager()
        tp = TurnEndPolicy()

        # Turn 1: cut
        state._sentence.last_cut_time = time.time() - 1.0
        self._make_speech_state(state, "停", 0.5)
        decision1 = chain.check(state, tp)
        assert decision1 is not None and decision1.should_cut

        # Immediately: interval not met, but strong intent overrides.
        state._sentence.last_cut_time = time.time()  # just now
        self._make_speech_state(state, "停", 0.5)
        decision2 = chain.check(state, tp)
        # InterruptIntentPolicy comes before MinIntervalPolicy in the chain,
        # so "停" (strong interrupt) still cuts regardless of interval.
        assert decision2 is not None and decision2.should_cut


# ---------------------------------------------------------------------------
# Round 8 R8.5.b: dual-path scoring consistency
# ---------------------------------------------------------------------------


class TestR8DualPathConsistency:
    """Verify Path A (predict_end_of_turn) and Path B (should_interrupt)
    share the same base score + context adjustment, and only differ by
    Path B's interruption-specific guards (cooldown, similarity,
    continuation-intent).

    Round 7 G0a unified the underlying scoring path; this test guards
    against a regression where the paths drift apart.
    """

    @pytest.mark.asyncio
    async def test_paths_agree_when_no_interrupt_guards_active(self):
        """Same text, fresh state → both paths produce the same score."""
        from eidolon.livekit.plugins.eot import ContextEnhancedEot

        ctx = ContextEnhancedEot()
        text = "今天天气怎么样"

        # Path A: p_complete_score (used by predict_end_of_turn)
        score_a = ctx.p_complete_score(text)

        # Path B: compute_score (used by should_interrupt). With fresh
        # state (no cooldown, no last_text, no continuation intent),
        # it should equal Path A's score.
        score_b = ctx.compute_score(text)

        assert abs(score_a - score_b) < 1e-6, (
            f"paths should agree on neutral input; A={score_a:.6f} "
            f"B={score_b:.6f} delta={score_a - score_b:.6f}"
        )

    @pytest.mark.asyncio
    async def test_path_b_zeros_during_cooldown_path_a_unaffected(self):
        """After an interrupt, compute_score (B) returns 0 during
        cooldown but p_complete_score (A) keeps reporting the raw
        score — Path A is for endpointing decisions, not interrupt-
        specific guards."""
        from eidolon.livekit.plugins.eot import ContextEnhancedEot

        ctx = ContextEnhancedEot(cooldown_period=10.0)
        text = "好的"

        # Trigger an interrupt to start cooldown.
        ctx.record_interrupt(text)

        score_a = ctx.p_complete_score(text)
        score_b = ctx.compute_score(text)

        # Path A is unguarded.
        assert score_a > 0
        # Path B is gated to 0 by cooldown.
        assert score_b == 0.0

    @pytest.mark.asyncio
    async def test_path_b_zeros_on_continuation_intent_path_a_unaffected(self):
        """Continuation intent ("再换一个") should NOT interrupt the
        agent (Path B → 0), but the framework can still consider it
        a complete user turn for endpointing (Path A unaffected).
        """
        from eidolon.livekit.plugins.eot import ContextEnhancedEot

        ctx = ContextEnhancedEot()
        text = "再换一个吧"

        score_a = ctx.p_complete_score(text)
        score_b = ctx.compute_score(text)

        assert score_a > 0
        assert score_b == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
