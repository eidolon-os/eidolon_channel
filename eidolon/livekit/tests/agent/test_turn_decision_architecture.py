"""Architecture locks for the single committed-turn decision authority."""

from __future__ import annotations

import inspect
from pathlib import Path

from eidolon.livekit.agent.session import decision_effects, semantic_interrupt
from eidolon.livekit.agent.turn_policy import attention, decider, intent_classifier, runtime
from eidolon.livekit.plugins.eot.impl.context_enhanced_eot import ContextEnhancedEot
from eidolon.livekit.plugins.eot.impl.eot_policy import PolicyChain
from eidolon.livekit.plugins.eot.models.base import EidolonEOTModel


def test_fixed_lexicons_are_not_imported_by_the_irreversible_decider() -> None:
    source = inspect.getsource(decider)

    assert "DEFAULT_TOPIC_SWITCH_LEXICON" not in source
    assert "_redirect_prefix_intent" not in source
    assert "_FAST_DEADLINE_SINGLE_CHAR_ACKS" not in source


def test_production_turn_modules_do_not_import_fixed_phrase_classifiers() -> None:
    sources = "\n".join(
        inspect.getsource(module) for module in (attention, decider, intent_classifier, runtime)
    )

    assert "LexiconInterruptClassifier" not in sources
    assert "hard_stop_intent" not in sources
    assert "hard_stop_prefix_intent" not in sources
    assert "DEFAULT_TOPIC_SWITCH_LEXICON" not in sources


def test_production_eot_chains_exclude_lexical_override_policies() -> None:
    semantic = {
        type(policy).__name__ for policy in PolicyChain.for_semantic_interruption()._policies
    }
    normal = {type(policy).__name__ for policy in PolicyChain.for_normal_turn_end()._policies}
    forbidden = {
        "BackchannelSuppressionPolicy",
        "NoiseLikeTranscriptPolicy",
        "InterruptIntentPolicy",
    }

    assert semantic.isdisjoint(forbidden)
    assert normal.isdisjoint(forbidden)


def test_retired_lexical_eot_modules_are_absent() -> None:
    impl_dir = Path(inspect.getfile(ContextEnhancedEot)).parent
    common_dir = impl_dir.parents[2] / "common"

    assert not (impl_dir / "constants.py").exists()
    assert not (impl_dir / "conversation_phase.py").exists()
    assert not (common_dir / "conversation_signals.py").exists()


def test_production_eot_score_does_not_apply_fixed_phrase_overrides() -> None:
    semantic_score = inspect.getsource(ContextEnhancedEot.semantic_completeness_score)
    interrupt_score = inspect.getsource(ContextEnhancedEot.compute_score)
    interrupt_entrypoint = inspect.getsource(EidolonEOTModel.should_interrupt)

    for source in (semantic_score, interrupt_score, interrupt_entrypoint):
        assert "CONTINUATION_INTENT_PATTERNS" not in source
        assert "FILLER_WORDS" not in source
        assert "is_strong_interrupt_intent" not in source
        assert "_get_context_adjustment" not in source


def test_semantic_handler_never_executes_from_intent_alone() -> None:
    source = inspect.getsource(semantic_interrupt.SemanticInterruptHandler)

    assert "hard_stop_intent" not in source
    assert "decision.intent is InterruptIntent.HARD_STOP" not in source


def test_provisional_effects_cannot_publish_next_turn_metadata() -> None:
    source = inspect.getsource(decision_effects.DecisionEffectApplier)

    assert "set_turn_decision_metadata" not in source
    assert "set_turn_control_metadata" not in source
    assert "publish_turn_control" not in source
