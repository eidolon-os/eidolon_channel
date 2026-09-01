"""Model, ONNX backend and public EOT adapter contract tests."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from livekit.agents.llm import ChatContext

from eidolon.livekit.plugins.eot import (
    ChineseModel,
    ContextEnhancedEot,
    MultilingualModel,
)
from eidolon.livekit.plugins.eot.impl.eot_backend import OnnxEotBackend
from eidolon.livekit.plugins.eot.impl.eot_manager import _resolve_default_model_dir


def test_bundled_model_artifacts_are_complete() -> None:
    model_dir = _resolve_default_model_dir()

    assert model_dir.exists()
    assert (model_dir / "chinese_best_model_q8.onnx").exists()
    assert (model_dir / "multilingual_best_model_q8.onnx").exists()
    assert (model_dir / "tokenizer").is_dir()


@pytest.fixture(scope="module")
def chinese_backend() -> OnnxEotBackend:
    return OnnxEotBackend(_resolve_default_model_dir(), prefer_multilingual=False)


def test_onnx_backend_loads(chinese_backend: OnnxEotBackend) -> None:
    assert chinese_backend._session is not None
    assert chinese_backend._tokenizer is not None


@pytest.mark.parametrize("text", ["你好", "我想查询今天的天气", "mixed language input"])
def test_onnx_backend_returns_probability(
    chinese_backend: OnnxEotBackend,
    text: str,
) -> None:
    assert 0.0 <= chinese_backend.score(text) <= 1.0


@pytest.mark.parametrize("text", ["", "   "])
def test_onnx_backend_empty_text_is_zero(
    chinese_backend: OnnxEotBackend,
    text: str,
) -> None:
    assert chinese_backend.score(text) == 0.0


def test_model_variants_select_the_expected_backend() -> None:
    assert ChineseModel().model == "chinese"
    assert MultilingualModel().model == "multilingual"


@pytest.mark.asyncio
async def test_framework_predict_end_of_turn_uses_last_user_message() -> None:
    model = ChineseModel()
    chat = ChatContext()
    chat.add_message(role="assistant", content=["上一条回复"])
    chat.add_message(role="user", content=["这是当前完整问题"])

    expected = model._context_eot.semantic_completeness_score("这是当前完整问题")
    assert await model.predict_end_of_turn(chat) == pytest.approx(expected)


@pytest.mark.asyncio
async def test_framework_predict_end_of_turn_handles_empty_context() -> None:
    assert await ChineseModel().predict_end_of_turn(ChatContext()) == 0.0


@pytest.mark.asyncio
async def test_framework_detector_metadata_contract() -> None:
    model = ChineseModel()

    assert model.provider == "eidolon"
    assert await model.supports_language("zh-CN") is True
    assert await model.unlikely_threshold("zh-CN") == pytest.approx(
        model._config.eot_unlikely_threshold
    )


def _ready_for_semantic_cut(model: ChineseModel) -> None:
    model.update_vad(True)
    model._state._vad.active_since = time.time() - 1.0
    model._state._sentence.start_time = time.time() - 1.0
    model._state._sentence.last_cut_time = time.time() - 1.0
    model._state._interrupt.last_interrupt_time = 0.0


def test_semantic_interrupt_high_model_score_cuts() -> None:
    model = ChineseModel()
    _ready_for_semantic_cut(model)
    model._context_eot.compute_score = lambda *_args, **_kwargs: 0.9

    assert model.should_interrupt("任意完整文本", vad_active=True) is True
    assert model.current_eot_score == pytest.approx(0.9)


def test_semantic_interrupt_low_model_score_holds() -> None:
    model = ChineseModel()
    _ready_for_semantic_cut(model)
    model._context_eot.compute_score = lambda *_args, **_kwargs: 0.3

    assert model.should_interrupt("任意完整文本", vad_active=True) is False
    assert model.current_eot_score == pytest.approx(0.3)


def test_interrupt_cut_records_cooldown_and_text() -> None:
    model = ChineseModel()
    _ready_for_semantic_cut(model)
    model._context_eot.compute_score = lambda *_args, **_kwargs: 0.9

    assert model.should_interrupt("原始流式文本", vad_active=True) is True
    state = model._context_eot.get_current_state()
    assert state["text"] == "原始流式文本"
    assert state["last_interrupt_time"] is not None


def test_dynamic_threshold_adapter_does_not_classify_wording() -> None:
    model = ChineseModel()

    left = model.get_dynamic_silence_threshold("文本甲", p_complete=0.5, is_final=False)
    right = model.get_dynamic_silence_threshold("完全不同的文本乙", p_complete=0.5, is_final=False)
    assert left == right == pytest.approx(1.0)


def test_model_reset_preserves_vad_but_clears_turn_state() -> None:
    model = ChineseModel()
    model.update_vad(True)
    model.update_asr("当前文本", is_final=True)

    model.reset()

    assert model._state.vad_active is True
    assert model._state.current_text == ""
    assert model.current_eot_score == 0.0


def test_path_a_and_path_b_share_learned_score_without_guards() -> None:
    context = ContextEnhancedEot()
    text = "完整问题"

    assert context.compute_score(text) == pytest.approx(context.p_complete_score(text))


def test_interrupt_guard_does_not_change_framework_score() -> None:
    context = ContextEnhancedEot(cooldown_period=10.0)
    context.record_interrupt("前一条文本")

    assert context.p_complete_score("下一条文本") > 0.0
    assert context.compute_score("下一条文本") == 0.0


def test_test_file_resolves_inside_current_worktree() -> None:
    assert Path(__file__).resolve().is_relative_to(Path.cwd().resolve())
