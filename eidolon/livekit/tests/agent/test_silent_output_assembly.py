from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from eidolon_sdk.biz.presentation import OutputSelection, SessionOutputPlan
from eidolon.livekit.agent.factory import SharedStageFactory
from eidolon.livekit.agent.shared.pipeline import BasePipeline
from eidolon.livekit.agent.full_duplex import StreamingPipeline
from eidolon.livekit.common.config.schema import EffectiveAgentConfig, ProvidersConfig


def silent_plan():
    return SessionOutputPlan(
        session_id="session-1",
        policy_revision=1,
        outputs=OutputSelection(expression=True),
        expression_profile="eidolon.face.v1",
    )


def test_production_factory_never_constructs_tts_for_silent_session(monkeypatch):
    tts_builder = Mock(side_effect=AssertionError("TTS constructor reached"))
    monkeypatch.setattr(SharedStageFactory, "_build_llm", lambda _: object())
    monkeypatch.setattr(SharedStageFactory, "_build_stt", lambda _: object())
    monkeypatch.setattr(SharedStageFactory, "_build_vad", lambda _: None)
    monkeypatch.setattr(SharedStageFactory, "_build_tts", tts_builder)
    monkeypatch.setattr(SharedStageFactory, "build_interrupt_classifier", lambda _: None)
    cfg = replace(
        EffectiveAgentConfig(),
        providers=ProvidersConfig(tts_provider=None, brain_provider="eidolon_agent"),
    )
    monkeypatch.setattr(
        "eidolon.livekit.agent.factory._build_runtime_services",
        lambda _: SimpleNamespace(resolve_room=AsyncMock()),
    )
    monkeypatch.setattr(
        "eidolon.livekit.agent.factory._build_device_token_source", lambda **_: lambda: "test"
    )
    monkeypatch.setattr(
        "eidolon.livekit.agent.eidolon_agent_rpc.EidolonAgentGrpcLlm", lambda **_: object()
    )
    factory = SharedStageFactory.from_config(
        cfg, runtime_session_id="session-1", output_plan=silent_plan()
    )
    assert factory.tts is None
    assert factory.outputs.expression and not factory.outputs.speech
    tts_builder.assert_not_called()


@pytest.mark.parametrize(
    "tts,plan,session",
    [
        (object(), silent_plan(), "session-1"),
        (None, None, "session-1"),
        (None, silent_plan(), "session-other"),
    ],
)
def test_stage_and_session_mismatch_are_rejected(tts, plan, session):
    with pytest.raises(ValueError):
        SharedStageFactory(
            llm=object(), stt=object(), tts=tts, runtime_session_id=session, output_plan=plan
        )


@pytest.mark.asyncio
async def test_silent_warmup_only_visits_input_stages():
    stt = SimpleNamespace(warmup=AsyncMock())
    factory = SharedStageFactory(
        llm=object(), stt=stt, tts=None, runtime_session_id="session-1", output_plan=silent_plan()
    )

    class InputOnlyPipeline(BasePipeline):
        async def run(self, room):
            pass

    pipeline = InputOnlyPipeline.__new__(InputOnlyPipeline)
    pipeline._factory = factory
    assert pipeline._lifecycle_stages() == [stt]
    await pipeline._warmup_stages()
    stt.warmup.assert_awaited_once()


def test_silent_session_suppresses_fixed_welcome_for_every_turn_mode():
    from eidolon.livekit.agent.half_duplex import HalfDuplexPttPipeline

    for kind in (StreamingPipeline, HalfDuplexPttPipeline):
        pipeline = kind.__new__(kind)
        pipeline._factory = SimpleNamespace(outputs=silent_plan().outputs)
        pipeline._session_intent = "user_initiated"
        pipeline._welcome_message = "Never synthesize me"
        assert pipeline._welcome_on_enter() is None


@pytest.mark.asyncio
async def test_silent_agent_has_no_transcription_node_side_effects():
    from eidolon.livekit.agent.session.policy_bound_agent import PolicyBoundAgent

    agent = PolicyBoundAgent(instructions="", outputs=silent_plan().outputs)

    async def text():
        raise AssertionError("forbidden transcript must not be consumed or published")
        yield "private answer"

    assert await agent.transcription_node(text(), {}) is None


def test_dispatch_output_plan_cannot_be_bound_to_another_session():
    import json
    from eidolon.livekit.agent.server import _resolve_output_plan

    ctx = SimpleNamespace(
        job=SimpleNamespace(metadata=json.dumps({"output_plan": silent_plan().model_dump()}))
    )
    assert _resolve_output_plan(ctx, "session-1") == silent_plan()
    with pytest.raises(ValueError, match="OUTPUT_PLAN_SESSION_MISMATCH"):
        _resolve_output_plan(ctx, "session-2")
