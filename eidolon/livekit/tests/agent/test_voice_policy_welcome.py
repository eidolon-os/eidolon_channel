"""Readiness cues must survive voice restrictions without allowing greetings."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from eidolon.livekit.agent.shared.pipeline import BasePipeline
from eidolon.livekit.common.welcome import WelcomeAudio


@pytest.mark.parametrize(
    "welcome,expected",
    [(WelcomeAudio(), True), ("你好", False), (WelcomeAudio("/tmp/custom.wav"), False)],
)
async def test_restricted_policy_preserves_only_builtin_readiness_cue(welcome, expected):
    admission = SimpleNamespace(authorize=AsyncMock(return_value=False))
    pipeline = SimpleNamespace(
        _factory=SimpleNamespace(
            companion_voice_admission=admission,
            runtime_context_resolver=AsyncMock(
                return_value=SimpleNamespace(companion_id="companion")
            ),
        ),
        _room=object(),
    )
    assert await BasePipeline._voice_policy_allows_welcome(pipeline, welcome) is expected
