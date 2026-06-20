"""Bailian TTS: task-failed recoverability classification.

Root cause (2026-06-20 real-room logs): a pre-warmed pool connection sits idle
between connect and the first ``run-task`` (run-task is deferred until the first
LLM tokens are aggregated). On a slow first token, CosyVoice rejects the
run-task — ``Invalid action('run-task')! Please follow the protocol!`` — which
was classified non-recoverable, so the framework dropped the reply (no audio).

A task-failed BEFORE any audio is a setup-phase failure and IS recoverable: the
framework replays the buffered input through a fresh ``_run`` (fresh pooled
connection). A failure AFTER audio is mid-synthesis and is NOT recoverable.
"""

from __future__ import annotations

import pytest
from livekit.agents.types import APIConnectOptions

from eidolon.livekit.plugins.tts.bailian.config import BailianTTSConfig
from eidolon.livekit.plugins.tts.bailian.tts import BailianTTS


def _tts() -> BailianTTS:
    return BailianTTS(
        BailianTTSConfig(api_key="test", pool_size=1, pool_size_bootstrap=1)
    )


def _task_failed_msg(message: str) -> dict:
    return {
        "header": {
            "event": "task-failed",
            "error_code": "InvalidParameter",
            "error_message": message,
        }
    }


@pytest.mark.asyncio
async def test_run_task_rejection_before_audio_is_recoverable() -> None:
    stream = _tts().stream(conn_options=APIConnectOptions())
    stream._first_provider_audio_emitted = False

    await stream._handle_json_event(
        _task_failed_msg("Invalid action('run-task')! Please follow the protocol!")
    )

    assert stream._task_failed_error is not None
    assert stream._task_failed_error.recoverable is True


@pytest.mark.asyncio
async def test_failure_after_audio_is_not_recoverable() -> None:
    stream = _tts().stream(conn_options=APIConnectOptions())
    # Audio already streamed to the user — retrying would re-emit speech.
    stream._first_provider_audio_emitted = True

    await stream._handle_json_event(
        _task_failed_msg("Invalid action('run-task')! Please follow the protocol!")
    )

    assert stream._task_failed_error is not None
    assert stream._task_failed_error.recoverable is False


@pytest.mark.asyncio
async def test_setup_timeout_is_recoverable() -> None:
    stream = _tts().stream(conn_options=APIConnectOptions())
    stream._first_provider_audio_emitted = False

    await stream._handle_json_event(_task_failed_msg("upstream task timeout"))

    assert stream._task_failed_error is not None
    assert stream._task_failed_error.recoverable is True
