"""Each hop's timeout, held to the turn budget it is derived from.

Before this, a turn's three hops carried 15 s, 10 s and 15 s — each longer than
a whole turn is allowed to take, and summing to about a minute. Nothing stated
what they should add up to, so nothing could be wrong.
"""

from __future__ import annotations

from eidolon_sdk.biz.contracts.turn_latency import (
    TURN_FIRST_AUDIO_BUDGET_S,
    generation_allowance_s,
    give_up_after_s,
    recognition_allowance_s,
    voice_allowance_s,
)

from eidolon.livekit.plugins.stt.local_asr.config import LocalAsrSTTConfig
from eidolon.livekit.plugins.tts.local_tts.config import LocalTtsConfig


def test_no_hop_waits_longer_than_a_whole_turn() -> None:
    """The property that was missing. A hop allowed to wait longer than the
    turn consumes the other hops' share, and the turn then fails with nobody
    at fault — which is what happened: recognition could hang 15 s inside a
    budget nobody had written down."""

    for name, waited in (
        ("local_asr final", LocalAsrSTTConfig().final_timeout_s),
        ("local_tts first frame", LocalTtsConfig().first_frame_timeout_s),
        ("generation first delta", give_up_after_s(generation_allowance_s())),
    ):
        assert waited < TURN_FIRST_AUDIO_BUDGET_S, (
            f"{name} waits {waited}s inside a {TURN_FIRST_AUDIO_BUDGET_S}s turn"
        )


def test_the_plugins_derive_their_timeouts_rather_than_naming_them() -> None:
    """A number written into a plugin is a number that does not move when the
    budget does."""

    assert LocalAsrSTTConfig().final_timeout_s == give_up_after_s(recognition_allowance_s())
    assert LocalTtsConfig().first_frame_timeout_s == give_up_after_s(voice_allowance_s())


def test_the_first_delta_deadline_is_not_the_connection_timeout() -> None:
    """The conflation that caused the failure.

    `conn_options.timeout` is a connection timeout, and `grpc_llm` used it for
    both opening the session and waiting for the first text chunk. Against a
    hosted model those are nearly the same; against a model on this Host's own
    cores a turn needing 11 s of prompt reading was cancelled at the
    framework's 10 s default, leaving `Cannot write to closing transport` and
    nothing to speak.
    """

    import inspect

    from eidolon.livekit.agent.eidolon_agent_rpc import grpc_llm

    source = inspect.getsource(grpc_llm)

    # The connection still uses conn_options; the first delta no longer does.
    assert grpc_llm.FIRST_DELTA_TIMEOUT_S == 30.0
    assert "first_delta_timeout = FIRST_DELTA_TIMEOUT_S" in source
    assert "first_delta_deadline = (\n                asyncio.get_running_loop().time() + first_delta_timeout" in source
    # And the old shape is gone: the deadline must not be built from `timeout`.
    assert "first_delta_deadline = asyncio.get_running_loop().time() + timeout" not in source


def test_a_cold_prompt_prefix_does_not_fit_and_says_so() -> None:
    """Measured on RK3588: a warm prefix produced the first text token in
    0.85 s and a cold one in 13 s. The allowance sits between them, so a Host
    whose prompt puts volatile content ahead of reusable content fails this
    hop — visibly, and attributed to generation rather than to the voice."""

    assert 0.85 < generation_allowance_s()
    assert 13.0 > give_up_after_s(generation_allowance_s())
