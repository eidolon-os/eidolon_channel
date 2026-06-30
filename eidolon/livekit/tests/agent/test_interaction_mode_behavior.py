from __future__ import annotations

from dataclasses import replace

from eidolon_sdk.biz.contracts import (
    INTERACTION_MODE_FULL_DUPLEX,
    INTERACTION_MODE_HALF_DUPLEX,
)

from eidolon.livekit.agent.session.interaction_mode import (
    FullDuplexInteractionMode,
    HalfDuplexInteractionMode,
    build_interaction_mode_behavior,
)
from eidolon.livekit.common.config import TurnPolicyConfig


def _native_policy() -> TurnPolicyConfig:
    return replace(TurnPolicyConfig(), interruption_owner="livekit_native_adaptive")


def test_factory_maps_known_modes() -> None:
    assert isinstance(
        build_interaction_mode_behavior(INTERACTION_MODE_FULL_DUPLEX),
        FullDuplexInteractionMode,
    )
    assert isinstance(
        build_interaction_mode_behavior(INTERACTION_MODE_HALF_DUPLEX),
        HalfDuplexInteractionMode,
    )


def test_factory_unknown_mode_falls_back_to_full_duplex() -> None:
    assert isinstance(build_interaction_mode_behavior("unknown"), FullDuplexInteractionMode)


def test_full_duplex_uses_eot_turn_detection_and_natural_evidence() -> None:
    behavior = FullDuplexInteractionMode()
    eot_model = object()

    assert behavior.turn_detection(lambda: eot_model) is eot_model
    assert behavior.treats_vad_silence_as_turn_boundary() is True
    assert behavior.starts_natural_interruption_candidate() is True
    assert behavior.applies_agent_echo_gate() is True
    assert behavior.allows_low_eot_defer() is True
    assert behavior.allows_playback_low_evidence_reject() is True


def test_half_duplex_uses_manual_ptt_boundaries() -> None:
    behavior = HalfDuplexInteractionMode()

    assert behavior.turn_detection(lambda: object()) == "manual"
    assert behavior.treats_vad_silence_as_turn_boundary() is False
    assert behavior.starts_natural_interruption_candidate() is False
    assert behavior.applies_agent_echo_gate() is False
    assert behavior.allows_low_eot_defer() is False
    assert behavior.allows_playback_low_evidence_reject() is False


def test_full_duplex_native_adaptive_turn_handling() -> None:
    interruption = FullDuplexInteractionMode().interruption_options(
        allow_interruptions=True,
        false_interruption_timeout=6.0,
        turn_policy=_native_policy(),
    )

    assert interruption["enabled"] is True
    assert interruption["discard_audio_if_uninterruptible"] is True
    assert interruption["mode"] == "adaptive"
    assert interruption["resume_false_interruption"] is True


def test_half_duplex_never_enables_native_adaptive_owner() -> None:
    interruption = HalfDuplexInteractionMode().interruption_options(
        allow_interruptions=False,
        false_interruption_timeout=6.0,
        turn_policy=_native_policy(),
    )

    assert interruption["enabled"] is False
    assert interruption["discard_audio_if_uninterruptible"] is False
    assert "mode" not in interruption
    assert "resume_false_interruption" not in interruption
