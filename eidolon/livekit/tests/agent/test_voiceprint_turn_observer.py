from __future__ import annotations

from dataclasses import dataclass

import pytest
from eidolon_sdk.biz.persona import ResolvedRuntimeIdentity as ResolvedContext

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.session.voiceprint import VoiceprintTurnObserver
from eidolon.livekit.common.speaker_verification import SpeakerSignal


@dataclass
class _Frame:
    samples_per_channel: int
    sample_rate: int = 16000

    @property
    def data(self) -> bytes:
        return b"\x01\x00" * self.samples_per_channel


class _Provider:
    provider = "3d_speaker"
    model = "campplus_zh_16k_common"


class _Service:
    def __init__(self, *, known: bool = True, score: float = 0.91) -> None:
        self._provider = _Provider()
        self.calls: list[dict] = []
        self.known = known
        self.score = score

    async def verify_turn(self, **kwargs):
        self.calls.append(kwargs)
        return SpeakerSignal(
            provider="3d_speaker",
            model="campplus_zh_16k_common",
            speaker_user_id=kwargs["user_id"] if self.known else None,
            known=self.known,
            score=self.score,
            owner_confidence=self.score,
            latency_ms=12.3,
            audio_ms=kwargs["audio_ms"],
            profile_id=f"vp_{kwargs['user_id']}_default",
        )


async def _resolve_context(_room):
    return ResolvedContext(
        owner_id="manson",
        companion_id="companion_1",
        memory_realm_id="realm_1",
        genome_id="genome_1",
        schema_version="eidolon.persona_genome",
        genome_hash="pg_voiceprint",
        realizer_version="eidolon.persona_realizer",
        device_id=None,
    )


async def _resolve_device_context(_room):
    return ResolvedContext(
        owner_id="manson",
        companion_id="companion_1",
        memory_realm_id="realm_1",
        genome_id="genome_1",
        schema_version="eidolon.persona_genome",
        genome_hash="pg_voiceprint",
        realizer_version="eidolon.persona_realizer",
        device_id="1c:db:d4:7a:ef:0c",
    )


@pytest.mark.asyncio
async def test_voiceprint_turn_observer_verifies_completed_turn() -> None:
    service = _Service()
    observer = VoiceprintTurnObserver(
        service=service,
        context_resolver=_resolve_context,
    )
    timeline = TurnTimeline("turn_1")

    observer.start_turn(timeline=timeline)
    observer.append_frame(_Frame(samples_per_channel=16000 * 2))
    task = observer.finish_turn()
    assert task is not None
    await task

    assert len(service.calls) == 1
    call = service.calls[0]
    assert call["tenant_id"] == "default"
    assert call["user_id"] == "manson"
    assert call["audio_ms"] == 2000
    assert len(call["audio"]) == 16000 * 2 * 2
    assert timeline.attrs["voiceprint"]["known"] is True
    assert timeline.attrs["voiceprint"]["score"] == 0.91
    assert timeline.attrs["voiceprint"]["cached"] is False
    assert timeline.attrs["voiceprint"]["commit_allowed"] is True


@pytest.mark.asyncio
async def test_voiceprint_turn_observer_reuses_accept_cache_for_short_turn() -> None:
    service = _Service(score=0.95)
    observer = VoiceprintTurnObserver(
        service=service,
        context_resolver=_resolve_context,
    )

    first = TurnTimeline("turn_1")
    observer.start_turn(timeline=first)
    observer.append_frame(_Frame(samples_per_channel=16000 * 4))
    task = observer.finish_turn()
    assert task is not None
    await task

    second = TurnTimeline("turn_2")
    observer.start_turn(timeline=second)
    observer.append_frame(_Frame(samples_per_channel=16000))
    task = observer.finish_turn()
    assert task is not None
    await task

    assert len(service.calls) == 1
    assert second.attrs["voiceprint"]["known"] is True
    assert second.attrs["voiceprint"]["audio_ms"] == 1000
    assert second.attrs["voiceprint"]["latency_ms"] == 0.0
    assert second.attrs["voiceprint"]["cached"] is True
    assert second.attrs["voiceprint"]["commit_allowed"] is True
    assert second.attrs["voiceprint"]["commit_reason"] == "cached_owner_context"


@pytest.mark.asyncio
async def test_voiceprint_accept_cache_short_audio_window_is_configurable() -> None:
    service = _Service(score=0.95)
    observer = VoiceprintTurnObserver(
        service=service,
        accept_cache_short_audio_max_ms=500,
        context_resolver=_resolve_context,
    )

    first = TurnTimeline("turn_1")
    observer.start_turn(timeline=first)
    observer.append_frame(_Frame(samples_per_channel=16000 * 4))
    task = observer.finish_turn()
    assert task is not None
    await task

    second = TurnTimeline("turn_2")
    observer.start_turn(timeline=second)
    observer.append_frame(_Frame(samples_per_channel=16000))
    task = observer.finish_turn()
    assert task is not None
    await task

    assert len(service.calls) == 2
    assert second.attrs["voiceprint"]["cached"] is False


@pytest.mark.asyncio
async def test_voiceprint_turn_observer_allows_provider_known_below_high_confidence() -> None:
    service = _Service(score=0.5734381675720215)
    observer = VoiceprintTurnObserver(
        service=service,
        context_resolver=_resolve_context,
    )
    timeline = TurnTimeline("turn_1")

    observer.start_turn(timeline=timeline)
    observer.append_frame(_Frame(samples_per_channel=16000 * 3))
    task = observer.finish_turn()
    assert task is not None
    await task

    assert timeline.attrs["voiceprint"]["known"] is True
    assert timeline.attrs["voiceprint"]["score"] == 0.5734381675720215
    assert timeline.attrs["voiceprint"]["commit_allowed"] is True
    assert (
        timeline.attrs["voiceprint"]["commit_reason"]
        == "owner_above_provider_threshold:0.573<0.580"
    )


@pytest.mark.asyncio
async def test_voiceprint_turn_observer_blocks_unknown_speaker() -> None:
    service = _Service(known=False, score=0.22)
    observer = VoiceprintTurnObserver(
        service=service,
        context_resolver=_resolve_context,
    )
    timeline = TurnTimeline("turn_1")

    observer.start_turn(timeline=timeline)
    observer.append_frame(_Frame(samples_per_channel=16000 * 3))
    task = observer.finish_turn()
    assert task is not None
    await task

    assert timeline.attrs["voiceprint"]["known"] is False
    assert timeline.attrs["voiceprint"]["commit_allowed"] is False
    assert timeline.attrs["voiceprint"]["commit_reason"] == "speaker_not_owner"


@pytest.mark.asyncio
async def test_voiceprint_turn_observer_allows_trusted_paired_device() -> None:
    service = _Service(known=False, score=0.146)
    observer = VoiceprintTurnObserver(
        service=service,
        context_resolver=_resolve_device_context,
    )
    timeline = TurnTimeline("turn_1")

    observer.start_turn(timeline=timeline)
    observer.append_frame(_Frame(samples_per_channel=16000 * 3))
    task = observer.finish_turn()
    assert task is not None
    await task

    assert len(service.calls) == 1
    assert service.calls[0]["user_id"] == "manson"
    assert timeline.attrs["voiceprint"]["known"] is False
    assert timeline.attrs["voiceprint"]["trusted_paired_device"] is True
    assert timeline.attrs["voiceprint"]["commit_allowed"] is True
    assert timeline.attrs["voiceprint"]["commit_reason"] == "trusted_paired_device"


async def _resolve_context_raises(_room):
    raise RuntimeError("admin down")


@pytest.mark.asyncio
async def test_voiceprint_fail_open_on_capture_error() -> None:
    """A verify/capture error (here: no audio captured — the web case) must NOT
    silently drop the user's turn. It fails OPEN: commit allowed, reason marked
    so it's auditable. Speaker was never proven owner, but blocking on OUR
    capture failure is worse than letting the owner's turn through."""
    service = _Service()
    observer = VoiceprintTurnObserver(
        service=service,
        context_resolver=_resolve_context,  # web user: device_id=None
    )
    timeline = TurnTimeline("turn_1")

    observer.start_turn(timeline=timeline)
    # No append_frame → audio buffer empty → "audio_not_captured".
    task = observer.finish_turn()
    assert task is not None
    await task

    assert service.calls == []  # never reached the verifier (no audio)
    vp = timeline.attrs["voiceprint"]
    assert vp["error"] == "audio_not_captured"
    assert vp["commit_allowed"] is True
    assert vp["commit_reason"] == "verify_error_failopen:audio_not_captured"


@pytest.mark.asyncio
async def test_voiceprint_context_error_still_blocks() -> None:
    """context_error is the one error that must STILL block — we can't run the
    turn without an agent context (task #9 surfaces it to the user)."""
    service = _Service()
    observer = VoiceprintTurnObserver(
        service=service,
        context_resolver=_resolve_context_raises,
    )
    timeline = TurnTimeline("turn_1")

    observer.start_turn(timeline=timeline)
    observer.append_frame(_Frame(samples_per_channel=16000 * 3))
    task = observer.finish_turn()
    assert task is not None
    await task

    vp = timeline.attrs["voiceprint"]
    assert vp["commit_allowed"] is False
    assert vp["commit_reason"].startswith("context_error")
