from __future__ import annotations

from dataclasses import dataclass

import pytest

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.runtime import ResolvedContext
from eidolon.livekit.agent.session.voiceprint import VoiceprintTurnObserver
from eidolon.livekit.agent.speaker_verification.signal import SpeakerSignal


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
        tenant_id="default",
        user_id="manson",
        agent_id="agent_1",
        template_id="caretaker_jiezhi",
        memory_mcp_url="http://127.0.0.1:8765/mcp",
        device_id=None,
    )


async def _resolve_device_context(_room):
    return ResolvedContext(
        tenant_id="default",
        user_id="manson",
        agent_id="agent_1",
        template_id="caretaker_jiezhi",
        memory_mcp_url="http://127.0.0.1:8765/mcp",
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
