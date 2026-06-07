"""ProviderEventObserver tests."""

from __future__ import annotations

from types import SimpleNamespace

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.session import ProviderEventObserver


class _FakeProvider:
    def __init__(self) -> None:
        self.handlers = {}

    def on(self, name: str, handler) -> None:
        self.handlers[name] = handler


def test_provider_event_observer_records_brain_and_tts_events() -> None:
    fake_llm = _FakeProvider()
    fake_tts = _FakeProvider()
    timeline = TurnTimeline("turn-provider")
    observer = ProviderEventObserver(
        factory=SimpleNamespace(
            llm=SimpleNamespace(llm=fake_llm),
            tts=SimpleNamespace(tts=fake_tts),
        ),
        get_timeline=lambda: timeline,
    )

    observer.install_brain_provider_event_observer()
    observer.install_tts_provider_event_observer()
    fake_llm.handlers["provider_event"](
        {
            "provider": "eidolon_agent_rpc",
            "event": "brain_first_delta",
            "timestamp": 20.25,
            "turn_id": "turn-provider",
            "request_id": "req-1",
        }
    )
    fake_tts.handlers["provider_event"](
        {
            "provider": "bailian",
            "model": "cosyvoice",
            "event": "tts_stream_started",
            "timestamp": 20.4,
        }
    )
    fake_tts.handlers["provider_event"](
        {
            "provider": "bailian",
            "model": "cosyvoice",
            "event": "tts_connection_acquired",
            "timestamp": 20.45,
        }
    )
    fake_tts.handlers["provider_event"](
        {
            "provider": "bailian",
            "model": "cosyvoice",
            "event": "tts_request_started",
            "timestamp": 20.5,
        }
    )
    fake_tts.handlers["provider_event"](
        {
            "provider": "bailian",
            "model": "cosyvoice",
            "event": "tts_first_text_sent",
            "timestamp": 20.55,
        }
    )
    fake_tts.handlers["provider_event"](
        {
            "provider": "bailian",
            "model": "cosyvoice",
            "event": "tts_provider_first_audio",
            "timestamp": 20.6,
        }
    )

    snap = timeline.snapshot()
    assert snap["timestamps"]["brain_first_delta_at"] == 20.25
    assert snap["timestamps"]["llm_first_delta_at"] == 20.25
    assert snap["timestamps"]["tts_stream_started_at"] == 20.4
    assert snap["timestamps"]["tts_connection_acquired_at"] == 20.45
    assert snap["timestamps"]["tts_request_started_at"] == 20.5
    assert snap["timestamps"]["tts_first_text_sent_at"] == 20.55
    assert snap["timestamps"]["tts_provider_first_audio_at"] == 20.6
    assert snap["attrs"]["brain_rpc"]["request_id"] == "req-1"
    assert snap["attrs"]["tts_stream"]["model"] == "cosyvoice"


def test_provider_event_observer_replays_pending_stt_events() -> None:
    fake_stt = _FakeProvider()
    timeline: TurnTimeline | None = None
    observer = ProviderEventObserver(
        factory=SimpleNamespace(stt=SimpleNamespace(stt=fake_stt)),
        get_timeline=lambda: timeline,
    )

    observer.install_stt_provider_event_observer()
    fake_stt.handlers["provider_event"](
        {
            "provider": "bailian",
            "model": "fun-asr",
            "event": "stt_turn_first_audio_sent",
            "timestamp": 30.05,
            "turn_id": "turn-stt",
            "stream_id": "stream-1",
        }
    )

    timeline = TurnTimeline("turn-stt")
    timeline.mark_at("speech_started_at", 30.1)
    observer.apply_pending_stt_provider_events()

    snap = timeline.snapshot()
    assert observer.pending_stt_provider_events == []
    assert snap["timestamps"]["stt_first_audio_sent_at"] == 30.05
    assert snap["attrs"]["stt_stream"]["stream_id"] == "stream-1"
