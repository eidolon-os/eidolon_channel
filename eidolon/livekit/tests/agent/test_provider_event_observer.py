"""ProviderEventObserver tests."""

from __future__ import annotations

import asyncio
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
    timeline.mark("turn_committed_at")
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
            "event": "brain_request_sent",
            "timestamp": 20.1,
            "turn_id": "turn-provider",
            "request_id": "req-1",
        }
    )
    fake_llm.handlers["provider_event"](
        {
            "provider": "eidolon_agent_rpc",
            "event": "brain_first_delta",
            "timestamp": 20.25,
            "turn_id": "turn-provider",
            "request_id": "req-1",
            "text_chars": 18,
            "framework_text_chars": 9,
            "user_text_source": "user_turn_coordinator",
            "text_overridden": True,
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
    assert snap["attrs"]["brain_rpc"]["user_text_source"] == "user_turn_coordinator"
    assert snap["attrs"]["brain_rpc"]["text_overridden"] is True
    assert snap["attrs"]["brain_rpc"]["text_chars"] == 18
    assert snap["attrs"]["brain_rpc"]["framework_text_chars"] == 9
    assert snap["attrs"]["tts_stream"]["model"] == "cosyvoice"


def test_provider_event_observer_marks_brain_speaking_without_delta() -> None:
    fake_llm = _FakeProvider()
    timeline = TurnTimeline("turn-silent")
    timeline.mark("turn_committed_at")
    flushes: list[str] = []
    observer = ProviderEventObserver(
        factory=SimpleNamespace(llm=SimpleNamespace(llm=fake_llm)),
        get_timeline=lambda: timeline,
        flush_timeline=lambda _timeline, reason: flushes.append(reason),
    )

    observer.install_brain_provider_event_observer()
    fake_llm.handlers["provider_event"](
        {
            "provider": "eidolon_agent_rpc",
            "event": "brain_request_started",
            "timestamp": 10.0,
        }
    )
    fake_llm.handlers["provider_event"](
        {
            "provider": "eidolon_agent_rpc",
            "event": "brain_state",
            "timestamp": 10.2,
            "state": "speaking",
        }
    )
    assert flushes == []
    fake_llm.handlers["provider_event"](
        {
            "provider": "eidolon_agent_rpc",
            "event": "brain_done",
            "timestamp": 11.0,
        }
    )

    snap = timeline.snapshot()
    assert snap["timestamps"]["brain_state_speaking_at"] == 10.2
    assert snap["attrs"]["agent_output"]["phase"] == "silent_failure"
    assert snap["attrs"]["agent_output"]["outcome"] == "brain_done_without_delta"
    assert snap["attrs"]["agent_output"]["silent_failure"] is True
    assert flushes == ["agent_output_brain_done_without_delta"]


def test_provider_event_observer_marks_brain_error_without_delta() -> None:
    fake_llm = _FakeProvider()
    timeline = TurnTimeline("turn-error")
    timeline.mark("turn_committed_at")
    flushes: list[str] = []
    observer = ProviderEventObserver(
        factory=SimpleNamespace(llm=SimpleNamespace(llm=fake_llm)),
        get_timeline=lambda: timeline,
        flush_timeline=lambda _timeline, reason: flushes.append(reason),
    )

    observer.install_brain_provider_event_observer()
    fake_llm.handlers["provider_event"](
        {
            "provider": "eidolon_agent_rpc",
            "event": "brain_error",
            "timestamp": 10.5,
            "code": "internal",
            "message": "internal: failed",
        }
    )

    snap = timeline.snapshot()
    assert snap["timestamps"]["brain_error_at"] == 10.5
    assert snap["attrs"]["agent_output"]["phase"] == "silent_failure"
    assert snap["attrs"]["agent_output"]["outcome"] == "brain_error"
    assert snap["attrs"]["agent_output"]["error_code"] == "internal"
    assert flushes == []


def test_provider_event_observer_accepts_retry_attempt_after_brain_error() -> None:
    fake_llm = _FakeProvider()
    timeline = TurnTimeline("turn-retry")
    timeline.mark("turn_committed_at")
    observer = ProviderEventObserver(
        factory=SimpleNamespace(llm=SimpleNamespace(llm=fake_llm)),
        get_timeline=lambda: timeline,
    )

    observer.install_brain_provider_event_observer()
    fake_llm.handlers["provider_event"](
        {
            "provider": "eidolon_agent_rpc",
            "event": "brain_request_sent",
            "timestamp": 10.0,
            "turn_id": "brain-turn-1",
            "request_id": "eidolon-brain-turn-1",
            "attempt": 1,
        }
    )
    fake_llm.handlers["provider_event"](
        {
            "provider": "eidolon_agent_rpc",
            "event": "brain_error",
            "timestamp": 10.05,
            "turn_id": "brain-turn-1",
            "request_id": "eidolon-brain-turn-1",
            "attempt": 1,
            "code": "first_delta_timeout",
            "message": "first delta timed out",
            "fatal": False,
        }
    )
    fake_llm.handlers["provider_event"](
        {
            "provider": "eidolon_agent_rpc",
            "event": "brain_request_sent",
            "timestamp": 10.1,
            "turn_id": "brain-turn-2",
            "request_id": "eidolon-brain-turn-2",
            "attempt": 2,
        }
    )
    fake_llm.handlers["provider_event"](
        {
            "provider": "eidolon_agent_rpc",
            "event": "brain_first_delta",
            "timestamp": 10.2,
            "turn_id": "brain-turn-2",
            "request_id": "eidolon-brain-turn-2",
            "attempt": 2,
        }
    )

    snap = timeline.snapshot()
    assert snap["timestamps"]["brain_request_sent_at"] == 10.0
    assert snap["timestamps"]["brain_first_delta_at"] == 10.2
    assert snap["attrs"]["brain_rpc"]["turn_id"] == "brain-turn-2"
    assert snap["attrs"]["brain_rpc"]["request_id"] == "eidolon-brain-turn-2"
    assert snap["attrs"]["brain_rpc"]["attempt"] == 2


def test_provider_event_observer_flushes_terminal_llm_error_without_delta() -> None:
    fake_llm = _FakeProvider()
    timeline = TurnTimeline("turn-llm-error")
    timeline.mark("turn_committed_at")
    flushes: list[str] = []
    observer = ProviderEventObserver(
        factory=SimpleNamespace(llm=SimpleNamespace(llm=fake_llm)),
        get_timeline=lambda: timeline,
        flush_timeline=lambda _timeline, reason: flushes.append(reason),
    )

    observer.install_llm_metrics_observer()
    fake_llm.handlers["error"](
        SimpleNamespace(
            timestamp=12.0,
            label="eidolon_agent_rpc",
            recoverable=False,
            error=RuntimeError("failed after retries"),
        )
    )

    snap = timeline.snapshot()
    assert snap["timestamps"]["llm_error_at"] == 12.0
    assert snap["attrs"]["llm_error"]["recoverable"] is False
    assert snap["attrs"]["agent_output"]["phase"] == "silent_failure"
    assert snap["attrs"]["agent_output"]["outcome"] == "llm_error_without_delta"
    assert flushes == ["agent_output_llm_error_without_delta"]


def test_provider_event_observer_snapshots_first_delta_timeout() -> None:
    async def _run() -> None:
        fake_llm = _FakeProvider()
        timeline = TurnTimeline("turn-timeout")
        timeline.mark("turn_committed_at")
        snapshots: list[tuple[str, dict]] = []
        observer = ProviderEventObserver(
            factory=SimpleNamespace(llm=SimpleNamespace(llm=fake_llm)),
            get_timeline=lambda: timeline,
            append_timeline_snapshot=lambda item, reason: snapshots.append(
                (reason, item.snapshot())
            ),
            first_delta_timeout_sec=0.01,
        )

        observer.install_brain_provider_event_observer()
        fake_llm.handlers["provider_event"](
            {
                "provider": "eidolon_agent_rpc",
                "event": "brain_request_sent",
                "timestamp": 20.0,
            }
        )
        await asyncio.sleep(0.03)

        snap = timeline.snapshot()
        assert "llm_first_delta_timeout_at" in snap["timestamps"]
        assert snap["attrs"]["agent_output"]["phase"] == "first_delta_timeout"
        assert snap["attrs"]["agent_output"]["risk"] == "awaiting_first_delta"
        assert snapshots[0][0] == "agent_output_first_delta_timeout"
        assert snapshots[0][1]["turn_id"] == "turn-timeout"

    asyncio.run(_run())


def test_provider_event_observer_cancels_first_delta_timeout_after_delta() -> None:
    async def _run() -> None:
        fake_llm = _FakeProvider()
        timeline = TurnTimeline("turn-no-timeout")
        timeline.mark("turn_committed_at")
        snapshots: list[str] = []
        observer = ProviderEventObserver(
            factory=SimpleNamespace(llm=SimpleNamespace(llm=fake_llm)),
            get_timeline=lambda: timeline,
            append_timeline_snapshot=lambda _item, reason: snapshots.append(reason),
            first_delta_timeout_sec=0.01,
        )

        observer.install_brain_provider_event_observer()
        fake_llm.handlers["provider_event"](
            {
                "provider": "eidolon_agent_rpc",
                "event": "brain_request_sent",
                "timestamp": 30.0,
            }
        )
        fake_llm.handlers["provider_event"](
            {
                "provider": "eidolon_agent_rpc",
                "event": "brain_first_delta",
                "timestamp": 30.005,
            }
        )
        await asyncio.sleep(0.03)

        snap = timeline.snapshot()
        assert "brain_first_delta_at" in snap["timestamps"]
        assert "llm_first_delta_timeout_at" not in snap["timestamps"]
        assert snapshots == []

    asyncio.run(_run())


def test_provider_event_observer_ignores_brain_done_on_open_user_turn() -> None:
    fake_llm = _FakeProvider()
    timeline = TurnTimeline("open-user-turn")
    timeline.mark("speech_started_at")
    flushes: list[str] = []
    observer = ProviderEventObserver(
        factory=SimpleNamespace(llm=SimpleNamespace(llm=fake_llm)),
        get_timeline=lambda: timeline,
        flush_timeline=lambda _timeline, reason: flushes.append(reason),
    )

    observer.install_brain_provider_event_observer()
    fake_llm.handlers["provider_event"](
        {
            "provider": "eidolon_agent_rpc",
            "event": "brain_done",
            "timestamp": 40.0,
            "turn_id": "old-brain-turn",
            "request_id": "eidolon-old-brain-turn",
        }
    )

    snap = timeline.snapshot()
    assert "brain_done_at" not in snap["timestamps"]
    assert "agent_output" not in snap["attrs"]
    assert "brain_rpc" not in snap["attrs"]
    assert flushes == []


def test_provider_event_observer_ignores_mismatched_brain_identity() -> None:
    fake_llm = _FakeProvider()
    timeline = TurnTimeline("turn-current")
    timeline.mark("turn_committed_at")
    flushes: list[str] = []
    observer = ProviderEventObserver(
        factory=SimpleNamespace(llm=SimpleNamespace(llm=fake_llm)),
        get_timeline=lambda: timeline,
        flush_timeline=lambda _timeline, reason: flushes.append(reason),
    )

    observer.install_brain_provider_event_observer()
    fake_llm.handlers["provider_event"](
        {
            "provider": "eidolon_agent_rpc",
            "event": "brain_request_sent",
            "timestamp": 50.0,
            "turn_id": "brain-current",
            "request_id": "eidolon-brain-current",
        }
    )
    fake_llm.handlers["provider_event"](
        {
            "provider": "eidolon_agent_rpc",
            "event": "brain_done",
            "timestamp": 50.4,
            "turn_id": "brain-old",
            "request_id": "eidolon-brain-old",
        }
    )

    snap = timeline.snapshot()
    assert snap["timestamps"]["brain_request_sent_at"] == 50.0
    assert "brain_done_at" not in snap["timestamps"]
    assert snap["attrs"]["brain_rpc"]["turn_id"] == "brain-current"
    assert snap["attrs"]["brain_rpc"]["request_id"] == "eidolon-brain-current"
    assert snap["attrs"]["brain_rpc"]["last_event"] == "brain_request_sent"
    assert flushes == []


def test_provider_event_observer_ignores_tts_on_open_user_turn() -> None:
    fake_tts = _FakeProvider()
    timeline = TurnTimeline("open-user-turn")
    timeline.mark("speech_started_at")
    observer = ProviderEventObserver(
        factory=SimpleNamespace(tts=SimpleNamespace(tts=fake_tts)),
        get_timeline=lambda: timeline,
    )

    observer.install_tts_provider_event_observer()
    fake_tts.handlers["provider_event"](
        {
            "provider": "bailian",
            "model": "cosyvoice",
            "event": "tts_provider_first_audio",
            "timestamp": 60.0,
        }
    )

    snap = timeline.snapshot()
    assert "tts_provider_first_audio_at" not in snap["timestamps"]
    assert "tts_stream" not in snap["attrs"]


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
