from __future__ import annotations

from types import SimpleNamespace

import pytest

from livekit.agents.types import NOT_GIVEN, TimedString
from livekit.agents.voice import agent_activity

from eidolon.livekit.plugins.stt.bailian import BailianFunASRSTT
from eidolon.livekit.plugins.stt.bailian.models import parse_funasr_message
from eidolon.livekit.plugins.stt.bailian.speech_stream import (
    BailianFunASRSpeechStream,
)


FUNASR_ALIGNED_SAMPLE = {
    "header": {
        "task_id": "2bf83b9a-baeb-4fda-8d9a-test",
        "event": "result-generated",
        "attributes": {},
    },
    "payload": {
        "output": {
            "sentence": {
                "begin_time": 170,
                "end_time": 920,
                "text": "好，我知道了",
                "heartbeat": False,
                "sentence_end": True,
                "sentence_id": 1,
                "words": [
                    {
                        "begin_time": 170,
                        "end_time": 295,
                        "text": "好",
                        "punctuation": "，",
                    },
                    {
                        "begin_time": 295,
                        "end_time": 503,
                        "text": "我",
                        "punctuation": "",
                    },
                    {
                        "begin_time": 503,
                        "end_time": 711,
                        "text": "知道",
                        "punctuation": "",
                    },
                    {
                        "begin_time": 711,
                        "end_time": 920,
                        "text": "了",
                        "punctuation": "",
                    },
                ],
            }
        },
        "usage": {
            "duration": 3,
        },
    },
}


def test_funasr_document_sample_maps_to_livekit_timed_strings() -> None:
    event_name, parsed = parse_funasr_message(FUNASR_ALIGNED_SAMPLE)

    assert event_name == "result-generated"
    assert len(parsed.sentences) == 1
    sentence = parsed.sentences[0]
    assert sentence.begin_time == 170
    assert sentence.end_time == 920
    assert sentence.text == "好，我知道了"

    words = BailianFunASRSpeechStream.build_timed_strings(sentence)

    assert [str(word) for word in words] == ["好", "我", "知道", "了"]
    assert all(isinstance(word, TimedString) for word in words)
    assert words[0].start_time == pytest.approx(0.170)
    assert words[0].end_time == pytest.approx(0.295)
    assert words[-1].start_time == pytest.approx(0.711)
    assert words[-1].end_time == pytest.approx(0.920)
    assert "，" not in str(words[0])


def test_bailian_stt_declares_word_aligned_transcript() -> None:
    stt = BailianFunASRSTT(api_url="ws://localhost:9999", api_key="test-key")

    assert stt.capabilities.streaming is True
    assert stt.capabilities.interim_results is True
    assert stt.capabilities.aligned_transcript == "word"


def test_livekit_native_adaptive_compatibility_accepts_bailian(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = object()
    monkeypatch.setattr(
        agent_activity.inference,
        "AdaptiveInterruptionDetector",
        lambda: detector,
    )

    fake_activity = SimpleNamespace(
        stt=BailianFunASRSTT(api_url="ws://localhost:9999", api_key="test-key"),
        vad=object(),
        llm=object(),
        allow_interruptions=True,
        _turn_detection=object(),
        _agent=SimpleNamespace(interruption_detection="adaptive"),
        _session=SimpleNamespace(interruption_detection=NOT_GIVEN),
    )

    resolved = agent_activity.AgentActivity._resolve_interruption_detection(
        fake_activity
    )

    assert resolved is detector
