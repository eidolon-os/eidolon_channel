"""ASR is before admission in both modes; diagnostics must not publish its text."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType
from eidolon.livekit.plugins.stt.bailian.models import FunASRResultGenerated, FunASRSentence
from eidolon.livekit.plugins.stt.bailian.speech_stream import BailianFunASRSpeechStream


@pytest.mark.asyncio
async def test_interim_and_final_reach_owner_but_not_logs_or_provider_events(caplog):
    stream = object.__new__(BailianFunASRSpeechStream)
    stream._SpeechData, stream._SpeechEvent, stream._SpeechEventType = (
        SpeechData,
        SpeechEvent,
        SpeechEventType,
    )
    stream._language, stream._stream_id = "zh", "test"
    stream._parser_confirmed, stream._last_interim_text = [], ""
    stream._first_final_seen = stream._first_partial_seen = False
    stream._cancel_preflight = Mock()
    stream._arm_preflight = Mock()
    stream._stt_ref = SimpleNamespace(emit_provider_event=Mock())
    stream._push_event = Mock()
    caplog.set_level("DEBUG")
    text = "尚未获准的私人内容"
    for final in (False, True):
        await stream._handle_result(
            FunASRResultGenerated(
                task_id="t",
                request_id="r",
                sentences=[FunASRSentence(text=text, sentence_id=1, sentence_end=final)],
            )
        )
    assert [c.args[0].alternatives[0].text for c in stream._push_event.call_args_list] == [
        text,
        text,
    ]
    assert text not in caplog.text
    assert text not in repr(stream._stt_ref.emit_provider_event.call_args_list)
