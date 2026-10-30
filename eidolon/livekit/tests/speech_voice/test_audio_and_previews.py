from __future__ import annotations

import asyncio
import base64
import io
import json
import wave

import numpy as np
import pytest
from eidolon_sdk.biz.speech_voice import VoiceRef

from eidolon.speech_voice import reference_audio
from eidolon.speech_voice.preview import PREVIEW_TEXTS, PreviewBusy, VoicePreviews
from eidolon.speech_voice.wav import wav_bytes, wav_pcm

REF = VoiceRef(provider="fake", model="m1", voice_id="warm-f")


def _tone_wav(seconds: float, *, rate: int = 44100, amplitude: float = 0.3) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    samples = (np.sin(2 * np.pi * 220 * t) * amplitude * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1), out.setsampwidth(2), out.setframerate(rate)
        out.writeframes(samples.tobytes())
    return buffer.getvalue()


@pytest.fixture
def speech_everywhere(monkeypatch):
    """The real FireRed VAD is exercised end to end; here every window counts as speech."""
    def flags(samples, size=512):
        return np.ones(len(samples) // size, dtype=bool), size
    monkeypatch.setattr(reference_audio, "_speech_flags", flags)


def test_a_reference_is_normalized_to_24k_mono(speech_everywhere) -> None:
    result = reference_audio.check_reference(_tone_wav(12))
    assert result.accepted and result.window_start_ms == 0 and 11900 <= result.window_ms <= 12000
    pcm, rate = wav_pcm(base64.b64decode(result.wav_base64))
    assert rate == 24000 and abs(len(pcm) // 2 - 12 * 24000) < 100


def test_a_long_file_is_cut_to_twenty_seconds_where_the_owner_says(speech_everywhere) -> None:
    result = reference_audio.check_reference(_tone_wav(40), start_ms=5000)
    assert (result.window_start_ms, result.window_ms) == (5000, 20000)
    late = reference_audio.check_reference(_tone_wav(40), start_ms=39000)
    assert late.window_start_ms == 20000


@pytest.mark.parametrize("data, problem", [
    (b"not audio at all", "unreadable"),
    (b"x" * (11 * 1024 * 1024), "too_large"),
])
def test_unusable_input_is_named(data, problem) -> None:
    assert reference_audio.check_reference(data).problems == [problem]


def test_length_quietness_and_clipping_are_judged(speech_everywhere) -> None:
    assert reference_audio.check_reference(_tone_wav(4)).problems == ["too_short"]
    assert reference_audio.check_reference(_tone_wav(70)).problems == ["too_long"]
    assert "too_quiet" in reference_audio.check_reference(_tone_wav(12, amplitude=0.001)).problems
    assert "clipping" in reference_audio.check_reference(_tone_wav(12, amplitude=1.5)).problems


def test_without_speech_a_reference_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(reference_audio, "_speech_flags",
                        lambda samples, size=512: (np.zeros(len(samples) // size, dtype=bool), size))
    assert reference_audio.check_reference(_tone_wav(12)).problems == ["too_little_speech"]


def test_wav_round_trip_and_the_aigc_chunk() -> None:
    data = wav_bytes(b"\x01\x00\x02\x00", 16000, aigc={"ContentProducer": "eidolon/fake"})
    assert wav_pcm(data) == (b"\x01\x00\x02\x00", 16000)
    chunk = data.index(b"AIGC")
    size = int.from_bytes(data[chunk + 4:chunk + 8], "little")
    assert json.loads(data[chunk + 8:chunk + 8 + size].rstrip(b"\0")) == {"Label": "1", "ContentProducer": "eidolon/fake"}
    with wave.open(io.BytesIO(data)) as parsed:  # an ordinary reader skips the extra chunk
        assert parsed.getframerate() == 16000 and parsed.readframes(2) == b"\x01\x00\x02\x00"


@pytest.mark.parametrize("producer", ["eidolon/odd", "eidolon/even"])
def test_every_chunk_is_found_by_a_reader_that_ignores_padding(producer) -> None:
    """Android's WAV extractor steps by each chunk's declared size alone."""
    data = wav_bytes(b"\x01\x00" * 8, 16000, aigc={"ContentProducer": producer})
    offset, seen = 12, []
    while offset + 8 <= len(data):
        seen.append(data[offset:offset + 4])
        offset += 8 + int.from_bytes(data[offset + 4:offset + 8], "little")
    assert seen == [b"fmt ", b"AIGC", b"data"]
    assert offset == len(data)


async def test_previews_are_made_once_marked_as_ai_and_bounded(tmp_path) -> None:
    calls = []

    async def synthesize(ref, text):
        calls.append(text)
        return b"\x01\x00" * 1600, 16000

    previews = VoicePreviews(tmp_path, synthesize=synthesize, propagator="eidolon-host-x")
    first = await previews.wav(REF, PREVIEW_TEXTS[0])
    again = await previews.wav(REF, PREVIEW_TEXTS[0])
    assert first == again and calls == [PREVIEW_TEXTS[0]]
    assert b"eidolon-host-x" in first and b"AIGC" in first
    await previews.wav(REF, "rendition", keep=False)
    await previews.wav(REF, "rendition", keep=False)
    assert calls.count("rendition") == 2


async def test_a_queue_beyond_four_is_turned_away(tmp_path) -> None:
    gate = asyncio.Event()

    async def slow(ref, text):
        await gate.wait()
        return b"\x01\x00" * 160, 16000

    previews = VoicePreviews(tmp_path, synthesize=slow, propagator="p", queue_limit=4)
    waiting = [asyncio.create_task(previews.wav(REF, f"t{i}")) for i in range(4)]
    await asyncio.sleep(0)
    with pytest.raises(PreviewBusy):
        await previews.wav(REF, "one too many")
    gate.set()
    await asyncio.gather(*waiting)
