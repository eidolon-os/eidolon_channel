"""Silero provider on the shared window-classifier state machine.

`data/speech_zh_16k.wav` is 2 s of Eidolon's own Mandarin TTS output.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest
from livekit import rtc
from livekit.agents.vad import VADEventType

from eidolon.livekit.agent.providers.vad import load_vad
from eidolon.livekit.common.config.schema import VadPolicyConfig
from eidolon.livekit.plugins.vad.firered import FireredPvadVAD
from eidolon.livekit.plugins.vad.silero import SileroClassifier, SileroVAD
from eidolon.livekit.plugins.vad.silero import classifier as classifier_module
from eidolon.livekit.plugins.vad.silero.classifier import DEFAULT_MODEL_PATH

_SPEECH = Path(__file__).parent / "data" / "speech_zh_16k.wav"


def _speech() -> np.ndarray:
    with wave.open(str(_SPEECH)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


def _probabilities(classifier: SileroClassifier, samples: np.ndarray) -> list[float]:
    f32 = samples.astype(np.float32) / 32768
    return [classifier(f32[i : i + 512]) for i in range(0, len(f32) - 511, 512)]


def test_shipped_model_is_the_verified_upstream_file():
    classifier = SileroClassifier.load()

    assert max(_probabilities(classifier.new_stream(), np.zeros(16000, dtype="<i2"))) < 0.1
    assert max(_probabilities(classifier.new_stream(), _speech())) > 0.9


def test_a_corrupt_model_fails_loudly(tmp_path, monkeypatch):
    pointer = tmp_path / "silero_vad_16k_op15.onnx"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\n")
    monkeypatch.setattr(classifier_module, "DEFAULT_MODEL_PATH", pointer)

    with pytest.raises(ValueError, match="checksum"):
        SileroClassifier.load()


def test_streams_keep_their_own_recurrent_state():
    shared = SileroClassifier.load()
    speech, quiet = shared.new_stream(), shared.new_stream()
    window = _speech()[:512].astype(np.float32) / 32768

    for _ in range(10):
        speech(window)
    before = quiet(np.zeros(512, dtype=np.float32))

    assert before == pytest.approx(shared.new_stream()(np.zeros(512, dtype=np.float32)))


def test_windows_of_the_wrong_size_are_refused():
    with pytest.raises(ValueError):
        SileroClassifier.load()(np.zeros(320, dtype=np.float32))


@pytest.mark.asyncio
async def test_speech_opens_and_closes_one_turn():
    vad = SileroVAD.load(min_silence_duration=0.5)
    stream = vad.stream()
    silence = np.zeros(8000, dtype="<i2")
    audio = np.concatenate([silence, _speech(), silence, silence])
    for i in range(0, len(audio), 320):
        chunk = audio[i : i + 320]
        stream.push_frame(rtc.AudioFrame(chunk.tobytes(), 16000, 1, len(chunk)))
    stream.end_input()

    events = []
    async for event in stream:
        if event.type is not VADEventType.INFERENCE_DONE:
            events.append(event.type)
    await stream.aclose()

    assert events == [VADEventType.START_OF_SPEECH, VADEventType.END_OF_SPEECH]


def test_configured_provider_selects_the_model():
    policy = VadPolicyConfig()

    assert isinstance(load_vad("silero", policy), SileroVAD)
    assert isinstance(load_vad("firered_pvad", policy), FireredPvadVAD)
    assert load_vad("none", policy) is None
    with pytest.raises(ValueError):
        load_vad("webrtc", policy)


def test_thresholds_come_from_the_shared_turn_policy():
    vad = load_vad("silero", VadPolicyConfig(activation_threshold=0.6, min_speech_duration_ms=250))

    assert vad._opts.activation_threshold == 0.6
    assert vad._opts.min_speech_duration == pytest.approx(0.25)


def test_a_provider_is_built_outside_any_running_loop():
    # The worker prewarm builds the VAD synchronously, before any session loop.
    assert isinstance(SileroVAD.load(), SileroVAD)


def test_each_model_file_loads_once_and_is_never_swapped(tmp_path):
    custom = tmp_path / "custom.onnx"
    custom.write_bytes(DEFAULT_MODEL_PATH.read_bytes())

    bundled = SileroVAD.load()._processor

    assert SileroVAD.load()._processor is bundled
    assert SileroVAD.load(model_path=str(custom))._processor is not bundled


def test_only_the_classifier_sample_rate_is_accepted():
    with pytest.raises(ValueError, match="16000 Hz"):
        SileroVAD.load(sample_rate=8000)
