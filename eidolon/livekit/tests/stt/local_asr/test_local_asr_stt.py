"""The local recognizer: same interface as a provider, none of the machinery.

What most of these check is what is *absent*. The pipeline above this plugin is
already provider-agnostic — `SttStage` "doesn't know which provider it wraps" —
so the thing worth pinning is that this implementation carries no endpoint, no
credential and no reconnect budget, and still answers the same events.
"""

from __future__ import annotations

import asyncio

import pytest

from eidolon.livekit.plugins.stt.local_asr import (
    LocalAsrEndpointError,
    LocalAsrSTT,
    LocalAsrSTTConfig,
    resolve_port,
    resolve_stream_url,
)


def _registry(tmp_path, body: str):
    path = tmp_path / "ports.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_the_endpoint_comes_from_the_hosts_registry(tmp_path) -> None:
    """The number is stated once, by the component that reserves it.

    Ops selects it for a Host by capability and writes it there. A port written
    into this plugin's configuration would be a second statement of a fact that
    already has an author — and the kind that goes stale quietly.
    """

    path = _registry(
        tmp_path, "hub:\n  api:\n    port: 8082\nport_roles:\n  asr_stream: 8768\n"
    )

    assert resolve_port(registry_path=path) == 8768
    assert resolve_stream_url(registry_path=path) == "ws://127.0.0.1:8768/v1/stream"


def test_a_host_without_the_capability_is_refused_not_guessed(tmp_path) -> None:
    """A missing role means this Host has no local recognition to reach.

    Falling back to a literal would turn that into a connection attempt against
    whatever else happens to be listening on the number we guessed.
    """

    path = _registry(tmp_path, "hub:\n  api:\n    port: 8082\n")

    with pytest.raises(LocalAsrEndpointError, match="reserves no 'asr_stream'"):
        resolve_port(registry_path=path)


def test_the_refusal_says_what_to_do_about_it(tmp_path) -> None:
    path = _registry(tmp_path, "hub:\n  api:\n    port: 8082\n")

    with pytest.raises(LocalAsrEndpointError) as raised:
        resolve_port(registry_path=path)

    message = str(raised.value)
    assert "local_asr" in message
    assert "Declare the capability" in message


def test_an_unset_registry_variable_says_so_rather_than_defaulting(monkeypatch) -> None:
    monkeypatch.delenv("EIDOLON_PORTS_FILE", raising=False)

    with pytest.raises(LocalAsrEndpointError, match="EIDOLON_PORTS_FILE"):
        resolve_port()


def test_it_declares_the_same_streaming_contract_a_provider_does() -> None:
    """The pipeline is provider-agnostic, so this has to look like the others.

    Interim results are why a local Host is worth having here: the streaming
    model answers while the words are still being spoken, and the offline pass
    replaces that with a punctuated final.
    """

    recognizer = LocalAsrSTT()

    assert recognizer.capabilities.streaming is True
    assert recognizer.capabilities.interim_results is True
    assert recognizer.provider == "local_asr"


def test_the_provider_name_is_the_capability_name() -> None:
    """One string, so "configured for local speech" and "can do local speech"
    compare directly instead of through a table that could disagree."""

    from eidolon_sdk.biz.contracts import local_asr as contract

    assert LocalAsrSTT().provider == contract.LOCAL_ASR_CAPABILITY


def test_it_carries_no_credential_and_no_address() -> None:
    """Every field a cloud provider needs and this one has no business holding."""

    names = set(vars(LocalAsrSTTConfig()))

    assert not names & {"api_key", "api_url", "url", "host", "region", "model"}


def test_the_audio_contract_is_the_protocols_and_not_a_choice() -> None:
    from eidolon_sdk.biz.contracts import local_asr as contract

    config = LocalAsrSTTConfig()
    assert config.sample_rate == contract.AUDIO_SAMPLE_RATE
    assert config.channels == contract.AUDIO_CHANNELS

    with pytest.raises(ValueError, match="16000 Hz"):
        LocalAsrSTTConfig(sample_rate=8000)
    with pytest.raises(ValueError, match="mono"):
        LocalAsrSTTConfig(channels=2)


def test_a_one_shot_recognize_is_refused_rather_than_faked() -> None:
    """This service answers while the words arrive; buffering a whole utterance
    to ask for it once is the shape it exists to avoid."""

    with pytest.raises(NotImplementedError, match="streaming recognizer"):
        asyncio.run(LocalAsrSTT()._recognize_impl())


def test_the_provider_is_selectable_and_the_pipeline_does_not_change(monkeypatch) -> None:
    """A branch in the factory, and nothing above it.

    `SttStage` wraps any LiveKit-compatible recognizer and, in its own words,
    "doesn't know which provider it wraps". So making a Host speak its own
    speech is one `elif` — not a second pipeline.
    """

    from eidolon.livekit.agent.factory import SharedStageFactory
    from eidolon.livekit.common.config.validators import STT_PROVIDERS

    assert "local_asr" in STT_PROVIDERS

    class _Config:
        providers = type("P", (), {"stt_provider": "local_asr"})()
        local_asr_stt = LocalAsrSTTConfig()

    stage = SharedStageFactory._build_stt(_Config())

    assert type(stage._stt).__name__ == "LocalAsrSTT"


def test_an_unknown_provider_names_every_one_this_build_has() -> None:
    """Including the local one, which the message used to be unable to mention."""

    from eidolon.livekit.agent.factory import SharedStageFactory

    class _Config:
        providers = type("P", (), {"stt_provider": "whisper"})()

    with pytest.raises(ValueError, match="local_asr"):
        SharedStageFactory._build_stt(_Config())
