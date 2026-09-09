"""This Host's own voice: same interface as a provider, none of the machinery.

What most of these check is what is *absent*. The pipeline above this plugin is
already provider-agnostic, so the thing worth pinning is that this
implementation carries no endpoint, no credential and no voice name, and still
answers as a streaming TTS.
"""

from __future__ import annotations

import pytest
from eidolon_sdk.biz.contracts import local_tts as contract

from eidolon.livekit.plugins.tts.local_tts import (
    LocalTTS,
    LocalTtsConfig,
    LocalTtsEndpointError,
    resolve_port,
    resolve_ready_url,
    resolve_stream_url,
)


def _registry(tmp_path, body: str):
    path = tmp_path / "ports.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_the_endpoint_comes_from_the_hosts_registry(tmp_path) -> None:
    """The number is stated once, by the component that reserves it, and Ops
    writes it into a Host's registry for the capabilities that Host declares."""

    path = _registry(
        tmp_path, "hub:\n  api:\n    port: 8082\nport_roles:\n  tts_stream: 8770\n"
    )

    assert resolve_port(registry_path=path) == 8770
    assert resolve_stream_url(registry_path=path) == "ws://127.0.0.1:8770/v1/stream"
    assert resolve_ready_url(registry_path=path) == "http://127.0.0.1:8770/readyz"


def test_a_host_without_the_capability_is_refused_not_guessed(tmp_path) -> None:
    """A missing role means this Host has no local synthesis to reach. Falling
    back to a literal would turn that into a connection attempt against
    whatever else happens to be listening on the number we guessed."""

    path = _registry(tmp_path, "hub:\n  api:\n    port: 8082\n")

    with pytest.raises(LocalTtsEndpointError, match="reserves no 'tts_stream'"):
        resolve_port(registry_path=path)


def test_the_refusal_says_what_to_do_about_it(tmp_path) -> None:
    path = _registry(tmp_path, "hub:\n  api:\n    port: 8082\n")

    with pytest.raises(LocalTtsEndpointError) as error:
        resolve_port(registry_path=path)

    message = str(error.value)
    assert "local_tts" in message
    assert "capabilities" in message or "Declare the capability" in message


def test_an_unset_registry_variable_says_so_rather_than_defaulting(monkeypatch) -> None:
    monkeypatch.delenv("EIDOLON_PORTS_FILE", raising=False)

    with pytest.raises(LocalTtsEndpointError, match="EIDOLON_PORTS_FILE"):
        resolve_port()


def test_the_provider_name_is_the_capability_name() -> None:
    """So that "the config asks for local speech" and "this Host can do local
    speech" compare as two identical strings."""

    from eidolon.livekit.common.config.validators import TTS_PROVIDERS
    from eidolon.livekit.plugins.tts.local_tts import PROVIDER

    assert PROVIDER == contract.LOCAL_TTS_CAPABILITY == "local_tts"
    assert PROVIDER in TTS_PROVIDERS


def test_it_declares_the_same_streaming_contract_a_provider_does(tmp_path) -> None:
    """The stage above it does not learn that this one is local."""

    tts = LocalTTS(stream_url="ws://127.0.0.1:8770/v1/stream")

    assert tts.capabilities.streaming is True
    assert tts.sample_rate == contract.AUDIO_SAMPLE_RATE
    assert tts.num_channels == contract.AUDIO_CHANNELS


def test_it_carries_no_credential_and_no_address() -> None:
    """Two of the three things every cloud provider's config is mostly about.
    A field for either would be a place for a stale value to live."""

    fields = set(LocalTtsConfig.__dataclass_fields__)

    credentials = {"api_key", "app_key", "access_key", "secret", "token", "app_id"}
    assert not (fields & credentials)
    # `first_token_timeout_s` and `inter_token_timeout_s` are the LLM's tokens,
    # not a credential — which is why this compares whole field names rather
    # than looking for a substring.
    addresses = {name for name in fields if "url" in name or "endpoint" in name}
    assert not addresses
    assert not {name for name in fields if "region" in name or "host" in name}


def test_a_one_shot_synthesize_is_refused_rather_than_faked() -> None:
    """The engine streams, and a plugin that buffered a whole reply to look
    non-streaming would throw away the reason to run it locally."""

    tts = LocalTTS(stream_url="ws://127.0.0.1:8770/v1/stream")

    with pytest.raises(NotImplementedError):
        tts.synthesize("你好")


def test_a_batch_longer_than_stays_audible_is_refused_at_construction() -> None:
    """At construction rather than by ear.

    The guard used to compare against `MAX_TEXT_CHARACTERS` (400), which is the
    length the service *refuses*. But the length that stays *audible* is 60:
    past it the Host's buffer runs dry mid-sentence and the listener hears a
    gap — 126 characters measured 246-372 ms below empty on every turn. So
    anything up to 400 passed a guard that was watching the wrong limit.
    """

    with pytest.raises(ValueError, match="without the audio breaking up"):
        LocalTtsConfig(hard_max_chars=contract.SAFE_TEXT_CHARACTERS + 1)


def test_the_default_batch_sits_exactly_on_the_audible_limit() -> None:
    """60 is not a coincidence: it is the contract's safe bound, so the default
    is the largest batch that was measured to stay whole."""

    assert LocalTtsConfig().hard_max_chars == contract.SAFE_TEXT_CHARACTERS


def test_the_first_sentence_may_be_shorter_than_the_rest() -> None:
    """It is the one the listener waits through the whole engine start for."""

    config = LocalTtsConfig()

    assert config.first_sentence_soft_min_chars < config.soft_min_chars
    assert config.first_sentence_flush_any_punct is True


def test_an_unknown_provider_names_every_one_this_build_has() -> None:
    from eidolon.livekit.common.config.validators import TTS_PROVIDERS

    assert {"bailian", "sensetime", "local_tts"} <= TTS_PROVIDERS


# -- the finish report, which nothing checked until it broke -----------------
#
# The service renamed `underruns` to `late_chunks`, the branch that read it
# went permanently dead, and both ends kept passing their own tests, because
# neither end's spelling of the word was checked against the other's. These
# read the names out of the contract, so the next rename fails here instead of
# in a log nobody is watching.


def _finished(**fields):
    from eidolon.livekit.plugins.tts.local_tts.tts import LocalSynthesizeStream

    event = {"type": contract.SYNTHESIS_FINISHED, contract.REQUEST_ID_FIELD: "r1"}
    event.update(fields)
    return LocalSynthesizeStream._report, event


def test_a_negative_buffer_floor_is_reported_because_it_was_audible(caplog) -> None:
    """Below zero is the one number that means the listener heard a gap."""

    report, event = _finished(
        **{
            contract.MINIMUM_BUFFER_MS_FIELD: -372.0,
            contract.AUDIO_SECONDS_FIELD: 26.2,
            contract.LATE_CHUNKS_FIELD: 26,
        }
    )
    with caplog.at_level("WARNING"):
        report(None, event)

    assert "audio broke up" in caplog.text
    assert "-372" in caplog.text


def test_late_chunks_alone_are_not_reported_as_a_dropout(caplog) -> None:
    """The misreading this replaced, pinned so it cannot come back.

    16.88 s of audio with a healthy +209 ms floor reported 16 late chunks on
    the board. Nothing was audible; nothing should be said.
    """

    report, event = _finished(
        **{
            contract.MINIMUM_BUFFER_MS_FIELD: 209.0,
            contract.AUDIO_SECONDS_FIELD: 16.88,
            contract.LATE_CHUNKS_FIELD: 16,
        }
    )
    with caplog.at_level("WARNING"):
        report(None, event)

    assert caplog.text == ""


def test_a_missing_floor_is_not_read_as_no_gap(caplog) -> None:
    """Absent means the Host did not measure it, which is not a verdict.

    An older Host, or one whose engine reported no floor, omits the field.
    Warning on that would cry wolf; claiming silence proves quiet would be
    the same mistake in the other direction, so this only declines to speak.
    """

    report, event = _finished(**{contract.AUDIO_SECONDS_FIELD: 4.08})
    with caplog.at_level("WARNING"):
        report(None, event)

    assert caplog.text == ""


def test_the_field_names_are_the_contracts_own() -> None:
    """The point of the whole exercise: one spelling, in one place.

    If these move, the service's mirror test and this one both fail, which is
    what was missing when `underruns` became `late_chunks`.
    """

    assert contract.MINIMUM_BUFFER_MS_FIELD == "minimum_buffer_ms"
    assert contract.LATE_CHUNKS_FIELD == "late_chunks"
    assert not hasattr(contract, "UNDERRUNS_FIELD")
