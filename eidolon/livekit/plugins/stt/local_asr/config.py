"""Configuration for recognition that runs on this Host.

Notice what is not here: a URL, a host, a port, an API key, a model name, a
region, a retry budget for a remote endpoint. None of them are this
provider's to hold.

* the endpoint is resolved from the Host's own port registry (`endpoint.py`)
* the model is whatever the Host's service has open, and it reports it
* there is no credential, because there is no third party
* a loopback service does not need a reconnect budget sized for the internet

What is left is the audio contract and two timeouts, and even the audio
contract is fixed by the protocol rather than chosen here.
"""

from __future__ import annotations

from dataclasses import dataclass

from eidolon_sdk.biz.contracts import local_asr as contract


@dataclass
class LocalAsrSTTConfig:
    #: Fixed by the protocol. Present so the pipeline can be told what it is
    #: rather than assuming, and validated against the contract on use.
    sample_rate: int = contract.AUDIO_SAMPLE_RATE
    channels: int = contract.AUDIO_CHANNELS

    #: Recognition is Chinese on this Host; the service reports the model.
    language: str = "zh"

    #: How long to wait for the service to accept a stream. Short: it is on
    #: this machine, and a Host whose own service is not answering has a
    #: problem that waiting will not fix.
    connect_timeout_s: float = 5.0

    #: How long to wait for the final answer after the utterance is closed.
    #: The two-pass shape re-decodes with an offline model and punctuates, so
    #: this is longer than the interim cadence, and still local.
    final_timeout_s: float = 15.0

    #: Milliseconds of audio per frame sent to the service. The service paces
    #: recognition on what it receives, so this is the interim cadence.
    frame_ms: int = 100

    #: An explicit port, for a source run with no Host registry. Left unset on
    #: a Host: the registry is the answer there, and a number written here
    #: would be a second statement of it.
    port: int | None = None

    def __post_init__(self) -> None:
        if self.sample_rate != contract.AUDIO_SAMPLE_RATE:
            raise ValueError(
                f"local_asr accepts {contract.AUDIO_SAMPLE_RATE} Hz audio only"
            )
        if self.channels != contract.AUDIO_CHANNELS:
            raise ValueError("local_asr accepts mono audio only")
        if not 10 <= self.frame_ms <= 1000:
            raise ValueError("local_asr frame_ms must be in 10..1000")
        if self.connect_timeout_s <= 0 or self.final_timeout_s <= 0:
            raise ValueError("local_asr timeouts must be positive")
        if self.port is not None and not 1 <= self.port <= 65535:
            raise ValueError("local_asr port is out of range")
