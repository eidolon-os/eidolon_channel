"""What this Host's synthesis provider can be asked to do differently.

Deliberately short. The cloud providers' configuration is mostly about the
cloud — credentials, regions, retry budgets, billing gates — and none of that
exists here. The endpoint is resolved rather than configured (see
`endpoint.py`), the audio format is the contract's, and the voice is the
service's own choice because the weights are what carry it.
"""

from __future__ import annotations

from dataclasses import dataclass

from eidolon_sdk.biz.contracts import local_tts as contract


@dataclass(frozen=True)
class LocalTtsConfig:
    #: Sentence batching. The engine says one utterance at a time and holds it
    #: in a fixed context, so tokens have to be gathered into sentences before
    #: they are sent — unlike the cloud providers, which take a token stream.
    #: The defaults are the aggregator's own; `hard_max_chars` is well under
    #: the contract's limit so a long sentence is split rather than refused.
    soft_min_chars: int = 12
    hard_max_chars: int = 60
    idle_ms: int = 300
    #: The first sentence of a reply is what the listener waits for, so it is
    #: allowed to be shorter than the rest.
    first_sentence_soft_min_chars: int = 8
    first_sentence_flush_any_punct: bool = True

    #: How long to wait for the first audio frame of a sentence. The engine
    #: needs about 2.9 s on RK3588 — the voice-profile precompute plus its own
    #: time to first PCM — so this is that with room, not a tight deadline.
    first_frame_timeout_s: float = 15.0
    #: How long a silence mid-sentence means the service has stopped. Audio
    #: arrives roughly every 200 ms while a sentence is being said.
    inter_frame_timeout_s: float = 20.0
    #: How long to wait for the first token from the LLM before giving up on
    #: the turn. Matches the cloud providers' own behaviour.
    first_token_timeout_s: float = 30.0
    inter_token_timeout_s: float = 10.0

    def __post_init__(self) -> None:
        if self.hard_max_chars > contract.MAX_TEXT_CHARACTERS:
            raise ValueError(
                f"hard_max_chars {self.hard_max_chars} exceeds what the service "
                f"will say in one request ({contract.MAX_TEXT_CHARACTERS}); the "
                "sentence would be refused after the turn had already started"
            )
