"""Speech recognition served by this Host, not by a provider."""

from __future__ import annotations

import logging

import aiohttp
from eidolon_sdk.biz.contracts import local_asr as contract
from livekit.agents import stt as lk_stt
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

from .config import LocalAsrSTTConfig
from .endpoint import resolve_ready_url, resolve_stream_url
from .speech_stream import LocalAsrSpeechStream

logger = logging.getLogger("eidolon.livekit.plugins.stt.local_asr")

PROVIDER = contract.LOCAL_ASR_CAPABILITY


class LocalAsrSTT(lk_stt.STT):
    """The Host's own streaming recognition, as a LiveKit STT plugin.

    Holds no credential and no address. The endpoint is resolved from this
    Host's port registry, which is written from the component contract that
    reserves the port — so this class has nothing to be configured with that
    could disagree with the Host it runs on.

    The provider name is the capability name: a Host that does not declare
    `local_asr` has no such service, and Ops refuses that configuration by
    comparing the two strings rather than consulting a table.
    """

    def __init__(
        self,
        *,
        config: LocalAsrSTTConfig | None = None,
        conn_options: APIConnectOptions | None = None,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=lk_stt.STTCapabilities(
                streaming=True,
                # The point of the two-pass shape: words appear while they are
                # still being spoken, then are replaced by a punctuated final.
                interim_results=True,
            ),
        )
        self._config = config or LocalAsrSTTConfig()
        self._conn_options = conn_options or DEFAULT_API_CONNECT_OPTIONS
        self._session = session
        self._owns_session = session is None

    @property
    def provider(self) -> str:
        return PROVIDER

    @property
    def label(self) -> str:
        return f"eidolon.{PROVIDER}"

    @property
    def language(self) -> str:
        return self._config.language

    @property
    def sample_rate(self) -> int:
        return self._config.sample_rate

    def stream_url(self) -> str:
        return resolve_stream_url(port=self._config.port)

    def ready_url(self) -> str:
        return resolve_ready_url(port=self._config.port)

    async def warmup(self) -> None:
        """Ask the Host whether its models are open, and say what answered.

        Not a health gate: a session that starts a moment before the service is
        ready is a slow first utterance, not a broken one. What this buys is
        that the log says which models this Host is listening with, instead of
        that being invisible until someone reads a transcript and wonders.
        """

        session = await self._ensure_session()
        try:
            async with session.get(
                self.ready_url(),
                timeout=aiohttp.ClientTimeout(total=self._config.connect_timeout_s),
            ) as answer:
                document = await answer.json()
        except Exception as exc:  # noqa: BLE001 - warmup never fails a session
            logger.warning(
                "local_asr warmup could not reach this Host's recognition: %s", exc
            )
            return
        served = document.get("protocol_version")
        if served != contract.LOCAL_ASR_PROTOCOL_VERSION:
            # Loud, because the alternative is mis-parsing. The service states
            # the version it serves precisely so a client need not guess.
            logger.error(
                "local_asr speaks protocol %s and this Host serves %s; "
                "recognition will not be attempted with a version this build "
                "cannot read",
                contract.LOCAL_ASR_PROTOCOL_VERSION,
                served,
            )
            return
        logger.info(
            "local_asr ready: backend=%s streaming=%s offline=%s punctuation=%s",
            document.get("backend"),
            document.get("model_id"),
            document.get("offline_model_id"),
            document.get("punctuation_model_id"),
        )

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        return self._session

    def stream(
        self,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions | None = None,
    ) -> LocalAsrSpeechStream:
        if language and language != self._config.language:
            logger.info(
                "local_asr recognizes %s; ignoring requested %s",
                self._config.language,
                language,
            )
        session = self._session
        if session is None or session.closed:
            session = aiohttp.ClientSession()
            self._session = session
            self._owns_session = True
        return LocalAsrSpeechStream(
            stt=self,
            config=self._config,
            stream_url=self.stream_url(),
            session=session,
        )

    async def _recognize_impl(self, *args, **kwargs):
        """Not offered. This provider declares `streaming` and nothing else.

        A one-shot recognize would have to buffer a whole utterance and then
        ask for it, which is the shape this service is built to avoid: it
        answers while the words arrive.
        """

        raise NotImplementedError(
            "local_asr is a streaming recognizer; use stream()"
        )

    async def aclose(self) -> None:
        if self._session is not None and self._owns_session and not self._session.closed:
            await self._session.close()
        self._session = None
