"""One cancellable recognition lease per utterance; never gates conversation."""

import asyncio
import json
import time
from eidolon_sdk.biz.speaker_attestation import sign_observation
from eidolon_sdk.biz.speaker_identity import SpeakerIdentityObservation
from .speaker_recognition import SpeakerRecognizer

TOPIC = "eidolon.speaker_observation"


class ConversationSpeaker:
    def __init__(
        self,
        *,
        recognizer: SpeakerRecognizer,
        owner_id: str,
        session_id: str,
        secret: str,
        publish,
        wait_ms: int = 300,
        lifetime_ms: int = 2000,
    ):
        self.recognizer, self.owner_id, self.session_id = recognizer, owner_id, session_id
        self.secret, self.publish = secret, publish
        self.wait_ms, self.lifetime_ms = wait_ms, lifetime_ms
        self._capture = None
        self._task = None
        self._key = None
        self._closed = False
        self._publish_tasks = set()

    def bind_capture(self, capture):
        if capture is not self._capture:
            self.cancel()
        self._capture = capture

    def cancel(self):
        if self._task is not None:
            self._task.cancel()
        self._task = None
        self._key = None

    def _evidence(self, coordinator):
        capture = self._capture
        candidate = coordinator.active
        if capture is None or capture.active_source is None or candidate is None:
            return None
        reference = (
            coordinator.capture_committed_audio_reference(
                candidate_id=candidate.candidate_id, source=capture.active_source
            )
            if candidate.state == "committed"
            else coordinator.capture_audio_reference(source=capture.active_source)
        )
        if reference.span is None:
            return None
        return capture.snapshot(candidate=candidate.candidate_id, span=reference.span)

    def begin(self, coordinator, turn_id):
        self.begin_evidence(self._evidence(coordinator), turn_id)

    def begin_evidence(self, evidence, turn_id):
        if self._closed:
            return
        key = (
            (turn_id, evidence.source, evidence.generation, evidence.start, evidence.end)
            if evidence
            else (turn_id, None)
        )
        if key == self._key:
            return
        self.cancel()
        self._key = key

        async def run():
            try:
                async with asyncio.timeout(self.lifetime_ms / 1000):
                    return await self.recognizer.recognize(
                        evidence,
                        owner_id=self.owner_id,
                        session_id=self.session_id,
                        turn_id=turn_id,
                    )
            except asyncio.CancelledError:
                raise
            except TimeoutError:
                return self.fallback(turn_id, "recognition_timeout")
            except Exception:
                return self.fallback(turn_id, "dependency_unavailable")

        self._task = asyncio.create_task(run())

    def fallback(self, turn_id, reason):
        now = time.time()
        return SpeakerIdentityObservation(
            status="unavailable",
            confidence="none",
            reason=reason,
            owner_id=self.owner_id,
            session_id=self.session_id,
            turn_id=turn_id,
            issued_at=now,
            expires_at=now + 10,
        )

    async def finish(self, coordinator, *, turn_id, text, llm, timeline):
        self.begin(coordinator, turn_id)
        key = self._key
        candidate = coordinator.active
        capture, source = self._capture, self._capture.active_source if self._capture else None
        generation = candidate.latest_generation_id if candidate else None
        value = await self.result(turn_id)
        if (
            self._closed
            or self._key != key
            or self._capture is not capture
            or (capture is not None and capture.active_source != source)
            or coordinator.active is not candidate
            or candidate is None
            or candidate.state != "committed"
            or candidate.latest_generation_id != generation
        ):
            return
        self.deliver(value, text=text, llm=llm, timeline=timeline)

    async def result(self, turn_id):
        task = self._task
        if task is None:
            return self.fallback(turn_id, "audio_evidence_unavailable")
        done, _ = await asyncio.wait({task}, timeout=self.wait_ms / 1000)
        if not done:
            # Cancel instead of queuing or attaching a late result to a new turn.
            task.cancel()
            self._task = None
            return self.fallback(turn_id, "reply_deadline")
        if task.cancelled():
            return self.fallback(turn_id, "recognition_cancelled")
        return task.result()

    def deliver(self, value, *, text, llm, timeline):
        # A long ASR tail may outlive an otherwise completed recognition task.
        # Expiry clears the identity; it must never abort the spoken request.
        if value.expires_at <= time.time():
            value = self.fallback(value.turn_id, "observation_expired")
        setter = getattr(llm, "set_speaker_observation", None)
        if callable(setter):
            setter(sign_observation(value, secret=self.secret, text=text))
        if timeline is not None:
            # No names/profile text, PCM, embeddings or signed token in traces.
            timeline.set_attr(
                "speaker_observation",
                {
                    "status": value.status,
                    "reason": value.reason,
                    "person_id": value.person_id,
                    "confidence": value.confidence,
                    "decision_version": value.decision_version,
                    "delivered_to_agent": callable(setter),
                },
            )
        self.publish(value)

    def submit(self, *_):
        # The framework hook awaits finish; diagnostic-only submit is retired.
        pass

    async def aclose(self):
        self._closed = True
        task = self._task
        self.cancel()
        pending = list(self._publish_tasks)
        for publish_task in pending:
            publish_task.cancel()
        if task is not None:
            pending.append(task)
        await asyncio.gather(*pending, return_exceptions=True)


async def create_conversation_speaker(factory, cfg, room, *, publish_room=None):
    """Reuse authoritative runtime identity and the Provider's actual endpoint."""
    import os
    from eidolon_sdk.biz.runtime import resolve_shared_secret
    from eidolon_sdk.biz.speaker_identity import SpeakerRecognitionPolicy
    from eidolon_sdk.core.http import create_async_client
    from eidolon.channel_provider.config import load_provider_config
    from .speaker_recognition import SpeakerRecognitionClient
    from pathlib import Path

    # Missing or invalid trial policy degrades identification, never conversation.
    policy = None
    try:
        policy = SpeakerRecognitionPolicy.model_validate_json(
            (Path(cfg.voiceprint.model_dir) / "recognition-controlled-trial.json").read_text()
        )
    except (ValueError, OSError):
        pass
    context = await factory.runtime_services.resolve_room(room)
    provider = load_provider_config()
    secret = resolve_shared_secret(cfg.runtime_authority.jwt_secret)
    http = create_async_client(timeout=2)
    client = SpeakerRecognitionClient(
        http,
        data_url=cfg.runtime_authority.data_api_url,
        data_token=os.environ.get(cfg.runtime_authority.data_service_token_env, ""),
        provider_url=f"http://127.0.0.1:{provider.http.port}",
        provider_token=provider.bearer_token,
    )
    target = publish_room or room
    tasks = set()

    def publish(value):
        payload = {
            "schema_version": 1,
            "session_id": factory.runtime_session_id,
            "conversation_id": factory.runtime_session_id,
            "turn_id": value.turn_id,
            "status": value.status,
            "reason": value.reason,
            "person_id": value.person_id,
            "display_name": value.display_name,
            "confidence": value.confidence,
            "issued_at": value.issued_at,
            "expires_at": value.expires_at,
        }

        async def send():
            try:
                await asyncio.wait_for(
                    target.local_participant.publish_data(
                        json.dumps(payload, ensure_ascii=False).encode(), topic=TOPIC, reliable=True
                    ),
                    1,
                )
            except Exception:
                pass

        task = asyncio.create_task(send())
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    observer = ConversationSpeaker(
        recognizer=SpeakerRecognizer(client, policy),
        owner_id=context.owner_id,
        session_id=factory.runtime_session_id,
        secret=secret,
        publish=publish,
        wait_ms=cfg.voiceprint.recognition_wait_ms,
    )
    observer._publish_tasks = tasks
    return observer, http
