"""ModelScope CAMPPlus speaker verification provider."""

from __future__ import annotations

import asyncio
import tempfile
import time
import wave
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from eidolon.livekit.agent.speaker_verification.signal import SpeakerSignal
from eidolon.livekit.agent.speaker_verification.store import (
    VoiceprintProfile,
    default_profile_id,
    default_voiceprint_root,
)


class ModelScopeCampPlusSpeakerVerificationProvider:
    """Verify voiced turns against admin-generated voiceprint embeddings."""

    provider = "3d_speaker"
    model = "campplus_zh_16k_common"

    def __init__(
        self,
        *,
        model_dir: str | Path,
        voiceprint_root: str | Path | None = None,
        threshold: float = 0.31,
        min_audio_ms: int = 1500,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser()
        self.voiceprint_root = Path(voiceprint_root).expanduser() if voiceprint_root else default_voiceprint_root()
        self.threshold = threshold
        self.min_audio_ms = min_audio_ms
        self._pipeline: Any | None = None
        self._profile_cache: dict[tuple[str, str, str], tuple[str, int, Any]] = {}

    async def warm_up(self) -> None:
        """Load the pipeline before the first user turn."""
        await asyncio.to_thread(self._load_pipeline)

    async def enroll(
        self,
        *,
        tenant_id: str,
        user_id: str,
        audio_segments: Sequence[bytes],
        sample_rate: int,
    ) -> VoiceprintProfile:
        duration_ms = sum(_pcm_duration_ms(segment, sample_rate=sample_rate) for segment in audio_segments)
        return VoiceprintProfile(
            profile_id=default_profile_id(user_id),
            tenant_id=tenant_id,
            user_id=user_id,
            provider=self.provider,
            model=self.model,
            sample_rate=sample_rate,
            duration_ms=duration_ms,
            threshold=self.threshold,
            quality={
                "accepted_segments": len([segment for segment in audio_segments if segment]),
                "rejected_segments": len([segment for segment in audio_segments if not segment]),
            },
            metadata={"status": "requires_admin_enrollment"},
        )

    async def verify(
        self,
        *,
        audio: bytes,
        sample_rate: int,
        profiles: Sequence[VoiceprintProfile],
        audio_ms: int | None = None,
    ) -> SpeakerSignal:
        started = time.monotonic()
        if audio_ms is not None and audio_ms < self.min_audio_ms:
            return SpeakerSignal.error_signal(
                provider=self.provider,
                model=self.model,
                error="audio_too_short",
                latency_ms=_elapsed_ms(started),
                audio_ms=audio_ms,
                profile_id=profiles[0].profile_id if profiles else None,
            )
        candidates = [profile for profile in profiles if profile.provider == self.provider]
        if not candidates:
            return SpeakerSignal.error_signal(
                provider=self.provider,
                model=self.model,
                error="profile_provider_mismatch",
                latency_ms=_elapsed_ms(started),
                audio_ms=audio_ms,
                profile_id=profiles[0].profile_id if profiles else None,
            )

        try:
            test_embedding = await asyncio.to_thread(
                self._extract_embedding_from_audio,
                audio,
                sample_rate,
            )
            scored = []
            for profile in candidates:
                profile_embedding = self._load_profile_embedding(profile)
                if profile_embedding is None:
                    continue
                scored.append((profile, _cosine(profile_embedding, test_embedding)))
        except Exception as exc:  # noqa: BLE001 - service will expose provider_error
            raise RuntimeError(f"campplus verify failed: {exc}") from exc

        if not scored:
            return SpeakerSignal.error_signal(
                provider=self.provider,
                model=self.model,
                error="embedding_not_found",
                latency_ms=_elapsed_ms(started),
                audio_ms=audio_ms,
                profile_id=candidates[0].profile_id,
            )
        best_profile, best_score = max(scored, key=lambda item: item[1])
        threshold = best_profile.threshold or self.threshold
        known = best_score >= threshold
        return SpeakerSignal(
            provider=self.provider,
            model=self.model,
            speaker_user_id=best_profile.user_id if known else None,
            known=known,
            owner_confidence=max(0.0, min(1.0, best_score)),
            score=best_score,
            latency_ms=_elapsed_ms(started),
            audio_ms=audio_ms,
            profile_id=best_profile.profile_id,
        )

    def _load_pipeline(self) -> Any:
        if self._pipeline is None:
            if not self.model_dir.is_dir():
                raise RuntimeError(f"3D-Speaker model dir not found: {self.model_dir}")
            from modelscope.pipelines import pipeline

            self._pipeline = pipeline(
                task="speaker-verification",
                model=str(self.model_dir),
            )
        return self._pipeline

    def _extract_embedding_from_audio(self, audio: bytes, sample_rate: int):
        with tempfile.TemporaryDirectory(prefix="eidolon-channel-voiceprint-") as tmp:
            wav_path = Path(tmp) / "turn.wav"
            if audio.startswith(b"RIFF"):
                wav_path.write_bytes(audio)
            else:
                _write_pcm_wav(wav_path, audio=audio, sample_rate=sample_rate)
            result = self._load_pipeline()([str(wav_path)], output_emb=True)
        embedding = result["embs"][0]
        return _normalize(embedding)

    def _load_profile_embedding(self, profile: VoiceprintProfile):
        if not profile.embedding_ref:
            return None
        path = (self.voiceprint_root / profile.embedding_ref).expanduser().resolve()
        root = self.voiceprint_root.expanduser().resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"embedding_ref escapes voiceprint root: {profile.embedding_ref}") from exc
        if not path.exists():
            return None
        stat = path.stat()
        cache_key = (profile.tenant_id, profile.user_id, profile.profile_id)
        cached = self._profile_cache.get(cache_key)
        if cached is not None:
            cached_ref, cached_mtime_ns, cached_vector = cached
            if cached_ref == profile.embedding_ref and cached_mtime_ns == stat.st_mtime_ns:
                return cached_vector
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        vector = _normalize(data.get("vector") or ())
        self._profile_cache[cache_key] = (profile.embedding_ref, stat.st_mtime_ns, vector)
        return vector


def _normalize(vector):
    import numpy as np

    array = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(array)
    if norm <= 0:
        raise RuntimeError("voiceprint embedding has zero norm")
    return array / norm


def _cosine(left, right) -> float:
    import numpy as np

    return float(np.dot(left, right))


def _write_pcm_wav(path: Path, *, audio: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio)


def _pcm_duration_ms(audio: bytes, *, sample_rate: int) -> int:
    if sample_rate <= 0:
        return 0
    return int((len(audio) // 2) * 1000 / sample_rate)


def _elapsed_ms(started: float) -> float:
    return (time.monotonic() - started) * 1000.0
