"""Filesystem-backed voiceprint profile metadata store."""

from __future__ import annotations

import json
from pathlib import Path

from eidolon.livekit.common.speaker_verification import (
    VoiceprintEmbedding,
    VoiceprintProfile,
    default_profile_id,
    default_voiceprint_root,
    validate_voiceprint_id,
)

__all__ = [
    "VoiceprintEmbedding",
    "VoiceprintProfile",
    "VoiceprintStore",
    "default_profile_id",
    "default_voiceprint_root",
]


class VoiceprintStore:
    """Small JSON metadata store under ``root/tenant_id/user_id``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()

    def profile_path(
        self,
        *,
        tenant_id: str,
        user_id: str,
        profile_id: str | None = None,
    ) -> Path:
        tenant_id = validate_voiceprint_id(tenant_id, label="tenant_id")
        user_id = validate_voiceprint_id(user_id, label="user_id")
        if profile_id is None or profile_id == default_profile_id(user_id):
            return self.root / tenant_id / user_id / "profile.json"
        profile_id = validate_voiceprint_id(profile_id, label="profile_id", max_len=128)
        return self.root / tenant_id / user_id / f"{profile_id}.json"

    def save_profile(self, profile: VoiceprintProfile) -> Path:
        path = self.profile_path(
            tenant_id=profile.tenant_id,
            user_id=profile.user_id,
            profile_id=profile.profile_id,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(profile.to_json(), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        return path

    def load_profile(
        self,
        *,
        tenant_id: str,
        user_id: str,
        profile_id: str | None = None,
    ) -> VoiceprintProfile | None:
        path = self.profile_path(
            tenant_id=tenant_id,
            user_id=user_id,
            profile_id=profile_id,
        )
        if not path.exists() and profile_id is None:
            legacy = (
                self.root
                / validate_voiceprint_id(tenant_id, label="tenant_id")
                / validate_voiceprint_id(user_id, label="user_id")
                / f"{default_profile_id(user_id)}.json"
            )
            path = legacy
        if not path.exists():
            return None
        return VoiceprintProfile.from_json(json.loads(path.read_text(encoding="utf-8")))

    def list_user_profiles(
        self,
        *,
        tenant_id: str,
        user_id: str,
    ) -> list[VoiceprintProfile]:
        base = self.profile_path(tenant_id=tenant_id, user_id=user_id).parent
        if not base.exists():
            return []
        profiles: list[VoiceprintProfile] = []
        for path in sorted(base.glob("*.json")):
            profiles.append(
                VoiceprintProfile.from_json(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            )
        return profiles

    def embedding_path(self, profile: VoiceprintProfile) -> Path | None:
        if not profile.embedding_ref:
            return None
        path = (self.root / profile.embedding_ref).expanduser().resolve()
        root = self.root.expanduser().resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"embedding_ref escapes voiceprint root: {profile.embedding_ref}") from exc
        return path

    def load_embedding(self, profile: VoiceprintProfile) -> VoiceprintEmbedding | None:
        path = self.embedding_path(profile)
        if path is None or not path.exists():
            return None
        return VoiceprintEmbedding.from_json(json.loads(path.read_text(encoding="utf-8")))

    def delete_profile(
        self,
        *,
        tenant_id: str,
        user_id: str,
        profile_id: str | None = None,
    ) -> bool:
        path = self.profile_path(
            tenant_id=tenant_id,
            user_id=user_id,
            profile_id=profile_id,
        )
        if not path.exists():
            return False
        path.unlink()
        return True
