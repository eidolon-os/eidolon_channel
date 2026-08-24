"""Strict, process-owned configuration for the Channel Provider service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml
from dotenv import load_dotenv

from .adapters.livekit import LiveKitConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SETTINGS = _REPO_ROOT / "config" / "channel-provider.yaml"
_DEFAULT_ENV = _REPO_ROOT / "config" / ".env"
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


@dataclass(frozen=True, slots=True)
class HttpConfig:
    host: str = "127.0.0.1"
    port: int = 8767


@dataclass(frozen=True, slots=True)
class StorageConfig:
    path: Path


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    http: HttpConfig
    storage: StorageConfig
    livekit: LiveKitConfig
    bearer_token: str
    # Adapter names in the order this deployment prefers them. Selection walks
    # this list and takes the first one that can carry the device's spec.
    adapter_preference: tuple[str, ...] = ("livekit",)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    return value


def _exact(value: dict[str, Any], *, name: str, allowed: set[str]) -> None:
    extra = value.keys() - allowed
    if extra:
        raise ValueError(f"unknown config field {name}.{sorted(extra)[0]}")


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, str) and (value.startswith("~") or "$" in value):
        expanded = os.path.expanduser(os.path.expandvars(value))
        if "$" in expanded:
            raise ValueError("config contains an unexpanded environment variable")
        return expanded
    return value


def _settings_path() -> Path:
    raw = os.environ.get("EIDOLON_CHANNEL_PROVIDER_SETTINGS_YAML", "").strip()
    path = Path(raw).expanduser() if raw else _DEFAULT_SETTINGS
    if not path.is_file():
        raise FileNotFoundError(f"Channel Provider settings are missing: {path}")
    return path.resolve()


def _load_environment() -> None:
    raw = (
        os.environ.get("EIDOLON_CHANNEL_PROVIDER_ENV_FILE", "").strip()
        or os.environ.get("EIDOLON_CHANNEL_ENV_FILE", "").strip()
    )
    path = Path(raw).expanduser() if raw else _DEFAULT_ENV
    if raw and not path.is_file():
        raise FileNotFoundError(f"Channel Provider environment file is missing: {path}")
    if path.is_file():
        load_dotenv(path, override=False)


def _required_secret(name: str, *, minimum: int = 1) -> str:
    value = os.environ.get(name, "")
    if len(value.encode()) < minimum:
        raise RuntimeError(f"{name} must contain at least {minimum} bytes")
    return value


def _url(
    value: Any,
    *,
    name: str,
    client: bool,
    allow_insecure_lan: bool = False,
) -> str:
    text = str(value or "").strip()
    parsed = urlparse(text)
    allowed = {"ws", "wss"} if client else {"http", "https"}
    if (
        parsed.scheme not in allowed
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or "$" in text
    ):
        raise ValueError(f"{name} must be a plain {sorted(allowed)} URL")
    if (
        parsed.scheme in {"http", "ws"}
        and parsed.hostname not in _LOOPBACK_HOSTS
        and not (client and allow_insecure_lan)
    ):
        raise ValueError(f"insecure {name} is allowed only on loopback")
    return text.rstrip("/")


def _allow_insecure_lan_client() -> bool:
    raw = os.environ.get("EIDOLON_CHANNEL_PROVIDER_ALLOW_INSECURE_LAN_CLIENT_URL", "0").strip()
    if raw not in {"0", "1"}:
        raise ValueError("EIDOLON_CHANNEL_PROVIDER_ALLOW_INSECURE_LAN_CLIENT_URL must be 0 or 1")
    return raw == "1"


def load_provider_config() -> ProviderConfig:
    _load_environment()
    source = yaml.safe_load(_settings_path().read_text(encoding="utf-8")) or {}
    root = _mapping(_expand(source), "root")
    _exact(root, name="root", allowed={"http", "storage", "livekit", "adapters"})
    adapters = _mapping(root.get("adapters"), "adapters")
    _exact(adapters, name="adapters", allowed={"preference"})

    http = _mapping(root.get("http"), "http")
    storage = _mapping(root.get("storage"), "storage")
    livekit = _mapping(root.get("livekit"), "livekit")
    _exact(http, name="http", allowed={"host", "port"})
    _exact(storage, name="storage", allowed={"path"})
    _exact(
        livekit,
        name="livekit",
        allowed={
            "api_url",
            "client_url",
            "room_prefix",
            "agent_name",
            "grant_ttl_seconds",
            "sample_rate",
            "channels",
        },
    )

    host = str(http.get("host", "127.0.0.1")).strip()
    port = int(http.get("port", 8767))
    if not host or not 1 <= port <= 65535:
        raise ValueError("http.host and http.port must define a valid listener")

    raw_state_path = storage.get("path")
    if not isinstance(raw_state_path, str) or not raw_state_path.strip():
        raise ValueError("storage.path is required")
    state_path = Path(raw_state_path).expanduser()
    if not state_path.is_absolute():
        state_path = (_REPO_ROOT / state_path).resolve()

    ttl = int(livekit.get("grant_ttl_seconds", 1800))
    if not 300 <= ttl <= 86400:
        raise ValueError("livekit.grant_ttl_seconds must be in 300..86400")
    sample_rate = int(livekit.get("sample_rate", 16000))
    channels = int(livekit.get("channels", 1))
    if not 8000 <= sample_rate <= 48000 or channels not in {1, 2}:
        raise ValueError("livekit audio must be 8..48kHz and mono or stereo")

    room_prefix = str(livekit.get("room_prefix", "eidolon-device")).strip()
    agent_name = str(livekit.get("agent_name", "eidolon")).strip()
    if not room_prefix or len(room_prefix) > 48 or not agent_name or len(agent_name) > 64:
        raise ValueError("LiveKit room or agent policy is invalid")

    preference = adapters.get("preference", ["livekit"])
    if (
        not isinstance(preference, list)
        or not preference
        or not all(isinstance(name, str) and name.strip() for name in preference)
        or len(set(preference)) != len(preference)
    ):
        raise ValueError("adapters.preference must be a non-empty list of distinct adapter names")

    return ProviderConfig(
        http=HttpConfig(host=host, port=port),
        storage=StorageConfig(path=state_path),
        livekit=LiveKitConfig(
            api_url=_url(livekit.get("api_url"), name="livekit.api_url", client=False),
            client_url=_url(
                livekit.get("client_url"),
                name="livekit.client_url",
                client=True,
                allow_insecure_lan=_allow_insecure_lan_client(),
            ),
            api_key=_required_secret("LIVEKIT_API_KEY"),
            api_secret=_required_secret("LIVEKIT_API_SECRET"),
            room_prefix=room_prefix,
            agent_name=agent_name,
            grant_ttl_seconds=ttl,
            sample_rate=sample_rate,
            channels=channels,
        ),
        bearer_token=_required_secret("EIDOLON_CHANNEL_PROVIDER_TOKEN", minimum=32),
        adapter_preference=tuple(name.strip() for name in preference),
    )
