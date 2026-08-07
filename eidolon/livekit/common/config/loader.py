"""Load the effective channel configuration from settings.yaml and .env."""

from __future__ import annotations

import logging
import os
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any, TypeVar

import yaml

from .profiles import profile_defaults
from .schema import (
    AgentBehaviorConfig,
    AvatarConfig,
    BailianSTTConfig,
    BailianTTSConfig,
    CoreConfig,
    EffectiveAgentConfig,
    LLMConfig,
    ObservabilityConfig,
    ProvidersConfig,
    RemoteAgentRpcConfig,
    RuntimeAuthorityConfig,
    SenseTimeSTTConfig,
    SenseTimeTTSConfig,
    TurnPolicyConfig,
    VoiceprintConfig,
    WorkerConfig,
)
from .validators import validate_effective_config

logger = logging.getLogger("agent.config")

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_YAML = _REPO_ROOT / "config" / "settings.yaml"
_DEFAULT_EXAMPLE_YAML = _REPO_ROOT / "config" / "settings.example.yaml"
_DEFAULT_ENV = _REPO_ROOT / "config" / ".env"

T = TypeVar("T")


def _resolve_settings_yaml() -> Path:
    explicit = os.environ.get("EIDOLON_CHANNEL_SETTINGS_YAML", "").strip()
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"EIDOLON_CHANNEL_SETTINGS_YAML missing: {p}")
        return p.resolve()
    if _DEFAULT_YAML.is_file():
        return _DEFAULT_YAML.resolve()
    if _DEFAULT_EXAMPLE_YAML.is_file():
        return _DEFAULT_EXAMPLE_YAML.resolve()
    raise FileNotFoundError(
        f"channel settings not found: {_DEFAULT_YAML}. Run ./deploy/dev/init.sh"
    )


def _resolve_settings_overlay_yaml() -> Path | None:
    explicit = os.environ.get("EIDOLON_CHANNEL_SETTINGS_OVERLAY_YAML", "").strip()
    if not explicit:
        return None
    p = Path(explicit).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"EIDOLON_CHANNEL_SETTINGS_OVERLAY_YAML missing: {p}")
    return p.resolve()


def _resolve_env_file() -> Path | None:
    raw = (
        os.environ.get("EIDOLON_CHANNEL_ENV_FILE", "").strip()
        or os.environ.get("EIDOLON_CHANNEL_LIVEKIT_ENV", "").strip()
    )
    if raw:
        p = Path(raw).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"channel env file missing: {p}")
        return p.resolve()
    if _DEFAULT_ENV.is_file():
        return _DEFAULT_ENV.resolve()
    return None


def _bootstrap_dotenv() -> None:
    from dotenv import load_dotenv

    env_path = _resolve_env_file()
    if env_path is None:
        return
    os.environ.setdefault("EIDOLON_CHANNEL_LIVEKIT_ENV", str(env_path))
    os.environ.setdefault("EIDOLON_CHANNEL_ENV_FILE", str(env_path))
    load_dotenv(env_path, override=False)


def _load_yaml() -> dict[str, Any]:
    data = yaml.safe_load(_resolve_settings_yaml().read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("channel settings.yaml must be a mapping")
    overlay_path = _resolve_settings_overlay_yaml()
    if overlay_path is None:
        return data
    overlay = yaml.safe_load(overlay_path.read_text(encoding="utf-8")) or {}
    if not isinstance(overlay, dict):
        raise ValueError("channel settings overlay YAML must be a mapping")
    return _deep_merge_dicts(data, overlay)


def _deep_merge_dicts(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(current, value)
        else:
            merged[key] = value
    return merged


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    sec = data.get(key) or {}
    return sec if isinstance(sec, dict) else {}


def _reject_unknown_fields(
    section_name: str,
    raw: dict[str, Any],
    *,
    allowed: set[str],
) -> None:
    for key in raw:
        if key in allowed:
            continue
        raise ValueError(f"unknown config field {section_name}.{key}")


def _secret(section: dict[str, Any], field: str, env_var: str) -> str:
    val = str(section.get(field) or "").strip()
    if val and val != env_var:
        raise ValueError(
            f"{field} must be empty or the placeholder {env_var}; set {env_var} in .env"
        )
    return os.environ.get(env_var, "").strip()


def _optional_float(raw: Any) -> float | None:
    if raw is None:
        return None
    s = str(raw).strip()
    return float(s) if s else None


def _optional_int(raw: Any) -> int | None:
    if raw is None:
        return None
    s = str(raw).strip()
    return int(s) if s else None


def _coerce_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    raise ValueError(f"expected string/list for lexicon, got {type(value).__name__}")


def _merge_dataclass(base: T, raw: dict[str, Any] | None) -> T:
    if not raw:
        return base
    if not is_dataclass(base):
        raise TypeError(f"{type(base).__name__} is not a dataclass")

    values: dict[str, Any] = {}
    field_names = {f.name for f in fields(base)}
    for key, value in raw.items():
        if key not in field_names:
            raise ValueError(f"unknown config field {type(base).__name__}.{key}")
        current = getattr(base, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be a mapping")
            values[key] = _merge_dataclass(current, value)
        elif isinstance(current, tuple):
            values[key] = _coerce_tuple(value)
        else:
            values[key] = value
    return replace(base, **values)


def _load_turn_policy(raw: dict[str, Any]) -> TurnPolicyConfig:
    profile = str(raw.get("profile") or "balanced_semantic").strip() or "balanced_semantic"
    base = profile_defaults(profile)
    return _merge_dataclass(base, raw)


def load_effective_config() -> EffectiveAgentConfig:
    """Load, validate, and return the only runtime config object."""
    _bootstrap_dotenv()
    y = _load_yaml()

    core_y = _section(y, "core")
    behavior_y = _section(y, "behavior") or _section(y, "agent_behavior")
    providers_y = _section(y, "providers")
    llm_y = _section(y, "llm")
    rpc_y = _section(y, "remote_agent_rpc")
    runtime_authority_y = _section(y, "runtime_authority")
    turn_y = _section(y, "turn_policy")
    obs_y = _section(y, "observability")
    voiceprint_y = _section(y, "voiceprint")
    worker_y = _section(y, "worker")
    avatar_y = _section(y, "avatar")
    bailian_stt_y = _section(y, "bailian_stt")
    bailian_tts_y = _section(y, "bailian_tts")
    sensetime_stt_y = _section(y, "sensetime_stt")
    sensetime_tts_y = _section(y, "sensetime_tts")

    _reject_unknown_fields(
        "core",
        core_y,
        allowed={"livekit_url", "api_key", "api_secret", "host", "port"},
    )
    _reject_unknown_fields(
        "behavior",
        behavior_y,
        allowed={
            "instructions",
            "welcome_message",
            "audio_sample_rate",
        },
    )
    _reject_unknown_fields(
        "llm",
        llm_y,
        allowed={
            "base_url",
            "model",
            "api_key",
            "temperature",
            "timeout",
            "max_completion_tokens",
        },
    )
    _reject_unknown_fields(
        "remote_agent_rpc",
        rpc_y,
        allowed={
            "target",
            "locale",
            "conversation_id_prefix",
            "tls_mode",
            "tls_ca_path",
            "tls_client_cert_path",
            "tls_client_key_path",
        },
    )
    _reject_unknown_fields(
        "runtime_authority",
        runtime_authority_y,
        allowed={
            "enabled",
            "kernel_api_url",
            "data_api_url",
            "data_service_token_env",
            "jwt_secret",
            "jwt_algorithm",
            "device_token_ttl_seconds",
            "http_timeout_sec",
            "http_connect_timeout_sec",
        },
    )

    cfg = EffectiveAgentConfig(
        core=CoreConfig(
            livekit_url=str(core_y.get("livekit_url") or "ws://localhost:7880"),
            api_key=_secret(core_y, "api_key", "LIVEKIT_API_KEY") or "devkey",
            api_secret=_secret(core_y, "api_secret", "LIVEKIT_API_SECRET") or "devkey_secret",
            host=str(core_y.get("host") or "0.0.0.0"),
            port=int(core_y.get("port") or 8766),
        ),
        behavior=AgentBehaviorConfig(
            instructions=str(behavior_y.get("instructions") or AgentBehaviorConfig().instructions),
            welcome_message=str(
                behavior_y.get("welcome_message")
                if behavior_y.get("welcome_message") is not None
                else AgentBehaviorConfig().welcome_message
            ),
            audio_sample_rate=int(behavior_y.get("audio_sample_rate") or 16000),
        ),
        providers=_merge_dataclass(ProvidersConfig(), providers_y),
        llm=LLMConfig(
            base_url=str(llm_y.get("base_url") or ""),
            model=str(llm_y.get("model") or "gpt-4o-mini"),
            api_key=_secret(llm_y, "api_key", "OPENAI_LLM_API_KEY"),
            temperature=_optional_float(llm_y.get("temperature")),
            timeout=_optional_float(llm_y.get("timeout")),
            max_completion_tokens=_optional_int(llm_y.get("max_completion_tokens")),
        ),
        remote_agent_rpc=RemoteAgentRpcConfig(
            target=str(rpc_y.get("target") or "").strip(),
            locale=str(rpc_y.get("locale") or "zh").strip() or "zh",
            conversation_id_prefix=str(rpc_y.get("conversation_id_prefix") or "livekit").strip()
            or "livekit",
            tls_mode=str(rpc_y.get("tls_mode") or "off").strip().lower() or "off",
            tls_ca_path=str(rpc_y.get("tls_ca_path") or "").strip(),
            tls_client_cert_path=str(rpc_y.get("tls_client_cert_path") or "").strip(),
            tls_client_key_path=str(rpc_y.get("tls_client_key_path") or "").strip(),
        ),
        runtime_authority=RuntimeAuthorityConfig(
            enabled=bool(runtime_authority_y.get("enabled", True)),
            kernel_api_url=str(
                runtime_authority_y.get("kernel_api_url") or "http://127.0.0.1:8083/api/kernel/v1"
            ).strip(),
            data_api_url=str(
                runtime_authority_y.get("data_api_url") or "http://127.0.0.1:8084"
            ).strip(),
            data_service_token_env=str(
                runtime_authority_y.get("data_service_token_env")
                or "EIDOLON_DATA_COMPANION_AUTHORITY_TOKEN"
            ).strip(),
            jwt_secret=_secret(
                runtime_authority_y,
                "jwt_secret",
                "PAIRING_JWT_SECRET",
            ),
            jwt_algorithm=str(runtime_authority_y.get("jwt_algorithm") or "HS256").strip(),
            device_token_ttl_seconds=int(
                runtime_authority_y.get("device_token_ttl_seconds") or 24 * 3600
            ),
            http_timeout_sec=float(runtime_authority_y.get("http_timeout_sec") or 10.0),
            http_connect_timeout_sec=float(
                runtime_authority_y.get("http_connect_timeout_sec") or 3.0
            ),
        ),
        turn_policy=_load_turn_policy(turn_y),
        observability=_merge_dataclass(ObservabilityConfig(), obs_y),
        voiceprint=_merge_dataclass(VoiceprintConfig(), voiceprint_y),
        worker=_merge_dataclass(WorkerConfig(), worker_y),
        avatar=_merge_dataclass(AvatarConfig(), avatar_y),
        # Non-secret Bailian STT/TTS config lives in settings.yaml; the API key
        # stays in .env (the BailianSTTConfig()/BailianTTSConfig() base reads it
        # from env via default_factory, and the YAML section — which must omit
        # api_key — overrides only the non-secret fields).
        bailian_stt=_merge_dataclass(BailianSTTConfig(), bailian_stt_y),
        bailian_tts=_merge_dataclass(BailianTTSConfig(), bailian_tts_y),
        sensetime_stt=_merge_dataclass(SenseTimeSTTConfig(), sensetime_stt_y),
        sensetime_tts=_merge_dataclass(SenseTimeTTSConfig(), sensetime_tts_y),
    )
    validate_effective_config(cfg)
    logger.info("[AgentConfig] effective_config=%s", cfg.sanitized_dict())
    return cfg


def load_agent_config() -> EffectiveAgentConfig:
    return load_effective_config()
