from __future__ import annotations

from eidolon.livekit.common.config.schema import (
    CoreConfig,
    EffectiveAgentConfig,
    RuntimeAuthorityConfig,
)


def test_effective_config_redacts_all_livekit_and_runtime_authority_secrets() -> None:
    cfg = EffectiveAgentConfig(
        core=CoreConfig(api_key="livekit-key", api_secret="livekit-secret"),
        runtime_authority=RuntimeAuthorityConfig(jwt_secret="runtime-jwt-secret"),
    )

    sanitized = cfg.sanitized_dict()

    assert sanitized["core"]["api_key"] == "***"
    assert sanitized["core"]["api_secret"] == "***"
    assert sanitized["runtime_authority"]["jwt_secret"] == "***"
    assert "livekit-key" not in repr(sanitized)
    assert "livekit-secret" not in repr(sanitized)
    assert "runtime-jwt-secret" not in repr(sanitized)
