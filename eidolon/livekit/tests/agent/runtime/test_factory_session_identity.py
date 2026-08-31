from __future__ import annotations

from types import SimpleNamespace

import pytest

from eidolon.livekit.agent.factory import _build_device_token_source


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        runtime_authority=SimpleNamespace(
            enabled=True,
            jwt_secret="configured-secret",
            jwt_algorithm="HS256",
            device_token_ttl_seconds=600,
        )
    )


def _services() -> SimpleNamespace:
    return SimpleNamespace(
        runtime=object(),
        mounts=object(),
        resolve_room=object(),
    )


def test_runtime_token_uses_unique_dispatch_session_not_stable_room(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    monkeypatch.setattr(
        "eidolon_sdk.biz.runtime.resolve_shared_secret",
        lambda _value: "resolved-secret",
    )

    def _resolver(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        "eidolon.livekit.agent.runtime.make_device_token_resolver",
        _resolver,
    )

    result = _build_device_token_source(
        cfg=_config(),
        livekit_room=SimpleNamespace(name="eidolon-device-stable"),
        runtime_services=_services(),
        runtime_session_id="esp32-dispatch-00000003",
    )

    assert result is sentinel
    assert captured["session_id"] == "esp32-dispatch-00000003"


def test_same_device_room_gets_a_new_runtime_session_on_each_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions: list[str] = []

    monkeypatch.setattr(
        "eidolon_sdk.biz.runtime.resolve_shared_secret",
        lambda _value: "resolved-secret",
    )

    def _resolver(**kwargs):
        sessions.append(str(kwargs["session_id"]))
        return object()

    monkeypatch.setattr(
        "eidolon.livekit.agent.runtime.make_device_token_resolver",
        _resolver,
    )
    stable_room = SimpleNamespace(name="eidolon-device-stable")

    for interaction_id in (
        "esp32-dispatch-00000002",
        "esp32-dispatch-00000003",
    ):
        _build_device_token_source(
            cfg=_config(),
            livekit_room=stable_room,
            runtime_services=_services(),
            runtime_session_id=interaction_id,
        )

    assert sessions == [
        "esp32-dispatch-00000002",
        "esp32-dispatch-00000003",
    ]


def test_runtime_token_rejects_missing_dispatch_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "eidolon_sdk.biz.runtime.resolve_shared_secret",
        lambda _value: "resolved-secret",
    )

    with pytest.raises(RuntimeError, match="named dispatch"):
        _build_device_token_source(
            cfg=_config(),
            livekit_room=SimpleNamespace(name="eidolon-device-stable"),
            runtime_services=_services(),
            runtime_session_id="",
        )
