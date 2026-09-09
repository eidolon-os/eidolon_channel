from __future__ import annotations

import textwrap

import pytest

from eidolon.channel_provider.config import load_provider_config


def _environment(monkeypatch, tmp_path, yaml_text: str) -> None:
    settings = tmp_path / "settings.yaml"
    settings.write_text(textwrap.dedent(yaml_text), encoding="utf-8")
    environment = tmp_path / "empty.env"
    environment.write_text("", encoding="utf-8")
    monkeypatch.setenv("EIDOLON_CHANNEL_PROVIDER_SETTINGS_YAML", str(settings))
    monkeypatch.setenv("EIDOLON_CHANNEL_PROVIDER_ENV_FILE", str(environment))
    monkeypatch.setenv("EIDOLON_CHANNEL_PROVIDER_TOKEN", "provider-token-" + "x" * 32)
    monkeypatch.setenv("LIVEKIT_API_KEY", "api-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "api-secret")
    monkeypatch.setenv("STATE_ROOT_FOR_TEST", str(tmp_path / "state"))
    monkeypatch.setenv("LIVEKIT_URL_FOR_TEST", "wss://livekit.example.test")


def test_config_expands_env_and_keeps_secrets_out_of_yaml(monkeypatch, tmp_path) -> None:
    _environment(
        monkeypatch,
        tmp_path,
        """
        http:
          host: 127.0.0.1
          port: 8767
        storage:
          path: $STATE_ROOT_FOR_TEST/provider.sqlite3
        livekit:
          api_url: http://127.0.0.1:7880
          client_url: $LIVEKIT_URL_FOR_TEST
        """,
    )

    config = load_provider_config()

    assert config.http.port == 8767
    assert config.storage.path == tmp_path / "state" / "provider.sqlite3"
    assert config.livekit.client_url == "wss://livekit.example.test"
    assert config.bearer_token.startswith("provider-token-")


@pytest.mark.parametrize(
    ("storage_path", "client_url"),
    [
        ("", "wss://livekit.example.test"),
        ("$MISSING_STATE_ROOT/provider.sqlite3", "wss://livekit.example.test"),
        ("state.sqlite3", "ws://livekit.example.test"),
        ("state.sqlite3", "wss://user:password@livekit.example.test"),
        ("state.sqlite3", "wss://livekit.example.test/unexpected-path"),
    ],
)
def test_config_rejects_unsafe_or_unexpanded_origins(
    monkeypatch, tmp_path, storage_path: str, client_url: str
) -> None:
    _environment(
        monkeypatch,
        tmp_path,
        f"""
        storage:
          path: {storage_path!r}
        livekit:
          api_url: http://127.0.0.1:7880
          client_url: {client_url!r}
        """,
    )

    with pytest.raises(ValueError):
        load_provider_config()


def test_config_allows_explicit_development_lan_client_url(monkeypatch, tmp_path) -> None:
    _environment(
        monkeypatch,
        tmp_path,
        """
        storage:
          path: $STATE_ROOT_FOR_TEST/provider.sqlite3
        livekit:
          api_url: http://127.0.0.1:7880
          client_url: ws://192.168.1.25:7880
        """,
    )
    monkeypatch.setenv("EIDOLON_CHANNEL_PROVIDER_ALLOW_INSECURE_LAN_CLIENT_URL", "1")

    assert load_provider_config().livekit.client_url == "ws://192.168.1.25:7880"


def test_config_rejects_ambiguous_insecure_lan_switch(monkeypatch, tmp_path) -> None:
    _environment(
        monkeypatch,
        tmp_path,
        """
        storage:
          path: $STATE_ROOT_FOR_TEST/provider.sqlite3
        livekit:
          api_url: http://127.0.0.1:7880
          client_url: wss://livekit.example.test
        """,
    )
    monkeypatch.setenv("EIDOLON_CHANNEL_PROVIDER_ALLOW_INSECURE_LAN_CLIENT_URL", "true")

    with pytest.raises(ValueError, match="must be 0 or 1"):
        load_provider_config()


def test_config_accepts_a_client_url_whose_host_is_not_yet_knowable(
    monkeypatch, tmp_path
) -> None:
    """`ws://:7880` says the host is decided when a binding is minted.

    Written that way because it is the truth: nothing at configuration time
    knows which of a Host's addresses a device will be able to reach, and an
    address observed once at deploy was handed to every device until the next
    deploy.
    """

    _environment(
        monkeypatch,
        tmp_path,
        """
        storage:
          path: $STATE_ROOT_FOR_TEST/provider.sqlite3
        livekit:
          api_url: http://127.0.0.1:7880
          client_url: ws://:7880
        """,
    )
    monkeypatch.setenv("EIDOLON_CHANNEL_PROVIDER_ALLOW_INSECURE_LAN_CLIENT_URL", "1")

    assert load_provider_config().livekit.client_url == "ws://:7880"


def test_config_refuses_a_client_url_with_neither_host_nor_port(
    monkeypatch, tmp_path
) -> None:
    """Deferring the host is not licence to omit the port too.

    A port is knowable at configuration time — it is this Host's own LiveKit
    listener — so leaving it out is a missing value, not a deferred one.
    """

    _environment(
        monkeypatch,
        tmp_path,
        """
        storage:
          path: $STATE_ROOT_FOR_TEST/provider.sqlite3
        livekit:
          api_url: http://127.0.0.1:7880
          client_url: ws://
        """,
    )
    monkeypatch.setenv("EIDOLON_CHANNEL_PROVIDER_ALLOW_INSECURE_LAN_CLIENT_URL", "1")

    with pytest.raises(ValueError, match="livekit.client_url"):
        load_provider_config()


def test_config_refuses_an_api_url_that_defers_its_host(monkeypatch, tmp_path) -> None:
    """Only the client URL may defer. `api_url` is this process reaching
    LiveKit over loopback, and loopback is known."""

    _environment(
        monkeypatch,
        tmp_path,
        """
        storage:
          path: $STATE_ROOT_FOR_TEST/provider.sqlite3
        livekit:
          api_url: http://:7880
          client_url: wss://livekit.example.test
        """,
    )

    with pytest.raises(ValueError, match="livekit.api_url"):
        load_provider_config()
