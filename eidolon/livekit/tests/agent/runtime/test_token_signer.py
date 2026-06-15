"""Phase 32.B: token_signer happy + error cases.

Schema correctness is critical — agent's PairingTokenVerifier will reject
any drift in field names/types. We decode the token with the same secret
to spot regressions immediately.
"""

from __future__ import annotations

from pathlib import Path

import jwt
import pytest

from eidolon.livekit.agent.runtime.token_signer import (
    resolve_shared_secret,
    sign_device_token,
)


def test_sign_token_payload_matches_verifier_schema():
    """Pin the exact fields agent's PairingTokenVerifier expects."""
    secret = "test-secret-with-enough-entropy-32b"
    token, exp = sign_device_token(
        secret=secret,
        device_id="dev-1",
        tenant_id="default",
        user_id="manson",
        template_id="caretaker",
    )
    payload = jwt.decode(token, secret, algorithms=["HS256"])
    assert payload["device_id"] == "dev-1"
    assert payload["tenant_id"] == "default"
    assert payload["user_id"] == "manson"
    assert payload["template_id"] == "caretaker"
    assert payload["scopes"] == ["device"]
    assert "jti" in payload and len(payload["jti"]) == 32
    assert payload["exp"] == int(exp.timestamp())
    assert payload["iat"] <= payload["exp"]


def test_sign_token_rejects_empty_secret():
    """Fail loud on empty secret — agent would reject anyway, better at
    sign time than after a round-trip."""
    with pytest.raises(ValueError, match="secret is required"):
        sign_device_token(
            secret="",
            device_id="d",
            tenant_id="t",
            user_id="u",
            template_id=None,
        )


def test_sign_token_null_template_id_round_trips():
    """ResolvedContext.template_id is None when admin has no preference;
    must propagate as JSON null, not the string 'None'."""
    secret = "test-secret-with-enough-entropy-32b"
    token, _exp = sign_device_token(
        secret=secret,
        device_id="d",
        tenant_id="t",
        user_id="u",
        template_id=None,
    )
    payload = jwt.decode(token, secret, algorithms=["HS256"])
    assert payload["template_id"] is None


# ---- resolve_shared_secret ------------------------------------------------


def test_resolve_secret_explicit_arg_wins(monkeypatch, tmp_path: Path):
    """Explicit env_value arg takes priority over file fallback. Test
    pins this so a future refactor that flips the order is caught."""
    secret_file = tmp_path / "jwt-secret"
    secret_file.write_text("from-file")
    monkeypatch.setattr(
        "eidolon.livekit.agent.runtime.token_signer._SHARED_SECRET_FILE",
        secret_file,
    )
    assert resolve_shared_secret("from-arg") == "from-arg"


def test_resolve_secret_falls_back_to_file(monkeypatch, tmp_path: Path):
    """Empty env arg → read ~/eidolon/run/jwt-secret. This is the
    dev-stack happy path that agent set up by auto-persisting."""
    secret_file = tmp_path / "jwt-secret"
    secret_file.write_text("from-file\n")  # newline tolerated
    monkeypatch.setattr(
        "eidolon.livekit.agent.runtime.token_signer._SHARED_SECRET_FILE",
        secret_file,
    )
    assert resolve_shared_secret("") == "from-file"


def test_resolve_secret_empty_when_neither_set(monkeypatch, tmp_path: Path):
    """No env arg, no file → empty string. Factory's caller decides
    whether to hard-fail or fall back to legacy static token."""
    nonexistent = tmp_path / "jwt-secret"  # never created
    monkeypatch.setattr(
        "eidolon.livekit.agent.runtime.token_signer._SHARED_SECRET_FILE",
        nonexistent,
    )
    assert resolve_shared_secret("") == ""
