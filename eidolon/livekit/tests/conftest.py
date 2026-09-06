"""Test configuration — load environment from .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

# Loopback WebSocket mocks (e.g. Bailian STT tests) must not go through SOCKS/HTTP proxies.
_no_proxy = os.environ.get("NO_PROXY", "").strip()
_loop = "localhost,127.0.0.1,::1"
os.environ["NO_PROXY"] = f"{_loop},{_no_proxy}" if _no_proxy else _loop

# AgentConfig.from_env requires EIDOLON_CHANNEL_LIVEKIT_ENV (no implicit default).
_tests_dir = Path(__file__).parent
_local_env = _tests_dir / ".env"
_fixture_env = _tests_dir / "fixtures" / "minimal_test.env"
_explicit_env = (
    os.environ.get("EIDOLON_CHANNEL_ENV_FILE", "").strip()
    or os.environ.get("EIDOLON_CHANNEL_LIVEKIT_ENV", "").strip()
)
_env = Path(_explicit_env) if _explicit_env else (
    _local_env if _local_env.is_file() else _fixture_env
)
if not os.environ.get("EIDOLON_CHANNEL_LIVEKIT_ENV", "").strip():
    os.environ["EIDOLON_CHANNEL_LIVEKIT_ENV"] = str(_env)

if _env.is_file():
    # An explicit integration environment must not be overwritten by dummy keys.
    load_dotenv(_env, override=not bool(_explicit_env))
    print(f"[conftest] Loaded env from {_env}")
else:
    print(
        f"[conftest] No env file at {_local_env} or {_fixture_env}; set "
        f"EIDOLON_CHANNEL_LIVEKIT_ENV to a valid env file before running tests "
        f"that call load_agent_config()"
    )
