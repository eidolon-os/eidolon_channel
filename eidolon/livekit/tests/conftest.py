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
_env = _local_env if _local_env.is_file() else _fixture_env
if not os.environ.get("EIDOLON_CHANNEL_LIVEKIT_ENV", "").strip():
    os.environ["EIDOLON_CHANNEL_LIVEKIT_ENV"] = str(_env)

if _env.is_file():
    load_dotenv(_env, override=True)
    print(
        f"[conftest] Loaded env from {_env}, "
        f"SENSETIME_TTS_API_KEY={os.environ.get('SENSETIME_TTS_API_KEY', 'NOT SET')[:20]}"
    )
else:
    print(
        f"[conftest] No env file at {_local_env} or {_fixture_env}; set "
        f"EIDOLON_CHANNEL_LIVEKIT_ENV to a valid env file before running tests "
        f"that call load_agent_config()"
    )
