"""Compose the optional online music capability for one Channel session."""

from __future__ import annotations
import logging
import os
from pathlib import Path
import aiohttp
from ...plugins.music.lizheng import LizhengMusicApi, LizhengMusicConfig
from ..session.music import SessionMusicRuntime
from ..output.music import MusicSession

logger = logging.getLogger("agent.session.music")


def music_cache_directory() -> Path:
    root = Path(os.environ.get("EIDOLON_CACHE_ROOT", "~/eidolon/cache")).expanduser()
    return root / "channel" / "music"


async def create_music_runtime(session: MusicSession) -> SessionMusicRuntime | None:
    if not any(
        os.getenv(key, "").strip() for key in ("LIZHENG_MUSIC_API_MAC", "LIZHENG_MUSIC_API_AES_KEY")
    ):
        return None
    http = None
    try:
        config = LizhengMusicConfig.from_environment()
        http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=55))
        api = LizhengMusicApi(config, http)
        return await SessionMusicRuntime.create(
            session, music_dir=music_cache_directory(), api=api, http=http
        )
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("online music disabled: initialization failed (%s)", type(exc).__name__)
        if http is not None:
            await http.close()
        return None
    except BaseException:
        if http is not None:
            await http.close()
        raise
