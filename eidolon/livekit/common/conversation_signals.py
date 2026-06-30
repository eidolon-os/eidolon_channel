"""Shared text-signal constants for realtime conversation control."""

from __future__ import annotations

BACKCHANNEL_WORDS: frozenset[str] = frozenset(
    {
        # Chinese acknowledgements
        "嗯",
        "嗯嗯",
        "嗯哼",
        "哦",
        "哦哦",
        "啊",
        "啊啊",
        "好",
        "好的",
        "好吧",
        "可以",
        "行",
        "行的",
        "对",
        "对啊",
        "对的",
        "对呀",
        "是",
        "是啊",
        "是的",
        "是呀",
        "没错",
        "嗯对",
        "嗯好",
        # English / Pinyin acknowledgements
        "ok",
        "okay",
        "yes",
        "yeah",
        "yep",
        "uh-huh",
        "mhm",
        "right",
        "sure",
    }
)
BACKCHANNEL_COMPOUND_CHARS = "嗯哦啊好对是"

NOISE_LIKE_TRANSCRIPTIONS: frozenset[str] = frozenset(
    {
        "啊",
        "嗯",
        "哈",
        "咳",
        "咳咳",
        "嗯哼",
        "啊啊",
        "啊啊啊",
        "啊啊啊啊",
        "嗯啊",
        "哎",
        "哎呀",
        "哦",
        "哦哦",
        "唉",
    }
)
REPEATED_NOISE_CHARS = "啊嗯哈咳哎哦唉"
