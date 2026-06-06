"""Shared default values for the effective channel config.

These values are deployment-tunable defaults. Runtime code should read them
through the typed config objects, while schema/profile code can import them to
avoid duplicating literal lexicons.
"""

from __future__ import annotations

DEFAULT_HARD_STOP_LEXICON: tuple[str, ...] = (
    "停",
    "停一下",
    "别说了",
    "不要说了",
    "打住",
    "闭嘴",
    "先别讲了",
)

DEFAULT_TOPIC_SWITCH_LEXICON: tuple[str, ...] = (
    "换个话题",
    "不聊这个",
    "别聊这个",
    "说点别的",
    "聊点别的",
    "我们聊点",
    "刚才那个不用了",
)

DEFAULT_CORRECTION_LEXICON: tuple[str, ...] = (
    "不是",
    "等一下",
    "我不是这个意思",
    "我刚才",
    "我刚才说",
    "我刚才说错了",
)
