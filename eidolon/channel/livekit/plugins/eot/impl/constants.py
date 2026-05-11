# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Turn Detection constants shared across EOT, TurnEndPolicy, and ContextEnhancedEot.
Copied from eidolon/pipeline/src/manager/turn_detection/constants.py
"""

# -----------------------------------------------------------------------------
# 1. Filler words (EOT incomplete / Policy invalid speech)
# -----------------------------------------------------------------------------
FILLER_WORDS = frozenset(
    {
        "嗯",
        "啊",
        "呃",
        "额",
        "哦",
        "噢",
        "喔",
        "咦",
        "欸",
        "哼",
        "哈",
        "嘿",
        "呵",
        "嘻",
        "哒",
        "哎",
        "哎呀",
        "哎哟",
        "哇",
        "靠",
        "天呐",
        "妈呀",
        "唔",
        "喂",
        "那个",
        "就是",
        "然后",
        "这个",
        "那个那个",
        "就是那个",
        "那个呢",
        "这样",
        "这样吧",
        "那个什么",
        "哦哦",
        "嗯嗯",
        "哈哈",
        "呵呵",
        "哈喽",
        "hello",
        "hi",
        "呢",
        "咯",
        "嘞",
        "嘛",
        "好",
        "好的",
        "行",
        "可以",
        "那行",
    }
)

# -----------------------------------------------------------------------------
# 2. Thinking / Conjunction endings (EOT force incomplete / Policy T_MAX)
# -----------------------------------------------------------------------------
INCOMPLETE_CONJUNCTIONS = (
    "因为",
    "所以",
    "但是",
    "而且",
    "那么",
    "如果",
    "除非",
    "只是",
    "还有",
    "然后",
    "或是",
    "哪怕",
    "之所以",
)
THINKING_WORDS = (
    "然后",
    "接着",
    "而且",
    "还有",
    "并且",
    "另外",
    "所以",
    "因为",
    "既然",
    "但是",
    "可是",
    "不过",
    "虽然",
    "尽管",
    "结果",
    "也就是",
    "也就是说",
    "其实",
    "实际上",
    "基本上",
    "简单来说",
    "比如说",
    "例如",
    "哪怕",
    "要是",
    "如果",
    "假设",
    "只要",
    "除非",
    "哪怕是",
    "即使",
    "特别是",
    "尤其是",
    "让我想想",
    "我想一下",
    "我想想看",
    "这个",
    "这个嘛",
    "那个",
    "那个就是",
    "可能是",
    "大概是",
    "也许是",
    "意思是",
    "我记得",
    "我感觉",
    "我觉得",
    "我的意思是",
    "怎么说呢",
    "也就是那个",
    "就那个",
    "就那",
    "再就是",
    "让我想想那个",
    "就是说",
    "我想下哈",
    "这个那个",
    "怎么讲",
    "叫什么来着",
)

# -----------------------------------------------------------------------------
# 3. Command / Short complete words (EOT complete 1.0 / Policy T_URGENT)
# -----------------------------------------------------------------------------
COMMAND_WORDS = (
    "停",
    "闭嘴",
    "停止",
    "退出",
    "再见",
    "拜拜",
    "滚",
    "安静",
    "大声点",
    "小声点",
)
ACTION_COMPLETE_WORDS = frozenset(
    {
        "播放",
        "打开",
        "取消",
        "跳过",
        "重试",
        "发送",
        "确认",
        "查询",
    }
)
SHORT_COMPLETE_WORDS = (
    frozenset(
        {
            "好",
            "行",
            "对",
            "是",
            "停",
            "等",
            "喂",
            "不",
            "要",
            "买",
            "查",
            "放",
            "闭嘴",
            "你好",
            "谢谢",
            "再见",
            "拜拜",
            "晚安",
            "好的",
            "可以",
            "没",
            "换",
            "没问题",
            "对的",
            "没错",
        }
    )
    | ACTION_COMPLETE_WORDS
)

# -----------------------------------------------------------------------------
# 4. EOT specific: Hesitation short phrases, dangling suffixes
# -----------------------------------------------------------------------------
INCOMPLETE_SHORT_PHRASES = frozenset(
    {
        "我看看",
        "我想想",
        "怎么说呢",
        "其实",
        "稍微等我一下",
        "就是那个",
        "让我想一想",
    }
)
DANGLING_SUFFIXES = (
    "是",
    "叫",
    "位于",
    "比如",
    "像是",
    "准备",
    "叫做",
    "它是",
    "包含",
    "到",
    "在",
    "有",
    "为",
    "和",
    "就",
    "要",
    "会",
    "能",
    "把",
    "的",
)

# -----------------------------------------------------------------------------
# 5. Policy specific: Question words, punctuation
# -----------------------------------------------------------------------------
QUESTION_WORDS = (
    "吗",
    "呢",
    "吧",
    "嘛",
    "么",
    "没",
    "哩",
    "不",
    "呀",
    "哈",
    "哇",
    "谁",
    "什么",
    "怎么",
    "哪里",
    "哪儿",
    "哪个",
    "哪些",
    "几时",
    "几点",
    "多少",
    "多大",
    "多远",
    "多久",
    "多深",
    "多重",
    "为啥",
    "为什么",
    "何必",
    "如何",
    "哪位",
    "几位",
    "几个",
    "啥时",
    "啥样",
    "咋样",
    "咋办",
    "还是",
    "是不是",
    "能不能",
    "会不会",
    "对不对",
    "好不好",
    "行不行",
    "可以吗",
    "要不要",
)
TERMINAL_PUNCTUATION = (
    "。",
    "？",
    "！",
    "」",
    "』",
    "）",
    "›",
    "】",
    ".",
    "?",
    "!",
    '"',
    "'",
    ")",
    "]",
    "}",
    "…",
    "...",
)

# Round 7 G3: question particle detection at sentence end.
#
# These are sentence-ending grammatical markers that strongly imply the user
# is asking a question and expects an immediate response, EVEN WHEN ASR did
# not insert a "?" at the end (which is common for streaming ASR). Treated
# as equivalent to terminal_punctuation for threshold purposes.
#
# Excludes ambiguous particles (吧 = suggestion, 啊 = exclamation, 呀/哇 =
# emphasis) that frequently appear in non-question contexts. False positives
# there would cause premature cuts on declarative sentences.
QUESTION_END_PARTICLES = ("吗", "呢", "么", "嘛")

# Bound interrogative forms — these make the sentence unambiguously a question
# when they appear at the end. Different from QUESTION_END_PARTICLES (single
# chars) — these are 2-3 char A-not-A patterns or full phrases.
QUESTION_BOUND_FORMS_END = (
    "是不是",
    "对不对",
    "好不好",
    "行不行",
    "能不能",
    "会不会",
    "要不要",
    "有没有",
    "可不可以",
)
WAIT_PUNCTUATION = ("，", ",", ";", "；", "、", "—", "——")
PUNCTUATION_RADIUS = 3

# -----------------------------------------------------------------------------
# 6. EOT specific: Sentence end punctuation -> score mapping
# -----------------------------------------------------------------------------
SENTENCE_END_PUNCS = {"。", "？", "！", ".", "?", "!"}
MID_PUNCS = {"，", "、", ",", ";"}

# -----------------------------------------------------------------------------
# 7. ContextEnhancedEot specific: Context adjustment rules
# -----------------------------------------------------------------------------
FOLLOWUP_INDICATORS = frozenset(
    {
        "呢",
        "那",
        "还有",
        "另外",
        "而且",
        "但是",
        "不过",
        "然后",
        "接着",
        "再说",
        "对了",
    }
)
SUMMARY_CLOSING_WORDS = frozenset(
    {
        "就这些",
        "就这么多",
        "说完了",
        "讲完了",
        "就这样吧",
        "好了",
        "搞定",
        "没了",
        "OK了",
        "没了没了",
    }
)
STRONG_ENDING_WORDS = (
    frozenset(
        {
            "谢谢",
            "拜拜",
            "再见",
            "好的",
            "行",
            "可以",
            "没问题",
            "就这样",
            "先这样",
            "挂了吧",
        }
    )
    | SUMMARY_CLOSING_WORDS
)
GREETING_WORDS = frozenset(
    {"你好", "您好", "哈喽", "嗨", "hello", "hi"}
)
SHORT_COMPLETE_PHRASES = frozenset(
    {
        "你好",
        "您好",
        "哈喽",
        "嗨",
        "谢谢",
        "拜拜",
        "再见",
        "好的",
        "行",
        "可以",
        "没问题",
        "就这样",
        "先这样",
        "北京",
        "上海",
        "广州",
        "深圳",
        "杭州",
        "成都",
        "武汉",
        "西安",
        "天气",
        "时间",
        "日期",
        "温度",
        "价格",
        "多少钱",
        "2024年",
        "2025年",
        "2023年",
    }
)
NOUN_PATTERNS = (
    r"^\d{4}年$",
    r"^[\u4e00-\u9fa5]{2,4}天气$",
    r"^[\u4e00-\u9fa5]{2,4}时间$",
    r"^[\u4e00-\u9fa5]{2,4}价格$",
)
HESITATION_PATTERNS = (
    ("那个", "嗯"),
    ("就是", "嗯"),
    ("那个", "那个"),
    ("嗯", "嗯"),
    ("啊", "啊"),
    ("呃", "呃"),
)

# -----------------------------------------------------------------------------
# 8. VAD + EOT coordination (stepped threshold / interrupt / tail / noise)
# -----------------------------------------------------------------------------
RAPID_REPEAT_INTERRUPT_WORDS = frozenset(
    {
        "停停停",
        "不对不对",
        "错了错了",
        "好了好了",
    }
)
STRONG_INTERRUPT_INTENT_WORDS = (
    frozenset(
        {
            "停",
            "闭嘴",
            "不对",
            "错了",
            "重说",
            "取消",
            "不要",
            "别说了",
            "闭嘴吧",
            "你先停下",
            "别播了",
            "不用说了",
            "不用了",
            "先等一下",
            "等下等下",
        }
    )
    | RAPID_REPEAT_INTERRUPT_WORDS
)
WEAK_INTERRUPT_INTENT_WORDS = frozenset(
    {
        "好的",
        "嗯",
        "那个",
        "然后",
        "等等",
        "等会",
        "哦",
        "那那个",
        "我想问",
        "其实是",
        "那什么",
        "对了",
    }
)
INTERRUPT_INTENT_WORDS = (
    STRONG_INTERRUPT_INTENT_WORDS | WEAK_INTERRUPT_INTENT_WORDS
)
WEAK_INTERRUPT_EOT_THRESHOLD = 0.6

# -----------------------------------------------------------------------------
# 8b. Continuation intent: phrases that signal the user wants the agent to
# continue/replace the current task rather than stop. These must NOT trigger
# interruption even if they contain strong interrupt substrings (e.g. "不对"
# in "不对，你要换一个新的"). Checked at priority 0 in should_interrupt().
# -----------------------------------------------------------------------------
CONTINUATION_INTENT_PATTERNS = frozenset(
    {
        # Explicit replacement / swap requests
        "换一个",
        "再换一个",
        "给我换一个",
        "给我再换一个",
        "换一个吧",
        "再换一个吧",
        "换别的",
        "换掉",
        "不要这个",
        "换一个话题",
        "换一个故事",
        "换一个笑话",
        "换一个段子",
        "这个不好笑",
        "不好笑",
        "太无聊了",
        "没意思",
        "来一个",
        # Explicit continuation requests
        "继续",
        "继续说",
        "继续讲",
        "接着说",
        "接着讲",
        "接着",
        "然后呢",
        "后来呢",
        # Negative-form corrections followed by a continuation request
        # (e.g. "不对，你换一个新的" means "no, switch to a new one")
        "不是，",
        "不是这个",
        "不是那样的",
    }
)
IGNORE_ASR_WORDS = frozenset({"欸", "哼", "呸", "啧", "唏"})
