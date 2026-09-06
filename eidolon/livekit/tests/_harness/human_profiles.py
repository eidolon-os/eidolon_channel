"""Speech delivery, not a hotword table: no profile text enters production policy."""
from dataclasses import dataclass


@dataclass(frozen=True)
class HumanProfile:
    name: str
    text: str
    interims: tuple[str, ...] = ()
    gap_ms: int = 80
    duration_ms: int = 650
    pauses: tuple[tuple[int, int], ...] = ()
    final_delay_ms: int = 0


HUMAN_PROFILES = (
    HumanProfile('fast', '请把今天和明天的天气都告诉我', ('请把今天', '请把今天和明天'), 35, 350),
    HumanProfile('slow', '我想了解一下明天的天气', ('我想', '我想了解', '我想了解一下', '我想了解一下明天'), 320, 1700),
    HumanProfile('repeated_words', '我我我想问一下明天明天的天气', ('我', '我我', '我我我想', '我我我想问一下明天'), 110, 1000),
    HumanProfile('repeated_interims', '帮我设置明天早上的提醒', ('帮我设置',) * 5, 80, 900),
    HumanProfile('self_correction', '订明天，不对，是后天去上海的票', ('订明天去北京的票', '订明天，不对', '订明天，不对，是后天去上海'), 140, 1200),
    HumanProfile('hesitation', '嗯那个我想问一下这件事情怎么办', ('嗯', '嗯那个', '嗯那个我想', '嗯那个我想问一下'), 190, 1300),
    HumanProfile('short_answer', '好', (), 80, 350),
    HumanProfile('single_digit', '7', (), 80, 350),
    HumanProfile('long_number', '13800138000', ('138', '1380013', '138001380'), 110, 1000),
    HumanProfile('english', 'Could you explain that again please', ('Could you', 'Could you explain'), 110, 950),
    HumanProfile('mixed_language', '帮我检查一下 API timeout 的原因', ('帮我检查 API', '帮我检查一下 API timeout'), 130, 1000),
    HumanProfile('quoted_command', '他说不要讲了但我还想听你继续解释', ('他说不要讲了', '他说不要讲了但我还想听'), 130, 1000),
    HumanProfile('negated_command', '不要停止，请继续刚才的解释', ('不要停止', '不要停止，请继续'), 100, 1000),
    HumanProfile('long_clause', '如果明天下雨而且风比较大的话我们就改到室内活动你觉得怎么样', ('如果明天下雨', '如果明天下雨而且风比较大的话', '如果明天下雨而且风比较大的话我们就改到室内活动'), 230, 1500),
    HumanProfile('within_turn_pause', '我想了一下还是选择第二个方案', ('我想了一下', '我想了一下还是'), 420, 1700, ((350, 1050),)),
    HumanProfile('late_final', '改成后天下午三点吧', ('改成明天下午三点', '改成后天下午三点吧'), 100, 600, final_delay_ms=900),
)
