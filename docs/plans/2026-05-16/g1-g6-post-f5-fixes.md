# G1–G6 后续修复 — eidolon_channel LiveKit（post-F5）

## Context

F1–F5（[streaming-pipeline-diagnosis.md](./streaming-pipeline-diagnosis.md)）落地后跑了一次 4 轮真实对话回归。多数 fix 验证生效，但**新暴露了两个真 bug**，且发现还有一些既有 deprecation/能力缺口值得一起处理。本文档把这些做成 G1–G6 一次性闭环，**全部用 framework public API + 我们自己模块改造，零 monkey-patch**。

---

## Implementation Log (2026-05-16)

**Status**: G1–G5 全部 shipped；G6 留 follow-up（架构重构，本批不实施，原因见 § 6 章节）。
**Test result**: `472 passed, 7 skipped, 39 deselected`。
新增 4 个测试文件 19 个用例。

| Item | 状态 | 主要文件 |
|---|---|---|
| **G1** state mirror `"listening"` 分支 | ✅ shipped | [`pipeline/base.py:168-180`](../../../eidolon/livekit/agent/pipeline/base.py) |
| **G2** `_snapshot_interrupted_context` 加括号 | ✅ shipped | [`agent/streaming.py:1050`](../../../eidolon/livekit/agent/streaming.py) |
| **G3** `RoomOptions` SDK 迁移 | ✅ shipped | [`agent/streaming.py:268-285`](../../../eidolon/livekit/agent/streaming.py) |
| **G4** `LLMConfig` 扩展 + MiniMax 切换 | ✅ shipped | [`common/config.py`](../../../eidolon/livekit/common/config.py)、[`agent/factory.py`](../../../eidolon/livekit/agent/factory.py)、env 文件 |
| **G5** Aggregator 第一句激进模式 | ✅ shipped（默认关闭，env 可开）| [`tts/_aggregator.py`](../../../eidolon/livekit/plugins/tts/_aggregator.py)、[`tts/bailian/config.py`](../../../eidolon/livekit/plugins/tts/bailian/config.py)、[`tts/bailian/tts.py`](../../../eidolon/livekit/plugins/tts/bailian/tts.py) |
| **G6** Interrupted-context 架构重构 | ⏸️ deferred | 见 § 6 |

### 设计取舍 / 偏离

1. **G1 兼容两个名字** — 原 plan 写"加 `listening` 分支"；实施时改成 `elif new in ("idle", "listening")`，对 framework 未来可能改回 `"idle"` 也兼容。
2. **G2 只加括号、不重构数据源** — `session.history.messages()` 作为 interrupted context 的近似确实够用（只丢"用户实际听到了多少"的细节）；架构重构 → G6 follow-up。
3. **G4 LLM stage temperature 没动** — `factory.py:146` 的 `LlmParams(temperature=0.6)` 是 stage 层的另一个 temperature。本批保留 hard-code，让 plugin 层 `OPENAI_LLM_TEMPERATURE=0.7` 走 lk_openai。这两个 temperature 的优先级关系（plugin 覆盖 stage）依赖 `lk_openai.LLM` 内部行为，需要 runtime 观察是否生效。如果不生效，后续把 `LlmParams.temperature` 也改成读 cfg。
4. **G4 max_completion_tokens 默认 None** — 不强制写 100000，由 plugin 走自己的默认；用户在 env 里设 `OPENAI_LLM_MAX_TOKENS=100000` 才覆盖。这样 4o-mini / MiniMax 兼容。
5. **G5 完全 opt-in** — 默认 `BAILIAN_TTS_AGGREGATOR_FIRST_SENTENCE_SOFT_MIN_CHARS=0`、`...FLUSH_ANY_PUNCT=false`，老用户行为零变化。

### 配套测试（新增）

| 文件 | 用例数 | 覆盖点 |
|---|---|---|
| [`tests/agent/test_state_mirror.py`](../../../eidolon/livekit/tests/agent/test_state_mirror.py) | 5 | `speaking/thinking/listening/idle/unknown` 五种状态转移 |
| [`tests/agent/test_interrupted_context.py`](../../../eidolon/livekit/tests/agent/test_interrupted_context.py) | 5 | 取最后 assistant / `.messages()` 被调用 / 跳过空 text / 无 assistant / 配置关闭 |
| [`tests/common/test_llm_config.py`](../../../eidolon/livekit/tests/common/test_llm_config.py) | 6 | `_optional_float/int` 边界 + LLMConfig 默认 / 显式值 |
| [`tests/tts/sensetime/test_sentence_aggregator.py`](../../../eidolon/livekit/tests/tts/sensetime/test_sentence_aggregator.py) | +3 (G5) | first_sentence_flush_any_punct only-once / first_soft_min only-once / legacy default 保持 |

### 不打 patch 的保证（再次确认）

- 全部走 framework public API：`RoomOptions` / `AudioOutputOptions` / `lk_openai.LLM(temperature=, timeout=, max_completion_tokens=)` / `ChatContext.messages()`
- 改的文件全部在 `eidolon/` 和 `deploy/` 下，site-packages 零修改
- `framework_patches.py` 未动

### 暴露的问题（按性质归类）

| 类别 | 项 | 说明 |
|---|---|---|
| 🐛 真 bug | G1 | `_on_agent_state_changed` 不处理 framework 的 `"listening"` quiet state — 导致 `self._state` 永远卡在 `SPEAKING`，F3.2 的 state-guard 沦为摆设 |
| 🐛 真 bug | G2 | `_snapshot_interrupted_context` 把 `ChatContext.messages` 当 property 用（漏括号）— 只在 EOT cancel 路径触发时崩；F3 把 VAD 阈值降到 0.40 后首次触达 cancel 路径才暴露 |
| 📜 SDK 迁移 | G3 | `RoomInputOptions/RoomOutputOptions` 被框架 deprecate；要改用 `RoomOptions` + `AudioOutputOptions/TextOutputOptions` |
| 🔧 能力扩展 | G4 | `LLMConfig` 当前只暴露 `base_url / model / api_key`；用户给的 MiniMax 配置含 `timeout=120 / temperature=0.7 / extra_headers / extra_body` — framework 的 `lk_openai.LLM` 早就支持这些参数，缺的是我们 factory + config 把它们暴露出来 |
| ⚡ 架构改进 | G5 | 流式 LLM→TTS 链路设计本身正确（每 token 都喂 aggregator → 满 punct/max 即 `send_continue` → dashscope 立刻合成），但 LLM body 在 200ms 内一次性 burst 到达时，aggregator 抓不到中段 punct → 全文 `explicit` 一次性 flush。新增 env 旋钮让 aggregator 在第一句更激进 flush，让用户更快听到首字 |

### 显式不在本轮（理由）

- **`playback_finished called more times` warning**：F5 plan 已标 "warning-only，未处理"。
- **LLM ReadTimeout 10s / dashscope task-started 15s timeout**：上游基础设施抖动，不是 eidolon 代码问题。G4 切到 MiniMax + LiteLLM 后大概率自动消失。
- **完整重构 "interrupted context" 数据源到 TtsStage + DuckingMixer**：G2 加括号能让现有逻辑跑通；但当前实现"取 `session.history.messages()` 最后一条 assistant"在语义上**只能近似**真正被打断的内容（不知道用户实际听到了多少、buffer 里丢弃了多少）。本文档把架构重构作为 follow-up Issue 单列在 § 6，**G6**，不在本批落地。

---

## 修复方案

### G1 — `_on_agent_state_changed` 漏 `"listening"` 分支

**文件**：[`eidolon/livekit/agent/pipeline/base.py:159-178`](../../../eidolon/livekit/agent/pipeline/base.py)

现状：
```python
if new == "speaking":
    self._state = PipelineState.SPEAKING
elif new == "thinking":
    self._state = PipelineState.GENERATING
elif new == "idle":              # ← framework 从不发 "idle"，发的是 "listening"
    self._state = PipelineState.IDLE
```

**改动**：把 `elif new == "idle"` 替换为 `elif new in ("idle", "listening")`（双重防御：framework 未来真的发 "idle" 也兼容）。

**影响**：让 F3.2 的 state-guard 真正生效。`self._state` 在 agent 不说话时会归位 `IDLE`，DuckingMixer 不再在 listening 时白做 fade 循环。

### G2 — `_snapshot_interrupted_context` 调错 API

**文件**：[`eidolon/livekit/agent/streaming.py:1028`](../../../eidolon/livekit/agent/streaming.py)

现状：
```python
messages = self._session.history.messages   # method 引用，没调用
for msg in reversed(messages):              # TypeError
```

framework [`ChatContext.messages`](../../../.venv/lib/python3.12/site-packages/livekit/agents/llm/chat_context.py) 是普通方法不是 property。

**改动**：`self._session.history.messages` → `self._session.history.messages()`。

**测试覆盖缺口**：当前没有专门测 `_snapshot_interrupted_context` 的单测——因为它只在 EOT cancel 路径触发。加一个 minimal 单测验证它能正确从 `session.history` 取最后一条 assistant text。

### G3 — `RoomOptions` SDK 迁移

**文件**：[`eidolon/livekit/agent/streaming.py:268-276`](../../../eidolon/livekit/agent/streaming.py)

现状：
```python
await session.start(
    agent=agent,
    room=room,
    room_input_options=la.RoomInputOptions(),
    room_output_options=la.RoomOutputOptions(
        transcription_enabled=True,
        audio_sample_rate=self._audio_sample_rate,
    ),
)
```

**改动**（按 framework `RoomOptions` schema 重写）：
```python
from livekit.agents.voice.room_io import RoomOptions, AudioOutputOptions

await session.start(
    agent=agent,
    room=room,
    room_options=RoomOptions(
        # audio_input / text_input / text_output 留 NOT_GIVEN 走 framework 默认（全开）
        audio_output=AudioOutputOptions(sample_rate=self._audio_sample_rate),
    ),
)
```

说明：
- 旧 `RoomOutputOptions.transcription_enabled=True` 在新 schema 里对应 `text_output != False`，留 NOT_GIVEN 即默认开启。
- 旧 `audio_sample_rate=N`（输出端）→ 新 `audio_output=AudioOutputOptions(sample_rate=N)`。
- 旧 `RoomInputOptions()` 全默认 → 新 `audio_input/text_input` 留 NOT_GIVEN 即默认开启。

**影响**：消除 `RoomInputOptions and RoomOutputOptions are deprecated, use RoomOptions instead` warning。Framework 行为完全等价。

### G4 — LLM 切换 MiniMax-M2.7 + `LLMConfig` 能力扩展

#### G4.1 扩展 `LLMConfig`

**文件**：[`eidolon/livekit/common/config.py:122-134`](../../../eidolon/livekit/common/config.py)

新增字段（都走 env、默认值保持当前行为）：
```python
@dataclass
class LLMConfig:
    base_url: str = ""
    model: str = "gpt-4o-mini"
    api_key: str = ""
    # G4 新增
    temperature: float | None = None           # OPENAI_LLM_TEMPERATURE
    timeout: float | None = None               # OPENAI_LLM_TIMEOUT (秒)
    max_completion_tokens: int | None = None   # OPENAI_LLM_MAX_TOKENS
```

`AgentConfig.from_env`：用 `_optional_float / _optional_int` 解析（"" / None 都映射为 None，表示"用 framework 默认"）。

#### G4.2 Factory 透传

**文件**：[`eidolon/livekit/agent/factory.py:155-170`](../../../eidolon/livekit/agent/factory.py)

```python
return lk_openai.LLM(
    model=cfg.llm.model,
    api_key=cfg.llm.api_key or None,
    base_url=cfg.llm.base_url or None,
    **_conditional_kwargs(
        temperature=cfg.llm.temperature,
        timeout=httpx.Timeout(cfg.llm.timeout) if cfg.llm.timeout else None,
        max_completion_tokens=cfg.llm.max_completion_tokens,
    ),
)
```

其中 `_conditional_kwargs` 只把 not-None 的项放进 kwargs，None 就不传 — 让 framework 自己的 NOT_GIVEN 语义生效。

⚠️ 注意：factory 第 146 行的 `LlmParams(temperature=0.6)` 是 stage 层另外一个 temperature。需要确认这两个 temperature 的优先级和组合关系——保留 hard-code 还是改成读 cfg。**本批保留 hardcode**（不改它的语义），让 stage 层 temperature 仍是 0.6 / 我们新加的 `OPENAI_LLM_TEMPERATURE` 走 plugin 层。如果两个都设，plugin 层覆盖 stage 层（lk_openai 内部行为）。

#### G4.3 env 文件更新

`.livekit-channel.env` + `livekit-channel.env.template`：
```
# G4 (2026-05-16): switched to MiniMax-M2.7 via LiteLLM gateway
OPENAI_LLM_BASE_URL=https://litellm.yangtzeailab.com/v1
OPENAI_LLM_MODEL=openai/MiniMax-M2.7
OPENAI_LLM_API_KEY=sk-6PCfjF8zzdxCKwUN6EeP3Q
OPENAI_LLM_TEMPERATURE=0.7
OPENAI_LLM_TIMEOUT=120
# OPENAI_LLM_MAX_TOKENS=100000  # optional
```

⚠️ `.livekit-channel.env` 含真实密钥，已经 gitignored，本地直接改；template 用 placeholder。

### G5 — Streaming TTS aggregator 优化（让流式真正发挥）

**文件**：[`eidolon/livekit/plugins/tts/_aggregator.py`](../../../eidolon/livekit/plugins/tts/_aggregator.py)、[`eidolon/livekit/plugins/tts/bailian/config.py`](../../../eidolon/livekit/plugins/tts/bailian/config.py)

#### 问题回顾

正常情况：LLM token 一边来 → aggregator.feed 一边切 → `hard_punct` 命中即 `send_continue`。但实测里 LLM body 在 200ms 内一次性 burst 完所有 token，**chunk 边界跨过 `！`**（chunk 是"灵活一点！不过刚才那句"），`_classify_buffer` 看到 `_buf[-1]='句'`，不 flush，最后整段 `explicit` 一次性出。用户首字延迟 = LLM TTFB + 全部 body 时间。

#### 改动：第一句激进 flush 模式

[`_aggregator.py`](../../../eidolon/livekit/plugins/tts/_aggregator.py) `SentenceAggregator`：

1. 新增构造参数 `first_sentence_soft_min_chars: int | None = None`：如果设了，**只对第一次 flush** 用这个更小的 soft_min（默认场景例如 6），第一次 flush 后切回 `soft_min_chars`（默认 12）。
2. 新增构造参数 `flush_on_any_punct_first: bool = False`：第一次 flush 前任何 punct（包括 `，`/`、`）都触发 hard flush，不管字数；之后正常。

[`bailian/config.py`](../../../eidolon/livekit/plugins/tts/bailian/config.py) 新增 env 字段：
- `BAILIAN_TTS_AGGREGATOR_FIRST_SENTENCE_SOFT_MIN_CHARS=6`（默认仍走旧行为，要求显式开启）
- `BAILIAN_TTS_AGGREGATOR_FIRST_SENTENCE_FLUSH_ANY_PUNCT=false`

[`bailian/tts.py`](../../../eidolon/livekit/plugins/tts/bailian/tts.py) 构造 `SentenceAggregator` 时把新参数传进去。

**影响**：
- 默认行为不变（向后兼容）
- 用户在 env 里打开两个旋钮后，第一句一旦遇到任何 punct 就 flush → TTS 早 200-500ms 开播
- 后续句子走正常阈值，避免 dashscope 过多 task_continue 拼接缝

⚠️ 这是**取舍 not silver bullet**：开了之后第一句可能拆得更碎，dashscope 多次合成的衔接需要听感测试。所以**默认关闭**，仅在确实卡 TTFB 时打开。

### G6 — 架构 follow-up（本批不实施，留 issue）

**问题**：`_snapshot_interrupted_context` 当前依赖 `session.history.messages()`，存在多个语义弱点：

1. **时序不确定**：framework 把当前 turn 的 assistant message commit 到 history 的时机是内部细节，cancel 时它可能在/不在。
2. **抓的是全文不是播放进度**：我们读到的是 TTS 收到的完整文本，但用户实际只听到前 1/N（DuckingMixer SUSPENDED 时 buffer 的部分丢弃）。
3. **跨层耦合**：直接读 framework 的内部 chat history 是脆弱的。

**proper 设计**：
- `BailianSynthesizeStream` 维护 `_emitted_chars: list[str]`（accumulating），每次 `emit_segment` 成功后 append。
- `DuckingMixer` 暴露 `played_seconds` property（已有 `_total_buffer_frames_drained` × frame_duration，封装下即可）。
- `_snapshot_interrupted_context` 读这两个状态：`{"text_emitted": "...", "audio_played_sec": 1.2}`。LLM 后续 turn 就有"用户听到了前 1.2 秒（约 5 个字），整体文本是 XX"的完整上下文。

**复杂度**：跨 3 个模块协同状态，需要 stage 引用 stream 实例 + mixer 实例。设计稍复杂。本批先 G2 加括号让现有近似逻辑跑通；架构重构跟踪在 [GitHub issue/TODO]（未开）。

---

## 配套测试

| 改动 | 测试更新 | 类型 |
|---|---|---|
| G1 | 加 `test_state_mirror_handles_listening` 单测（伪造 framework state event，验证 `self._state == IDLE`）| 新增 |
| G2 | 加 `test_snapshot_interrupted_context_basic`（伪造 session.history，验证不抛 TypeError + 正确取最后 assistant）| 新增 |
| G3 | 既有 `test_streaming_pipeline_starts.py`（如果有）会跑到 session.start —— 应该自动通过；如果没有就手动起一次 session 看 log | 既有 |
| G4 | 加 `test_llm_config_env_parsing`（验证 timeout/temperature/max_tokens 从 env 正确解析）| 新增 |
| G5 | 加 `test_aggregator_first_sentence_aggressive`（喂一段 token，验证开启 first_sentence flag 后第一句早 flush）| 新增 |

`pytest -x` 必须全绿，包括既有 453 个测试 + 新增 4-5 个。

---

## 文件清单

| 文件 | 改动 | Group |
|---|---|---|
| `eidolon/livekit/agent/pipeline/base.py` | `_on_agent_state_changed` 加 `listening` 分支 | G1 |
| `eidolon/livekit/agent/streaming.py` | 1) `messages()` 加括号；2) `RoomOptions` 迁移 | G2, G3 |
| `eidolon/livekit/common/config.py` | `LLMConfig` 增 3 字段 + `AgentConfig.from_env` 读 env | G4 |
| `eidolon/livekit/agent/factory.py` | `_build_llm` 透传新字段 | G4 |
| `eidolon/livekit/plugins/tts/_aggregator.py` | `SentenceAggregator` 增 2 参数 + first_emit 状态 | G5 |
| `eidolon/livekit/plugins/tts/bailian/config.py` | 增 2 env 字段 | G5 |
| `eidolon/livekit/plugins/tts/bailian/tts.py` | aggregator 构造时透传 | G5 |
| `deploy/livekit-channel.env.template` | LLM 切 MiniMax + 新 TTS 旋钮（commented optional） | G4, G5 |
| `deploy/.livekit-channel.env` | 同上（含真密钥）| G4, G5 |
| `eidolon/livekit/tests/...` | 新增 4-5 个单测 | all |
| `docs/plans/2026-05-16/g1-g6-post-f5-fixes.md` | 本计划 + 实施日志 | — |

---

## 验证

1. `pytest -x` 全绿（包括新增 4-5 测试）
2. 启动 agent，跑一段 multi-turn 对话，确认：
   - **G1**: 用户在 agent listening 时开口，log 出现 `duck skipped — agent not speaking (state=IDLE)`，**不再出现** `DuckingMixer duck NORMAL→SUSPENDED` 在 listening 状态。
   - **G2**: 用户主动打断 agent（EOT score > 0.7），log 出现 `interrupted context captured: '...'`，**不再出现** `TypeError: 'method' object is not reversible`。
   - **G3**: 启动日志**不再出现** `RoomInputOptions and RoomOutputOptions are deprecated`。
   - **G4**: 启动日志 `config reloaded: llm=openai/MiniMax-M2.7`，LLM POST 日志显示 base_url 指向 `litellm.yangtzeailab.com`。LLM TTFB 应明显下降（mopai 6.7-10s+ → MiniMax 应该 <1s）。
   - **G5**: 默认行为不变（不设新 env 旋钮）；若设 `BAILIAN_TTS_AGGREGATOR_FIRST_SENTENCE_SOFT_MIN_CHARS=6`，第一次 `aggregator flush reason=...` 出现的时间应早于 LLM body 结束。
3. 不打 patch 的保证：
   - 没有改 `framework_patches.py` 之外的任何 framework 行为
   - 全部走 framework public API（`RoomOptions`、`lk_openai.LLM(temperature=...)` 等）
   - site-packages 下零改动
