# G6–G9 架构性 follow-up — eidolon_channel LiveKit

## Context

G1–G5（[../2026-05-16/g1-g6-post-f5-fixes.md](../2026-05-16/g1-g6-post-f5-fixes.md)）落地并跑了 multi-turn 回归后，**新暴露了 4 个更深层的架构 / 协议问题**。本文档把它们做成 G6–G9 一批 plan，**全部走 framework public API + 我们模块内重构，零新增 monkey-patch**（现存 `_framework_patches.py` 那一项仍按原状保留——它的职责与本批互不相干，见 § Q3）。

### 暴露问题来源

回归运行（2026-05-17 00:05–00:06）日志要点：
- 用户在欢迎语 ~2s 处说"停"——VAD 检测到 + DuckingMixer SUSPENDED，**但 STT 完全没产出**。诊断：framework 的 `aec_warmup_duration=3.0s` 在 `agent_state="speaking"` 时把 STT 旁路掉。
- Bailian STT WS 在该 session **首次**出现服务端单边关（`ConnectionClosedError: no close frame`），原因疑似 long-lived 单 task 模式遇到 AEC 空窗 + DashScope 服务端超时。
- `final transcript not received after timeout` 在用户**没有 ASR 文本**（"停"被 AEC 吞）时也触发——framework 用了**后一句话的 INTERIM** 当 FINAL 提交，产生**幽灵 LLM call**。

### 与上一批的依赖关系

| Item | 类型 | 优先级 |
|---|---|---|
| **G6** Interrupted-context 数据源从 framework history 切到 TtsStage+DuckingMixer | 架构积压 | P3（已积压一轮）|
| **G7** Bailian STT 协议改 per-utterance + 引入 STT 连接池 | 协议正确性 + 可靠性 | P2 |
| **G8** No-ASR-no-commit 守卫 | 真 bug | **P0（最即效，1 行改）**|
| **G9** `aec_warmup_duration` env 化 | 配置缺口 | **P0（1 行改）** |

---

## G8 — No-ASR-no-commit 守卫（P0，最简单）

### 根因

[`streaming.py:587-606`](../../../eidolon/livekit/agent/streaming.py)：
```python
if self._session is not None:
    if self._latest_asr_text:
        eot_model.record_turn(...)
    eot_model.reset()
    self._inject_interrupted_context()
    self._session.commit_user_turn(transcript_timeout=...)
```

`record_turn` 已经被 `if self._latest_asr_text` 保护，但 **`commit_user_turn` 没有**——任何 VAD-end 都会触发它。

### 真实灾难路径

```
t0    AEC warmup active；用户说"停" → VAD 检测到 → VAD-end
t0    commit_user_turn(transcript_timeout=5.0) fires；framework 等 FINAL
t1=t0+3s  用户开始说第二句"我刚才说了停..."
t1+    INTERIM 进 framework._audio_interim_transcript（全局，无段隔离）
t5=t0+5s  framework timeout → 抓 _audio_interim_transcript = "我刚才" → 当 FINAL
t5+   LLM 用"我刚才"调用（幽灵 turn），同时第二句话还在采集中
```

framework 的 `_audio_interim_transcript` **设计上没有 VAD 段隔离**——是个全局字符串，被任何 INTERIM 覆写。我们改不了它（framework 内部）。但我们可以**不喂这个错误的 commit** 给它。

### 修法

[`streaming.py:587`](../../../eidolon/livekit/agent/streaming.py) 改为：

```python
if self._session is not None:
    if self._latest_asr_text:
        eot_model.record_turn(...)
        eot_model.reset()
        self._inject_interrupted_context()
        self._session.commit_user_turn(
            transcript_timeout=self._stt_commit_transcript_timeout,
        )
    else:
        # G8 (2026-05-17): VAD 检测到说话但 ASR 完全没文本，
        # 几种情况：① AEC warmup 窗口吞了音频；② 短促噪音；
        # ③ STT 服务端临时抖动。任何一种都不该触发 LLM。
        # 之前的代码无条件 commit，framework 会等 5s 超时，
        # 然后把后续 utterance 的 INTERIM 拿来"补位"——制造跨段污染
        # 和幽灵 LLM 调用。
        eot_model.reset()
        logger.info(
            "[StreamingPipeline] VAD-end with empty ASR — skipping "
            "commit_user_turn (AEC window / noise / STT hiccup)"
        )
```

**影响**：
- 消除幽灵 LLM call
- 消除 `final transcript not received after timeout` 在 AEC 期间的虚报
- 不影响正常路径——只在 ASR 真空才生效

**测试**：单测 `test_streaming_skips_commit_on_empty_asr`——伪造 VAD-end 事件 + 空 `_latest_asr_text`，断言 `session.commit_user_turn` 未被调用。

---

## G9 — `aec_warmup_duration` env 化（P0，配置缺口）

### 现状 vs 实际播放时间

实测日志：
```
t=0     speaking 开始
t=3.0   aec warmup 过期（默认值）
t=4.5   欢迎语实际播完
```

**默认 3s 只盖前 2/3**，最后 1.5s 实际是可打断的——半保护状态，反而困惑。23 字短句尚且如此，长一点的欢迎语漏更多。

### 三档场景的合理值

| 场景 | duration | 说明 |
|---|---|---|
| Power-user / 老用户 | `0` 或 `None` | 欢迎语全程可打断 |
| 主流（建议默认） | `0.5`–`1.0` | 防开口噪音误打断，保护起手 |
| 客服 / 新手引导 | 对齐 welcome 时长（~5–6s） | 欢迎语全程不可打断 |

### 修法

1. [`common/config.py`](../../../eidolon/livekit/common/config.py) `AgentBehaviorConfig` 加字段：
   ```python
   aec_warmup_duration: float | None = 1.0   # G9: framework 默认 3.0，太长
   ```
   `None` 显式表示禁用，`0.0` 等价。

2. `AgentConfig.from_env` 读 env：
   ```python
   aec_warmup_duration=_optional_float(get("AGENT_AEC_WARMUP_DURATION", "1.0")),
   ```
   解析逻辑：`""`/`"none"` → None；其他 → float。

3. [`agent/server.py`](../../../eidolon/livekit/agent/server.py) 透传到 `StreamingPipeline`。

4. [`agent/streaming.py:216`](../../../eidolon/livekit/agent/streaming.py) `AgentSession(...)` 加入：
   ```python
   aec_warmup_duration=self._aec_warmup_duration,
   ```

5. env 文件：
   ```
   # G9 fix (2026-05-17): 欢迎语前多少秒不可被用户打断（防开口噪音误触）。
   # 0 = 全程可打断（power-user）；建议主流场景 0.5-1.0；客服场景对齐 welcome 时长。
   AGENT_AEC_WARMUP_DURATION=1.0
   ```

**为什么不是 monkey-patch**：[`agent_session.py:236`](.venv/.../voice/agent_session.py) `aec_warmup_duration: float | None = 3.0` 本来就是 public API 关键字参数。我们只是没用上。

**测试**：单测验证 env 解析（含 `"none"` 字面量）+ 端到端验证启动 log 里 `aec warmup active, disabling interruptions for 1.00s` 出现且 timing 与设置一致。

---

## G7 — Bailian STT 协议改 per-utterance + 引入 STT 连接池（P2）

### 触发证据 & 协议分析

回归日志：STT WS 在第一次 user turn 结束 200ms 后被服务端单边关（`no close frame`）。从我们这边看：
- 客户端 PING 间隔 20s，远没到
- AEC warmup 窗口（02.645–05.645）期间，STT WS 上几乎无音频流动
- 全程 task_id 复用 = 单 task 跨多 turn

DashScope FunASR 文档推荐模式（待二次验证）：**一个 utterance 一个 task**——`run-task → audio → finish-task → wait task-finished → close`。当前 long-lived 单 task 是非主流用法，触发服务端的"客户端可能已经走了"判定。

### 与 TTS 池可对偶的解决方案

引入 **STT 连接池**（复用 [`tts/_pool.py`](../../../eidolon/livekit/plugins/tts/_pool.py) 的 `TTSConnectionPool`——其实是 provider-agnostic 的，名字误导）：

```
启动 warmup：预热 N 个 STT WS（each: connect + run-task + 等 task-started）
        ↓
        N 条 idle WS 等着收音频，每条独立 task_id

VAD-start：pool.acquire() → 拿一个 warm STT 连接（~0ms 延迟）
           send_loop 推音频
VAD-end：  send_audio_finish (DashScope 没 finish-task 协议则发空帧)
           等 task-finished
           pool.mark_dirty(conn) → 后台开新连接补池
```

**和你的目标对齐**：
- ✅ 降低 pipeline 耗时：池让 acquire 0ms（vs 每次 cold start 300-500ms）
- ✅ STT 链路可靠：每个 turn 独立 task，单个连接挂掉只丢一轮，不丢整 session
- ✅ 解决 Q3 的跨段污染根因：每轮独立 task → framework 的 `_audio_interim_transcript` 不会跨段串

### 预热的隐患

预热的 WS 上没有音频流动；如果 DashScope 对 `task-started` 后无音频也有空闲超时（待验证），需要：
- **方案 A**：预热时不发 `run-task`，只建 WS 不开 task；VAD-start 时再发 `run-task`（+ 100ms 等 task-started 的延迟）
- **方案 B**：预热完整流程，但发周期性 silence audio（每 20s 一次 100ms 静音）保活
- **方案 C**：预热深度 limit 1-2（不预热太多 idle 连接）

### 实施前置

1. **查 DashScope FunASR 官方文档**，确认：
   - 是否真的推荐 per-utterance task 模式
   - `task-started` 之后多久不发音频会被关
   - WS 层 ping/pong 协议要求
2. 写一个 standalone repro 脚本验证：建 WS + run-task + 静默 30s，看是否被关
3. 验证完再决定预热方案 A/B/C

### 改动范围（大）

| 文件 | 变化 |
|---|---|
| [`plugins/stt/bailian/connection_manager.py`](../../../eidolon/livekit/plugins/stt/bailian/connection_manager.py) | 改成单次 utterance 生命周期；加 `finish_task_and_close()` |
| [`plugins/stt/bailian/speech_stream.py`](../../../eidolon/livekit/plugins/stt/bailian/speech_stream.py) | 改为：监听 VAD 事件 → `pool.acquire` → push → 监听 VAD-end → `pool.mark_dirty`；不再 long-lived |
| [`plugins/stt/bailian/stt.py`](../../../eidolon/livekit/plugins/stt/bailian/stt.py) | 加 `_pool: TTSConnectionPool` 实例 + `warmup()` / `shutdown()` |
| [`plugins/stt/bailian/config.py`](../../../eidolon/livekit/plugins/stt/bailian/config.py) | 加 `pool_size`、`pool_size_bootstrap`、`pool_max_idle_sec` 等（同 TTS 那批）|
| env 文件 | 加 `BAILIAN_STT_POOL_*` 系列旋钮 |
| tests | 新增 STT 池行为测试（镜像 TTS 池的测试）|

工作量：**约 1 day**。不在本批立刻做，需要 DashScope 文档调研 + 真实压测。

---

## G6 — Interrupted-context 数据源重构（P3，架构积压）

### 当前问题（重复 G1–G6 plan 已说过的内容，这里压缩）

[`streaming.py:1050`](../../../eidolon/livekit/agent/streaming.py) `_snapshot_interrupted_context` 直接读 `self._session.history.messages()` 取最后一条 assistant message。

三个语义弱点：
1. **时序不确定**：framework 何时 commit assistant message 到 history 是内部细节
2. **抓的是全文不是播放进度**：用户实际只听到前 1/N（被 DuckingMixer fade-out + cancel 吞）
3. **跨层耦合**：依赖 framework 内部 chat history 的稳定性（G2 加括号事件就是个信号）

### 重构方案

引入 `InterruptedContextTracker`（新模块），由三个数据源协同：

| 数据源 | 提供什么 |
|---|---|
| `BailianSynthesizeStream._emitted_chars: list[str]` | 累计已发给 TTS 的文本（每次 `emit_segment` 成功 append）|
| `DuckingMixer.played_seconds: float`（新 property，基于 `_total_buffer_frames_drained × frame_dur`）| 实际播给用户的秒数 |
| Bailian/SenseTime TTS 平均朗读速度（从 config 估算或运行时学习）| chars-per-second |

合成：
```python
interrupted_context = {
    "text_emitted": "全文 A...",
    "audio_played_sec": 1.23,
    "estimated_chars_heard": round(played_sec * chars_per_sec),
    "text_heard_approx": "前 N 个字...",
    "text_lost": "中间被打断的 M 个字...",
}
```

这种 context 比"最后一条 assistant 全文"对 LLM 续接更有信息量。

### 复杂度评估

跨 3 个模块协同状态：
- `BailianSynthesizeStream` 加状态 + 暴露 API
- `DuckingMixer` 加 property
- 新建 `InterruptedContextTracker` 模块负责融合
- `_snapshot_interrupted_context` 改为读它

工作量：**1-2 day**。架构改进，非紧急。可延后到生产数据证明这个 context 真的需要更精准时再上。

---

## 实施顺序

| 优先级 | Items | 工作量 | 收益 |
|---|---|---|---|
| **P0**（本批）| G8 + G9 | <1 hour 合并 | 消除幽灵 LLM call + 欢迎语行为可配 |
| **P2**（次批，需调研）| G7 | ~1 day + DashScope 文档调研 | STT 可靠性 + Q3 根因彻底解决 |
| **P3**（积压）| G6 | ~1-2 day | 更精准的 interrupted context |

---

## 验证

### G8
- 单测 `test_streaming_skips_commit_on_empty_asr`：mock VAD-end + empty `_latest_asr_text`，断言 `session.commit_user_turn` 0 次调用
- 端到端：跑 multi-turn 对话，在欢迎语前 3s 内开口"停"，日志应出现 `VAD-end with empty ASR — skipping commit_user_turn`，**不再出现** `final transcript not received after timeout`

### G9
- 单测 `test_aec_warmup_duration_from_env`：env 设 `"0"` / `"none"` / `"2.5"`，验证 None / None / 2.5
- 端到端：env 设 `AGENT_AEC_WARMUP_DURATION=0.5`，启动日志应有 `aec warmup active, disabling interruptions for 0.50s`，可在欢迎语开播 0.6s 后说话被正常 STT

### G7（次批）
- 真实压测：100 轮对话 × 4 turn，统计 STT WS 中途关闭率（目标 < 0.1%）
- 首字延迟：池中 acquire 后到第一帧 audio 推送 < 5ms

### G6（积压）
- 单测：mock TtsStage 累计 emit + DuckingMixer played_seconds，断言 `interrupted_context` 输出符合预期
- 端到端：用户在 agent 说到一半打断，下一轮 LLM payload 里 `interrupted_context.text_heard_approx` 应≈实际听到的部分

---

## 关键文件索引（按 G 编号）

### G8
- [`eidolon/livekit/agent/streaming.py:587-606`](../../../eidolon/livekit/agent/streaming.py) — `_on_user_state_changed` commit 路径

### G9
- [`eidolon/livekit/common/config.py`](../../../eidolon/livekit/common/config.py) — `AgentBehaviorConfig` 加字段 + `from_env` 解析
- [`eidolon/livekit/agent/server.py`](../../../eidolon/livekit/agent/server.py) — 透传
- [`eidolon/livekit/agent/streaming.py:216-222`](../../../eidolon/livekit/agent/streaming.py) — `AgentSession(aec_warmup_duration=...)`
- `deploy/livekit-channel.env*` — 旋钮暴露

### G7
- 整个 [`eidolon/livekit/plugins/stt/bailian/`](../../../eidolon/livekit/plugins/stt/bailian/) 目录
- [`eidolon/livekit/plugins/tts/_pool.py`](../../../eidolon/livekit/plugins/tts/_pool.py) — 复用（rename 或独立 STT 版本）

### G6
- [`eidolon/livekit/agent/streaming.py:1031-1066`](../../../eidolon/livekit/agent/streaming.py) — `_snapshot_interrupted_context`
- [`eidolon/livekit/plugins/tts/bailian/tts.py:357-391`](../../../eidolon/livekit/plugins/tts/bailian/tts.py) — `emit_segment` 加 `_emitted_chars` tracking
- [`eidolon/livekit/agent/ducking.py`](../../../eidolon/livekit/agent/ducking.py) — 加 `played_seconds` property
- 新文件 `eidolon/livekit/agent/interrupted_context.py` — `InterruptedContextTracker`

---

## 不打 patch 的保证（再次确认）

- `aec_warmup_duration`、`AudioOutputOptions`、`commit_user_turn(transcript_timeout=...)` 都是 framework public 关键字参数
- `_framework_patches.py` 这一项（disable audio-activity auto-interrupt）**保持原样**，与本批职责互不相干（一个管 interrupt 路径，本批管 commit 路径 / STT 协议 / interrupted-context 数据源）
- 全部改动在 `eidolon/`、`deploy/`、`docs/` 下；site-packages 零变化
