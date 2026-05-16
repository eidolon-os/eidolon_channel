# 多轮对话日志诊断 — eidolon_channel LiveKit

## Context

测试一段 4 轮中文多轮对话（13:30:24–13:31:44）。链路 STT(Bailian) → EOT(FireRed) → LLM(deepseek-v4-flash) → TTS(Bailian CosyVoice) 整体能跑通，但出现 **6 类警告/错误、3 处打断行为不符预期、每轮多次触发"双 FINAL→双 LLM 调用"**，以及一次 **TTS WebSocket 因长时间空闲被服务端关闭、且没有重试导致整轮无声回复**。

本文档**仅作根因诊断 + 修复优先级**，不修改代码。Plan 由 4 个问题对应 4 个章节，最后一节给出按收益/成本排序的修复路径。

---

## 第 4 问：四轮对话的事件时间线

### Round 0 — 进房欢迎语（agent 主动）

| t | 事件 |
|---|---|
| 30:44.430 | `TtsStage warming up`，目标池=4 |
| 30:49.559 | warmup complete（**耗时 5.1s** — 4 个 WS 并行 ≈ 单个 1.3s） |
| 30:49.569 | `VoiceAgent on_enter welcome="你好！我是你的 AI 助手..."` |
| 30:49.608 | TTS pool acquire（fast path，剩 3）|
| 30:49.609 | aggregator flush `reason=hard_punct_or_max` |
| 30:50.677 | agent_state: **initializing → listening** |
| 30:50.737 | agent_state: **listening → speaking** |
| 30:52.116 | TTS synth complete: text=25 chars / pcm=72480 B |
| 30:55.268 | ⚠️ `_SegmentSynchronizerImpl.playback_finished called before text/audio input is done` |
| 30:55.269 | ⚠️ `playback_finished called more times than playback segments were captured` |
| 30:55.270 | agent_state: **speaking → listening**（欢迎语播完） |

### Round 1 — 用户："你叫什么。"（**正常 happy path**）

| t | 事件 |
|---|---|
| 31:05.933 | VAD: start of speech |
| 31:05.935 | DuckingMixer **NORMAL→SUSPENDED**（但当时 agent_state=listening，没有可 duck 的音频） |
| 31:06.736 | duck resolved `reason=timeout action=unduck`（**0.80s 超时**） |
| 31:06.741 | VAD: end of speech（说话 ~0.8s） |
| 31:07.142 | INTERIM "你叫"（score 0.0 — too_short） |
| 31:07.533 | INTERIM "你叫什么"（score 0.129） |
| 31:08.431 | **FINAL "你叫什么。"**（score 0.209）— STT FINAL 比 VAD-end **晚 1.69s** |
| 31:09.424 | LLM POST `/chat/completions` |
| 31:11.280 | LLM 200 OK（**1.86s**） |
| 31:11.419 | aggregator flush "我叫 AI 助手..." |
| 31:11.597 | aggregator flush "怎么称呼你呢？" |
| 31:12.218 | agent_state: **thinking → speaking** |
| 31:13.990 | TTS complete: 30 chars / 93760 B |

**评价**：4.06s 端到端（VAD-end → speaking 开始）；其中 STT 占 1.69s、LLM 占 1.86s、TTS 首字 ~0.6s。

### Round 2 — 用户："我不相信你啊。"（**双 FINAL → 双 LLM**）

| t | 事件 |
|---|---|
| 31:17.433 | VAD: start of speech（**agent 正在说 reply1**） |
| 31:17.434 | DuckingMixer NORMAL→SUSPENDED |
| 31:18.221 | INTERIM "我"（score 0）；`should_interrupt: cut=False — Low VAD confidence avg=0.42 < 0.55` → **打断被否决** |
| 31:18.221 | `EOT(duck): score=0.00 mid-band, holding SUSPENDED` |
| 31:18.234 | duck resolved `reason=timeout action=unduck`（reply1 已自然结束） |
| 31:18.368 | agent_state: speaking → listening |
| 31:18.481 | VAD: end of speech |
| 31:18.990 | INTERIM "我不想信你啊" |
| 31:19.583 | VAD: start of speech 再起（用户接着第二句） |
| 31:20.385 | duck timeout, unduck |
| **31:20.482** | ⚠️ **`final transcript not received after timeout` → 把 INTERIM `我不想信你啊` 强制晋级 FINAL** |
| 31:20.540 | **LLM 调用 #1**（payload 含 "我不想信你啊"） |
| 31:20.628 | Bailian 真正 FINAL "我不相信你啊。"（带句号、错字订正） |
| 31:21.135 | **LLM #1 被 cancel**（`CancelledError`） |
| 31:21.145 | **LLM 调用 #2**（payload 含修正后的 "我不相信你啊。"） |
| 31:21.148 | TTS pool acquire（剩 2，2 个 refill 在飞） |

**评价**：一次用户讲话触发了 **2 次 LLM 调用**，第一次浪费。原因：STT FINAL 比框架的 `transcript_timeout=2.0s` 慢。

### Round 3 — 用户："你到底是谁。"（**双 FINAL + TTS 死池**）

| t | 事件 |
|---|---|
| 31:21.804 | INTERIM **"You"**（English 误识别！score 0.585） |
| 31:21.820 | INTERIM "你到底是" |
| 31:22.256 | FINAL "你到底是谁。"（end_time=26680） |
| 31:22.768 | 上一个 LLM (round 2) 被 cancel |
| 31:22.775 | **LLM 调用 #3** |
| 31:22.780 | TTS pool acquire（剩 1，2 个 refill 在飞） |
| 31:24.623 | LLM 200 OK（**1.85s**） |
| 31:24.976 | aggregator flush（巨长一句："我明白你的顾虑。其实我是由深度求索公司创造的 AI 助手...") |
| **31:24.976** | ❌ **`BailianTTSError("WebSocket disconnected", recoverable=True)`** |
| 31:25.050 | 第二段 flush 也失败，同一条死 WS |
| 31:25.050 | aggregator log + 丢弃文本，**LLM 后续 token 永远不会被合成** |

**评价**：用户**完全听不到这一轮的回复**。被选中的 WS 是 30:49.559 warmup 时建立的——idle 了 **33 秒**，远超 dashscope 服务端默认 30s 空闲超时。

### Round 4 — 用户："你到底是谁？你说说话呀。"（**用户重发 + 又一次双 FINAL**）

| t | 事件 |
|---|---|
| 31:33.133 | VAD: start of speech（用户没听到回复，重问） |
| 31:33.712 | INTERIM **"That"**（English 误识别又一次） |
| 31:34.983 | INTERIM "你到底是谁你说" |
| 31:35.080 | VAD: end of speech |
| 31:36.544 | INTERIM "你到底是谁？你说说话呀" |
| **31:37.081** | ⚠️ `final transcript not received after timeout` → 晋级 INTERIM |
| 31:37.099 | **LLM 调用 #4**（无句号版本） |
| 31:37.122 | Bailian 真正 FINAL "你到底是谁？你说说话呀。"（带句号） |
| 31:37.632 | LLM #4 cancel |
| 31:37.643 | **LLM 调用 #5**（带句号版本） |
| 31:39.652 | LLM 200 OK |
| 31:39.708 | aggregator flush "我理解你的怀疑！" |
| 31:40.483 | agent_state: thinking → speaking（这次池子有新鲜连接，TTS 成功） |

**全局统计**：4 轮用户对话 → **5 次 LLM 调用，其中 3 次被 cancel**；1 次 TTS 整轮失败。

---

## 第 1 问：警告与错误的根因（按严重度）

### 1A ❌ TTS WebSocket disconnected（最严重，用户能感知）

**症状**：[`bailian/tts_client.py:146`](eidolon/livekit/plugins/tts/bailian/tts_client.py#L146) 抛 `BailianTTSError("WebSocket disconnected", recoverable=True)`，aggregator [`_aggregator.py:208-212`](eidolon/livekit/plugins/tts/_aggregator.py#L208) 只 `logger.exception()` 就丢弃文本。

**三个叠加成因**：

1. **池连接零保活** — [`_pool.py`](eidolon/livekit/plugins/tts/_pool.py) 的 `TTSConnectionPool` 把 warmup 后的 WS 放进 `asyncio.Queue` 就再也不动。没有 ping/heartbeat、没有 max-idle 驱逐、没有 acquire 时 `ws.closed` 检查。33s 前 warmup 的 WS 被 acquire 时已经被 dashscope 服务端关掉。
2. **dashscope 服务端约 30s 空闲超时**（aiohttp ping 间隔默认不会主动开，要看服务端是否发 keep-alive 协商）。
3. **`recoverable=True` 是装饰品** — 全代码搜索 `e.recoverable` 只在 [`tts.py:383-391`](eidolon/livekit/plugins/tts/bailian/tts.py#L383) 出现一次，且**条件是"已在关闭流程或 exit_event 设置"，不是用来重试**。整个项目没有一处看 `recoverable` 后从池里换一条新连接重发。

**用户行为后果**：当前 round 整轮无声 → 用户重问 → 双 FINAL → 双 LLM。

### 1B ⚠️ `final transcript not received after timeout` → 双 LLM 调用

**症状**：framework 在 [`livekit/agents/voice/audio_recognition.py:699-731`](.venv) 等 STT FINAL，默认 `transcript_timeout=2.0s`；超时后把最新 INTERIM 强制当 FINAL，触发一次 LLM；几百毫秒后真正 FINAL 又到，触发第二次 LLM、第一次被 cancel。

**根因**：
- Bailian FunASR FINAL 比 VAD-end **晚 0.4–1.7s**，长句子稳定 > 2.0s（Round 4 长句晚到 2.0s）。
- framework 的 `transcript_timeout` 在框架里**硬编码 2.0s**，没暴露 env 旋钮，只能通过 monkey-patch（项目已有 [framework_patches.py](eidolon/livekit/agent/framework_patches.py)）或子类化 `AgentSessionOptions` 来改。
- 项目本身的 `AGENT_FALSE_INTERRUPTION_TIMEOUT=6.0` 是**另一个**机制（误打断恢复），跟这里无关。

**用户感知**：每次说长句子都触发 1 次"白调用"的 LLM。LLM 计费翻倍，agent_state 在 thinking↔listening 来回跳。

### 1C ⚠️ `_SegmentSynchronizerImpl.playback_finished called before text/audio input is done` + `called more times than playback segments`

**症状**：欢迎语播放结束时（30:55）warning 配对出现。

**根因**：framework 的 `TranscriptSynchronizer` 跟项目的 `DuckingMixer` 串联时，`playback_finished` 计数路径有重复触发——DuckingMixer 自己也会汇报"播完了"，加上 RoomIO 原生 sink 也汇报一次。无业务影响（warning-only），但说明输出链路有不必要的双触发。

### 1D ⚠️ `resume_false_interruption is enabled, but the audio output does not support pause, ignored` — **你配的 6.0s 是死代码**

**症状**：session 启动时打一次的 warning。表面看像 informational，但**它意味着你 env 里 `AGENT_FALSE_INTERRUPTION_TIMEOUT=6.0` 完全没生效**。

#### 代码路径（确认过的）

1. ✅ 6.0s 配置**确实**被项目正确地传给了 framework：
   - [`common/config.py:111`](eidolon/livekit/common/config.py#L111) `false_interruption_timeout: float | None = 6.0`
   - [`agent/server.py:169`](eidolon/livekit/agent/server.py#L169) → `StreamingPipeline(false_interruption_timeout=cfg.behavior.false_interruption_timeout)`
   - [`agent/streaming.py:216-222`](eidolon/livekit/agent/streaming.py#L216) → `AgentSession(turn_handling={"interruption": {"false_interruption_timeout": self._false_interruption_timeout, ...}})`
2. ❌ Framework 在 [`voice/agent_session.py:832-840`](.venv) 启动时检查 `self.output.audio.can_pause`，发现是 `False`，打出这条 warning **并把整个 false-interruption 恢复机制 ignored**。
3. ❌ `pause=False` 来自 [`agent/ducking.py:119`](eidolon/livekit/agent/ducking.py#L119) — DuckingMixer 显式声明不支持 framework 那种 pause/resume 协议。

#### 项目自己有一套替代机制，但**和你的 6.0s 没有关系**

DuckingMixer 内部走的是 fade+buffer：

- `duck()` 进入 SUSPENDED → 50ms fade-out → 之后的 TTS 帧进入 `_buffer` 而不外发
- `unduck()` 触发 → 200ms fade-in → 后续 `capture_frame` 会 `_drain_buffer()` 把 SUSPENDED 期间 buffer 的帧补播出去（[`ducking.py:218-241`](eidolon/livekit/agent/ducking.py#L218)、`_drain_buffer` 在 335-358 行）。
- 触发 unduck 的窗口是 **`duck_suspend_timeout_sec=0.8s`**（在 [`eot/config.py`](eidolon/livekit/plugins/eot/config.py)），由 [`streaming.py:_duck_suspend_timeout_fallback`](eidolon/livekit/agent/streaming.py) 兜底——**和你那个 6.0s 完全独立、没有任何继承关系**。

#### 实际"坏影响"

| 你以为的行为 | 真实行为 |
|---|---|
| 用户哼一声 ≤ 6s 后没继续 → agent 接着把原来那句讲完 | 用户哼一声 → 0.8s 后 DuckingMixer timeout unduck（不管你说 6s 还是 600s） |
| 6.0s 内多说几个字也允许 agent 软恢复 | 超过 0.8s 直接默认 unduck，根本到不了 6s |
| `_buffer` 把 SUSPENDED 期间的 TTS 帧补播 | 日志里**每次都是 `buffered=0 frames (0.000s)`** —— framework 在更上层已经把 TTS 出帧停了，buffer 里其实没东西 |

**结论：你配的 6.0s 是字面意义的死代码**。真正决定打断恢复窗口的是 `duck_suspend_timeout_sec`（默认 0.8s）。要想"误打断 6 秒内自动恢复"成立，要么：

- (A) 把 `duck_suspend_timeout_sec` 调大到 6s（最简单，但 0.8s 是经过调优的体感值，调到 6s 会让用户感觉 agent 反应慢）；
- (B) 让 DuckingMixer 实现 framework 的 `pause()/resume()` 协议（`pause=True`），让 framework 的 6.0s 机制真正接管（**是 P3 推荐项**，工作量较大但是"对齐预期"的正确做法）。

DuckingMixer 的 buffer/replay 代码**写了但用不上**——因为 framework 在更高层已经在 interrupt 时停 TTS 出帧，到 mixer 这里时帧已经没生成。要让 buffer 真正有内容补播，得让 DuckingMixer 成为 pause 协议的承担者，TTS 继续往 mixer 里灌帧、mixer 自己暂停外发。这正是 P3 的设计意图。

### 1E ⚠️ `empty text in latest sentence`（4 次）

**症状**：每个新 utterance 的第一条 STT 消息都是空文本。

**根因**：Bailian FunASR 每个新 sentence_id 的第一条 `result-generated` 服务端只送框（begin_time、空 sentence）。无害。

### 1F 🔍 DEBUG `flush audio emitter due to slow audio generation`（2 次）

**症状**：framework 调度认为 TTS 出帧慢于实时。

**根因**：Bailian TTS 首字 0.6-1.5s 才出第一帧 pcm；framework 在还没收到第一帧前先 flush 一次。属于正常的流式 TTS 等待，**不是 bug**，但暗示**长句单段合成**首字延迟较高（Round 3 的巨长一句更明显）。可通过减小 `BAILIAN_TTS_AGGREGATOR_HARD_MAX_CHARS=80` 缩短单段提早送 TTS。

### 1G 🔍 中英文 STT 误识别（"You"、"That"）

**症状**：开口前 100-200ms 的杂音被 Bailian 识成英文短词。

**根因**：Bailian 服务端**默认 auto language detection**。我们在 env 里设了 `BAILIAN_STT_LANGUAGE=zh`，但需要确认 [`bailian/stt_stream.py`](eidolon/livekit/plugins/stt/bailian/stt_stream.py) 的 `run-task` payload 是否把 `language` 字段真正传给服务端，还是只在客户端记录。Round 3/4 都出现，干扰短句的 EOT 评分。

---

## 第 2 问：哪些标记/流程没有开启

| 机制 | 状态 | 关键文件 |
|---|---|---|
| **TTS 池保活（ping/heartbeat）** | ❌ 未实现 | `_pool.py` — `TTSConnectionPool` 无周期 ping、无 max-idle 驱逐 |
| **TTS 池 acquire 时的 liveness 检查** | ❌ 未实现 | `_pool.py:241` `get_nowait()` 直接弹出，不看 `ws.closed` |
| **`BailianTTSError(recoverable=True)` 的重试** | ❌ 未实现 | `bailian/tts.py:383` 只在退出流程里看 recoverable，不重发 |
| **framework `resume_false_interruption`（误打断自动续播）** | ❌ 被禁用 | `ducking.py:119` `pause=False`，framework 检查后 warning 并忽略；项目改用 DuckingMixer 自己的软恢复（这部分**是**启用的） |
| **STT `transcript_timeout` 可配置化** | ❌ 框架硬编码 2.0s | `audio_recognition.py:701` 没 env 旋钮 |
| **Bailian STT `language=zh` 是否传给服务端** | ⚠️ 待确认 | `bailian/stt_stream.py` 的 run-task payload |
| **`agent_state != speaking` 时跳过 ducking** | ❌ 没有这个守卫 | `streaming.py:880-918` `_duck_and_arm_timeout()` 无条件 duck |

---

## 第 3 问：打断行为为什么都不符合预期

日志里**每次** duck 都是 `reason=timeout action=unduck`，没有一次真正 cancel agent 的发言。

### 原因 1：VAD 置信度门限 0.55 过严

[`eidolon/livekit/plugins/eot/impl/eot_policy.py:295-312`](eidolon/livekit/plugins/eot/impl/eot_policy.py#L295) 的 `MinSpeakingDurationPolicy`：

```
if recent_avg_vad_confidence() < min_avg_vad_confidence (默认 0.55):
    return cut=False  # 早于 EOT 评分检查
```

日志实证：Round 2 用户讲 "我不相信你啊" — EOT score 0.654（>0.55 中段），但 VAD avg=0.42 → cut 被否决。**EOT 评分根本没机会决定是否打断**。

### 原因 2：duck 在 `agent_state=listening` 时也触发

Round 1 的用户讲话发生时 agent_state 已经是 `listening`（没有正在播的音频），却依旧 `NORMAL→SUSPENDED`，再 timeout-unduck。**纯浪费 fade-out/fade-in 计算**，对用户没有感知，但浪费 CPU 和 200ms+50ms 的淡入淡出窗口。

[`streaming.py:880-918`](eidolon/livekit/agent/streaming.py#L880) `_duck_and_arm_timeout()` 在 `user_state: listening→speaking` 触发时无条件 `mixer.duck()`，没有检查 `self._state == SPEAKING`。

### 原因 3：framework 的 `resume_false_interruption` 不可用

如 1D 所述：6.0s 的误打断恢复窗口实际没开。**用户真打断了 agent 又改主意时（VAD 起又落、没说话），无法自动恢复 agent 原来的发言**——只能靠 DuckingMixer 的 0.80s timeout-unduck 兜底（resume_thr=0.20），且不会真正"续播"被吞掉的句子，只是淡入回到正常音量。

### 综合结果

- "用户开口" → 立即 duck（哪怕 agent 没在说）
- 0.80s 后 → 默认 unduck（要么 EOT 评分中段、要么 VAD 太低否决了 cancel）
- 真要打断 → 需要 EOT score ≥ 0.7 **且** VAD avg ≥ 0.55 — 中文短句很难同时满足
- 误打断（用户开口又闭嘴） → 没有自动续播，被 fade-out 吞掉的那段 audio 永久丢失

---

## 彻底修复方案（**零 framework 修改、零 monkey-patch**）

修复全部在我们自己的代码里。每一项都走 framework 的 public API 或在我们自己的模块内完成。

### F1 — STT FINAL 超时（修 1B：消除双 LLM 调用）

**根因定位**：[`streaming.py:583`](eidolon/livekit/agent/streaming.py#L583) 调 `self._session.commit_user_turn()` **不传参**。framework 的 [`agent_session.py:1219`](.venv/.../voice/agent_session.py) `commit_user_turn(transcript_timeout=2.0, ...)` 是 public API，参数完全开放，我们之前没传所以吃了 framework 默认 2.0s。

**修法**：
1. [`common/config.py`](eidolon/livekit/common/config.py) `AgentBehaviorConfig` 加字段 `stt_commit_transcript_timeout: float = 5.0`（env: `AGENT_STT_COMMIT_TIMEOUT`），默认 5.0s 覆盖中文长句的 FINAL 延迟分布（实测最长 2.0s+，留余量）。
2. [`streaming.py:113`](eidolon/livekit/agent/streaming.py#L113) 构造函数收这个值。
3. [`streaming.py:583`](eidolon/livekit/agent/streaming.py#L583)：
   ```python
   self._session.commit_user_turn(
       transcript_timeout=self._stt_commit_transcript_timeout,
       stt_flush_duration=2.0,
   )
   ```
4. `deploy/livekit-channel.env.template` + `.livekit-channel.env` 加上对应 env 注释。

**为什么不是 monkey-patch**：因为 framework 早就把 timeout 暴露为 `commit_user_turn` 的关键字参数。我们只是把它从代码里硬调到 env 配置。

**预期收益**：每轮双 LLM 调用减为 1 次；LLM 成本/延迟降一半；agent_state 不再 thinking↔listening 反复横跳。

### F2 — TTS 池保活 + recoverable 重试（修 1A：消除整轮无声）

**根因定位**：池在 [`_pool.py`](eidolon/livekit/plugins/tts/_pool.py) 内零保活，[`bailian/tts.py:381`](eidolon/livekit/plugins/tts/bailian/tts.py#L381) 的 `emit_segment` 抓到 `recoverable=True` 后**只在退出流程里**检查，没用 recoverable 的语义重试。

**修法**（两步走，独立且互补）：

1. **池侧 — 增加 freshness 守卫**（[`_pool.py`](eidolon/livekit/plugins/tts/_pool.py)）：
   - 在 `_ready` 队列里改存 `(conn, created_at_monotonic)` 元组而不是裸 conn。
   - `acquire()` 弹出后检查 `now - created_at > max_idle_sec`（新增字段，默认 25s，env: `BAILIAN_TTS_POOL_MAX_IDLE_SEC`）。若 stale，立刻 `dispose(conn)`、`schedule_refill(1)` 一次、再走 inline-warm 路径返回新连接。
   - 触发 `refill_one` 是已经有的代码 ([`_pool.py:_schedule_refill`](eidolon/livekit/plugins/tts/_pool.py))，复用。
   - **为什么用 age 而不用 ping**：dashscope 服务端默认 30s 空闲超时；age 阈值 25s 给一点安全余量。Ping 也行但更复杂（aiohttp WSClient 的 heartbeat 协商不稳定）。

2. **客户端 — 真正消费 `recoverable=True`**（[`bailian/tts.py:381 emit_segment`](eidolon/livekit/plugins/tts/bailian/tts.py#L381)）：
   ```python
   try:
       await client.send_continue(cleaned)
   except BailianTTSError as e:
       if e.recoverable and not self._exit_event.is_set() and not self._retried_once:
           self._retried_once = True
           # release dead conn, acquire a fresh one, retry once
           new_client = await self._pool.acquire()
           self._client = new_client
           await new_client.send_continue(cleaned)
       else:
           raise
   ```
   - 关键约束：**只重试一次**（避免无限循环）；重试失败照常抛，让 `_aggregator._flush_locked` 看到。
   - `self._retried_once` 在每个新 synthesize stream 开始时 reset。

3. **Aggregator — 失败语义降级**（[`_aggregator.py:208`](eidolon/livekit/plugins/tts/_aggregator.py#L208)）：
   - 当前 `logger.exception()` 后吞掉。改成：把异常 set 到 stream 的 `error_event` 上（如果有的话，否则新增），让上层 `synthesize` task 能感知"这一轮 TTS 失败了"，至少在 metrics 里能体现，不要静默吞。

**预期收益**：33s 老连接被 acquire 时立刻被换掉，根本不到 `send_continue`；即使漏过，`recoverable=True` 也会触发一次自动重试。整轮无声彻底消失。

### F3 — VAD 阈值 + state 守卫（修第 3 问的 #1、#2）

**根因定位**：
- [`eot/config.py:60`](eidolon/livekit/plugins/eot/config.py#L60) `min_avg_vad_confidence = 0.55` 太严，中文短句很难达到。
- [`eot/impl/eot_policy.py:295-312`](eidolon/livekit/plugins/eot/impl/eot_policy.py#L295) 的 `MinSpeakingDurationPolicy` 早于评分策略 short-circuit。
- [`streaming.py:_duck_and_arm_timeout`](eidolon/livekit/agent/streaming.py) 无 state 守卫。

**修法**：
1. **暴露 VAD 阈值为 env**：[`eot/config.py`](eidolon/livekit/plugins/eot/config.py) 把 `min_avg_vad_confidence` 改成 `field(default_factory=lambda: float(os.environ.get("EIDOLON_EOT_MIN_VAD_CONFIDENCE", "0.40")))`。**默认从 0.55 改 0.40**（覆盖实测 avg=0.42 的真打断场景）。template/env 文件加注释。
2. **State 守卫**：[`streaming.py:_duck_and_arm_timeout`](eidolon/livekit/agent/streaming.py) 函数入口：
   ```python
   if self._state != PipelineState.SPEAKING:
       logger.debug("duck skipped: agent not speaking (state=%s)", self._state)
       return
   ```
   agent 没在说话就根本不进 SUSPENDED，避免日志里那种 "duck on listening" 的浪费。

**预期收益**：用户能真的打断 agent；agent 处于 listening 时用户开口不再触发不必要的 fade-out/fade-in 计算。

### F4 — DuckingMixer 实现 framework pause 协议（修 1D：让 `false_interruption_timeout=6.0` 真正生效）

**根因定位**（见第 1D 章节展开）：framework 的 false-interruption 自动恢复要求 `audio.can_pause=True`，但 [`ducking.py:119`](eidolon/livekit/agent/ducking.py#L119) 写死 `pause=False`，framework warning 并 ignored 6.0s，最终窗口只剩 DuckingMixer 自己 0.8s 的 `duck_suspend_timeout_sec`。

**修法**：
1. **改 capability**：[`ducking.py:119`](eidolon/livekit/agent/ducking.py#L119) `pause=False` → `pause=True`。
2. **实现 pause/resume 协议方法**：[`ducking.py`](eidolon/livekit/agent/ducking.py) DuckingMixer 加：
   ```python
   def pause(self) -> None:
       """Framework-driven pause. Stops draining frames to inner sink, 
       continues buffering inbound TTS frames so they can be drained on resume."""
       # state transition: NORMAL → PAUSED_BY_FRAMEWORK
       # similar to SUSPENDED but no fade (framework will handle resume timing)
       ...

   def resume(self) -> None:
       """Framework-driven resume. Drain buffered frames with fade-in."""
       ...
   ```
3. **状态机对齐**：现在 DuckingMixer 已经有 NORMAL/SUSPENDED 两态，加 PAUSED_BY_FRAMEWORK 第三态。三态互斥；当用户真打断（高 EOT 分）时 DuckingMixer 自己进 SUSPENDED；当 framework 调 pause 时进 PAUSED_BY_FRAMEWORK。两者不会同时发生（framework 不会在 EOT 高分时调 pause——它信任已经被中断了）。
4. **删/调整 0.8s timeout 兜底**：[`streaming.py:_duck_suspend_timeout_fallback`](eidolon/livekit/agent/streaming.py) 现在的 0.8s 兜底应该和 framework 的 6.0s false-interruption 不冲突。建议保留 0.8s 作为 "EOT 评分中段 / 拿不到判定时" 的最短兜底，让 framework 的 6.0s 接管"高确信度是 false interrupt"的延迟恢复。

**风险**：DuckingMixer 与 framework false-interruption 的状态机有 overlap，必须把两者的进入/退出条件理清。建议先在 dev env 跑 50 轮对话回归再上。

**预期收益**：你 env 里配的 6.0s 真正生效；用户起一个嗯/啊瞬间又闭嘴时，agent 从原句继续往下说，体感连贯。

### F5 — 锦上添花（不阻塞主流程）

- **验证 Bailian STT `language=zh` 服务端传参**：读 [`bailian/stt_stream.py`](eidolon/livekit/plugins/stt/bailian/stt_stream.py) 的 `run-task` 构造，确认 `language` 真的在 `parameters` 字段里发出去（log 里 #1 message 看不到 language 字段，怀疑没传）。修复后消除 "You" / "That" 类英文误识别。
- **`playback_finished called more times` 双触发**：[`ducking.py`](eidolon/livekit/agent/ducking.py) 与 framework `TranscriptSynchronizer` 都汇报；查清楚谁先谁后，让 DuckingMixer 不重复汇报。Warning-only。
- **TTS warmup 5.1s 优化**：分离"启动池"和"运行时目标池"——新增 `BAILIAN_TTS_POOL_SIZE_BOOTSTRAP=3`（启动时同步并行 warm 的连接数），保留 `BAILIAN_TTS_POOL_SIZE=4` 作为运行时目标（启动后后台慢慢补到 4）。冷启动从 5.1s 降到 ~3.5s（3 个并行 warmup ≈ 单个 1.3s × 1 批），首句体验立刻可用。2 太少（用户第一次说话还没说完池就空了），3-4 是合理下限。
- **EOT/Duck timeout 暴露为 env**：`EIDOLON_DUCK_SUSPEND_TIMEOUT_SEC`、`EIDOLON_EOT_DUCK_EARLY_CANCEL_SCORE`、`EIDOLON_EOT_DUCK_EARLY_RESUME_SCORE`——方便部署期调参不改代码。

## 实施顺序建议

按 ROI 与互相依赖来排：

1. **F1** 先做（5 分钟级改动，立刻见效，消除双 LLM）
2. **F2** 紧跟（半天工作量，消除整轮无声，是用户最严重的感知问题）
3. **F3** 同步可做（独立模块，1 小时级别）
4. **F4** 单独立项（设计 + 实现 + 回归，1-2 天工作量）
5. **F5** 跟在主线之后

---

## Verification（实施后怎么验证）

1. **F1 验证**：跑同样的 4 轮对话，grep `final transcript not received after timeout` 应消失；统计 `agent_state: thinking → listening` 后紧接着又 `listening → thinking` 的次数：改前 ~3 次/4 轮，改后应为 0；LLM 调用次数应是 4（每轮 1 次），不再是 5。
2. **F2 验证**：agent 启动后挂 60s 不说话再喊一句长话。日志里应看到 `pool.acquire: replaced stale conn` 或类似，不再看到 `WebSocket disconnected` 然后丢段。若 stale 守卫漏过，recoverable retry 应在 log 里留下 `pool.acquire: stale conn replaced via recoverable retry`。
3. **F3 验证**：(a) agent 长答时插一句"停一下"，改前 100% timeout-unduck，改后期望看到 `duck resolved reason=early_cancel action=cancel`；(b) agent 处于 listening 时用户开口，日志不再有 `duck NORMAL→SUSPENDED`，应看到 `duck skipped: agent not speaking`。
4. **F4 验证**：用户起头 "嗯..."（约 0.3s）后闭嘴，agent 原句被 framework pause；6s 内无后续音 → framework 调 DuckingMixer.resume → 用户听到 agent 把原句**接着**讲完（而不是 fade-out 永久丢失）。同时 framework 启动 warning `resume_false_interruption ignored` 应消失。

## 不打 patch 的保证

所有修复点都在以下我们自己的模块内：

- [`eidolon/livekit/common/config.py`](eidolon/livekit/common/config.py)（加 env 字段）
- [`eidolon/livekit/agent/streaming.py`](eidolon/livekit/agent/streaming.py)（传 commit timeout、state 守卫）
- [`eidolon/livekit/plugins/tts/_pool.py`](eidolon/livekit/plugins/tts/_pool.py)（freshness 守卫）
- [`eidolon/livekit/plugins/tts/bailian/tts.py`](eidolon/livekit/plugins/tts/bailian/tts.py)（recoverable retry）
- [`eidolon/livekit/plugins/tts/_aggregator.py`](eidolon/livekit/plugins/tts/_aggregator.py)（失败语义降级）
- [`eidolon/livekit/plugins/eot/config.py`](eidolon/livekit/plugins/eot/config.py)（VAD 阈值 env 化）
- [`eidolon/livekit/agent/ducking.py`](eidolon/livekit/agent/ducking.py)（pause/resume 协议）
- `deploy/livekit-channel.env*`（env 旋钮暴露）

`eidolon/livekit/agent/framework_patches.py` **不动**，site-packages 下的 framework 代码**不动**。framework 自己提供的所有 public API（`commit_user_turn(transcript_timeout=...)`、`AudioOutputCapabilities(pause=True)`、`turn_handling.interruption.false_interruption_timeout`）都被正确使用——之前只是没把入参传齐。

---

## 关键文件索引

- [`eidolon/livekit/plugins/tts/_pool.py`](eidolon/livekit/plugins/tts/_pool.py) — TTS 池（无保活）
- [`eidolon/livekit/plugins/tts/bailian/tts.py:381`](eidolon/livekit/plugins/tts/bailian/tts.py#L381) — `emit_segment`，recoverable 错误处理点
- [`eidolon/livekit/plugins/tts/bailian/tts_client.py:122,146`](eidolon/livekit/plugins/tts/bailian/tts_client.py#L122) — `send_continue` 抛 disconnect
- [`eidolon/livekit/plugins/tts/_aggregator.py:208`](eidolon/livekit/plugins/tts/_aggregator.py#L208) — `_flush_locked` 吞掉异常
- [`eidolon/livekit/agent/framework_patches.py`](eidolon/livekit/agent/framework_patches.py) — 框架补丁机制（用来调 `transcript_timeout`）
- [`eidolon/livekit/agent/streaming.py:880-918`](eidolon/livekit/agent/streaming.py#L880) — `_duck_and_arm_timeout`（缺 state 守卫）
- [`eidolon/livekit/agent/ducking.py:119`](eidolon/livekit/agent/ducking.py#L119) — `pause=False`
- [`eidolon/livekit/plugins/eot/impl/eot_policy.py:295-312`](eidolon/livekit/plugins/eot/impl/eot_policy.py#L295) — VAD 置信度门限
- [`eidolon/livekit/plugins/eot/config.py:60`](eidolon/livekit/plugins/eot/config.py#L60) — `min_avg_vad_confidence=0.55`
- `.venv/lib/python3.12/site-packages/livekit/agents/voice/audio_recognition.py:699-731` — framework `transcript_timeout` 硬编码 2.0s
- `.venv/lib/python3.12/site-packages/livekit/agents/voice/agent_session.py:832-840` — `resume_false_interruption` can_pause 检查
