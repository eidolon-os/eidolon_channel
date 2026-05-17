# G17–G23：流畅打断 + 输出状态机重构（分阶段）

## 0. TL;DR

**目标**：把"等置信度爬坡"的级联换成"立刻 MUTED + 首信号决策（≤500ms）"，对齐到业界顶级体验；同时把输出层状态机从 `DuckingMixer`（含 SUSPENDED/buffer/drain）收敛成 `OutputController`（3 态），代码结构清晰。

**策略**：分两阶段，**用户感知层（Phase 1）先 ship、工程清理层（Phase 2）后跟进**。Phase 1 完成后体验已对齐顶级，Phase 2 是 pure cleanup，每步风险都低。

```
Phase 1（2-3 天）：体验对齐到顶级
  G20 → ChatMessage.create AttributeError 修复
  G17a → drain stale buffer bug 修复（cancel 路径直接丢，不 drain）
  G18a → 触发器从"等稳定值"换"首信号"+ 500ms 硬预算
  G21 → interrupted context 数据源换 _pushed_text
  G22 → played_seconds reset 时机换到 agent_state→speaking
  → 真机回归 20 轮

Phase 2（2-3 天，独立可延后）：架构清晰度
  G17b → DuckingMixer 重命名 OutputController
  G18b → 抽出 InterruptDecider 独立模块
  G18c → EOT 角色削减（只保留用户回合结束检测）
  G17c → 删除 SUSPENDED 状态、_buffer、_drain_buffer、长 fade
  G_env → env 清理（删 EIDOLON_DUCK_*_SCORE / TIMEOUT / AGENT_FALSE_INTERRUPTION_TIMEOUT）

Phase 3（可选，单独立项）：性能优化
  G23 → 评估 Pipecat Smart Turn v3 替换/补强 FireRed EOT 用户回合检测
```

## 1. 业界对照（决策依据）

| 产品 | 打断时 TTS 处理 | 假打断恢复 | 是否有专用 ML 模型 |
|---|---|---|---|
| OpenAI Realtime API | `response.cancel` 立即丢 in-flight | server-VAD threshold 前置过滤；无 resume | 端到端模型一体化 |
| LiveKit Adaptive | 决策完才停（agent 多说 100-200ms） | `resume_false_interruption` 要 `audio.can_pause=True` | **专用音频 CNN**（拒 51% 假打断），闭源 |
| Pipecat default | InterruptionFrame 立即 reset + clear buffer | 无；issue #3985 在讨论 graceful release | 无（barge-in 端）；用 Smart Turn v3 做 input 侧 |
| **G17-G22（我们）** | 立刻 MUTED，500ms 内决策 cancel/rollback | TTS 上游未取消，rollback 后从当前位置继续 | 无（用 STT INTERIM 作语义信号 + backchannel filter） |

## 2. 误触发处理双维度对比

**总体验 = (1 - 前置过滤拒绝率) × 单次误触发代价**

| | 前置过滤手段 | 拒绝率估计 | 单次代价 | 综合 |
|---|---|---|---|---|
| OpenAI Realtime | VAD threshold（钝）| 30% | 巨大（丢回复） | 大 |
| LiveKit Adaptive | **音频 CNN** | **70%** | 中（要 resume 协议生效） | 中小 |
| Pipecat default | min_words / min_duration | 40% | 巨大（丢回复） | 大 |
| **我们** | backchannel filter + STT INTERIM 语义 | 55% | **小（500ms gap + 续播）** | **小** |

**定位**：
- 我们与 LiveKit Adaptive 大致并列第一，路径不同
- LiveKit 靠"拒绝得准"（少发生），我们靠"代价小"（发生了也无感）
- 进一步胜出 LiveKit 需要**自研 CNN 前置过滤**，超出本 plan 范围

## 3. CNN 前置过滤现状评估

调研结论：**没有开箱即用的"interrupt vs noise"预训练模型**。

| 候选 | 解决问题 | 适用性 |
|---|---|---|
| Pipecat Smart Turn v3 | 用户回合结束检测（input 侧 EOT） | ✅ 但**不是 barge-in**；本 plan 不引入，列入 G23 |
| VAP (Ekstedt) | 双方未来 2s 语音活动预测 | ⚠️ 研究级，需改造，R&D 工作 |
| LiveKit Adaptive CNN | 真/假打断专用 | ❌ 闭源，不可用 |

**本期决策**：不引入 CNN 前置过滤。500ms 决策窗 + 续播质量在没有专用模型的前提下已经是开源世界天花板。

## 4. 当前 5 秒延迟根因（澄清：不是级联慢，是 gates 等稳定值）

| 当前 gate | 实测耗时 | 原因 |
|---|---|---|
| VAD-起 → duck SUSPENDED | ~10ms | 本来就快 |
| Backchannel filter "嗯" | ~50ms | 本来就快 |
| **VAD avg 爬到 0.55** | **~1000ms** | 滑动平均需 ~1s 帧累积 |
| **EOT 评分进入 cut 区间** | **~1500ms** | 模型需看完整前缀才稳定 |
| **`duck_suspend_timeout_sec=0.8`** 软取消兜底 | **~800ms** | 没更早信号就硬等 |
| drain stale buffer（独立 bug） | ~2000ms | SUSPENDED→NORMAL 补播陈旧帧 |
| **合计** | **~5.3s** | |

级联本身不慢；慢在每个 gate"等到稳定/置信"。换成"首信号触发"即可压到 500ms 内。

## 5. 设计目标

| 指标 | 目标 |
|---|---|
| VAD-起 → MUTED 生效 | ≤30ms |
| 决策窗（MUTED → CANCELLED 或 NORMAL） | ≤500ms |
| 真打断后 TTS 帧到达用户 | 0ms（被丢弃） |
| 假打断恢复后 agent 继续说话 | TTS 从当前位置继续，不补播被静音段 |
| 业界对齐 | OpenAI / LiveKit Adaptive 主流 |

## 6. Phase 1 — 用户感知层（先 ship）

### G20：`ChatMessage.create` AttributeError 修复（1 行排雷）

**文件**：`eidolon/livekit/agent/streaming.py`

**问题**：当前代码调用 `ChatMessage.create(role=..., text=..., id=...)` 在 livekit-agents 1.5.x 中是 pydantic 模型直接构造，没有 `.create` 类方法。

**修复**：
```python
# 旧
msg = ChatMessage.create(role="user", text=..., id=...)
# 新
msg = ChatMessage(role="user", content=[text], id=...)
```

参照 site-packages 中 `ChatMessage` 实际字段调整。

**测试**：`test_chat_message_construction.py`

### G17a：drain stale buffer bug 修复（最关键的用户感知修复）

**文件**：`eidolon/livekit/agent/ducking.py`

**问题**：当 cancel 后 DuckingMixer 解除 SUSPENDED，`_drain_buffer()` 把 SUSPENDED 期间累积的陈旧 TTS 帧补播 → 与用户新讲话音频重叠（实测 2 秒）。

**修复**：
- `cancel()` 方法（或对应路径）显式调用 `self._buffer.clear()` 后再转 NORMAL
- 区分两条恢复路径：
  - 真打断 → `cancel_and_drop()`：清空 buffer，转 NORMAL（用户不应听到陈旧帧）
  - 假打断（soft unduck）→ 暂时保留 drain 行为，Phase 2 一并下线
- 加 metric：`drained_frames_after_cancel`（应恒为 0）

**测试**：`test_cancel_drops_buffer.py`

### G18a：触发器从"等稳定值"换"首信号" + 500ms 硬预算

**文件**：
- `eidolon/livekit/plugins/eot/impl/eot_policy.py`（移除 VAD 置信度短路）
- `eidolon/livekit/agent/streaming.py`（`_duck_and_arm_timeout` 改造）
- `eidolon/livekit/plugins/eot/config.py`（删 `duck_suspend_timeout_sec`）

**核心修改**：

1. **触发源换"首信号"**：
   - VAD：首条 prob>0.5 的帧即触发 mute（不要求 `min_avg_vad_confidence` 爬坡）
   - 删除 `MinSpeakingDurationPolicy` 中的 VAD 置信度短路逻辑

2. **决策依据换"首条 INTERIM 的 backchannel filter"**：
   - 不再等 EOT score 稳定多个 INTERIM
   - 首条 INTERIM 非 backchannel 且字符 ≥2 → cancel
   - 首条 INTERIM 是 backchannel → 等下一条
   - VAD-止 + 无 INTERIM → rollback

3. **500ms 硬预算**：
   - `EIDOLON_INTERRUPT_DECISION_MS=500`（新 env）
   - 超时仍未决 + VAD 仍在 → trust VAD，强制 cancel
   - 删除 `duck_suspend_timeout_sec` env 和 0.8s 软兜底

**测试**：
- `test_first_signal_trigger.py`：VAD prob 单次 0.6 立刻 mute
- `test_decision_deadline.py`：500ms 后仍无 INTERIM 但 VAD 在 → cancel
- `test_backchannel_delays_not_blocks.py`：首条 "嗯" 等下一条
- `test_rollback_no_drain.py`：VAD-止 + 无 INTERIM → rollback，buffer 不补播

### G21：interrupted context 数据源换 `_pushed_text`

**文件**：
- `eidolon/livekit/plugins/tts/bailian/tts.py`（暴露 `_pushed_text`）
- `eidolon/livekit/agent/streaming.py`（`_snapshot_interrupted_context` 改读源）

**问题**：当前 `_snapshot_interrupted_context` 从 `session.history` 取消息，但 in-progress TTS 文本还没回写到 history，所以经常取到上一轮甚至 welcome。

**修复**：
- `BailianSynthesizeStream` 维护 `_pushed_text: list[str]`，记录已 push 给 TTS 的所有 token
- `streaming.py` 的 `_snapshot_interrupted_context` 直接从 active synthesize stream 读
- 配合 `played_seconds` 估算"用户实际听到的前 N 字"

**测试**：`test_interrupted_context_pushed_text.py`

### G22：`played_seconds` reset 时机修正

**文件**：`eidolon/livekit/agent/ducking.py`

**问题**：当前 `_played_samples_this_turn` 在 `duck()` 入口 reset，同一轮多次 duck-unduck 会清零前面播放计数。

**修复**：
- reset 时机从 `duck()` 入口改为 `agent_state→speaking` 事件
- 新增 `on_agent_started_speaking()` 公开方法，由 `streaming.py` 在 `pipeline.state == SPEAKING` 时调用

**测试**：`test_played_seconds_reset_on_speaking.py`

### Phase 1 真机验收

20 轮多场景对话录音：
- 5 轮"用户真打断"（≤500ms 切断）
- 5 轮"用户 backchannel"（"嗯/啊"应不打断，500ms 内恢复）
- 5 轮"用户假打断"（开口又闭嘴，500ms 内恢复且无音频重叠）
- 5 轮"环境噪声"（关门/键盘）应不触发 mute 或 ≤500ms 恢复

指标：
- `interrupt_decision_ms` P50 ≤300ms、P95 ≤500ms
- `audio_overlap_seconds` 恒为 0
- `drained_frames_after_cancel` 恒为 0

## 7. Phase 2 — 架构清晰度（独立可延后）

### G17b：`DuckingMixer` → `OutputController` 重命名

**文件**：
- `eidolon/livekit/agent/ducking.py` → `output_controller.py`
- 类名 `DuckingMixer` → `OutputController`
- 引用更新：`streaming.py`、测试文件

**理由**：Phase 1 之后 SUSPENDED 已不被业务路径使用，类的语义已经变成"输出控制器"而非"音量调节器"，命名应反映现实。

### G18b：抽出 `InterruptDecider` 独立模块

**文件**：新建 `eidolon/livekit/agent/interrupt_decider.py`

**职责**：把 Phase 1 散落在 `streaming.py` 和 `eot_policy.py` 中的决策逻辑（VAD 起、首条 INTERIM 判定、backchannel filter、500ms 预算）收敛到一个模块。

**接口**：
```python
class InterruptDecider:
    def on_vad_start(self, agent_state) -> None
    def on_stt_interim(self, text: str) -> None
    def on_vad_end(self) -> None
    # 内部：500ms decision deadline timer
    # 输出：调用 OutputController.cancel() / rollback()
```

**测试**：5 个核心用例覆盖决策矩阵（cancel / rollback / deadline-trust-vad / backchannel-delays / state-guard）

### G18c：EOT 角色削减

**文件**：
- `eidolon/livekit/plugins/eot/impl/eot_policy.py`
- `eidolon/livekit/plugins/eot/config.py`

**修改**：
- EOT 模型继续运行，**只用于用户回合结束检测**（input pipeline，决定何时 `commit_user_turn`）
- 移除所有 EOT score → cut 决策路径
- `MinSpeakingDurationPolicy` 简化为单纯的"最短发声时长"过滤（不再短路 VAD 置信度）

### G17c：删除 SUSPENDED / `_buffer` / `_drain_buffer` / 长 fade

**文件**：`eidolon/livekit/agent/output_controller.py`

**修改**：
- 状态枚举从 NORMAL/SUSPENDED/CANCELLED 改为 NORMAL/MUTED/CANCELLED
- 删除 `_buffer` 队列、`_drain_buffer()` 方法
- `fade_in_ms=200` / `fade_out_ms=50` 改为统一 ~30ms anti-click 短斜坡
- `pause=False` capability（不再宣称支持 framework pause）

**测试**：
- `test_output_controller_state.py`：3 态转移矩阵
- `test_output_controller_discard.py`：MUTED 期间帧丢弃
- `test_output_controller_anticlick.py`：30ms 斜坡完整性
- 删除 `test_ducking_played_seconds.py` 中 SUSPENDED/fade/buffer 相关用例

### G_env：env 清理

删除（Phase 2 完成后已无读者）：
```bash
EIDOLON_EOT_MIN_VAD_CONFIDENCE      # F3 引入，VAD 爬坡 gate 下线
EIDOLON_DUCK_SUSPEND_TIMEOUT_SEC    # 软兜底下线
EIDOLON_EOT_DUCK_EARLY_CANCEL_SCORE # EOT 不参与 cut
EIDOLON_EOT_DUCK_EARLY_RESUME_SCORE # 同上
AGENT_FALSE_INTERRUPTION_TIMEOUT    # F4 framework resume 不再依赖
```

更新文档（`deploy/livekit-channel.env.template`、注释、README）。

## 8. Phase 3 — 性能优化（可选，单独立项）

### G23：评估 Pipecat Smart Turn v3 替换/补强 FireRed EOT

**背景**：调研发现 Smart Turn v3（Apache-2.0，HuggingFace `pipecat-ai/smart-turn-v3`）：
- Whisper Tiny backbone，8M 参数
- 8MB ONNX int8，CPU 推理 12ms
- 23 语言含中文
- 与 VAD 模型并行使用，不冲突

**评估范围**（非本期工作）：
1. 在我们的中文测试集上跑准确率对比（vs 当前 FireRed EOT）
2. 整合到 input pipeline 的工程成本
3. 与 FireRed pVAD 是否冲突
4. 综合 latency / 准确率 trade-off

**输出**：评估报告 + 决策（替换 / 补强 / 维持现状）。**不阻塞 Phase 1/2**。

## 9. 配置项变更总览

### 新增（Phase 1）

```bash
EIDOLON_INTERRUPT_DECISION_MS=500          # 决策窗硬上限
EIDOLON_INTERRUPT_MIN_INTERIM_CHARS=2      # 触发 cancel 的最少字符
```

### 删除（Phase 2）

```bash
EIDOLON_EOT_MIN_VAD_CONFIDENCE
EIDOLON_DUCK_SUSPEND_TIMEOUT_SEC
EIDOLON_EOT_DUCK_EARLY_CANCEL_SCORE
EIDOLON_EOT_DUCK_EARLY_RESUME_SCORE
AGENT_FALSE_INTERRUPTION_TIMEOUT
```

### 重要性提升

- `AGENT_AEC_WARMUP_DURATION`：现在是唯一防回声手段，默认 1.0s；回声明显的硬件可调至 2.0s

## 10. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 回声触发误 mute（移除 VAD 置信度 gate 后） | 依赖 LiveKit WebRTC AEC + `AGENT_AEC_WARMUP_DURATION`；真机环境跑 20 轮回归 |
| 环境噪声（TV/空调）触发误 mute | InterruptDecider 的 500ms 决策内若无 INTERIM 自动 rollback，单次误判代价 ≤500ms |
| Backchannel "嗯/啊" 触发 mute 后用户不继续 | 首条 INTERIM 为 backchannel → 等下一条 → VAD-止前没下一条 → rollback。用户感觉"agent 顿了一下"≤500ms |
| Phase 1 后 SUSPENDED 路径仍被部分代码引用 | Phase 2 G17c 之前先加 deprecation log，验证生产无访问后再删 |
| TTS in-flight 帧丢弃竞态 | asyncio 单线程原子标志，无需锁 |

## 11. 测试与回归

### Phase 1 单元测试
- `test_chat_message_construction.py`（G20）
- `test_cancel_drops_buffer.py`（G17a）
- `test_first_signal_trigger.py`、`test_decision_deadline.py`、`test_backchannel_delays_not_blocks.py`、`test_rollback_no_drain.py`（G18a）
- `test_interrupted_context_pushed_text.py`（G21）
- `test_played_seconds_reset_on_speaking.py`（G22）

### Phase 1 真机回归
- 20 轮多场景录音对比（见 6.6）
- 指标：`interrupt_decision_ms` P50/P95、`audio_overlap_seconds`、`drained_frames_after_cancel`

### Phase 2 单元测试
- `test_output_controller_state.py`、`test_output_controller_discard.py`、`test_output_controller_anticlick.py`（G17c）
- `test_interrupt_decider_*`（G18b，5 个核心用例）
- 删除：`test_ducking_*` 中走 SUSPENDED/fade/buffer 路径的用例

### Phase 2 集成测试
- 全套 Phase 1 真机回归重跑，体验应保持一致（架构清理不应改变用户感知）

## 12. 不打 patch 保证

修改全在我们自己的模块内：

**Phase 1**：
- `eidolon/livekit/agent/streaming.py`
- `eidolon/livekit/agent/ducking.py`
- `eidolon/livekit/plugins/eot/impl/eot_policy.py`
- `eidolon/livekit/plugins/eot/config.py`
- `eidolon/livekit/plugins/tts/bailian/tts.py`
- `eidolon/livekit/common/config.py`
- `deploy/livekit-channel.env*`

**Phase 2**：
- `eidolon/livekit/agent/output_controller.py`（rename from ducking.py）
- `eidolon/livekit/agent/interrupt_decider.py`（新）
- 上述 Phase 1 文件的进一步清理

`eidolon/livekit/agent/framework_patches.py` 不动，site-packages 不动。framework 所有 public API（`commit_user_turn(transcript_timeout=...)`、`AudioOutputCapabilities`、`turn_handling`）继续走标准接口。

## 13. 实施顺序

1. **G20** — 1 行排雷
2. **G17a** — drain bug（最高用户感知收益）
3. **G18a** — 首信号 + 500ms 预算（核心体验改善）
4. **G21、G22** — 配套上下文与计数修复
5. **Phase 1 真机回归** — 验证 ≤500ms 决策、零音频重叠
6. **Phase 2 起步** — G17b 重命名（纯 cosmetic，零行为变化，验证测试管线）
7. **G18b** — 抽出 InterruptDecider
8. **G18c** — EOT 角色削减
9. **G17c** — 删除死代码（SUSPENDED/buffer/drain/长 fade）
10. **G_env** — env 清理 + 文档更新
11. **Phase 2 集成回归** — 真机重跑确认无回归
12. **G23**（可选，单独立项） — Smart Turn v3 评估

---

**起点**：G20（1 行修复，5 分钟）
**Phase 1 完成里程碑**：真机回归通过，体验对齐顶级
**Phase 2 完成里程碑**：删除路径下线，env 清理完成，测试全绿
**最终态**：≤500ms 决策 + 单一 OutputController 状态机 + InterruptDecider 模块化 + EOT 仅做用户回合检测
