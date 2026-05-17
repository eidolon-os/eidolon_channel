# G23 — STT Transcript Gate: 消除跨 turn transcript 污染

> Status: planned (2026-05-18)
> Owner: agent layer
> ROI: 修一个用户可感知的偶发 bug（"嗯，听到了。"幻觉响应 + 双 LLM 调用浪费）
> Scope: ~350 LOC（含测试），STT-agnostic 设计

## 1. Bug 一句话定义

Long-running STT stream（Bailian / SenseTime / Azure realtime / Google streaming 这一类）
**跨用户 turn 的 transcript 事件被 framework 累积进同一个 user message**，导致 LLM 看
到污染输入、产生幻觉响应（生产日志观察到 "嗯，听到了。"）+ 浪费一次 LLM 调用。

## 2. 根因（framework 代码位置）

`.venv/lib/python3.12/site-packages/livekit/agents/voice/audio_recognition.py`：

### 污染源 A — INTERIM 末尾拼接（line 715-730）

```python
if self._audio_interim_transcript:                  # 当前 pending INTERIM 非空
    self._audio_transcript = (
        f"{self._audio_transcript} {self._audio_interim_transcript}".strip()
    )
```

**触发**：FINAL 已到 → 用户说下一句首词的 INTERIM 进来 → 我们的
`commit_user_turn` 触发（VAD-end 驱动）→ framework 把这条 INTERIM 拼到上一句末尾
→ LLM 收到 `"嗯，我...睡不着。 你有"` 这种污染字符串。

### 污染源 B — 多 FINAL 累积（line 839）

```python
elif ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
    ...
    self._audio_transcript += f" {transcript}"      # 每个 FINAL 都追加
```

**触发**：两句话间隔短到 VAD-end 都没起就出了第二个 FINAL（生产 log 暂未观察到，
但 SenseTime 风险更高——它只发 FINAL 不发 INTERIM）。

### 生产证据（2026-05-18 log）

```
01:21:52.785  FINAL  "嗯，我还能睡几个小时...睡不着。"  (sentence_id=6)
01:21:52.913  INTERIM "你有"                             (sentence_id=7 - 新句子)
01:21:53.107  VAD: end of speech
01:21:53.108  user_state speaking → listening → commit_user_turn
01:21:53.128  ★ LLM 调用 #1, user content = "嗯...睡不着。 你有"  ← 污染！
01:21:55.367  TTS aggregator flush "嗯，听到了。"        ← LLM #1 的幻觉响应
01:21:55.667  FINAL  "你有什么建议吗。"                  (sentence_id=7 真 FINAL)
01:21:55.684  LLM #1 CancelledError
01:21:55.692  ★ LLM 调用 #2，正确响应                    ← 但 #1 的 TTS 部分已播放
```

用户感知：听到 "嗯，听到了。" + 停顿 + "几点的火车？给你几个促眠建议..." 两段奇怪
的回复。

## 3. 解决方案：`STTTranscriptGate`（STT-agnostic 装饰器）

### 3.1 架构插入点

```
                              用户音频
                                 ↓
┌─────────────────────────────────────────────────────────┐
│  Bailian / SenseTime / Azure / Google 等任意 STT plugin │  实际协议
│  (concrete impl of lk_stt.STT)                          │
└────────────────────────┬────────────────────────────────┘
                         │  inner.stream() → RecognizeStream
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STTTranscriptGate(lk_stt.STT)            ← 新增        │  事件过滤
│  - 装饰器，转发所有方法到 inner                          │
│  - 覆盖 stream() 返回 _FilteringRecognizeStream         │
└────────────────────────┬────────────────────────────────┘
                         │  事件流（过滤后）
                         ▼
┌─────────────────────────────────────────────────────────┐
│  SttStage  ← 不动；现在 .stt 指向 gate 而不是裸 plugin   │  pipeline 便利层
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
                  framework AgentSession
```

### 3.2 装饰器模式说明

**两层都继承 `lk_stt.STT`** — 这是装饰器模式的标准做法：

- `BailianFunASRSTT` / `SenseTimeSTT` **不变**，继续 `extends lk_stt.STT`
- `STTTranscriptGate` 也 `extends lk_stt.STT`，持有 `inner: lk_stt.STT`
- `STTTranscriptGate` 不重写业务逻辑，只在事件流上加过滤层
- 所有真正的 STT 工作（WS 协议、API key、metrics）仍在 inner 里

### 3.3 过滤规则

**单一规则**：当 inner 发出 FINAL_TRANSCRIPT 事件后，在 `suppress_window_ms`（默认
200ms）窗口内，**抑制后续的 INTERIM_TRANSCRIPT 和 FINAL_TRANSCRIPT** 事件——
不转发给 framework 的 audio_recognition。

**为什么这个规则能修两个污染源**：

- **污染源 A**（INTERIM 末尾拼接）：
  被抑制的 INTERIM 不会进 framework 的 `_audio_interim_transcript` →
  commit 时 line 715 的 `if self._audio_interim_transcript:` 为假 → 不拼接。

- **污染源 B**（多 FINAL 累积）：
  被抑制的第二个 FINAL 不会进 framework 的 `_audio_transcript` →
  line 839 的 `+=` 不执行 → 不累积。

**为什么 200ms 是合理的窗口**：

- Bailian 实测：上一句 FINAL → 下一句首 INTERIM 间隔 ≈ 128ms（生产 log 实证）
- SenseTime 实测：类似量级
- 500ms 太长会过度过滤
- 100ms 太短可能漏掉

**为什么这种"过早抑制"是安全的**：

- 被抑制的 INTERIM 后面同一句的 INTERIM 会再出（cumulative protocol，下一条
  INTERIM 内容就是 "你有什么"，重复了 "你有"）。
- 被抑制的 FINAL 通过下次 VAD-end 边界另外打 `commit_user_turn` 处理。
- 即"丢弃的事件"在后续事件流里有同等信息可用。

### 3.4 抑制后续 FINAL 的特殊考量

被抑制的 FINAL **会丢失语义信息**（其后的 INTERIM 也被抑制）——但这正是我们要的，
因为：
- 用户两句间隔 <200ms 在自然对话里少见
- 即便发生，**第二句的 INTERIM 在 200ms 后会重新被允许通过**
- 第二次 VAD-end 触发新的 `commit_user_turn` 时，这次的 INTERIM 已经累积到完整
  状态，会作为新 user turn 正常进入 LLM

也就是说：连说两句的极端用户体验上，**会变成两次独立 LLM 调用**，但不会丢字。

## 4. 实现

### 4.1 文件位置

```
eidolon/livekit/plugins/stt/_transcript_gate.py    ← 新增 ~150 LOC
  ├── class STTTranscriptGate(lk_stt.STT)
  └── class _FilteringRecognizeStream(lk_stt.RecognizeStream)

eidolon/livekit/agent/factory.py                    ← +3 行 wrap
eidolon/livekit/common/config.py                    ← +2 个 env 字段
eidolon/livekit/tests/stt/test_transcript_gate.py   ← 新增 ~200 LOC
deploy/livekit-channel.env.template                 ← +3 行注释
deploy/.livekit-channel.env                         ← +3 行注释
```

**为什么放 `plugins/stt/_transcript_gate.py`**：与 `plugins/tts/_pool.py` /
`plugins/tts/_aggregator.py` / `plugins/stt/bailian/_gate.py` 同一约定，前置下划线
= 跨 provider 共享的工具，不是 provider 自身。

### 4.2 类骨架

```python
class STTTranscriptGate(lk_stt.STT):
    """STT-agnostic decorator that suppresses cross-turn transcript events.

    Sits between any lk_stt.STT plugin and framework's audio_recognition.
    Filter rule: when a FINAL is emitted, any INTERIM/FINAL arriving within
    `suppress_window_ms` is dropped — preventing framework from concatenating
    next-turn transcripts onto the just-finished turn.
    """

    def __init__(self, inner: lk_stt.STT, *, suppress_window_ms: int = 200) -> None:
        super().__init__(capabilities=inner.capabilities)
        self._inner = inner
        self._suppress_window_ms = max(0, int(suppress_window_ms))

    # Metadata forwarding
    @property
    def label(self) -> str: return f"STTTranscriptGate({self._inner.label})"
    @property
    def model(self) -> str: return self._inner.model
    @property
    def provider(self) -> str: return self._inner.provider

    # Forward methods that aren't lifecycle-related
    async def _recognize_impl(self, buffer, *, language, conn_options):
        return await self._inner._recognize_impl(
            buffer, language=language, conn_options=conn_options
        )

    # Plugin-specific lifecycle hooks (warmup, shutdown) — forward if present
    async def warmup(self) -> None:
        if hasattr(self._inner, "warmup"):
            await self._inner.warmup()

    async def shutdown(self) -> None:
        if hasattr(self._inner, "shutdown"):
            await self._inner.shutdown()

    # The actual gate point
    def stream(self, *, language=NOT_GIVEN, conn_options=DEFAULT_API_CONNECT_OPTIONS,
               sample_rate=NOT_GIVEN):
        inner_stream = self._inner.stream(
            language=language, conn_options=conn_options, sample_rate=sample_rate
        )
        return _FilteringRecognizeStream(
            stt=self,
            inner=inner_stream,
            suppress_window_ms=self._suppress_window_ms,
        )


class _FilteringRecognizeStream(lk_stt.RecognizeStream):
    """The stream object actually given to framework. Iterates inner stream
    events, applies time-window gate, re-emits to outer event channel."""

    def __init__(self, *, stt, inner, suppress_window_ms):
        super().__init__(stt=stt, conn_options=inner._conn_options)
        self._inner = inner
        self._suppress_window_sec = suppress_window_ms / 1000.0
        self._last_final_time: float | None = None
        self._suppressed_count: int = 0

    async def _run(self) -> None:
        # 1. forward audio input from outer to inner
        forward_task = asyncio.create_task(self._forward_input())

        try:
            # 2. iterate inner events, gate, re-emit to outer
            async for ev in self._inner:
                if not self._should_pass(ev):
                    self._suppressed_count += 1
                    logger.debug(
                        "[STTTranscriptGate] suppressed %s within %.0fms of FINAL: %r",
                        ev.type.name,
                        self._suppress_window_sec * 1000,
                        ev.alternatives[0].text if ev.alternatives else "",
                    )
                    continue
                if ev.type == lk_stt.SpeechEventType.FINAL_TRANSCRIPT:
                    self._last_final_time = time.monotonic()
                self._event_ch.send_nowait(ev)
        finally:
            forward_task.cancel()
            await asyncio.gather(forward_task, return_exceptions=True)

    async def _forward_input(self) -> None:
        async for item in self._input_ch:
            if isinstance(item, self._FlushSentinel):
                self._inner.flush()
            else:
                self._inner.push_frame(item)
        self._inner.end_input()

    def _should_pass(self, ev) -> bool:
        # Always pass non-transcript events (START/END_OF_SPEECH, metrics)
        if ev.type not in (
            lk_stt.SpeechEventType.INTERIM_TRANSCRIPT,
            lk_stt.SpeechEventType.FINAL_TRANSCRIPT,
        ):
            return True
        # No prior FINAL? Always pass — this is the first utterance.
        if self._last_final_time is None:
            return True
        # Within suppression window of a recent FINAL? Suppress.
        elapsed = time.monotonic() - self._last_final_time
        return elapsed >= self._suppress_window_sec

    @property
    def suppressed_count(self) -> int:
        return self._suppressed_count
```

### 4.3 Factory 接入

```python
# eidolon/livekit/agent/factory.py — _build_stt
inner = ...  # 原来构造 BailianFunASRSTT / SenseTimeSTT 的逻辑保留

if cfg.behavior.stt_transcript_gate_enabled:
    from eidolon.livekit.plugins.stt._transcript_gate import STTTranscriptGate
    stt_to_wrap = STTTranscriptGate(
        inner,
        suppress_window_ms=cfg.behavior.stt_gate_suppress_window_ms,
    )
else:
    stt_to_wrap = inner

return SttStage(stt_to_wrap, params=params)
```

### 4.4 Config 字段

```python
# eidolon/livekit/common/config.py — AgentBehaviorConfig
stt_transcript_gate_enabled: bool = False  # default off; enable after verification
stt_gate_suppress_window_ms: int = 200
```

Env 读取（参照现有约定）：
```python
stt_transcript_gate_enabled = get("EIDOLON_STT_TRANSCRIPT_GATE_ENABLED", "false").lower() == "true"
stt_gate_suppress_window_ms = int(get("EIDOLON_STT_GATE_SUPPRESS_WINDOW_MS", "200"))
```

### 4.5 Env 模板

```bash
# ---------- STT transcript gate (G23, 2026-05-18) ----------
# 跨 turn transcript 污染防护：当用户连说两句间隔很短，framework 会把第二句首
# INTERIM 拼到第一句末尾，导致 LLM 收到污染输入 + 双 LLM 调用浪费。
# 默认 false（等真机回归通过后会改 true）
EIDOLON_STT_TRANSCRIPT_GATE_ENABLED=false
# FINAL 之后多少 ms 内的 INTERIM/FINAL 被丢弃
EIDOLON_STT_GATE_SUPPRESS_WINDOW_MS=200
```

## 5. 测试

### 5.1 单元测试（`tests/stt/test_transcript_gate.py`，~200 LOC）

10 个核心用例：

1. **`test_gate_pass_through_when_no_prior_final`** — 未发生过 FINAL，
   所有事件透传
2. **`test_gate_suppresses_interim_within_window`** — FINAL 后 100ms 的
   INTERIM 被丢弃
3. **`test_gate_passes_interim_outside_window`** — FINAL 后 300ms 的
   INTERIM 通过（>200ms 窗）
4. **`test_gate_suppresses_second_final_within_window`** — FINAL 后 150ms
   的 FINAL 也被丢弃
5. **`test_gate_forwards_input_audio`** — outer push_frame → inner push_frame
6. **`test_gate_forwards_flush_sentinel`** — outer flush → inner flush
7. **`test_gate_preserves_event_order`** — 未抑制事件按时序到达 outer
8. **`test_gate_capabilities_forwarded`** — gate.capabilities is inner.capabilities
9. **`test_gate_suppressed_count_metric`** — 计数正确
10. **`test_gate_window_zero_disables_suppression`** — window=0 等价于全部透传

### 5.2 真机回归

3 段录音手动测试：

- **A. 单句长 Chinese 5 秒（无连续两句）**：suppressed_count == 0
- **B. 两句中间停顿 100ms**：第二句 INTERIM 被吃 → 用户感知为两次独立 LLM
  调用、两次独立 TTS 回复（可接受）
- **C. 两句中间停顿 500ms**：完全独立两次 LLM，gate 不参与（在窗外）

## 6. 风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| 用户语速极快连说两句（间隔 <200ms） | 中文不常见 | 缩窗到 150ms；首字屏显晚 200ms 但 LLM 响应正确 |
| 抑制了正常 INTERIM 导致 G18a 首信号 cancel 触发慢 | 低 | G18a 触发场景是 agent 正在说、用户开口——FINAL 距上一轮 >2s，远超 200ms |
| 跟 G16 STT VAD gate 冲突 | 无 | G16 是音频侧 gate；本 gate 是 transcript 侧（FINAL 后短窗丢事件）。完全独立 |
| Framework 后续修了 line 715/839 → 本层冗余 | 中 | env 一行关掉，代码留着不影响 |
| 测试覆盖不到的边界 | 低 | RecognizeStream `_run` 跟 `_input_ch` 的生命周期需要小心处理 cancel |

## 7. 实施顺序

1. 写 `_transcript_gate.py` 实现
2. 写 unit tests（10 个）
3. 跑单测全部 green
4. Factory 接入 + config 字段
5. Env 模板更新
6. 跑全套测试 + 真机回归
7. Commit + push（默认 env=false）

## 8. 默认开关

**第一版默认 `false`**——稳健派，等真机回归通过 + 跑 1-2 天日志确认 suppressed_count
分布合理（应该是 0 或 1 per session，不会过高）后，再 commit 第二版改默认 `true`。

## 9. 不打 patch 保证

修改全部在我们模块内：

- `eidolon/livekit/plugins/stt/_transcript_gate.py`（新）
- `eidolon/livekit/agent/factory.py`（3 行 wrap）
- `eidolon/livekit/common/config.py`（2 个 env 字段）
- `eidolon/livekit/tests/stt/test_transcript_gate.py`（新）
- `deploy/livekit-channel.env.template`、`deploy/.livekit-channel.env`（注释 + env）

`audio_recognition.py` line 715/839 的 framework bug **不动**——我们在它上游过滤
掉污染源，让它没有可以累积的东西。`_framework_patches.py` 不动。
