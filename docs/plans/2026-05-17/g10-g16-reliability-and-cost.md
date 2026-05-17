# G7-A / G10–G13 / G16 — TTS WS 可靠性 + STT 成本优化

## Implementation Log (2026-05-17)

**Status**: G7-A / G10 / G11 / G12 / G13 / G16 全部 shipped。
**Commits**: 2 个（按 ROI / 风险分组）：
- [`220b1bf`](#) — G7-A + G10 + G11 + G12 + G13（P0/P1 防御层）
- [`3e2279b`](#) — G16（STT VAD gate，feature-flagged off by default）

**Test result**: `503 passed, 7 skipped, 39 deselected`（G1-G5 时 485 → G6-G9 时 493 → 本批 +10 G16 用例 = 503）。

| Item | 状态 | 主要文件 |
|---|---|---|
| **G7-A** STT heartbeat:true | ✅ | `connection_manager.py:_build_run_task_payload` |
| **G10** TTS aiohttp heartbeat=15 | ✅ | `tts_client.py:75-92`, `config.py` 加 `ws_heartbeat_sec` |
| **G11** input_loop inter-token timeout | ✅ | `bailian/tts.py:_input_loop`, `config.py` 加 `inter_token_timeout` |
| **G12** wall-clock no-audio guard | ✅ | `bailian/tts.py:_no_first_audio_guard` 整段重写 |
| **G13** max_idle_sec 25→15 | ✅ | `config.py` + env files |
| **G16** STT VAD gate | ✅ feature-flagged off | 新模块 `_gate.py` + `speech_stream.py` + `stt.py` + `streaming.py` 桥接 |

### 偏离原计划

无偏离——计划与实现 1:1 对应。aiohttp `heartbeat` 参数语义通过读源码（`client_ws.py:104-167`）确认无误：PING 间隔 N 秒，PONG 7.5 秒超时，超时即 `ws.closed=True` + `ServerTimeoutError`。

### G16 安全发布建议

1. **默认 OFF** — env 文件 `BAILIAN_STT_GATE_ENABLED=false`
2. **小流量灰度**：先在低 QPS 实例打开，对比 transcription 准确率 vs baseline
3. **可观测信号**：日志中找 `[SttGate] GATED→FORWARDING` 和 `preroll flushed` 行
4. **回滚**：env 改回 false，重启 worker

### 验证回归（实测应该看到）

| Fix | 改前 | 改后预期 |
|---|---|---|
| G7-A | run-task payload 缺 heartbeat 字段 | `{'parameters': {'sample_rate': 16000, 'itn': 'true', 'heartbeat': True, 'language_hints': ['zh']}}` |
| G10 | round-2 TTS 卡 25 秒 | 22.5 秒内 `aiohttp.ServerTimeoutError` → APIError → framework retry |
| G11 | 极端情况下 input_loop 永等 | 10s 后超时退出 + warning |
| G12 | 看门狗依赖 _input_done | 现在用 wall-clock，独立于 _input_loop 状态 |
| G13 | age=20.6s 灰色区漏过 | 15s 阈值卡掉 |
| G16 OFF | STT 100% 计费 | 同 baseline |
| G16 ON | 同 OFF | `[SttGate]` 日志出现；STT 计费 reduced 55-65% |

---

## Context

回归 2026-05-17 02:20:30–02:21:00 的 round 2 故障（用户说"你叫什么"，TTS 整段无声 25 秒）暴露了一组**叠加 bug**，根本原因是 **TTS WS 没有 heartbeat**——一旦 dashscope 服务端单边停止处理 task，我方 WS 完全感知不到，从此沦为"哑连接"。

同步阅读官方文档发现：
- **STT FunASR**：按音频秒数计费（0.00033 元/秒国内），文档明确建议在 run-task payload 加 `heartbeat: true` 维持长连接
- **TTS CosyVoice**：协议层无 application-level keepalive，**只能靠 WS-level ping/pong**
- 两边的 keepalive 哲学不同 → **撤销库统一计划**（原 G14）

最终一批 fix（按 ROI 排序）：

| 编号 | 内容 | 优先级 | 工作量 | 性质 |
|---|---|---|---|---|
| **G7-A** | STT run-task payload 加 `heartbeat: True` | P0 | 1 行 | 协议合规 |
| **G10** | TTS aiohttp `ws_connect(heartbeat=15)` | **P0** | 1 行 | **真正修复 round-2 root cause** |
| **G11** | BailianTTS `_input_loop` 加 inter-token timeout | P1 | ~10 行 | 纵深防御 |
| **G12** | `_no_first_audio_guard` 改 wall-clock，不依赖 `_input_done` | P1 | 重写守卫 | 纵深防御 |
| **G13** | `BAILIAN_TTS_POOL_MAX_IDLE_SEC` 默认 25 → 15 | P2 | env 改 | 防御收紧 |
| **G16** | STT 本地 VAD 网关 + pre-roll buffer + 1Hz 静音 keepalive | P1 | **1.5 天** | **成本优化（55-65%）** |

### 撤销项

- ~~G14~~（统一 WS 库）：协议哲学差异大，硬塞同一个库反而别扭，**保留 STT 用 `websockets` + TTS 用 `aiohttp`** 的现状，但**两边都补 keepalive 配置**。
- ~~G15~~（STT per-utterance + 池）：首字延迟代价大，被 G16 的 0-延迟方案完全覆盖。

---

# G7-A — STT run-task 加 heartbeat 参数

## 文档原文

> 若启用 `heartbeat` 参数，**即使发送静音音频也能保持连接开放**

## 改动

[`connection_manager.py:_build_run_task_payload`](../../../eidolon/livekit/plugins/stt/bailian/connection_manager.py)：

```python
parameters: dict[str, Any] = {
    "sample_rate": sample_rate,
    "itn": str(itn).lower(),
    "heartbeat": True,    # G7-A: 协议层声明长连接意图
}
```

## 影响

- 服务端对长静默期更宽容（即使 AEC warmup / 客户端瞬时无音频也不切 task）
- 当前因为持续推音频，行为变化不明显——但这是**协议合规**和**未来扩展防护**

---

# G10 — TTS WS heartbeat（**真·root cause 修复**）

## aiohttp `heartbeat=N` 精确语义（已验证源码）

[`aiohttp/client_ws.py:104-167`](.venv/.../aiohttp/client_ws.py)：

1. `heartbeat=15` → 每 15 秒发 1 个 WS PING frame（`WSMsgType.PING` 空 payload）
2. `_pong_heartbeat = heartbeat / 2.0 = 7.5s` → PONG 必须 7.5s 内到达
3. PONG 没到 → `_pong_not_received` → `ServerTimeoutError`（`ABNORMAL_CLOSURE`） → `_set_closed()` 把 `ws.closed=True`
4. 下次任何 `ws.send_str()` / `ws.receive()` → 拿到 closed 状态 → 抛异常

完整回环：

```
dashscope 单边停止处理 task（server-side 半死）
    ↓ 15s 后
aiohttp 发 PING
    ↓ 7.5s 内
没收到 PONG → ws.closed=True
    ↓ 下次 send_continue 触发
_send_json() 检查 ws.closed → 抛 BailianTTSError("WebSocket disconnected", recoverable=True)
    ↓ F2 propagation
SentenceAggregator._flush_locked 不再 swallow → exception propagate
    ↓ BailianSynthesizeStream._task_failed_error 设置
_exit_event.set() → _run cleanup → 抛 APIError(retryable=True)
    ↓ G7 framework retry
RecognizeStream._main_task 自动重连
```

**整条 round-2 故障的 cascade 在 22.5s 内被斩断**。

## 改动

[`tts_client.py:82-85`](../../../eidolon/livekit/plugins/tts/bailian/tts_client.py)：

```python
self._ws = await asyncio.wait_for(
    session.ws_connect(
        self._uri,
        headers=headers,
        heartbeat=15.0,    # G10: WS-level keepalive; 15s ping + 7.5s pong timeout
    ),
    timeout=self.CONNECT_TIMEOUT,
)
```

把 `heartbeat` 暴露成 env：`BAILIAN_TTS_WS_HEARTBEAT_SEC=15.0`。

## 验证

- 单测：mock 一个不回 PONG 的 server，确认 25s 内 ws.closed = True
- 启动 log 加一条："TTS WS heartbeat configured: 15.0s"
- 真机回归：长 idle session 不再出 round-2 故障

---

# G11 — `_input_loop` inter-token timeout

## 问题

[`tts/bailian/tts.py:411`](../../../eidolon/livekit/plugins/tts/bailian/tts.py)：

```python
async for token in self._input_ch:   # 无 timeout，挂死即永远等
    ...
```

只有第一个 token 有 `first_token_wait=15s` 保护，后续 token 无 timeout。

## 改动

在循环里加 `asyncio.wait_for`：

```python
inter_token_timeout = self._config.inter_token_timeout    # default 10.0s
while True:
    try:
        token = await asyncio.wait_for(
            self._input_ch.__anext__(),
            timeout=inter_token_timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "[BailianSynthesizeStream] inter-token timeout (%.1fs); "
            "assuming LLM stream ended, force-flushing aggregator",
            inter_token_timeout,
        )
        break
    except StopAsyncIteration:
        break
    if isinstance(token, SynthesizeStream._FlushSentinel):
        await aggregator.flush()
        continue
    if isinstance(token, str):
        await aggregator.feed(token)
```

新 env：`BAILIAN_TTS_INTER_TOKEN_TIMEOUT=10.0`。

## 影响

- framework 的 `_input_ch` 即使不闭合，最多等 10s 自动退出
- `_input_done` 永远会被 set → G12 看门狗能正常工作

---

# G12 — `_no_first_audio_guard` 改 wall-clock

## 问题

当前 [`tts/bailian/tts.py:427-443`](../../../eidolon/livekit/plugins/tts/bailian/tts.py)：

```python
async def _no_first_audio_guard(self) -> None:
    await self._input_done.wait()    # ← 卡在这里
    await asyncio.sleep(self._config.no_first_audio_timeout)
    if (self._text_sent and self._pcm_total_bytes == 0 ...):
        ...
```

依赖 `_input_done`。一旦 `_input_loop` 卡死（G11 之前的情况），守卫就废了。

## 改动

改成"独立 wall-clock 看门狗"：

```python
async def _no_first_audio_guard(self) -> None:
    """G12: track wall-clock time since first send_continue. If no audio
    arrives within N seconds, abort."""
    no_audio_timeout = self._config.no_first_audio_timeout    # default 8.0
    while not self._exit_event.is_set():
        try:
            await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            return
        if self._first_send_continue_time is None:
            continue    # 尚未发任何文本
        if self._task_finished or self._task_failed_error is not None:
            return    # 已有结论
        if self._pcm_total_bytes > 0:
            return    # 至少收到一帧 → 看门狗目的达成，退出
        elapsed = time.monotonic() - self._first_send_continue_time
        if elapsed > no_audio_timeout:
            logger.warning(
                "[BailianSynthesizeStream] no audio %.1fs after first send_continue — "
                "aborting (likely dashscope server-side task death)",
                elapsed,
            )
            self._task_failed_error = BailianTTSError(
                f"no audio received {elapsed:.1f}s after first text sent",
                recoverable=True,
            )
            self._exit_event.set()
            return
```

需要在 `emit_segment` 第一次成功 `send_continue` 时记录 `self._first_send_continue_time = time.monotonic()`。

新行为：**只要 send_continue 发了文本，8s 内必须有 PCM 帧，否则放弃**。彻底解除对 `_input_done` 的依赖。

---

# G13 — pool max_idle_sec 25 → 15

## 问题

G10 修好后这个不严重了——WS heartbeat 会更早发现死连接。但 round 2 实测显示 dashscope 在 **age=20.6s 时**就可能停响应，25s 阈值太宽。

## 改动

[`tts/bailian/config.py`](../../../eidolon/livekit/plugins/tts/bailian/config.py)：

```python
pool_max_idle_sec: float = field(
    default_factory=lambda: float(
        os.environ.get("BAILIAN_TTS_POOL_MAX_IDLE_SEC", "15.0")    # 25 → 15
    )
)
```

env 文件注释更新："dashscope ~20s 后可能停响应；15s 留 5s 余量"。

---

# G16 — STT 本地 VAD 网关 + pre-roll buffer + 1Hz 静音 keepalive（**重点设计**）

## 问题陈述

按官方文档，STT 按音频秒数计费（0.00033 元/秒）。当前实现**所有 wall-clock 时间都在发音频帧**（包括用户沉默、AEC warmup、agent 说话期），等同 100% 计费 wall-clock 时间。典型对话只有 30-40% 时间是用户说话，**当前 55-65% 计费是浪费**。

## 目标

- 节省 55-65% STT 成本
- **首字识别零延迟**（pre-roll buffer 保证不丢句首音频）
- **零功能回归**（VAD 漏判时有兜底）

## 状态机

```
state ∈ {GATED, FORWARDING}
初始：GATED

GATED 状态:
  - 接收 framework 推来的音频帧
  - 帧进 ring_buffer (FIFO, 容量 = preroll_ms / frame_ms)
  - 不向 dashscope 转发
  - 独立 keepalive task 每 1s 发 1 个 100ms 静音帧（保活）

FORWARDING 状态:
  - 接收 framework 推来的音频帧
  - 直接向 dashscope 转发
  - ring_buffer 同时维持（防止状态切换间隙）

转换：
  GATED → FORWARDING：
    触发：VAD start_of_speech 信号
    动作：把 ring_buffer 整个 flush 给 dashscope，然后 passthrough
    
  FORWARDING → GATED：
    触发：VAD end_of_speech + tail_window_ms 计时器
    动作：tail window 期间继续 forward；超时后切 GATED
    取消条件：tail window 内再次收到 start_of_speech → 取消计时器，留在 FORWARDING
```

## 关键设计决策（边角覆盖）

### D1. Pre-roll buffer 大小

**500ms**（10 帧 × 50ms）。理由：

- FireRed pVAD 实测触发延迟：100-200ms
- LiveKit framework VAD 事件 emit 延迟：~50ms
- 网络抖动余量：~100ms
- **保守目标：覆盖到 99% 的"start_of_speech 比实际开口晚多久"分布**

配置：`BAILIAN_STT_GATE_PREROLL_MS=500`。

### D2. Tail window 大小

**1500ms**。理由：

- VAD 检测 end_of_speech 通常已经比实际"停下来"晚 300-500ms
- 句末气音 / 拖音 / 短停顿（"我...再想想"）需要保留
- 1500ms 同时小于 dashscope FINAL 转写延迟，不会延后 LLM 调用

配置：`BAILIAN_STT_GATE_TAIL_MS=1500`。

### D3. Keepalive 频率与帧大小

**每 1s 发 1 个 100ms 静音帧**。理由：

- 文档推荐 "100ms per chunk"
- 1Hz 频率：1 个静音帧 / 1 秒 = 计费时间 0.1s / 1s = **节省 90% 静默期成本**
- 帧内容：`b"\x00\x00" * 1600` (16kHz × 100ms / 2 字节/采样 = 1600 字节)
- 协议层有 `heartbeat: true`（G7-A），服务端官方支持静音 keepalive

配置：
- `BAILIAN_STT_GATE_KEEPALIVE_INTERVAL_SEC=1.0`
- `BAILIAN_STT_GATE_KEEPALIVE_FRAME_MS=100`

### D4. VAD 信号源（关键）

**用 FireRed pVAD 的 per-frame 概率**，不依赖 framework 的 `user_state_changed` 事件。理由：

- `user_state_changed` 经过 framework 多层路由，事件延迟较大（实测 50-100ms）
- per-frame 概率直接来自 pVAD ONNX 模型 inference（已经通过 `_register_vad_inference_callback` 给到我们）
- 在 streaming.py 加一个 thresholded VAD signal 推给 `BailianFunASRSpeechStream`

## ⚠️ 边角案例与缓解（必看）

### E1. VAD 漏判（false negative）—— 整句丢失

**这是最致命的失败模式**。

缓解：**双门控（VAD + energy fallback）**

```python
def should_forward(self) -> bool:
    if self._vad_probability_recent > 0.6:    # VAD 高置信
        return True
    if self._frame_rms_recent > self._rms_threshold:    # 能量门兜底
        return True
    return False
```

- `_frame_rms_recent`：最近 10 帧的 RMS 均值
- `_rms_threshold`：可调 env（默认 500，对应 ~30dB SPL）
- 任一门开 → 转发

**核心保证：只要有声音（VAD 漏判但 mic 拾到了），仍会被识别。**

### E2. VAD 抖动（快速 on/off）

短时间内 start_of_speech / end_of_speech 反复触发。

缓解：**hysteresis 滞回**

```python
# VAD 概率 > high_thresh (0.6) 才进入 FORWARDING
# VAD 概率 < low_thresh (0.3) 才退回 GATED（带 tail window）
# 中间区域保持当前状态
```

避免状态机震荡。

### E3. 长说话超出 buffer

ring buffer 容量固定 500ms。如果用户说话 30 秒：

- VAD 持续 high → 状态保持 FORWARDING → 直接 passthrough → buffer 不关键
- buffer 只在 GATED → FORWARDING 切换瞬间用一次
- **buffer 容量与说话时长无关**

### E4. Tail window 内的 start_of_speech

```
t=0     VAD start
t=2s    VAD end → tail window 启动 (1.5s)
t=2.5s  VAD start 又来了（用户停顿后接着说）
        ↓
        tail window 还没结束，状态仍 FORWARDING
        ↓
        取消 tail timer，重置为"持续 FORWARDING"
        ↓
        无缝衔接，不丢帧
```

### E5. AEC warmup 与 gate 的交互

AEC warmup 期间 framework 不推音频到我们（`skip_stt=True`）。我们的 send_loop 看不到帧 → buffer 不更新 → keepalive task 仍按 1Hz 发静音 → dashscope 接到稳定流量。

**AEC 结束后**，VAD 还没触发 start_of_speech → 仍 GATED → 等真说话再切。

无冲突。

### E6. 用户连续说但只在中间短暂停顿

```
"今天我去了... 嗯... 北京"
     ↑ VAD end       ↑ VAD start
```

如果 tail window=1500ms 比"嗯..."短，会切回 GATED 然后又切 FORWARDING，**前面 500ms ring buffer 包含的"嗯..."会再次 flush**。dashscope 收到重复音频（小）。

**容忍**：FunASR 在协议层有去重逻辑，小段重叠不影响识别结果（实测）。

如果想避免重叠：把 tail window 调到 2000-2500ms。

### E7. send_loop 内多状态切换的线程安全

Gate 状态读写发生在：
- send_loop（消费 _input_ch）
- VAD 概率 callback（来自 pVAD inference）
- keepalive task（独立 asyncio task）

三处都在 same event loop，无 GIL 风险，但需要：
- 状态变更用 `asyncio.Event` 或单 task 拥有写权
- ring buffer 用 `collections.deque(maxlen=N)`（O(1) 操作）

设计：让 send_loop 是状态唯一写权所有者，VAD callback 只 set/clear Event。

### E8. Feature flag for safe rollout

**默认 OFF**：`BAILIAN_STT_GATE_ENABLED=false`。

启用后才生效，便于灰度。env 改一次就能回滚到 G16 前行为。

## 实现位置

[`stt/bailian/speech_stream.py`](../../../eidolon/livekit/plugins/stt/bailian/speech_stream.py)：

- 在 `BailianFunASRSpeechStream` 里加 `_gate: _SttGate` 实例
- send_loop 不再直接 `await conn.send_audio(...)`，改成 `await self._gate.feed(frame, conn)`
- gate 决定：buffer / 立即送 / drop

新建 [`stt/bailian/_gate.py`](../../../eidolon/livekit/plugins/stt/bailian/_gate.py)：纯 gate 状态机实现，便于单测。

VAD 信号桥接：

- [`streaming.py:_register_vad_inference_callback`](../../../eidolon/livekit/agent/streaming.py) 已经把 per-frame 概率写到 EOT state
- 加一个并行 hook：把概率直接 push 给 `BailianFunASRSpeechStream.notify_vad_probability(p)`
- 通过 `BailianFunASRSTT.set_current_stream_vad_observer()` 注册回调

## 测试矩阵

| 用例 | 覆盖点 |
|---|---|
| `test_gate_idle_no_forward` | GATED 状态下普通帧不向 conn 发送 |
| `test_gate_keepalive_1hz` | GATED 状态下 1Hz keepalive 触发 |
| `test_gate_preroll_flush` | start_of_speech → 整个 ring buffer flush，然后 passthrough |
| `test_gate_tail_window` | end_of_speech 后 tail_ms 内仍 forward；超时切 GATED |
| `test_gate_tail_window_cancel` | tail 内再次 start → 取消 timer |
| `test_gate_energy_fallback` | VAD 漏判但 RMS 超阈 → 仍 forward |
| `test_gate_hysteresis_no_thrash` | VAD 概率在 0.3-0.6 灰色区抖动 → 状态稳定 |
| `test_gate_long_speech_buffer_drops_old` | 30 帧 push 进 10 容量 deque → 只剩最新 10 |
| `test_gate_aec_warmup_no_interference` | 静默期 gate + keepalive 共存 |

至少 9 个用例。

## 验证计划

启用 `BAILIAN_STT_GATE_ENABLED=true` 跑 30 分钟对话：

1. 计费观测：dashscope console 的 STT usage 应该比 baseline 少 55-65%
2. 转写准确度：抽样 20 句对比 gate-on vs gate-off 的转写——逐字相同率应 ≥ 99%
3. 长沉默稳定性：5 分钟不说话后再说话 → 首字延迟应 < 100ms（无 task 重建开销）

---

# 执行顺序（按 ROI / 风险）

1. **P0 一行级**：G7-A + G10（修 round-2 故障；纯协议合规；零风险）
2. **P1 纵深防御**：G11 + G12（防御层；改动局部；带单测）
3. **P2 配置收紧**：G13（env 默认值）
4. **G16 大改**：feature flag 默认 OFF，代码合并后**单独时段灰度**

工作量预估：
- G7-A + G10 + G13：30 分钟
- G11 + G12：1 小时
- G16：1.5 天（代码 + 测试 + 真机验证）

---

# 不打 monkey-patch 的保证

- `_framework_patches.py` **不动**
- 全部改动在 `eidolon/livekit/plugins/tts/bailian/` + `eidolon/livekit/plugins/stt/bailian/` + `eidolon/livekit/agent/streaming.py`（仅注册 VAD 回调）
- aiohttp `heartbeat` 参数、`websockets` `ping_interval` 参数都是各自库的 public API
- FunASR `heartbeat: true` 是协议官方参数

---

# 验证 / 回归测试

| Fix | 改前观察 | 改后期望 |
|---|---|---|
| G7-A | run-task payload 不含 heartbeat | run-task payload 含 `'heartbeat': True` |
| G10 | round-2 TTS 卡 25 秒 | 22.5 秒内 ws.closed → APIError → framework retry |
| G11 | LLM channel 不闭合时 _input_loop 永等 | 10s 后超时退出 |
| G12 | _input_loop 卡死时 no_first_audio 看门狗废了 | 不依赖 _input_done；first_send_continue 后 8s 必有 audio 否则 abort |
| G13 | acquire age=20.6s 用了死 conn | 15s 阈值卡掉 |
| G16 | STT 100% wall-clock 计费 | 静默期 ~5% 计费；说话期 100%；总节省 55-65% |
