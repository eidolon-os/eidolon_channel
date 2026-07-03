# LiveKit Agent Server 架构分析

> 最近更新: 2026-07-03
> 代码路径: `eidolon/livekit/agent/`

---

## 1. 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Eidolon Agent Server (eidolon_daemon)                  │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                         server.py (main entry point)                    │   │
│  │                                                                             │   │
│  │   AgentConfig ──loads──> .env + env vars                                   │   │
│  │                                                                             │   │
│  │   _build_llm() ──> livekit.plugins.openai.LLM (OpenAI-compatible)           │   │
│  │   _build_stt() ──> SttStage + BailianFunASRSTT (百练 FunASR)              │   │
│  │   _build_tts() ──> TtsStage + SenseTimeTTS (商汤 SenseAudio)              │   │
│  │   _build_vad() ──> FireredPvadVAD.load() (FireRed pVAD)                   │   │
│  │                                                                             │   │
│  │   AgentServer(ws_url, api_key, api_secret, host, port)                      │   │
│  │        │                                                                    │   │
│  │        └───> server.rtc_session(_on_session)  注册会话回调                   │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                       │                                         │
│                                       ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                    LiveKit Agent Framework (livekit.agents)                 │   │
│  │                                                                             │   │
│  │   AgentServer ────────────────────────────> LiveKit Cloud/Self-Hosted      │   │
│  │        │                                        Server (端口 7880)          │   │
│  │        │                                                                    │   │
│  │        └── _on_session() 触发 ──spawn worker──> 每个 Room 分配一个 worker     │   │
│  │                                           (macOS spawn 模式会重新 import)  │   │
│  │                                                                             │   │
│  │   在 worker 内:                                                            │   │
│  │   _on_session(ctx: JobContext)                                              │   │
│  │        │                                                                    │   │
│  │        ▼                                                                  │   │
│  │   run_agent(ctx, cfg)                                                      │   │
│  │        │                                                                    │   │
│  │        ├── SharedStageFactory(VoiceConfig)                                  │   │
│  │        │        ├── stt_stage (BailianFunASRSTT)                             │   │
│  │        │        ├── llm (livekit.plugins.openai.LLM)                         │   │
│  │        │        ├── tts_stage (SenseTimeTTS)                                 │   │
│  │        │        └── vad (FireredPvadVAD)                                     │   │
│  │        │                                                                    │   │
│  │        ├── AgentSession(                                                    │   │
│  │        │        stt=stt, llm=llm, tts=tts,                                  │   │
│  │        │        vad=vad,                                                    │   │
│  │        │        turn_detection=ChineseModel()                                │   │
│  │        │   )                                                               │   │
│  │        │        │                                                           │   │
│  │        │        ├── RoomIO(session, room)                                    │   │
│  │        │        │        │                                                  │   │
│  │        │        │        └───> room.local_publish_audio()  发布音频到房间     │   │
│  │        │        │                                                           │   │
│  │        │        └── session.start(agent, room, ...)  开始会话                 │   │
│  │        │                                                                    │   │
│  │        ├── StreamingPipeline (AGENT_MODE=streaming, 默认)                     │   │
│  │        │    或                                                               │   │
│  │        └── BatchPipeline   (AGENT_MODE=batch)                                │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 当前目录结构与边界

```
eidolon/livekit/agent/
├── README.md                 # 代码地图: entrypoints / integration / session / policy / output
├── server.py                 # 主入口: AgentServer 启动、配置加载、session 回调注册
├── factory.py                # SharedStageFactory: 统一创建 stt/llm/tts/vad/turn_detection
├── batch.py                  # BatchPipeline: 批量音频 blob 处理
├── integration/
│   ├── __init__.py           # LiveKit/framework 外部契约边界
│   ├── framework_patches.py  # LiveKit internal API patch，升级时唯一审计点
│   └── client_audio_state.py # client.audio_state data-channel payload parser/model
├── pipeline/
│   ├── base.py               # VoicePipeline 抽象基类
│   ├── stt.py                # SttStage: 封装 STT provider
│   ├── tts.py                # TtsStage: 封装 TTS provider
│   ├── vad.py                # VadStage: 封装 VAD provider
│   ├── llm.py                # LLM stage / remote-agent bridge
│   └── types.py              # PipelineMode, PipelineState, callbacks 等类型
├── turn_policy/
│   ├── attention.py          # client audio_state 与注意力判定
│   ├── constants.py          # 打断词表与 intent pattern 的代码默认值
│   ├── decider.py            # InterruptDecision 汇总入口
│   ├── evidence.py           # transcript/eot/preflight evidence 结构
│   ├── intent_classifier.py  # 轻量 intent 分类
│   ├── runtime.py            # turn policy 运行时状态
│   └── tiers/
│       ├── chain.py          # Tier 0-4 policy chain
│       ├── model.py          # TierDecision/TierEvidence
│       ├── tier0_hard_stop.py
│       ├── tier1_redirect.py
│       ├── tier2_interruption.py
│       ├── tier3_noise.py
│       └── tier4_attention.py
├── output/
│   ├── controller.py         # OutputController: TTS/播放句柄、取消、指标
│   ├── ducking.py            # OutputDuckingController: duck 安装/状态迁移/timeout
│   └── filler.py             # FillerManager: 填充语管理与播放
├── half_duplex/
│   ├── control.py            # PTT turn_status / no-turn terminal helpers
│   ├── pipeline.py           # HalfDuplexPttPipeline: segment PTT room pipeline
│   ├── ptt_segment.py        # PTT hold 内完整音频段采集
│   ├── ptt_transcriber.py    # release 后一次性 STT
│   └── ptt_turn_controller.py # segment PTT 状态机
├── full_duplex/
│   ├── client_preempt.py     # ExplicitClientPreemptHandler: full-duplex explicit client preempt
│   ├── semantic_interrupt_gate.py # SemanticInterruptGate: transcript-triggered semantic interrupt gate
│   ├── transcript_admission.py # TranscriptAdmissionGate: residual/echo transcript entry gate
│   └── pipeline.py           # StreamingPipeline: full-duplex realtime AgentSession pipeline
├── session/
│   ├── __init__.py           # package marker only; no broad component re-export facade
│   ├── agent_state.py        # AgentStateEffectHandler: agent state side effects
│   ├── attention_effects.py  # AttentionEffectHandler: attention admission effects
│   ├── client_control.py     # session-local eidolon.control envelope / event helpers
│   ├── decision_effects.py   # DecisionEffectApplier: decision timeline/metadata/effects
│   ├── duck_timeout.py       # DuckSuspendTimeoutHandler: duck deadline policy effects
│   ├── eot_model.py          # shared EOT model cache/loading helper
│   ├── messages.py           # LiveKit chat/message text helper
│   ├── provider_events.py    # STT/TTS provider event 观测
│   ├── idle.py               # IdleWatchdog: 空闲定时与主动问候
│   ├── room_data.py          # LiveKit data packet 解析与分发
│   ├── interruption.py       # SoftInterruptController: 软打断补偿路径
│   ├── semantic_interrupt.py # SemanticInterruptHandler: STT/EOT 打断热路径副作用
│   ├── signals.py            # SessionSignalBridge: VAD/STT provider 信号桥接
│   ├── transcript_echo.py    # TranscriptEchoGate: full-duplex TTS echo content gate
│   ├── user_turn_coordinator.py # UserTurnCoordinator: 用户 turn 候选、合并、提交/拒绝决策
│   └── turn_commit.py        # UserTurnCommitter: VAD-end commit guard
├── context/
│   └── interrupted.py        # InterruptedContextManager: 被打断回复注入上下文
├── runtime/
│   ├── interaction_mode.py   # session metadata parsing + mode/intent policy
│   └── resolver.py           # tenant/user/template 解析
├── eidolon_agent_rpc/
│   ├── grpc_llm.py           # remote Agent LLM adapter
│   └── session.py            # remote Agent session client
├── speaker_verification/
│   ├── service.py            # voiceprint service orchestration
│   └── store.py              # voiceprint metadata store; model providers live in plugins/
└── observability/
    ├── metrics.py            # 指标聚合
    └── timeline.py           # timeline event 记录
```

### 2.1 边界原则

`full_duplex/pipeline.py` 是 full-duplex 实时会话的主编排器，负责把 LiveKit `AgentSession`、Room、pipeline stage、打断决策、输出控制和观测串起来。根部 `streaming.py` 兼容入口已删除；新代码必须从 `eidolon.livekit.agent.full_duplex` 导入 full-duplex pipeline。full-duplex pipeline 可以持有流程状态，但不应承载可独立测试的副作用模块。

`turn_policy/` 负责“是否打断、如何标注 tier、是否 rollback/observe”的决策。这里应尽量保持输入输出结构化，不直接操作 LiveKit Room、播放句柄或 chat context。

`integration/` 负责 LiveKit/framework 边界代码，包括 internal API patch 和 data-channel payload parser。任何需要碰外部协议、框架 underscore API、线缆格式的代码都优先放在这里，业务逻辑层只消费结构化对象。

`output/` 负责 Agent 输出侧副作用，包括 TTS 播放控制、取消、填充语、输出状态和相关 metrics。未来如果继续收敛 duck/mute/unduck，也应优先放在这个边界内。

`session/` 负责 LiveKit 会话事件的局部处理，例如 provider event、room data packet、idle watchdog、软打断 fallback。`UserTurnCoordinator` 也位于这里：它是用户 turn 候选的纯决策层，负责 transcript revision、短停顿合并、低 EOT 等待、voiceprint commit/reject 和去重状态；`TranscriptEchoGate` 负责 full-duplex 播放中 transcript 与当前 TTS 文本的内容回声判定；它们都不直接调用 LiveKit API，副作用仍由 `StreamingPipeline` 执行。session helpers 可以调用 `StreamingPipeline` 注入的回调，但不应反向拥有主流程。

`full_duplex/transcript_admission.py` 是 full-duplex STT transcript 进入 turn/evidence 逻辑前的入口门禁。当前只拥有两类无副作用裁决：voiceprint ownership 后的 post-turn residual transcript 抑制，以及播放中 agent 自身 TTS echo transcript 抑制。它不负责 EOT、commit、cancel/resume，也不处理 half-duplex PTT。

`full_duplex/semantic_interrupt_gate.py` 是 full-duplex transcript 触发 semantic interruption owner 前的纯门禁。它只判断当前 transcript 是否处在可打断窗口、是否被 cancel 后残留抑制、是否需要 attention admission；真正的 EOT/intent 决策和输出副作用仍由 `SemanticInterruptHandler`、`TurnPolicyRuntime` 与 effect handlers 执行。

#### 2.1.1 产品交互模式边界

Eidolon Channel 当前有两条一等体验路径，代码上必须分开表达：

1. **Push-to-talk / half-duplex**
   - 入口证据：`client.audio_state.ptt`、设备 playback state、服务端采集到的 press-to-release 音频段。
   - 所有按钮/手势只是显式输入信号，不是设备侧决策。ESP32 不判断“要不要打断”。
   - ESP32 release 后保留一个很短的采集尾窗（`EIDOLON_PTT_RELEASE_TAIL_MS`），尾窗结束后才发布 `ptt=false`，避免截断末尾音节。
   - `server.py` 按 session `interaction_mode=half_duplex` 直接进入 `HalfDuplexPttPipeline`；half-duplex 不再通过 `StreamingPipeline` 的 manual/STT transcript owner。
   - `HalfDuplexPttTurnController` 是 half-duplex turn owner：press 打开音频段，release 关闭音频段，一次性 STT 转写后只输出一个 terminal outcome：`commit`、`tap_to_stop` 或 `reject`。
   - PTT 专用阈值使用 `turn_policy.ptt`：`segment_stt_strategy`、`segment_min_audio_ms`、`segment_max_audio_ms`、`segment_min_rms_ppm`、`segment_tap_to_stop_max_audio_ms`。
   - 空按 / 无有效语音是显式协议结果：Channel 发布 session-local `eidolon.control` / `op=ptt.turn_status`，ESP32 只清理 UI 状态并 ACK，不参与 turn 裁决。
   - `session/client_control.py` 是 full-duplex streaming path 与 half-duplex segment path 共享的 `eidolon.control` envelope 与 timeline event helper；PTT 专用 `ptt.turn_status` payload / no-turn terminal 规则位于 `half_duplex/control.py`。
   - `full_duplex/client_preempt.py` 只处理 full-duplex explicit client preempt bridge，用于把客户端显式控制转换成输出抢占副作用；它不拥有 half-duplex PTT turn lifecycle。
   - PTT/tap-to-stop 是高优先级 explicit evidence；发生在 agent playback 时由 half-duplex owner 抢占输出并发送 `playback.stop`；发生在空闲时则按音频段长度/能量裁决为空按或真实 turn。

2. **流式自然语言 / full-duplex**
   - 入口证据：VAD speech start/end、STT interim/final、EOT score、client acoustic/playback telemetry、voiceprint、echo/backchannel/noise/hard-stop intent。
   - VAD speech start 只负责快速 soft duck / suspend；terminal decision 由 `TurnPolicyRuntime` + `InterruptionOrchestrator` + session handlers 统一输出。
   - 关键 terminal outcomes：`cancel`（hard-stop/真实插话）、`resume`/rollback（backchannel、false-start、noise）、`commit`（真实用户 turn）、`reject`（echo/低证据/非 owner 等）。
   - backchannel 和 false-start 的产品目标是快速恢复 agent 输出且不污染 context ledger；topic switch/correction/normal interrupt 的目标是稳定后 cancel，并只提交真实用户 turn。

`StreamingPipeline` 的实现位于 `full_duplex/pipeline.py`，是 full-duplex realtime path，不承载 half-duplex PTT 状态机，也不再保留 half-duplex direct-construction fallback。新增产品体验时，优先判断它属于 explicit client control、natural full-duplex evidence、turn ledger，还是 output side-effect，再放入对应模块；half-duplex PTT 上层逻辑放入 `half_duplex/`。

`context/` 负责对 conversation/chat context 的局部改写。当前只放被打断回复注入，后续如果扩展 memory recall/write 的会话内上下文拼装，也应先判断是否属于 agent 项目还是上游 brain 项目。

`pipeline/` 只封装 STT/TTS/VAD/LLM stage 的 provider-neutral 接口，避免把实时会话策略写进 provider stage。

### 2.2 导入规则

根目录只保留 entrypoints 和共享公共入口；不再 re-export `StreamingPipeline` / `HalfDuplexPttPipeline` / `BatchPipeline`，也不再保留 `streaming.py` 兼容 shim。新代码必须从 `full_duplex.*`、`half_duplex.*`、`integration.*`、`output.*`、`turn_policy.*`、`session.*`、`context.*` 等边界包直接导入。`session/__init__.py` 只作为 package marker，不聚合导出组件；session helper 必须从具体模块导入，例如 `session.room_data`、`session.client_control`、`session.interruption_orchestrator`。

### 2.3 Plugin 目录结构

```
eidolon/livekit/plugins/
├── __init__.py
├── stt/
│   └── bailian/
│       ├── stt.py           # BailianFunASRSTT (实现 livekit.agents.stt.STT)
│       ├── speech_stream.py # BailianFunASRSpeechStream (流式 WebSocket)
│       ├── connection_manager.py  # BailianConnectionManager (WebSocket 管理)
│       ├── registry.py
│       ├── config.py         # BailianSTTConfig
│       ├── models.py        # FunASR JSON 消息解析
│       └── test_*.py
├── tts/
│   └── sensetime/
│       ├── tts.py           # SenseTimeTTS + SenseTimeSynthesizeStream
│       ├── tts_client.py    # SenseTimeTTSClient (WebSocket 客户端)
│       ├── protocol.py      # WebSocket 协议常量
│       ├── config.py        # SenseTimeTTSConfig
│       └── test_*.py
├── vad/
│   └── firered/
│       ├── vad.py           # VAD + VADStream (实现 livekit.agents.vad.VAD)
│       ├── processor.py     # PvadProcessor (ONNX 推理, 共享单例)
│       ├── config.py        # FireredPvadConfig
│       └── test_*.py
└── eot/
    ├── __init__.py
    ├── _plugin.py          # EidolonEOTPlugin (LiveKit 插件注册)
    ├── config.py           # EidolonEOTConfig
    ├── log.py
    ├── version.py
    ├── models/
    │   ├── base.py         # EidolonEOTModel (实现 LiveKit _TurnDetector 协议)
    │   ├── chinese.py       # ChineseModel (中文 EOT)
    │   └── multilingual.py  # MultilingualModel (多语言 EOT)
    ├── impl/
    │   ├── eot_manager.py          # EotManager (ONNX 推理单例)
    │   ├── context_enhanced_eot.py # ContextEnhancedEot (上下文增强)
    │   ├── turn_end_policy.py      # TurnEndPolicy (动态静默阈值)
    │   ├── state.py                # TurnDetectionStateManager (VAD/ASR 状态)
    │   ├── eot_policy.py          # PolicyChain (切句决策规则)
    │   └── eot_backend.py
    ├── firered/
    │   └── eot.py          # FireRedChatEOT (keyword fallback EOT)
    └── test_eot_plugin.py
```

---

## 3. Pipeline 运行路径对比

| | Full-duplex `StreamingPipeline` | Half-duplex `HalfDuplexPttPipeline` | BatchPipeline |
|---|---|---|---|
| 触发条件 | `AGENT_MODE=streaming` 且 `interaction_mode=full_duplex` | `AGENT_MODE=streaming` 且 `interaction_mode=half_duplex` | `AGENT_MODE=batch` |
| 音频处理 | Open-mic 实时流式音频，VAD/STT/EOT 持续运行 | PTT press/release 内服务端采集完整音频段，release 后一次性 STT | 客户端上传完整音频 blob |
| turn owner | `TurnPolicyRuntime` + `InterruptionOrchestrator` + full-duplex session handlers | `HalfDuplexPttTurnController` | 无实时 turn owner |
| 打断能力 | 支持 barge-in / backchannel / false resume | 支持 PTT/tap-to-stop 抢占播放；不消费 streaming transcript/EOT | 不支持 |
| EOT 检测 | ChineseModel (EidolonEOTModel) | 不参与 PTT terminal decision | 无 |
| 编排方式 | `AgentSession` 管理实时输入输出，Channel owner 裁决打断/上下文 | `AgentSession` 只承载回复播放；PTT 音频段由 half-duplex pipeline 独立采集/转写/提交 | 手动顺序: STT→LLM→TTS |
| 适用场景 | 自然流式对话、barge-in、backchannel | 触屏 PTT、一问一答、可预期打断播放 | 异步音频处理、消息回复 |

### Full-duplex 数据流

```
用户音频 → LiveKit Room → AgentSession
  → VAD START_OF_SPEECH: 标记用户开始说话，必要时先 duck 当前 Agent 输出
  → STT INTERIM/FINAL: 形成 transcript evidence
  → turn_policy Tier 0-4: hard-stop / redirect / normal / noise / attention observe
  → StreamingPipeline 应用决策:
       cancel: 取消当前 Agent 输出并提交用户输入
       rollback: 恢复被 duck 的输出，不把短反馈当成打断
       observe: 只记录环境人声或低置信信号
  → VAD END_OF_SPEECH + EOT score: 判定用户是否说完
  → LLM 流式生成 → TTS 流式合成 → OutputController 发布音频到 Room
```

### Full-duplex 打断分层

当前实时打断是五层策略链，目标是在“足够快”和“不误杀自然陪伴感”之间折中：

| Tier | 典型输入 | 目标动作 | 延迟目标 | 主要代码 |
|---|---|---|---|---|
| Tier 0 hard stop | “别说了”、“停”、“打住” | 立即 cancel | 最快，尽量不等 EOT | `turn_policy/tiers/tier0_hard_stop.py` |
| Tier 1 redirect/correct | “不是”、“等一下”、“换个话题” | 快速 cancel，但要求更强 evidence | 约 100-300ms 稳定窗口 | `turn_policy/tiers/tier1_redirect.py` |
| Tier 2 normal interruption | 普通完整插话 | 等 EOT/final/preflight 或稳定文本后 cancel | 约 500-800ms，取决于 STT/EOT | `turn_policy/tiers/tier2_interruption.py` |
| Tier 3 noise/backchannel | “嗯”、“好”、“啊”、咳嗽 | rollback / unduck | 短暂 duck 后恢复 | `turn_policy/tiers/tier3_noise.py` |
| Tier 4 attention observe | agent 正在说话时的环境人声 | observe / ignore | 不 cancel | `turn_policy/tiers/tier4_attention.py` |

`integration/client_audio_state.py` 提供来自 Web/硬件客户端的播放态信号。`attention.enforce=true` 时，如果客户端明确处于 Agent speaking，普通环境人声默认不会直接进入 EOT cancel；只有 Tier 0、Tier 1、PTT/manual interrupt 或足够强的语义 evidence 才会更快取消。

### Batch 模式数据流

```
用户上传音频 blob → track_subscribed → 等待 track_ended
  → STT.recognize() (一次性 WebSocket)
  → LLM.chat() (OpenAI API)
  → TTS.synthesize() (流式 WebSocket)
  → room.local_publish_audio() → 发布到 Room
```

---

## 4. Plugin 架构详解

### 4.1 STT Plugin (Bailian FunASR)

```
BailianFunASRSTT (stt.py) ──实现 livekit.agents.stt.STT 接口
     │
     ├── stream() ──> BailianFunASRSpeechStream (speech_stream.py)
     │        │
     │        ├── AudioByteStream (accumulate audio chunks → 100ms 帧化)
     │        ├── BailianConnectionManager (WebSocket 连接管理)
     │        │        │
     │        │        └───> WebSocket: wss://dashscope.aliyuncs.com/api-ws/v1/inference
     │        │
     │        ├── send_loop: push_frame() → audio_bytes → WebSocket.send()
     │        └── recv_loop: WebSocket.recv() → FunASR JSON → SpeechEvent
     │                │
     │                ├── INTERIM_TRANSCRIPT → 实时中间结果
     │                └── FINAL_TRANSCRIPT  → 最终识别文字
     │
     └── _recognize_impl() ──> BailianConnectionManager (一次性 WebSocket)
                                  用于 BatchPipeline 的整段音频识别
```

**关键配置:**
- `model`: `fun-asr-realtime-2026-02-28`
- `api_url`: `wss://dashscope.aliyuncs.com/api-ws/v1/inference`
- `language`: 默认 `"zh"`
- `itn`: 逆文本规范化 (数字转阿拉伯数字, 默认开启)

---

### 4.2 TTS Plugin (SenseTime SenseAudio)

```
SenseTimeTTS (tts.py) ──实现 livekit.agents.tts.TTS 接口
     │
     ├── synthesize(text) ──> 内部调用 stream()
     │
     └── stream() ──> SenseTimeSynthesizeStream
              │
              ├── AudioByteStream (60ms PCM framing)
              ├── SenseTimeTTSClient (WebSocket 连接)
              │        │
              │        └───> WebSocket: wss://api.senseaudio.cn/ws/v1/t2a_v2
              │
              ├── send_loop: _input_ch (token text) → send_task_continue()
              └── recv_loop: task_continue 音频块 → output_emitter.push()
                      │
                      ├── 接收 hex PCM 音频数据
                      ├── 丢弃 leading silence (全零片段)
                      ├── 帧化为 60ms 块
                      └── output_emitter.push(pcm_bytes) → AudioFrame
```

**关键配置:**
- `model`: `SenseAudio-TTS-1.0`
- `voice`: `female_0033_a`
- `sample_rate`: `16000`
- `speed`: `1.0`

---

### 4.3 VAD Plugin (FireRed pVAD)

```
FireredPvadVAD.load() (vad.py) ──实现 livekit.agents.vad.VAD 接口
     │
     ├── VAD.load() ──> PvadProcessor (共享 ONNX session, lazy load)
     │
     └── stream() ──> VADStream
              │
              ├── PvadProcessor (ONNX 推理, 单例共享)
              ├── SpeakerEmbExtractor (说话人自适应, update_speaker())
              │
              ├── ThreadPoolExecutor (max_workers=1)
              ├── ExpFilter (alpha=0.5, 平滑概率曲线)
              │
              └── 状态机:
                  WAITING ──speech_prob >= 0.5 且持续 >= 100ms──> SPEAKING
                    ↑            │
                    └────── silence >= 400ms ──┘

              VADEvent 触发:
                START_OF_SPEECH ──> AgentSession 唤醒 STT
                END_OF_SPEECH  ──> AgentSession 触发 LLM

              关键参数:
                activation_threshold: 0.5 (语音概率阈值)
                min_speech_duration: 0.1s (最小语音时长)
                min_silence_duration: 0.4s (最小静音时长)
                prefix_padding_duration: 0.5s (语音前缀补白)
                max_buffered_speech: 60s (最大缓冲语音)
```

**设计亮点:** ONNX session 是**类级别共享单例** (`VAD._processor`)，多个 VADStream 实例共享同一个模型权重，只有流状态缓冲区是 per-stream 的。

---

### 4.4 EOT Plugin (End-of-Turn 检测)

```
ChineseModel (models/chinese.py) ──继承 EidolonEOTModel
     │
     └── EidolonEOTModel (models/base.py)
              │
              ├── EotManager (impl/eot_manager.py) ── ONNX 推理单例
              ├── ContextEnhancedEot ── 上下文增强打分
              ├── TurnEndPolicy ── 动态静默阈值
              ├── TurnDetectionStateManager ── VAD/ASR 状态
              └── PolicyChain ── 切句决策规则

实现 LiveKit _TurnDetector 协议:
  predict_end_of_turn(chat_ctx) → score [0.0, 1.0]
  unlikely_threshold(language)  → 阈值
  supports_language(language)  → True

分数含义:
  1.0 = 用户非常可能继续说话 (agent 不应打断)
  0.0 = 用户非常可能已经说完 (agent 可以开始回复)
```

**EOT 决策机制:**

```
用户说完 → VAD 触发 END_OF_SPEECH
  → STT 返回 FINAL_TRANSCRIPT
    → EOT Model 分析用户文本
      → predict_end_of_turn() 产出用户是否结束的 evidence
        → turn_policy/tiers 结合 transcript、client_audio_state、preflight intent
          → cancel / rollback / observe / submit
            → submit 时 AgentSession 开始 LLM 生成回复
              → TTS 开始合成并通过 OutputController 发布音频
```

**备选 FireRedChatEOT (keyword fallback):**

```
end_punct (。！？!」』"'") → 0.05 (句号感叹号问号 = 很可能是结尾)
mid_punct (，；:)：  → 0.8  (逗号分号 = 还在句子中间)
no_punct            → 0.95 (无标点 = 还没结束)
问号: 短文本 → 0.7, 长文本 → 0.2
```

---

## 5. 启动与连接时序图

```
┌──────────────┐     ┌────────────────────────┐     ┌──────────────────┐     ┌────────────────────────────┐
│   运维人员    │     │   Eidolon Agent Server  │     │   LiveKit Server  │     │   客户端 (Web/App/SDK)       │
└──────┬───────┘     └───────────┬──────────────┘     └────────┬─────────┘     └──────────────┬───────────────┘
       │                         │                             │                          │
       │ python -m eidolon.livekit.agent.server        │                          │
       │────────────────────────>│                             │                          │
       │                         │                             │                          │
       │                         │  1. AgentConfig.from_env()  │                          │
       │                         │     加载 .env + env vars    │                          │
       │                         │                             │                          │
       │                         │  2. _build_llm/stt/tts/vad │                          │
       │                         │     创建各 stage 实例       │                          │
       │                         │                             │                          │
       │                         │  3. AgentServer(ws_url, ...)│                          │
       │                         │───────────────────────────>│  WebSocket 连接 (7880)    │
       │                         │                             │                          │
       │                         │  4. server.rtc_session(     │  注册 _on_session 回调    │
       │                         │      _on_session)          │                          │
       │                         │                             │                          │
       │                         │  5. asyncio.run(server.run)│  进入事件循环             │
       │                         │     [保持连接]              │  [等待 job 调度]         │
       │                         │                             │                          │
       │                         │                             │       客户端加入 Room      │
       │                         │                             │<─────────────────────────│
       │                         │                             │                          │
       │                         │                             │  6. on_job_started 事件   │
       │                         │<────────────────────────────│                          │
       │                         │                             │                          │
       │                         │  7. _on_session(ctx)        │                          │
       │                         │     (在 worker 进程中)       │                          │
       │                         │                             │                          │
       │                         │  8. AgentConfig.from_env()   │                          │
       │                         │     (spawn 模式下重新加载)   │                          │
       │                         │                             │                          │
       │                         │  9. run_agent(ctx, cfg)      │                          │
       │                         │                             │                          │
       │                         │  10. SharedStageFactory      │                          │
       │                         │      (创建 stt/llm/tts/vad) │                          │
       │                         │                             │                          │
       │                         │  11. StreamingPipeline /     │                          │
       │                         │      BatchPipeline          │                          │
       │                         │                             │                          │
       │                         │  12. AgentSession(stt,llm,  │                          │
       │                         │      tts,vad,              │                          │
       │                         │      turn_detection=...)    │                          │
       │                         │                             │                          │
       │                         │  13. RoomIO(session, room) │                          │
       │                         │      await room_io.start()  │                          │
       │                         │                             │                          │
       │                         │  14. session.start(agent,  │                          │
       │                         │      room, ...)            │                          │
       │                         │                             │                          │
       │                         │     Agent 作为 participant  │                          │
       │                         │     加入 Room              │                          │
       │                         │───────────────────────────>│  Agent join Room           │
       │                         │                             │                          │
       │                         │  15. while room.is_connected│  [开始音频事件循环]       │
       │                         │     await asyncio.sleep(1) │                          │
       │                         │                             │                          │
       │                         │                             │       客户端开始说话        │
       │                         │                             │<─────────────────────────│
       │                         │                             │                          │
       │                         │  16. 音频流进入 Room        │                          │
       │                         │<────────────────────────────│  audio_frames             │
       │                         │                             │                          │
       │                         │  17. VAD stream 检测语音    │                          │
       │                         │     START_OF_SPEECH         │                          │
       │                         │                             │                          │
       │                         │  18. STT stream 实时转写    │                          │
       │                         │     INTERIM_TRANSCRIPT       │                          │
       │                         │───────────────────────────>│  transcription (可选)       │
       │                         │                             │                          │
       │                         │  19. VAD 检测语音结束       │                          │
       │                         │     END_OF_SPEECH           │                          │
       │                         │                             │                          │
       │                         │  20. STT stream 返回        │                          │
       │                         │     FINAL_TRANSCRIPT        │                          │
       │                         │                             │                          │
       │                         │  21. turn_policy + EOT      │                          │
       │                         │     判定 cancel/rollback/   │                          │
       │                         │     observe/submit          │                          │
       │                         │                             │                          │
       │                         │  22. LLM.chat(chat_ctx)     │                          │
       │                         │     流式生成回复 token       │                          │
       │                         │                             │                          │
       │                         │  23. TTS stream 流式合成    │                          │
       │                         │     AudioFrame PCM          │                          │
       │                         │                             │                          │
       │                         │  24. room.local_publish_   │                          │
       │                         │      audio(frame)           │                          │
       │                         │───────────────────────────>│  Agent 音频流发布         │
       │                         │                             │                          │
       │                         │  [循环等待下一个用户语音]    │                          │
       │                         │     while room.is_connected│                          │
       │                         │                             │                          │
```

---

## 6. Full-duplex 模式完整音频流链路

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     实时音频流 (StreamingPipeline)                          │
│                                                                          │
│  客户端麦克风 ──PCM 16kHz──> LiveKit Room ──audio_frames──> Agent Server   │
│                                       │                                   │
│                                       ▼                                   │
│                              ┌───────────────┐                          │
│                              │     VAD       │ FireRed pVAD               │
│                              │ FireredPvadVAD.stream()                  │
│                              │               │                           │
│                              │  状态机:      │                           │
│                              │  WAITING ──────── speech_prob >= 0.5      │
│                              │     │              且持续 >= 100ms        │
│                              │     │                                       │
│                              │     └───────────── silence >= 400ms ────  │
│                              └───────┬───────────────┘                   │
│                                      │                                   │
│                     START_OF_SPEECH / END_OF_SPEECH 事件                  │
│                                      │                                   │
│                                      ▼                                   │
│                              ┌───────────────┐                           │
│                              │     STT       │ BailianFunASRSTT.stream()  │
│                              │ BailianFunASR │                           │
│                              │               │                           │
│                              │  WebSocket:   │                           │
│                              │  dashscope    │                           │
│                              │  .aliyuncs.com│                           │
│                              └───────┬───────┘                           │
│                                      │                                   │
│              INTERIM_TRANSCRIPT ──── + ─── FINAL_TRANSCRIPT               │
│                      (实时中间文字)          (最终识别结果)                  │
│                                      │                                   │
│                                      ▼                                   │
│                              ┌───────────────┐                           │
│                              │ turn_policy   │ Tier 0-4 decision chain   │
│                              │ + EOT evidence│                           │
│                              │               │                           │
│                              │ hard stop → cancel                        │
│                              │ noise → rollback/unduck                   │
│                              │ normal → wait EOT/final/stability         │
│                              └───────┬───────┘                           │
│                                      │                                   │
│                       用户说完且未取消当前输出后:                         │
│                                      │                                   │
│                                      ▼                                   │
│                              ┌───────────────┐                           │
│                              │     LLM       │ livekit.plugins.openai.LLM │
│                              │  OpenAI-style │                           │
│                              │               │                           │
│                              │  流式 token 输出│                          │
│                              │  chat_ctx 上下文│                          │
│                              └───────┬───────┘                           │
│                                      │                                   │
│                                      ▼                                   │
│                              ┌───────────────┐                           │
│                              │     TTS       │ SenseTimeTTS.stream()      │
│                              │  SenseAudio   │                           │
│                              │               │                           │
│                              │  WebSocket:   │                           │
│                              │  api.senseaudio│                          │
│                              │  .cn/ws/v1    │                           │
│                              │  /t2a_v2      │                           │
│                              └───────┬───────┘                           │
│                                      │                                   │
│                              AudioFrame PCM 16kHz                         │
│                                      │                                   │
│                                      ▼                                   │
│                              ┌───────────────┐                           │
│                              │  RoomIO       │                           │
│                              │               │                           │
│                              │ room.local_   │                           │
│                              │ publish_audio │                           │
│                              └───────┬───────┘                           │
│                                      │                                   │
│                                      ▼                                   │
│                              LiveKit Room ──audio_frames──> 所有参与者    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 7. BatchPipeline 音频处理流程

```
用户 (上传音频 blob)
    │
    ▼
LiveKit Room ──track_subscribed 事件──> BatchPipeline
    │
    ▼
等待 track_ended (最多 30s 超时)
    │
    ▼
收集所有 audio_frames → _frames_to_pcm_blob() → PCM bytes
    │
    ▼
Step 1: STT
    BailianFunASRSTT._recognize_impl()
        │
        ├── BailianConnectionManager.connect()
        │       │
        │       └───> WebSocket: wss://dashscope.aliyuncs.com/api-ws/v1/inference
        │
        ├── conn.send_audio(audio_bytes)
        ├── conn.finish()
        └── 等待 server 返回 RESULT_GENERATED
                    │
                    ▼
            all_sentences[] → text = "".join(s.text)
                    │
                    ▼
            FINAL_TRANSCRIPT → transcript
    │
    ▼
Step 2: LLM
    LivekitLlmStage.chat(LlmInput(text=transcript))
        │
        ├── _chat_ctx.add_message(user)
        ├── llm.chat(chat_ctx=_chat_ctx, tools=fnc_ctx)
        │       │
        │       └───> OpenAI-compatible API (cfg.llm_base_url)
        │
        └── stream → full_text → LlmOutput(text=full_text)
    │
    ▼
Step 3: TTS
    TtsStage.synthesize(response_text)
        │
        ├── _tts.synthesize(text) → SenseTimeSynthesizeStream
        │       │
        │       ├── client.connect()
        │       │       │
        │       │       └───> WebSocket: wss://api.senseaudio.cn/ws/v1/t2a_v2
        │       │
        │       ├── send_task_continue(tokens)
        │       └── recv_loop: task_continue → hex PCM → AudioFrame
        │
        └── async for frame in synthesize():
                │
                └── room.local_publish_audio(frame)
                    │
                    ▼
            发布到 LiveKit Room
```

---

## 8. 关键设计决策

### 8.1 macOS multiprocessing spawn 兼容性

在 macOS 上，`AgentServer` 使用 `spawn` 模式创建 worker 子进程。`spawn` 会重新 `import` 整个模块，导致模块级全局变量被重置。

```python
# server.py
_agent_config: AgentConfig | None = None  # 主进程中设置

async def _on_session(ctx):
    cfg = _agent_config
    if cfg is None:
        # spawn 模式下重新加载 (因为 _agent_config 在子进程中为 None)
        cfg = AgentConfig.from_env()
    await run_agent(ctx, cfg)
```

### 8.2 Plugin 预注册 (避免 multiprocessing 冲突)

```python
# server.py 在 main() 之前执行
try:
    from livekit.plugins import openai as _  # noqa: F401
except ImportError:
    logger.warning("livekit-plugins-openai not installed, LLM support unavailable")
```

在主进程 import LiveKit 插件可以避免子进程中的 import 竞争条件。

### 8.3 插件即 LiveKit 接口实现

所有 STT/TTS/VAD/EOT 插件都**直接实现 LiveKit 的标准接口** (`livekit.agents.stt.STT`, `livekit.agents.tts.TTS`, `livekit.agents.vad.VAD`, `_TurnDetector`)，因此可以直接传入 `AgentSession`，无需额外适配层。

### 8.4 VAD 单例共享

`FireredPvadVAD` 的 ONNX session 是**类级别单例** (`VAD._processor`)，所有 VADStream 实例共享同一个模型权重，只有 per-stream 的状态缓冲区 (mel_buffer, gru_buffer) 是独立的。

### 8.5 VAD speaker 自适应

`VADStream.update_speaker(audio_16k_samples)` 允许在第一次用户说话后，用 1-5 秒的音频更新说话人嵌入向量，从而提高对该特定说话人的 VAD 准确率。

### 8.6 配置优先级

```
环境变量 > .env 文件 > 代码默认值
```

`AgentConfig.from_env()` 先读取 `.env` 文件，再通过 `os.environ.get()` 让环境变量覆盖。

---

## 9. 配置项说明

**单一真相源**：所有 env 变量、默认值与注释见 [`config/.env.example`](../../../config/.env.example)。下面只列核心运行时分组，详细 plugin-specific 配置请直接读该文件。

| 分组 | 关键变量 | 说明 |
|---|---|---|
| LiveKit 连接 | `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | 连接 LiveKit Server 的鉴权三元组 |
| Agent 监听 | `AGENT_HOST` / `AGENT_PORT` / `AGENT_MODE` | Agent server 绑定地址与流水线模式（`streaming` \| `batch`） |
| Agent 行为 | `AGENT_INSTRUCTIONS` / `AGENT_WELCOME_MESSAGE` / `AGENT_FALSE_INTERRUPTION_TIMEOUT` / `AGENT_AUDIO_SAMPLE_RATE` | 系统提示、欢迎语、误打断恢复阈值、链路采样率 |
| Provider 选择 | `STT_PROVIDER` / `TTS_PROVIDER` / `VAD_PROVIDER` | 选择哪一家插件——`bailian` \| `sensetime` / `firered_pvad` \| `firered` \| `silero` \| `none` |
| LLM (OpenAI 兼容) | `OPENAI_LLM_BASE_URL` / `OPENAI_LLM_MODEL` / `OPENAI_LLM_API_KEY` | LLM provider 的 endpoint、模型名与密钥 |
| Remote Agent (可选) | `REMOTE_AGENT_RPC_TARGET` / `REMOTE_AGENT_RPC_LOCALE` | 非空时走 gRPC `RemoteAgent.Session`，绕过上面的 OpenAI 兼容 LLM |
| Plugin-specific | `BAILIAN_STT_*` / `BAILIAN_TTS_*` / `SENSETIME_STT_*` / `SENSETIME_TTS_*` | 各 provider 的 URL、密钥、采样率、池大小、聚合阈值等——详见 `config/.env.example` |
| 模型路径覆盖 | `EIDOLON_EOT_MODEL_DIR` / `EIDOLON_FIRERED_PVAD_MODEL_DIR` / `EIDOLON_EOT_DEBUG_LOG` | 留空使用插件自带 bundled 模型 |

**命名约定**：`<PROVIDER>_<STAGE>_<FIELD>` 用于 plugin 配置；`<STAGE>_PROVIDER` 选择激活的 provider；`LIVEKIT_*` / `AGENT_*` 用于平台与 agent 身份。所有 env 文件值都可被 shell 环境变量覆盖（优先级更高）。

---

## 10. 启动命令

日常本地开发推荐通过 **eidolon_admin** 的 supervisord（`deploy/supervisor/available/channel.conf` + `with-env.sh` 加载 `config/.env`），或全栈 `./deploy/dev/run_all.sh`。

单独调试 worker（须先 `./deploy/dev/init.sh` 生成 `config/.env`）：

```bash
cd eidolon_channel && source .venv/bin/activate
EIDOLON_ENV=dev python -m eidolon.livekit.agent.server
```

配置加载顺序见上文「8.6 配置优先级」；默认读取 `config/.env`，也可用 `EIDOLON_CHANNEL_ENV_FILE` 覆盖路径。
