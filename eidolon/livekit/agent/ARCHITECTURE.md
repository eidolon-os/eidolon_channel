# LiveKit Agent Server 架构分析

> 最近更新: 2026-07-21
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
│  │        ├── resolve session metadata: interaction_mode / session_intent       │   │
│  │        │                                                                    │   │
│  │        └── _use_ptt_pipeline(interaction_mode) routes to a pipeline:         │   │
│  │             ├── ptt         → HalfDuplexPttPipeline: button segment turns    │   │
│  │             ├── half_duplex → StreamingPipeline: no barge-in                 │   │
│  │             └── full_duplex → StreamingPipeline: barge-in                    │   │
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
├── integration/
│   ├── __init__.py           # LiveKit/framework 外部契约边界
│   └── client_audio_state.py # client.audio_state data-channel payload parser/model
├── shared/
│   ├── pipeline.py           # BasePipeline: half/full duplex pipeline 共享基类
│   └── types.py              # PipelineState, callbacks, turn id 等共享类型
├── providers/
│   ├── stt.py                # SttStage: 封装 STT provider
│   ├── tts.py                # TtsStage: 封装 TTS provider
│   ├── vad.py                # VadStage: 封装 VAD provider
│   └── llm.py                # LLM stage / remote-agent bridge
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
│   ├── client_audio.py      # FullDuplexClientAudioStateView / RoomDataBridge
│   ├── client_preempt.py     # ExplicitClientPreemptHandler: full-duplex explicit client preempt
│   ├── context_ledger.py     # FullDuplexContextLedger: interrupted context runtime wiring
│   ├── framework_completed_turn.py # LiveKit completed-turn 唯一产品终态门禁
│   ├── interruption_effects.py # FullDuplexInterruptionEffects: output/framework effects
│   ├── lifecycle.py          # FullDuplexSessionLifecycle: AgentSession run/start/shutdown
│   ├── output_flow.py        # FullDuplexOutputFlow: duck mixer install + VAD duck arming
│   ├── playback_turn_evidence.py # playback-overlap completed-turn pure decision contract
│   ├── semantic_interrupt_gate.py # SemanticInterruptGate: transcript-triggered semantic interrupt gate
│   ├── session_turn_boundary.py # canonical user text / context-error boundary effects
│   ├── turn_completion.py    # completed-turn 与 voiceprint 状态编排入口
│   ├── turn_completion_policy.py # pure voiceprint result selection contract
│   ├── transcript_admission.py # TranscriptAdmissionGate: residual/echo transcript entry gate
│   ├── transcript_event.py   # FullDuplexTranscriptEvent: LiveKit transcript event normalization
│   ├── transcript_handler.py # FullDuplexTranscriptHandler: STT transcript entry routing
│   ├── transcript_hypothesis_reconciler.py # provider hypothesis → acoustic generation coverage
│   ├── transcript_recorder.py # accepted transcript side effects + generation reconciliation
│   ├── speech_lifecycle.py # FullDuplexSpeechLifecycle: VAD speech segment lifecycle
│   ├── user_state_event.py   # FullDuplexUserStateEvent: LiveKit user_state event normalization
│   ├── user_state_handler.py # FullDuplexUserStateHandler: user_state entry routing
│   ├── voiceprint_commit_state.py # voiceprint commit task/result/timeline state adapter
│   └── pipeline.py           # StreamingPipeline: full_duplex + half_duplex 流式 AgentSession pipeline
├── session/
│   ├── __init__.py           # package marker only; no broad component re-export facade
│   ├── agent_state.py        # AgentStateEffectHandler: agent state side effects
│   ├── attention_effects.py  # AttentionEffectHandler: attention admission effects
│   ├── client_control.py     # session-local eidolon.control envelope / event helpers
│   ├── decision_effects.py   # DecisionEffectApplier: decision timeline/metadata/effects
│   ├── duck_timeout.py       # DuckSuspendTimeoutHandler: duck deadline policy effects
│   ├── eot_model.py          # shared EOT model cache/loading helper
│   ├── messages.py           # LiveKit chat/message text helper
│   ├── provider_events.py    # ProviderEventObserver: LLM/brain/STT/TTS timeline 观测
│   ├── idle.py               # IdleWatchdog: 空闲定时与主动问候
│   ├── room_data.py          # LiveKit data packet 解析与分发
│   ├── interruption.py       # SoftInterruptController: 软打断补偿路径
│   ├── semantic_interrupt.py # SemanticInterruptHandler: STT/EOT 打断热路径副作用
│   ├── signals.py            # SessionSignalBridge: VAD/STT provider 信号桥接
│   ├── transcript_echo.py    # TranscriptEchoGate: full-duplex TTS echo content gate
│   ├── transcript_revision.py # shared pure transcript revision comparison
│   └── user_turn_coordinator.py # transcript revision assembler + product terminal ledger
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
    ├── timeline.py           # timeline event 记录
    └── turn_events.py        # bounded Channel session/turn event projection
```

### 2.1 边界原则

`full_duplex/pipeline.py` 是 `StreamingPipeline` 的实现，同时服务 `full_duplex` 与 `half_duplex` 两种流式会话（二者只在 barge-in / `allow_interruptions` 上不同），负责把 LiveKit `AgentSession`、Room、pipeline stage、打断决策、输出控制和观测串起来。根部 `streaming.py` 兼容入口已删除；新代码必须从 `eidolon.livekit.agent.full_duplex` 导入 streaming pipeline。streaming pipeline 可以持有流程状态，但不应承载可独立测试的副作用模块。

`turn_policy/` 负责“是否打断、如何标注 tier、是否 rollback/observe”的决策。这里应尽量保持输入输出结构化，不直接操作 LiveKit Room、播放句柄或 chat context。

`integration/` 负责 LiveKit/framework 的公开契约边界和 data-channel payload parser。禁止依赖或修改框架 underscore/private API；如果公开事件需要补充 provider 能力，应在 STT adapter boundary 映射为 provider-neutral evidence，业务逻辑层只消费结构化对象。

`output/` 负责 Agent 输出侧副作用，包括 TTS 播放控制、取消、填充语、输出状态和相关 metrics。未来如果继续收敛 duck/mute/unduck，也应优先放在这个边界内。

`session/` 负责 LiveKit 会话事件的局部处理，例如 provider event、room data packet、idle watchdog、软打断 fallback。`ProviderEventObserver` 是 LLM/brain/STT/TTS provider event 的观测 owner，负责 observer install、pending STT replay、STT turn-audio observe 与 timeline recording；pending STT provider event 的保留窗口、speech-start 前置归因窗口和最大缓存条数由 `observability.stt_pending_provider_event_window_ms` / `observability.stt_pending_provider_event_preroll_ms` / `observability.stt_pending_provider_event_max_count` 配置，不再硬编码在 observer 内。`StreamingPipeline` 不再保留 provider event 代理方法。`UserTurnCoordinator` 也位于这里：它只组装 transcript revision、记录 VAD segment、去重 framework completion，并维护 `open / committed / rejected` 三态；它不根据本地 timer、EOT 分数或 voiceprint callback 直接提交。`TranscriptEchoGate` 负责 full-duplex 播放中 transcript 与当前 TTS 文本的内容回声判定；`AssistantSpeechLedger` 只是在 TTS in-flight text 暂不可读时提供短窗口 fixed-speech 兜底。它们都不直接调用 LiveKit API，副作用由 full-duplex runtime owners 执行。

`full_duplex/transcript_admission.py` 是 full-duplex STT transcript 进入 turn/evidence 逻辑前的入口门禁。当前只拥有两类无副作用裁决：voiceprint ownership 后的 post-turn residual transcript 抑制，以及播放中 agent 自身 TTS echo transcript 抑制。它不负责 EOT、commit、cancel/resume，也不处理 `ptt` 分段路径。

`full_duplex/transcript_event.py` 是 LiveKit `user_input_transcribed` 事件的归一化边界。`FullDuplexTranscriptHandler` 在入口把框架事件转成 `FullDuplexTranscriptEvent`，后续 admission、recording、semantic gate 都消费稳定字段，避免 `event.transcript` / `getattr(event, ...)` 散落。

`full_duplex/transcript_handler.py` 是 full-duplex STT transcript 的入口路由。它按顺序执行 admission、accepted transcript recording、semantic interrupt gate、attention admission 和 base transcript forward；它不拥有 EOT、commit、cancel/resume 的 terminal decision。

`full_duplex/transcript_recorder.py` 是 accepted STT transcript 副作用 owner。它负责把已接收 transcript 写入 latest ASR、interruption owner、`UserTurnCoordinator`、timeline mark 和 EOT ASR update；它不做 transcript admission、semantic interrupt gate 或 final commit 裁决。provider 可选的 utterance identity、sequence、source span、word timing 与 boundary 先映射为 `TranscriptEvidence`；有稳定 identity/span 时优先据此归并，没有时才由 `TranscriptHypothesisReconciler` 使用文本等价 fallback。coordinator 只记录显式覆盖关系，不把未收到 FINAL 的旧 segment 伪装成 final。

`full_duplex/user_state_event.py` 是 LiveKit `user_state_changed` 事件的归一化边界。`StreamingPipeline` 只消费 `FullDuplexUserStateEvent.old_state/new_state` 与 `started_speaking/stopped_speaking` 判断；VAD start/end 后续的 speech segment lifecycle 交给 `FullDuplexSpeechLifecycle`。

`full_duplex/user_state_handler.py` 是 full-duplex `user_state_changed` 的入口路由。它负责 companion UI 状态映射、STT presence 信号和 speaking start/stop 分发；它不拥有 VAD start/end 后续的 turn commit、voiceprint、EOT 或 interruption terminal decision。

`full_duplex/speech_lifecycle.py` 是 full-duplex VAD speech segment 生命周期 owner。它在 speech start 时打开或续接候选、建立 timeline、同步 EOT/VAD、启动 voiceprint 采集并触发快速 duck/candidate；speech stop 只记录 acoustic boundary、结束本段 voiceprint 和处理输出侧 interruption evidence。它不 commit、reject、clear 用户输入，也不启动低 EOT deferred timer。

`full_duplex/interruption_effects.py` 是 full-duplex interruption output side-effect adapter。它承接 cancel / rollback / hold / explicit preempt 后对 LiveKit `AgentSession.interrupt()`、ducking output、soft interrupt timer、stable-signal recheck、`playback.stop` control 和 interrupted-context snapshot 的副作用；当 `InterruptionOrchestrator` 判定 semantic cancel 需要继续收集用户 speech 时，它只停止/取消输出并把候选切到 collecting 状态，真正的用户 turn commit 仍在 speech stop 后完成。它不做 turn policy、semantic classification、user-turn commit 或 context ledger 裁决。

`full_duplex/output_flow.py` 是 full-duplex output ducking flow owner。它只负责在 `AgentSession.start()` 后安装 `OutputDuckingController`，以及在 attention/VAD speech-start 触发时执行 `duck -> mark interrupt_started -> arm duck deadline`。cancel、rollback、hold、explicit preempt 的 terminal output effect 仍由 `FullDuplexInterruptionEffects` 执行。

`full_duplex/turn_completion.py` 是 full-duplex completed-turn 编排入口。它只管理 candidate/completed voiceprint task/result，并把 `on_user_turn_completed` 委托给 `FullDuplexFrameworkCompletedTurnGate`；不再拥有 VAD-end commit、低 EOT timer、post-speech direct commit 或 `clear_user_turn()`。

`full_duplex/turn_completion_policy.py` 是 voiceprint completion 的纯 contract，只负责从多个 candidate voiceprint 结果中选择有效结果；可信设备、provider error、wrong-speaker 和 fail-open/fail-closed 已由 `VoiceprintTurnObserver` 输出为 `commit_allowed / commit_reason`，framework 不再二次改写该结论。旧 low-EOT/short-statement defer policy 已删除。

`full_duplex/voiceprint_commit_state.py` 是 full-duplex voiceprint task/result/timeline 的 runtime state adapter。它只管理 candidate tasks 和 completed-turn task/result；不再存在 pending direct-commit task 集合。

`full_duplex/framework_completed_turn.py` 是 LiveKit framework `on_user_turn_completed` hook 的唯一产品终态 owner。它只消费 voiceprint owner 结果和 `InterruptionOrchestrator` 按 turn-id 保存的 typed verdict，再对齐 canonical text 并决定是否进入 LLM；不再从 timeline dict、中文短语、字符数或 EOT 重新推断 interruption 结果。这个边界原子收口同一个产品结果的三种投影：`UserTurnCoordinator` 的 `committed/rejected`、`FullDuplexStateMachine` 的 terminal transition，以及 timeline terminal flush；任何 reject 都不能只关闭其中一层。它不调用 `clear_user_turn()`，避免清掉已经开始的下一段音频。

当打断 owner 为 `channel` 且 `turn_policy.interrupt.intent_provider=llm` 时，现有 `SemanticInterruptHandler` 只在全双工重叠输入的 final 到达后异步获取语义证据，复用 `SharedStageFactory` 的 LLM 适配器。结果绑定候选、声学 generation、SpeechHandle 和 final 文本；过期结果不能操作新回答。附和/不确定结果在用户停止说话后恢复原缓冲，纯停止不进入回复生成，明确接管交给原有裁决 owner。超时或错误也恢复原播报，不冒充高置信度判断。共享配置保留 `none`，沿用原 EOT 策略的兼容路径及其已知语义缺口；`llm` 由集成测试显式启用，尚未通过生产 RPC/RTC 链路验收。

`server.run_agent` 先解析 participant metadata，再通过既有 `apply_interaction_mode` 得到 session policy，最后用该策略构造 factory 和 pipeline。half duplex/PTT 将 `intent_provider` 设为 `none`，既不构造也不调用可选意图模型，不修改共享配置；选择 SDK 原生打断时，factory 的既有 owner 门禁同样禁止构造通道意图客户端。仅通道拥有打断且显式启用模型意图的全双工沿用 stage warmup 预热分类模型，预热有截止时间且不消费用户输入。意图客户端只对已知的 DeepSeek 官方 endpoint 自动设置供应商专用思考参数，其他 endpoint 仅透传显式配置，不根据模型名称猜测协议。

`InterruptAwareTurnDetector` 安装于通道拥有打断的全双工，包括 `none` 和 `llm` 两种配置；half duplex/PTT 和 SDK 原生打断不安装。它通过 SDK 公开 turn-detector 协议，先对当前候选转写取快照，再等候在途意图裁决（如有）和已取消 SpeechHandle 的公开 `wait_for_playout()` 收尾，随后仍由原 EOT 模型返回完成度。该适配用于对齐 Channel 取消与 SDK 回复调度，避免新问题在进入完成 hook 前被不可打断的旧播报丢弃；不接管 SDK 调度。完整句最小等待绑定既有 0.25 秒策略，不完整句最大等待继承 Session 配置（SDK 默认 3 秒）。

`full_duplex/playback_turn_evidence.py` 是 active interruption candidate 在 framework-final evidence 到达时使用的纯决策 contract。它只消费 typed `turn_policy.Decision`，输出 `should_apply / continue_to_llm / reason`；不读取 timeline dict、LiveKit、Room、AgentSession 或 pipeline 私有状态。由 policy 确认的 `NORMAL_INTERRUPT`、`CORRECTION`、`TOPIC_SWITCH` 接管共用 `intent_requires_reply`，在 CANCEL 后成为用户 turn，不依赖关键词；hard-stop 与 rollback 只终结 interruption，不进入 LLM。

`full_duplex/context_ledger.py` 是 full-duplex interrupted context ledger wiring。底层 capture/consume 算法仍由 `context/InterruptedContextManager` 负责；这里只把 full-duplex runtime 的 `AgentSession`、TTS factory、ducking playback offset、EOT config 和 timeline observability 传入，避免 `StreamingPipeline` 直接知道 context ledger 细节。ledger 只标注本次 output cancel 新捕获的上下文，不会用旧 pending context 重标后续候选；上下文跨 rejected control turn 保留，只在 framework gate 接受下一条真实用户 turn 时 claim 一次并写入 LiveKit 为该次生成提供的临时 `turn_ctx`。它不写持久 `session.history`，因此只帮助当前续答，不会污染长期对话历史，也不会被 hard-stop、wrong-speaker、noise 或重复 completion 消费。

`full_duplex/lifecycle.py` 是 full-duplex AgentSession 生命周期 owner。它负责 `AgentSession` 创建、event handler 绑定、stage warmup、RoomData/Voiceprint bridge 安装、`session.start()`、duck mixer/filler 输出准备、EOT session start、idle watchdog/proactive consumer 启停、session close prompt room teardown，以及 shutdown 顺序。打断 owner 只通过公开 `turn_handling.interruption` 配置选择：LiveKit native-adaptive owner 自行检测和执行；Channel owner 关闭框架自动打断但保留输入音频，在策略确认后调用公开 `session.interrupt(force=True)`。`StreamingPipeline` 保留 `run()` / `shutdown()` 公共入口，但不再承载这些生命周期私有步骤。

`full_duplex/semantic_interrupt_gate.py` 是 full-duplex transcript 触发 semantic interruption owner 前的纯门禁。它只判断当前 transcript 是否处在可打断窗口、是否被 cancel 后残留抑制、是否需要 attention admission；真正的 EOT/intent 决策和输出副作用仍由 `SemanticInterruptHandler`、`TurnPolicyRuntime` 与 effect handlers 执行。

`full_duplex/state_machine.py` 是 full-duplex turn contract 的无副作用可观测状态机。它把 timeline 统一标注为 `idle`、`user_speech_open`、`provisional_duck`、`evidence_arbitration`、`accepted_interruption`、`rejected_interruption`、`user_turn_pending`、`user_turn_committed`、`user_turn_rejected` 等阶段，并为每次 transition 标出 `side_effect=none|reversible|irreversible`。当前 contract 原则是：VAD 后的 duck/suspend 属于可回滚阶段；`AgentSession.interrupt()`、`playback.stop`、framework completed-turn 放行、用户 turn commit 属于不可逆或高副作用阶段，必须由 evidence/artifact gate 或 explicit client preempt 后的 terminal owner 触发。

`session/user_turn_coordinator.py` 的 owner ledger 是 full-duplex 用户 turn ownership 的纯状态边界。每个 candidate 只经历 `open / committed / rejected`，并记录 provisional、accepted、rejected、merged transition。它不分类语义、不包含中文短语表，也不删除已经 admission 的 transcript segment。在 framework completed 前出现的新 VAD speech 仅可在 `turn_policy.eot.speech_merge_grace_ms` 指定的声学连续窗口内续接同一 open candidate；coordinator 没有私有时间默认值，超窗 speech 会 supersede 旧 candidate。STT `FINAL` 关闭一个 revision stream；其后的 `INTERIM` 新建 sentence segment，避免旧 `final_text` 永久遮蔽后续文本。真正的 `open → committed` 转移同时写入 candidate `committed_at` 与 timeline `turn_committed_at`，使 commit 状态和延迟观测共享同一时钟边界。

#### 2.1.1 产品交互模式边界

`interaction_mode` 是 session 级 **三选一互斥** 模式（设备 header 声明 → hub 盖进 participant metadata → Channel 在构造 `AgentSession` 前解析一次；缺失/不可解析降级到安全的 `half_duplex`，该模式从不 barge-in）。路由入口是 `server.py` 的 `_use_ptt_pipeline(interaction_mode)`：只有 `ptt` 走 `HalfDuplexPttPipeline`（按钮分段轮次）；`half_duplex` 与 `full_duplex` 都走 `StreamingPipeline`（同一套流式 STT + EOT 轮次提交），二者只在 barge-in 上不同——barge-in 完全由 `allow_interruptions` 承载（`runtime/interaction_mode.py` 的 `apply_interaction_mode` 对 `half_duplex` 与 `ptt` 都返回 `False`），而不是靠另一条 pipeline。三种模式代码上必须分开表达：

1. **Push-to-talk（`ptt` → `HalfDuplexPttPipeline`）** — 按住说话（hold-to-talk）：只在设备按钮按住期间开麦，release 是显式的 turn 结束；其余时间关麦。面向可穿戴 / 按键设备（如 waveshare 2.06）。
   - 入口证据：`client.audio_state.ptt`、设备 playback state、服务端采集到的 press-to-release 音频段。
   - 所有按钮/手势只是显式输入信号，不是设备侧决策。设备不判断“要不要打断”。
   - 设备 release 后保留一个很短的采集尾窗（`EIDOLON_PTT_RELEASE_TAIL_MS`），尾窗结束后才发布 `ptt=false`，避免截断末尾音节。
   - `server.py` 只对 `interaction_mode=ptt` 进入 `HalfDuplexPttPipeline`；ptt 路径不消费 `StreamingPipeline` 的流式 STT transcript / EOT owner。
   - `HalfDuplexPttTurnController` 是 ptt turn owner：press 打开音频段，release 关闭音频段，一次性 STT 转写后只输出一个 terminal outcome：`commit`、`tap_to_stop` 或 `reject`。
   - PTT 专用阈值使用 `turn_policy.ptt`：`segment_stt_strategy`、`segment_min_audio_ms`、`segment_max_audio_ms`、`segment_min_rms_ppm`、`segment_tap_to_stop_max_audio_ms`。
   - 空按 / 无有效语音是显式协议结果：Channel 发布 session-local `eidolon.control` / `op=ptt.turn_status`，设备只清理 UI 状态并 ACK，不参与 turn 裁决。
   - `session/client_control.py` 是 streaming path 与 ptt segment path 共享的 `eidolon.control` envelope 与 timeline event helper；PTT 专用 `ptt.turn_status` payload / no-turn terminal 规则位于 `half_duplex/control.py`。
   - PTT/tap-to-stop 是高优先级 explicit evidence；发生在 agent playback 时由 ptt owner 抢占输出并发送 `playback.stop`；发生在空闲时则按音频段长度/能量裁决为空按或真实 turn。

2. **半双工自动录音（`half_duplex` → `StreamingPipeline`，无 barge-in）** — session 开始后自动录音（无按钮），设备 **无可用 AEC 参考**（如 m5stack-stackchan），agent 播放期间设备关麦，因此不可被打断。
   - 与 `full_duplex` 共用同一条 `StreamingPipeline`：流式 VAD/STT + EOT 轮次提交完全相同；唯一差别是 barge-in 关闭——`apply_interaction_mode` 返回 `allow_interruptions=False` + `attention.enabled=False` + `interrupt.intent_provider=none`（不做 barge-in / evidence-gate / 打断猜测，也不构造意图客户端）。
   - `StreamingPipeline` 保留真实 `interaction_mode=half_duplex`（不再改写成 `full_duplex`）；barge-in 关闭只经由 `allow_interruptions`，其余流式编排与 full-duplex 一致。
   - 因为 mic 在播放期关闭，实践中不会出现“播放中打断”证据；轮次边界仍来自 VAD/EOT 提交，而不是按钮 release。

3. **流式自然语言 / 全双工（`full_duplex` → `StreamingPipeline`，barge-in）** — open mic + 设备硬件 AEC（如 esp-box-3），用户可以在 agent 说话时插话。
   - 入口证据：VAD speech start/end、STT interim/final、EOT score、client acoustic/playback telemetry、voiceprint、echo/backchannel/noise/hard-stop intent。
   - `allow_interruptions=True` + attention soft-interrupt + content echo gate。
   - `FullDuplexSpeechLifecycle` 负责 VAD speech start/end 的语音段生命周期；speech start 只负责快速 soft duck / suspend 和候选打开，terminal decision 由 `TurnPolicyRuntime` + `InterruptionOrchestrator` + session handlers 统一输出。
   - `FullDuplexInterruptionEffects` 负责 terminal decision 之后的输出/框架副作用；它不判断“要不要打断”。
   - 关键 terminal outcomes：`cancel`（hard-stop/真实插话）、`resume`/rollback（backchannel、false-start、noise）、`commit`（真实用户 turn）、`reject`（echo/低证据/非 owner 等）。
   - backchannel 和 false-start 的产品目标是快速恢复 agent 输出且不污染 context ledger；topic switch/correction/normal interrupt 的目标是稳定后 cancel，并只提交真实用户 turn。
   - `full_duplex/client_preempt.py` 只处理 full-duplex explicit client preempt bridge，用于把客户端显式控制转换成输出抢占副作用；它不拥有 ptt segment turn lifecycle。无 speech/VAD timeline 的 explicit client PTT 会由 `StreamingPipeline` 创建 control-only timeline，记录 `cancel/hard_stop/client_ptt`、`playback.stop` 与 interrupted-context evidence，不等待 STT，也不提交用户 turn。
   - `full_duplex/client_audio.py` 拥有 full-duplex `client.audio_state` 的新鲜度视图、播放态判断，以及 room data -> explicit preempt handler 的桥接；pipeline 不再重复实现这些判断。

#### 2.1.2 Full-duplex contract 收敛状态（2026-07-17）

本轮已完成产品输入终态 owner 切换：

- 保留 LiveKit automatic turn lifecycle 和 Eidolon 自有 EOT/semantic interruption 策略，不切换 manual mode。
- `FINAL -> 后续 INTERIM` 按新 sentence segment 组装，framework completion 使用 coordinator canonical text。
- VAD-stop、output interruption、voiceprint callback、timer 都不能 commit/reject/clear 用户 turn；正常 commit 只来自 framework-completed gate。
- 删除 `turn_commit.py`、`deferred_commit_state.py`、`post_speech_interruption.py` 及对应测试；删除 6 个仅服务旧 defer/statement merge 的配置字段。
- framework 严格尊重 `VoiceprintTurnObserver` 已输出的 `commit_allowed / commit_reason`：可信配对设备与 provider error 的 fail-open、wrong-speaker 的 reject 都只在 observer policy 产生一次；framework 不再对 inconclusive 结果二次 fail-open。
- interruption deadline 只返回 provisional decision，`DecisionEffectApplier` 是唯一的最终决策记录点；无 transcript 候选使用 owner 的短 timeout 上限，不再被通用 duck buffer 延长或出现 `rollback -> hold -> rollback` 重复翻转。
- `InterruptionOrchestrator.resolve()` 生成并按 turn-id 保存 `confirmed_cancel / rejected_resume / expired_resume / rejected_candidate` typed verdict；framework terminal gate 只消费 verdict，已删除 playback artifact 字符启发式、timeline dict fallback 和对应旧 latency 字段。
- 已删除 coordinator 中“我再说/重新说”等短语表与 meta-tail drop。参数化回归用中文单字、英文短词和 meta language 证明 terminal boundary 对文本内容无感。
- 从真机十三轮轨迹提取的精确事件序列已固定为回归：`好啊。 -> 那你记下来吧，这是我们约定`、`好的。 -> 那太好`、`OK呀。 -> 到时候我还可以带几个朋友`，以及 interim-only 和重复 framework completion。
- candidate timeline 在 speech open 时显式记录 `interruption_target.response_turn_id`；candidate 侧的 duck/cancel/ACK 与 response 侧的 interrupted context/terminal outcome 通过该关系 join，不再按时间邻近猜测。播放已经结束后的无 target hard-stop 只算 unscoped control turn，不能污染响应打断延迟。
- 当前完整 agent 回归 `808 passed`，完整 benchmark 回归 `138 passed`；本次 context/turn/HIL 涉及的组合回归 `169 passed`；Ruff 与 diff check 通过。

2026-07-17 Box-3 dogfood 已验证正常欢迎、普通提问、播放中 hard-stop、打断后继续提问和 session 正常结束。首个“停，不要说了”在播放中触发 cancel（约 `244.9ms`）与设备 `playback.stop`（约 `242.1ms`），没有进入 Brain；随后“你还能听到我吗”正常 commit 并回复。最后一个“不要说了”发生在播放接近完成后，没有 response target，也没有进入 Brain；设备最终 `idle_normal_end` 并回到待命态。

该轨迹同时暴露并由架构约束修复了两个问题：framework gate 虽拒绝 hard-stop，但此前没有同步投影 FSM `user_turn_rejected`，会让下一轮从残留 `user_turn_pending` 开始；interrupted context 此前只有 capture 没有消费入口，而且旧 pending context 会被后续无输出控制轮重复标注。现在 reject terminal 在同一边界完成三层收口，context 在下一条 accepted turn 的临时 `turn_ctx` 中 exactly-once 消费。

此前 `box3-terminal-purity-20260716-r8` 的实房间结果仍作为 interruption-confidence 基线，两例均 `real_call_verified=true`：

- backchannel 无 transcript 候选在 `800.7ms` rollback，功能与 `<=900ms` 体验门禁通过；
- 主人完整追问通过 `interruption_verdict:continue` 正确进入 LLM，没有被短文本/播放 artifact gate 拒绝；可行动转写约 `539.0ms`，但仍到 final 才在 `1954.3ms` cancel，用户音频结束到新 agent 音频为 `2637ms`。

加载本次实现的新 worker 已于 2026-07-17 完成第二轮 Box-3 复核。两次播放中 hard-stop 均显式关联目标 response，candidate 最终同时落为 coordinator `rejected`、FSM `user_turn_rejected` 和 `channel.turn.rejected`，`unexpected_transition_count=0`，且没有 Brain turn。后续 accepted turn 分别在约 `6.0s`、`7.7s` 后 exactly-once 消费 interrupted context；“你继续说”正确续接被打断的“没吃饭”回复，持久消息表没有 system hint。session 最终 `idle_normal_end`。

本里程碑按功能与架构正确性收口。两次 hard-stop 的 speech-start -> confirmed-cancel 为 `495.9ms`、`573.8ms`；soft duck 约 `0.37ms`。第二个样本超过 HIL 单样本严格门禁 `500ms`，但低于 live-room P95 acceptable `800ms`。该结果作为后续性能采样项保留，不重新打开 terminal ownership，也不引入短语 fast-path。EOT（用户是否说完）与 interruption confidence（当前声音是否真在抢话）继续作为两条独立证据轴；Eidolon 继续拥有 terminal policy，LiveKit 或学习型打断模型只能作为结构化 evidence，不接管轮次 owner。

#### 2.1.3 自动化链路与真机 dogfood 的一致性

| 链路 | 真实组件 | 未覆盖部分 | 用途 |
| --- | --- | --- | --- |
| `policy` | Eidolon policy / evidence 纯函数 | RTC、provider、设备 | 穷举决策矩阵，不作 dogfood gate |
| `headless` | Channel session 编排与 effect contract | 真实 RTC/provider/设备 | 验证 owner 和副作用顺序 |
| `component` | 真实 VAD/STT/TTS provider 直调 | LiveKit room、worker 调度、设备 | provider 可用性与分段延迟 |
| `livekit_room` | 真实 LiveKit room/音轨、持久 worker、VAD/STT、Eidolon turn policy/owner、identity resolver、brain、TTS、timeline | ESP32 I2S/AEC/AFE、真实 Wi-Fi/时钟抖动、物理扬声器回馈；`client.audio_state` 与 `playback.stop ACK` 为模拟 | dogfood 必要前置 gate，但不等于 dogfood |
| Box-3 dogfood | 以上全部 + 真设备采集/AEC/上行/播放/控制执行 | 不具备自动可重放性 | 只在 `livekit_room` 体验 gate 通过后做短程验证 |

自动化必须用已注册的真实 user/device identity 做 preflight，并要求 `real_call_verified=true`。转写归因只把 benchmark participant 的 transcript 计为 user，agent 同步的 TTS/welcome transcript 不得冒充用户 STT；timeline 路径必须展开 `~`，否则会产生“实际 worker 有记录，报告说无 timeline”的假阴性。

Dogfood 诊断必须把设备与服务端串成同一条证据链。每个 full-duplex timeline 固定记录
`room_name`、runtime `participant_identity`、Channel `turn_id`，并由 brain provider
event 继续关联 brain `turn_id/request_id`。Channel `turn_id` 同时作为下一次 Brain
`StartTurn.trace_id`，retry 不换 trace；Agent 持久化后由 Mission Control 按 trace 合并。
`observability/turn_events.py` 只用有界 `put_nowait` 投影 safe phase/milestone/terminal
事实，专用 writer 才接触 SQLite；队列满只累计 dropped count，不阻塞语音热路径。
候选用户轮次与正在生成/播放的响应允许在 full-duplex 打断窗口内并存，不能再共享
一个可替换的 timeline 指针。`session/agent_output_coordinator.py` 是响应 timeline 的
唯一所有者：STT/VAD/EOT 继续写当前候选，Brain/LLM/TTS/agent playback 只写已 commit
的响应。接受打断时先以 `interrupted_by_user` 关闭被打断响应，再独立关闭 hard-stop /
redirect 候选；非可恢复 LLM/TTS/session error 直接关闭它所绑定的响应为 failed。
session close 必须分别清算仍存活的响应和候选，任何一个都不得依靠下一轮覆盖来结束。
设备现有 `client.audio_state.seq` 不参与策略，
但 Channel 会保留最近 16 个状态事件，并累计 `client_audio_state_gap_count` /
`client_audio_state_reordered_count`；这用于区分“设备未发或链路丢包”和“包已到达但
STT/EOT/Brain/TTS 未继续”。原始证据保存在 `~/eidolon/logs/channel/worker.log` 与
`~/eidolon/logs/channel/turn-timeline.jsonl`，真机串口日志用于补充 I2S/AEC、采集 RMS、
播放 RMS 和本地 `playback.stop` 执行结果。上述字段均为 observe-only，不得反向成为
terminal owner 或语义启发式。

`StreamingPipeline` 的实现位于 `full_duplex/pipeline.py`，是 `full_duplex` 与 `half_duplex` 共用的流式 realtime path（二者只在 barge-in / `allow_interruptions` 上不同），不承载 `ptt` segment 状态机，也不再保留 direct-construction fallback。新增产品体验时，优先判断它属于 explicit client control、natural full-duplex evidence、turn ledger，还是 output side-effect，再放入对应模块；`ptt` 分段轮次上层逻辑放入 `half_duplex/`。

`context/` 负责对 conversation/chat context 的局部改写。当前只放被打断回复注入，后续如果扩展 memory recall/write 的会话内上下文拼装，也应先判断是否属于 agent 项目还是上游 brain 项目。

`providers/` 只封装 STT/TTS/VAD/LLM stage 的 provider-neutral 接口，避免把实时会话策略写进 provider stage。具体模型集成与模型资源继续放在 `eidolon.livekit.plugins`。

`shared/` 只放 half/full duplex 都会消费的运行时基础类型，例如 `BasePipeline`、`PipelineState`、callbacks 和 turn-id helper。

### 2.2 导入规则

根目录只保留 entrypoints 和共享公共入口；不再 re-export `StreamingPipeline` / `HalfDuplexPttPipeline`，也不再保留 `streaming.py` 兼容 shim。新代码必须从 `full_duplex.*`、`half_duplex.*`、`providers.*`、`shared.*`、`integration.*`、`output.*`、`turn_policy.*`、`session.*`、`context.*` 等边界包直接导入。`session/__init__.py` 只作为 package marker，不聚合导出组件；session helper 必须从具体模块导入，例如 `session.room_data`、`session.client_control`、`session.interruption_orchestrator`。

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

路由：`_use_ptt_pipeline(interaction_mode)` 只把 `ptt` 送入 `HalfDuplexPttPipeline`；`full_duplex` 与 `half_duplex` 共用 `StreamingPipeline`，仅 barge-in（`allow_interruptions`）不同。

| | `StreamingPipeline`（`full_duplex` / `half_duplex`） | `HalfDuplexPttPipeline`（`ptt`） |
|---|---|---|
| 触发条件 | participant metadata `interaction_mode=full_duplex` 或 `half_duplex` | participant metadata `interaction_mode=ptt` |
| 音频处理 | Open-mic 实时流式音频，VAD/STT/EOT 持续运行（`half_duplex` 播放期设备关麦） | 按钮 press/release 内服务端采集完整音频段，release 后一次性 STT |
| turn owner | `TurnPolicyRuntime` + `InterruptionOrchestrator` + full-duplex session handlers | `HalfDuplexPttTurnController` |
| 打断能力 | `full_duplex` 支持 barge-in / backchannel / false resume；`half_duplex` 关闭 barge-in（`allow_interruptions=False`） | 支持 PTT/tap-to-stop 抢占播放；不消费 streaming transcript/EOT |
| EOT 检测 | ChineseModel (EidolonEOTModel) | 不参与 PTT terminal decision |
| 编排方式 | `AgentSession` 管理实时输入输出，Channel owner 裁决打断/上下文 | `AgentSession` 只承载回复播放；按钮音频段由 ptt pipeline 独立采集/转写/提交 |
| 适用场景 | `full_duplex`：自然流式对话、barge-in、backchannel（open mic + AEC）；`half_duplex`：无 AEC 设备自动录音、一问一答、播放期不可打断 | 触屏/按键 PTT、按住说话、可预期打断播放 |

### Full-duplex 数据流

`half_duplex` 走同一条 `StreamingPipeline` 数据流，唯一差别是不 barge-in（`allow_interruptions=False`）且设备播放期关麦，因此下图的 duck / cancel / rollback 打断分支在 `half_duplex` 不触发，轮次靠 VAD END_OF_SPEECH + EOT 提交。

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

### PTT segment 数据流（`ptt`）

```
client ptt_pressed → HalfDuplexPttTurnController 打开 hold 窗口
  → PttSegmentRecorder 收集 press/release 内音频
  → client ptt_released → 关闭 segment，按时长/RMS 判定 turn 或 tap-to-stop
  → PttSegmentTranscriber 对闭合音频段一次性转写
  → AgentSession.generate_reply() → TTS 播放
  → PTT/tap-to-stop 可在播放中抢占输出
```

### Benchmark 模式边界

`benchmark/cases/` 也按产品模式拆分：

- `full_duplex/`：open-mic natural conversation、barge-in/backchannel/false-start/ambient guard，以及 full-duplex explicit client control。
- `half_duplex/`：push-to-talk（PTT）segment owner；必须用 `--livekit-interaction-mode ptt` 跑（只有 `ptt` 路由到 `HalfDuplexPttPipeline`）。目录名与 suite-set key 沿用旧的 `half_duplex` 标签（与 `HalfDuplexPttPipeline` 类 / `half_duplex/` 包一致），但交互模式是 `ptt`。
- `shared/`：不绑定单一 room mode 的 deterministic/shared regression。
- `legacy/`：历史 suite；默认 E2E gate 不运行。

E2E A/B runner 通过 `suite_mode` 校验 `--livekit-interaction-mode`，并要求 room gate case 显式声明 `agent_audio_response`。报告按 `functional_outcome_passed` 与 `experience_slo_passed` 分层，避免把“动作正确但慢”和“terminal decision 断路”混为一类。

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
                                  用于组件级 one-shot 识别能力
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
       │                         │  11. resolve metadata then   │                          │
       │                         │      choose half/full pipe   │                          │
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
| Agent 监听 | `AGENT_HOST` / `AGENT_PORT` | Agent server 绑定地址 |
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
