# Agent 目录逻辑边界设计

> 日期：2026-06-07  
> 范围：`eidolon/livekit/agent/`  
> 目标：先定义清晰的目录职责边界，再分阶段整理代码结构，避免把 `streaming.py` 机械拆成一组没有架构语义的文件。

## 背景

当前 `agent/` 已经有一些自然形成的边界：

- `turn_policy/`：打断、attention、intent、evidence、tier policy。
- `pipeline/`：STT/TTS/VAD/LLM 的 provider-neutral stage。
- `observability/`：timeline 与 metrics。
- `runtime/`：admin client、device/user resolver、token signer。
- `eidolon_agent_rpc/`：连接 Eidolon Agent 后端 brain。

主要维护压力来自根目录混合了多类职责，尤其是 `streaming.py` 已经承担了过多角色：

- LiveKit session 启动和 AgentSession 编排。
- room data/client audio state 监听。
- idle watchdog。
- provider event observability。
- attention gate。
- EOT/interrupt decision 热路径。
- duck/unduck/cancel 等输出副作用。
- interrupted context snapshot/injection。
- turn control metadata 发布。

因此后续重构不应先按函数块随意搬家，而应先确立逻辑边界。

## 目标目录模型

```text
eidolon/livekit/agent/
  server.py
  factory.py

  session/
    streaming.py
    batch.py
    room_events.py
    provider_events.py
    idle.py

  pipeline/
    base.py
    llm.py
    stt.py
    tts.py
    types.py
    vad.py

  turn_policy/
    attention.py
    constants.py
    decider.py
    evidence.py
    intent_classifier.py
    runtime.py
    tiers/

  output/
    controller.py
    filler.py

  context/
    interrupted.py

  identity/
    admin_client.py
    resolver.py
    token_signer.py

  brain_client/
    grpc_llm.py
    session.py
    v1/

  observability/
    metrics.py
    timeline.py

  compat/
    ducking.py
    interrupt_decider.py
```

这不是一次性迁移目标，而是后续判断文件归属的架构基线。

## 分层职责

### `server.py`

负责 LiveKit Agent Server 的进程入口、配置加载、Job/session 生命周期入口。

允许：

- 读取 `AgentConfig`。
- 注册 LiveKit session/job handler。
- 构建高层 factory 并启动 session pipeline。

不应负责：

- 实时打断策略。
- 输出 duck/cancel 细节。
- provider event 解析。
- user/device 身份解析细节。

### `factory.py`

负责从配置构建 STT/TTS/VAD/LLM/brain client 等运行组件。

允许：

- provider selection。
- stage construction。
- benchmark/audio-only component construction。

不应负责：

- session 内实时状态机。
- turn policy 决策。
- observability timeline 写入。

### `session/`

负责一个 LiveKit room/session 的编排。

允许：

- 管理 `AgentSession`、room、participant、data packet。
- 调用 `pipeline/` stage、`turn_policy/` runtime、`output/` controller。
- 处理 session 生命周期、idle watchdog、room data event、provider event binding。

不应负责：

- 语义判断规则本身。
- STT/TTS/VAD provider 具体实现。
- 输出 buffer 算法细节。
- device token 签名和 admin API shape。

建议子模块：

- `session/streaming.py`：`StreamingPipeline` 的主壳，保留编排入口。
- `session/batch.py`：`BatchPipeline`。
- `session/room_events.py`：room data、client audio state、participant identity。
- `session/provider_events.py`：STT/TTS/LLM provider event 归一化到 timeline。
- `session/idle.py`：idle watchdog 和 idle disconnect。

### `pipeline/`

负责音频/文本处理 stage 抽象，是 provider-neutral wrapper。

允许：

- STT/TTS/VAD/LLM stage 输入输出适配。
- LiveKit framework stage wrapper。

不应负责：

- 是否打断。
- 谁是当前用户。
- room data/control signal。
- memory/persona/brain routing。

### `turn_policy/`

负责“是否听、是否打断、打断属于哪一层、需要等什么证据”。

允许：

- attention admission。
- intent classification。
- transcript evidence gate。
- interrupt decision。
- tier policy chain。
- EOT 相关配置转换。

不应负责：

- 调用 LiveKit cancel/interrupt。
- 操作输出 buffer。
- 写 room data。
- 连接 STT/TTS provider。

重要原则：

- 尽量保持纯策略层。
- 输入是信号和配置，输出是 `Decision`/`AttentionDecision`/metadata-friendly control signal。
- 副作用由 `session/` 或 `output/` 执行。

### `output/`

负责 agent 输出音频的播放控制。

允许：

- duck/unduck。
- buffered frames。
- cancel/drop buffered audio。
- filler audio。
- played seconds 统计。

不应负责：

- 判断用户意图是 hard stop、topic switch 还是 backchannel。
- 解析 STT transcript。
- 处理 room data。

### `context/`

负责 session 内上下文处理。

当前优先承载：

- interrupted context snapshot。
- interrupted context injection。

未来可承载：

- memory hint 注入前的 session-local 上下文整理。
- 被打断回答的摘要或回收策略。

不应负责：

- 直接查长期 memory store。
- 直接调用 LLM 仲裁。
- 输出音频控制。

### `identity/`

负责 user/device/session 身份解析和 token。

当前 `runtime/` 更像这一层，后续可以逐步迁移：

- `admin_client.py`
- `resolver.py`
- `token_signer.py`

不应负责：

- 实时语音状态机。
- turn policy。
- brain RPC streaming。

### `brain_client/`

负责连接 Eidolon Agent 后端 brain。

当前 `eidolon_agent_rpc/` 更像这一层，后续可以逐步迁移：

- `grpc_llm.py`
- `session.py`
- `v1/`

不应负责：

- LiveKit room/session 细节。
- STT/TTS/VAD provider 细节。
- interrupt policy。

### `observability/`

负责 timeline、metrics、provider latency segment、未来 OpenTelemetry/export shape。

允许：

- 记录和归一化事件。
- 计算延迟。
- 输出 snapshot。

不应负责：

- 参与决策。
- 改变输出播放。
- 触发 interrupt。

### `compat/`

负责旧 import path 的兼容桥。

当前候选：

- `ducking.py`
- `interrupt_decider.py`

原则：

- deprecated wrapper 可以保留一段时间。
- 新代码不应再从 compat 导入。
- 兼容模块应尽量薄，只 re-export 或发 warning。

## 推荐迁移顺序

### Phase 0：文档与边界冻结

本文件作为边界基线。后续每次重构前先判断目标代码属于哪一层。

验收：

- 不改运行时代码。
- 只新增/更新文档。

### Phase 1：无行为变化的轻量归类

目标：

- 保留现有 public import path。
- 新增目标 package，但先通过 wrapper 或 re-export 保持兼容。
- 不拆复杂逻辑。

候选：

- `output_controller.py` 迁移到 `output/controller.py`。
- `filler.py` 迁移到 `output/filler.py`。
- 根目录保留 wrapper。

验收：

- 全量测试通过。
- 旧 import path 测试仍通过。
- commit message 明确为 no behavior change。

### Phase 2：拆 `streaming.py` 的纯辅助职责

优先拆副作用较低的部分：

- provider event observer -> `session/provider_events.py`
- idle watchdog -> `session/idle.py`
- room data/client audio state -> `session/room_events.py`

验收：

- `StreamingPipeline` 主类仍是唯一编排入口。
- 拆出的类不直接做 turn policy 语义判断。
- 全量测试通过。

### Phase 3：拆打断副作用和上下文处理

目标：

- duck/cancel/rollback side effects -> `output/` 或 `session/interruption.py`
- interrupted context snapshot/injection -> `context/interrupted.py`

验收：

- `turn_policy/` 仍只产出 decision。
- `output/` 执行播放侧副作用。
- `session/` 负责把 decision 路由到 side effect。

### Phase 4：包重命名与兼容收尾

目标：

- `runtime/` -> `identity/`
- `eidolon_agent_rpc/` -> `brain_client/`
- deprecated wrappers 移入 `compat/`

验收：

- public imports 有明确迁移期。
- 测试覆盖旧路径与新路径。
- 文档和 ARCHITECTURE 更新。

## 迁移约束

- 每个 commit 尽量只移动一个职责域。
- 优先使用 wrapper/re-export 降低破坏性。
- 不在目录重构 commit 中混入策略行为变化。
- 不把 `turn_policy/` 写成 LiveKit side-effect 层。
- 不把 `pipeline/` 写成 session 状态机。
- 不把 `observability/` 写成策略判断层。
- 每一步至少跑相关单测；涉及 `StreamingPipeline` 时跑 `eidolon/livekit/tests`。

## 近期建议

### 2026-06-07 Phase 1 执行记录

已完成第一步低风险归类：

- 新建 `output/`。
- 将 `OutputController` 移入 `output/controller.py`。
- 将 `FillerManager` 移入 `output/filler.py`。
- 保留根目录 `output_controller.py`、`filler.py` 作为兼容 re-export。
- 更新 `StreamingPipeline` 新代码 import 到 `agent.output`。
- 新增 import boundary 测试，保护新旧路径一致性。

### 2026-06-07 Phase 2 执行记录

已完成第一块 `streaming.py` 职责抽取：

- 新建 `session/`。
- 新增 `session/provider_events.py`。
- 将 LLM metrics、brain provider event、STT provider event、TTS provider event 归一化逻辑移入 `ProviderEventObserver`。
- `StreamingPipeline` 保留薄 wrapper，兼容既有测试中的私有方法调用。
- 新增 `ProviderEventObserver` 直接单测。

下一步继续 Phase 2：

1. 抽 idle watchdog。
2. 抽 room data/client audio state handler。
3. 每一步独立 commit，保持行为不变。

这一步收益明确，风险较低，也能验证目录边界设计是否顺手。
