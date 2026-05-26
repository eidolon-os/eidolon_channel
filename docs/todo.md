# Channel TODO

调试 / 开发阶段记录：本仓库已识别但**当前计划不修**的项。每条标注现状、影响、推荐修复
路径（上线前再做）。

---

## 已知问题但暂不修

### JWT device_token 30 天过期，无续期 / 重读机制

- **现状**：`scripts/provision_eidolon_token.py` 默认 `ttl_days=30`。`EidolonAgentGrpcLlm`
  只校验 token 非空，不解析 `exp` 字段。token 过期那一刻 brain 返回 `UNAUTHENTICATED`，
  channel 把它包成 `APIConnectionError(retryable=True)` 让 LiveKit framework 反复重试。
- **影响**：生产部署后约第 31 天某时刻所有对话突然失败。health probe（gRPC 长连还活）
  显示绿色，无提前预警。
- **推荐修复（上线前）**：
  1. **boot 时检查**：`EidolonAgentGrpcLlm.__init__` 在 token 非空校验之后，**不验签**地
     `jwt.decode(token, options={"verify_signature": False})` 读 `exp`。
     - ≥ 7 天 → DEBUG 日志
     - < 7 天 → `logger.warning("device_token expires in N days")`
     - < 1 天 → `raise ValueError("device_token expires within 24h, rotate before
       restarting; see scripts/provision_eidolon_token.py")`
  2. **自动 rotation**（需 brain 端配合，见 `B-2` brain 侧建议）：admin HTTP 暴露
     `POST /api/admin/devices/<id>/rotate`，worker 在 token 剩 7 天时主动调一次。

---

## 等外部条件就绪的项

### SubscribeProactive（brain 主动开口）

- **是什么**：陪伴智能体的杀手特性——brain 决定主动说"还在吗？""我注意到你 5 分钟没说话"。
  brain proto 已经有 `SubscribeProactive` 这个 server-stream RPC。
- **channel 这边缺什么**：`StreamingPipeline` 当前只接收 LLM 流式 token，没有"外部注入语句到
  TTS pipeline"的通道。
- **工作量**：3-5 天。涉及：
  1. Adapter 层加 `SubscribeProactive` 流订阅
  2. `StreamingPipeline` 加外部注入接口（接 `ProactiveEvent.text` 当一句模拟 LLM 回复）
  3. 中断 / barge-in 逻辑要兼容主动语句被打断的场景

### PushSignal（情绪 / 韵律 / 视觉信号上行）

- **是什么**：channel 推情绪 / 韵律 / 视觉信号给 brain，brain 在 context compile 阶段融合
  生成更贴心的回复。brain proto 已有 `PushSignal` RPC。
- **channel 这边缺什么**：当前没有信号源——VAD/EOT 只输出"说话开始/结束"，没有情绪/韵律分类器。
- **何时做**：等 G24+ 多模态信号源就位（情绪分类器、视觉 tag 检测等）。

### LiveKit 工具透传到 brain

- **是什么**：LiveKit framework 给 LLM 注册工具时（`EmitEvent`、`LookupCalendar` 等），
  channel 应该把工具列表传给 brain，brain 决定何时调用，channel 执行后回填结果。
- **当前**：`EidolonAgentGrpcLlm.chat()` 收到 `tools` 参数直接 `logger.debug` 吞掉。
- **协议障碍**：`eidolon.proto` 的 `StartTurn` 没有工具字段。需要 brain 团队接 proto 改动
  （见 `B-3` brain 侧建议）。
- **长期方案**：proto 加 `StartTurn.channel_tools: repeated ToolDescriptor`；brain
  `TurnEngine` 合并工具集；channel 通过 `TOOL_CALL` 事件收回调，新增 `ToolResult` ChatRequest
  帧回填结果。
