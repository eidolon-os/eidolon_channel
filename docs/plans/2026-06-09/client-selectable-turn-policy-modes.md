# Client 连接时选择打断体验模式方案

日期：2026-06-09

## 背景

当前 `eidolon_channel` 的实时打断体验由两层配置组合决定：

- `turn_policy.profile`：控制底层时序和阈值，例如 VAD、EOT、decision timeout、稳定窗口。
- `turn_policy.interrupt_mode`：控制打断策略性格，例如是否首个有效 interim 就 cancel、是否启用 attention enforce、是否保留 weak followup hold。

现在这两项都来自 `settings.yaml`，在 channel worker 启动后固定。未来希望 Web client 或 ESP32 client 在连接 agent 时，可以主动选择自己的打断体验。

## 当前推荐方向

先做最小、安全的一步：**client 连接时只允许选择 `interrupt_mode`，不允许选择 `profile`**。

示例 participant metadata：

```json
{
  "kind": "user",
  "turn_policy": {
    "interrupt_mode": "responsive"
  }
}
```

服务端处理方式：

1. `settings.yaml` 仍然是全局默认配置来源。
2. `turn_policy.profile` 继续使用启动配置，例如 `balanced_semantic`。
3. client metadata 只允许覆盖 `interrupt_mode`，值必须在 allowlist 内。
4. 每个 LiveKit job/session 创建自己的 `TurnPolicyConfig` 和 `TurnPolicyRuntime`。
5. 日志和 timeline 记录最终生效的 `profile` 与 `interrupt_mode`，方便 dogfood 对照。

## 为什么先只开放 interrupt_mode

`interrupt_mode` 是轻量策略开关，不改变 VAD/EOT/STT/TTS 的模型加载参数。

因此开放它的风险较低：

- 不需要新增 VAD cache。
- 不需要新增 EOT cache。
- 不影响 worker 启动时已有的模型预热。
- 不改变 `SharedStageFactory` 中按启动配置创建 VAD/STT/TTS 的路径。
- 每个 session 只新增一份轻量 `TurnPolicyRuntime`，成本很低。

这一步可以支持如下体验：

```yaml
profile: balanced_semantic
interrupt_mode: balanced
```

当前稳健体验。

```yaml
profile: balanced_semantic
interrupt_mode: responsive
```

同一套底层时序参数，但更接近早期高敏感打断体验。

## 暂不开放 profile 的原因

`profile` 会改变底层时序参数，尤其是：

- `vad.min_speech_duration_ms`
- `vad.min_silence_duration_ms`
- `eot.tail_hang_silence_ms`
- `interrupt.decision_timeout_ms`
- `interrupt.normal_interrupt_stability_window_ms`

当前 worker prewarm 会按启动配置预热 VAD。若 client 连接时选择不同 `profile`，可能出现“会话策略参数”和“预热 VAD 参数”不一致。

EOT 当前也按 `eot_kwargs_from_turn_policy()` 做共享模型 cache。现在是单槽 cache，适合启动配置固定的场景；如果多个 profile 在不同 session 间来回切换，未来应改成按参数 key 的多实例 cache 或 LRU cache。

## 未来完整组合方案

当确认 client-selectable `interrupt_mode` 有价值后，可以进入第二阶段：开放完整 `profile + interrupt_mode` 组合。

示例：

```json
{
  "kind": "user",
  "turn_policy": {
    "profile": "fast_e2e_like",
    "interrupt_mode": "responsive"
  }
}
```

届时需要补齐：

1. `profile` allowlist 校验。
2. session-level effective policy 构造器。
3. VAD 按 `provider + vad 参数` 缓存。
4. EOT 按 `eot_kwargs` 缓存，从单槽 cache 改成 dict/LRU。
5. benchmark 覆盖典型组合：
   - `balanced_semantic + balanced`
   - `balanced_semantic + responsive`
   - `fast_e2e_like + balanced`
   - `fast_e2e_like + responsive`
   - `patient_companion + balanced`
6. 日志和 timeline 统一记录 `profile`、`interrupt_mode`、VAD/EOT cache key。

## 实施分期

### Phase 1：只开放 interrupt_mode

目标：最小改动，让每个 client/session 可以选择高敏感或稳健打断策略。

预计改动：

- 新增 participant metadata 解析。
- 新增 session turn policy override 构造函数。
- 在 `server.run_agent()` 创建 `StreamingPipeline` 前应用 override。
- 补 config/metadata 校验测试。
- 补 policy/runtime 测试，确认默认配置不变。

预估工作量：0.5-1 天。

### Phase 2：开放 profile + interrupt_mode

目标：支持完整二维体验矩阵。

预计改动：

- session policy 构造支持 `profile`。
- VAD per-profile cache。
- EOT multi-key cache。
- benchmark/report 扩展组合维度。

预估工作量：2-3 天。

### Phase 3：运行时动态切换

目标：同一个 session 中途通过 data channel 切换体验模式。

这一阶段不建议近期做。它需要处理正在进行的 VAD/EOT/STT/duck/turn runtime 状态迁移，复杂度明显高于连接时固定模式。

预估工作量：3-5 天以上，且需要更重的 live dogfood。

## 决策

近期优先选择 Phase 1。

即：client 连接时可选择 `interrupt_mode`；`profile` 仍由启动配置决定。这样既能满足“同一个部署中不同 client 拥有不同打断性格”的诉求，又不破坏当前 VAD/EOT 预热和缓存架构。
