# 意图分类接口与可配置实现（2026-10-17）

本次实现将 Channel 的异步打断意图分类抽象为 `InterruptIntentClassifier`，保留 `LlmInterruptClassifier`，新增 `SystemOneInterruptClassifier`。两者由已有 `SharedStageFactory` 按配置构建，业务 Agent、EOT 和 VAD 不随意图 provider 切换。独立验证阶段生产配置仍选 `llm`；之后已按用户授权在 Pi5 启用 d3，见第 6 节。

## 1. 架构边界

```text
配置加载与校验 → SharedStageFactory
                    ├─ llm       → LlmInterruptClassifier
                    └─ systemone → SystemOneInterruptClassifier
                                      ↓
                          InterruptIntentClassifier
                          classify / warmup / aclose
                                      ↓
                          IntentClassification
                                      ↓
                     SemanticInterruptHandler → 现有轮次策略
```

| 层 | 职责 | 本次处理 |
|---|---|---|
| 异步抽象接口 | `classify(text, assistant_text)` 返回分类证据；统一预热、关闭生命周期 | 使用 Python ABC，工厂与处理器依赖此接口 |
| LLM 实现 | 通过既有 LiveKit LLM 适配器获得离散标签 | 保留模型配置、提示词、24 token 限制、DeepSeek thinking 设置及无重试策略 |
| SystemOne 实现 | HTTP `/v1/systemone` choice 请求，校验结构与概率 | 复用 httpx 连接池；无重试、无跨模型回退；取消向上传递 |
| 输入与输出契约 | 同一提示词、四分类、用户文本和助手末尾 2,000 字符 | 两种实现共用；新增可选原始概率字段，兼容 SDK 结果类型 |
| Channel owner | 请求身份、超时、候选有效性、播放和提交决策 | 使用现有处理器与策略，不增加轮次状态机 |
| 同步提示接口 | 无模型的旧 EOT 路径使用同步提示 | 明确命名为 `InterruptIntentHintClassifier`，与异步模型接口分开 |

`keep / stop / reply / wait` 语义不变。SystemOne 概率写入现有候选 timeline；不加固定 0.8 门槛。概率是模型原始分数，不代表经过校准的正确率。

所有消费模型证据的分支通过共同的 `uses_model_intent` 判断，避免新 provider 绕过语义打断流程。选择 SystemOne 配合远端业务 Agent 时，不再因为意图分类而强制要求 `llm.model/base_url`；如果回答本身仍使用 `direct_llm`，这些字段仍必需。

## 2. 配置切换

LLM 沿用已有 `llm` 配置：

```yaml
turn_policy:
  interrupt:
    intent_provider: llm
    intent_timeout_ms: 2500
```

SystemOne 使用独立服务配置，不改回答模型：

```yaml
turn_policy:
  interrupt:
    intent_provider: systemone
    intent_timeout_ms: 2500
    systemone:
      endpoint: http://decider.yangtzeailab.com/d3/v1/systemone
      model: d3
```

提供 `config/overlays/intent_d3.yaml`，使用现有 `EIDOLON_CHANNEL_SETTINGS_OVERLAY_YAML` 加载。若主机已经使用 BOX3 或其他 overlay，将上述字段合入当前 overlay，保留其余主机参数；现有加载器只有一个 overlay 路径。

切回只需将 `intent_provider` 改回 `llm`，或移除 d3 overlay 恢复原配置。配置在服务/worker 启动时读取，需要通过正常运行流程重新加载；不支持正在进行的会话热切换。独立验证阶段没有发布或重启服务；之后 Pi5 的授权发布见第 6 节，Mac 未切换。

同一个 SystemOne 实现也可通过 endpoint/model 指向符合相同协议的其他服务；不会在代码里按 `d3` 或 `pplx` 写业务分支。本轮真实服务验证对象是 d3。

## 3. 同时修复的异步边界

前轮故障注入已经证明以下两项是现有实现的问题，非 d3 引入：

1. provider 吞掉超时取消后返回停止意图：原先可能在预算之外接受。现在接受前检查单调时钟，过期结果转换为既有 unavailable/UNCERTAIN 处理。
2. 同一候选文本 A → B → A：旧请求可能因文本重新相同而生效。现在还校验当前 asyncio task 身份，旧请求永久失去控制权。

两项对 LLM 和 SystemOne 都适用，已分别加入确定性回归。沿用现有错误处理语义：UNCERTAIN 在 VAD 静止后可能恢复旧播放，并不代表持续静默等待。

## 4. 验证方法与结果

结果目录：`../diagnostics/2026-10-17-intent-provider-integration`（相对工作区，不是 Channel 仓库内部）。

- 接口/配置/策略：抽象约束、四分类、输入边界、低概率正确结果、非法结构/概率、网络/HTTP/JSON 错误、取消、超时、资源释放、配置切换与恢复、native owner 与半双工路径、过期结果与 A→B→A。
- SDK 回归：复用真实 LiveKit AgentSession 和 Channel 控制流程；对不同返回时机的场景同时运行 `llm` 与 `systemone` 配置。
- 真实模型质量：冻结的 72 条合成测试用例，经正式工厂和统一接口调用。没有重新调提示词或按测试结果选择阈值。
- 真实模型 + SDK：八类场景，生产 2,500 ms 意图预算及冻结的其余策略，真实 FireRed、真实分类 HTTP；外部 ASR、回答 LLM、TTS 使用脚本替身。

冻结用例 SHA-256：`b62630e9d983f1d828172b44702bceddbc85e2de442ce2ffe8aed2512e9901ee`。

当前已完成的验证：

| 验证 | Mac | Pi5 |
|---|---|---|
| 接口、配置、策略回归 | 219 通过 | 218 通过 + 缺失配置补齐后 1 项复核通过 |
| SDK 时序/播放回归 | 55 通过 | 55 通过 |
| 真实分类服务 | DS、d3 各 72 次 | d3 72 次 |
| 真实分类服务 + SDK 场景 | DS、d3 各 8 类，均完成，无分类请求错误 | d3 8 类，均完成，无分类请求错误 |

Pi5 首轮快照缺少提示音资源，补齐只读资源后仍有一项缺少产品配置文件；复制现有产品配置后该项复核通过。失败原始日志已保留，未通过改动产品代码或跳过断言使其通过。

真实服务的 72 条合成用例结果（p95 使用排序样本下标 `int(.95 * (n - 1))`）：

| 主机 / 模型 | 与预设标签一致 | 请求错误 | p50 | p95 |
|---|---:|---:|---:|---:|
| Mac / DS | 72/72 | 0 | 1,073.5 ms | 1,265.3 ms |
| Mac / d3 | 71/72 | 0 | 102.8 ms | 138.1 ms |
| Pi5 / d3 | 71/72 | 0 | 96.9 ms | 139.8 ms |

这是分类调用耗时，不能直接当作用户感知响应延迟；运行环境期间还存在独立 SDK 回归进程。用例为前轮冻结的人工编写数据，不是新的盲测集。d3 两个主机都在 `test-093` 上与预设标签不同：“算了，我还不知道怎么表达。”预设 wait，返回 stop（约 0.453）。这里有自然语言歧义，不应仅凭 71/72 宣称已达到生产准确率。

真实服务 + SDK 观察（d3 在 Mac、Pi5 两端一致；DS 本轮在 Mac）：

| 输入场景 | d3 | DS |
|---|---|---|
| 附和、继续 | 保留旧回答，无新回答 | 相同 |
| 只停止 | 中断旧回答，无新回答 | 相同 |
| 换一种方式解释 | 中断旧回答，新回答恰好一次 | 相同 |
| “他说‘别说了’是什么意思？” | reply；中断旧回答，新回答恰好一次 | 相同 |
| “先别回答，我还要补充” | stop；中断旧回答，无新回答 | 相同 |
| 未完成表达、指向不明 | UNCERTAIN；保留旧回答，无新回答 | 相同 |
| “我刚才说先别回答，现在请回答” | keep；无新回答 | reply；新回答一次 |

最后一项没有先建立真实等待状态，只是当前文本声称之前要求等待，所以不能当作完整“等待→恢复”验收。倒数三类属于前轮已指出的产品语义边界；本次接口替换未扩展它们。引用问题在没有 0.8 门槛时正常触发新回复。

24 条真实 SDK 轨迹进一步做了结果断言：每条恰好一次分类，全部无分类异常；模型标签与中断/新回答次数一致。继续、停止、两类新问题共 12 条核心场景通过行为断言；其余 12 条作为产品边界观察，不计为“等待/恢复功能验收通过”。

最终接口类型导出后，Mac、Pi5 各复核 60 项接口测试通过（是上述测试的子集，不重复累加）。`ruff`、`git diff --check` 和 `uv lock --check --offline` 通过；httpx 原本已存在于依赖锁定中，此次仅声明为直接依赖，没有升级其他包。Pi5 最后检查 Channel/Agent 均 active，48.3°C，`throttled=0x0`。

## 5. 结论边界

本次交付的是可配置的模型接口和实际接入验证，不改变意图识别的产品范围。目前仍主要覆盖助手输出/打断窗口；普通空闲发起、持续等待、等待后的恢复不是本次扩展内容。真实服务 + SDK 测试不等于 BOX3 的 RTC、AEC、麦克风和扬声器体验验收，也不构成长期服务可用性结论。


## 6. Pi5 / BOX-3 发布结果

用户授权后，标准 Ops 可回退流程分两阶段执行：先激活支持接口的 `pi5-intent-provider-20261017-1`，保持 LLM；再激活配置发布 `pi5-intent-d3-20261017-1`，选择 SystemOne d3。两次均为 `activated`、`doctor=healthy`、`app_ready=app_ready`，没有数据库迁移。

- 实现提交：`8480fb5`；最终 Channel 发布提交：`1ca3438c96e7236fce88c12039074405c573ad6d`。
- Ops Pi5 配置提交：`dac9461`。只在 Pi5 host overlay 选择 `systemone`、d3 endpoint 和 model；共享 Channel 默认及 Mac 仍为 LLM。
- 其他组件固定为此前 Pi5 发布的精确 revision，未随本地 HEAD 升级。
- 最终 Ops status：当前和实际运行版本均为 `pi5-intent-d3-20261017-1`，21/21 服务正常。
- 从实际发布的安装包和 `/etc/eidolon/channel.yaml` 构建分类器，确认 `SystemOneInterruptClassifier`、model=d3、预算 2,500 ms。继续、只停止、引用“别说了”的新问题三次真实调用均通过；耗时分别 297.4 / 98.7 / 101.9 ms，首条包含新连接开销。
- 引用问题 reply 概率约 0.703，仍正常采纳；没有引入 0.8 门槛。
- 保留 LLM 兼容发布与 Ops 配置快照，便于回退。完整收据：工作区 `diagnostics/2026-10-17-intent-provider-integration/results/pi5-d3-deployment.json`。

现在可以开始 BOX-3 真机测试。上述部署后探测不包含物理麦克风、扬声器、AEC 或实际用户打断效果；这些由下一轮 BOX-3 操作验证。
