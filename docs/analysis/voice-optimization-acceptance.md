# 语音对话优化：阶段交付与剩余验收

当前结论（2026-09-08）：**按用户确认的范围，本阶段工程交付已收尾；原完整体验目标尚未通过验收。** 本次收尾是停止扩展实现、交付已验证修复并归档剩余项，不是将失败改判为通过。测试数量不作为完成度指标。

交付结果：三项工程修复已提交为 `d110d2b`，整合提交 `942ccc0` 已部署到 Mac 测试环境并完成三模式基础真实 Agent RPC 验证。保留 Channel 的编排职责和 LiveKit 公开接口，未修改 LiveKit SDK。本次仅补交验收记录与[关键结果归档](voice-validation/stage-closeout.json)，不新增运行逻辑。

测试环境定位：用户明确 Mac host 只是便于执行测试的环境，优化目标仍是通用级联对话能力及三模式体验。后续部署与实测记录不改变这一范围，OPI5 Max 部署后补硬件相关验收。

人工验证补充：用户确认当前 Mac host 已通过 mobile pad 进行了真实对话，对话链路已打通。这是用户报告的实际交互验证；尚未记录该会话的模式、运行版本及延迟测量，因此不推定三模式、六项语义反例或 2500 ms 门槛已通过。Mac + mobile pad 的打通状态与待部署的 OPI5 Max 验收分别记录。

后续日志核对已确认 Mac 存在全双工、真实 Agent RPC 对话，具体版本差异与完成度等待见文末续验；不再将 OPI5 Max 未部署作为 Mac 侧工作的前置条件。

原完整体验目标（2026-09-07 确认，以下保留作为后续验收要求）：沿用下述三个验收关口，先解决影响首段播报的确定性工程缺陷，再评估既有模型／框架能力能否改善条件句和打断语义。生产 RPC 与目标设备验收作为交付依赖明确跟踪。维持 Channel 的编排职责、三种模式及 LiveKit 公开接口，不新增状态机、热词规则或模型提示词变体。候选方案必须同时检查完整句／短答延迟和误打断，不能只让单个反例通过。

以下三个原代码与体验验收关口保持不变。只有三项全部通过，才能宣布完整体验目标达成；本阶段工程交付收尾不代表这些关口已通过，不得通过降低门槛、删除反例或保留 xfail 来宣布达标。

| 关口 | 固定通过条件 | 当前证据 |
| --- | --- | --- |
| 三种模式的功能 | 既有人声画像和三模式回归通过；PTT 只在松键提交、半双工不抢占播报、全双工取消／恢复后不丢失或重复回复；真实语音级联通过 | 最新核心回归 1117 通过；场景总回归 87 通过、6 项既有语义反例未通过，归入下一关。三模式 idle TTS 回归通过；真实级联 15/15 是第二十二批历史证据。本阶段另完成三模式基础 Agent RPC 验证，见续验记录 |
| 完成度与打断语义 | 现有条件句停顿（三种配置）和三个默认意图场景全部通过；判定须有明确上下文，不能把含糊短句固化为全局词规则；既有完整句和短答延迟门槛保持 | `--runxfail` 实跑为 6 失败、5 通过，仍阻断 |
| 默认全双工真实回复 | 原日期更正、费用请求音频，在同样配置和 2500 ms 门槛下，完整提交、恰好一次新回复，并在停声后取得新 TTS 对应的 RTC 音频 | 历史 direct LLM 三次重复为 4/6 通过；最新部署服务 Agent RPC 日期 2788／2842／2580 ms、费用 2956／2895／2694 ms，六次功能验证通过但延迟均失败。两种配置分别记录，仍阻断；第二十二批 2996／6100 ms 历史失败保留 |

直接暴露语义阻断的命令（使用已有模型目录环境）：

```sh
.venv/bin/python -B -m pytest -q --runxfail -p no:cacheprovider -p no:cov \
  eidolon/livekit/tests/scenarios/test_complete_turn_latency.py \
  eidolon/livekit/tests/scenarios/test_human_speaking.py \
  -k 'incomplete_turn or overlap_intent_contract'
```

后续阶段启动时仅处理上述关口的失败及改动引入的回归。冻结新增策略分支、意图提示词变体和新测试框架。补充针对性反例只能用于验证正在修复的根因，不能替代关口。

生产 Agent RPC 的身份缺口已在下述续验中通过在线设备元数据和权威接口解析补齐，三模式基础用例已有真实 RPC 证据；不能用 direct LLM 的通过代替。此前自动预检的特定设备身份返回未指定应答 Companion，该结果不用于否定用户后来确认的 Mac + mobile pad 真实对话。实体设备 AEC、长输出专项验收仍单列；OPI5 Max 待部署后验证。

本轮证据：[第二十二批记录](voice-validation/phase22-validation.json)。历史失败继续保留。


## 本阶段收尾与后续进入条件（2026-09-08）

收尾依据是已验证根因修复、版本整合部署与验收证据归档，不以继续堆叠实验作为进度。最新核心回归 1117 通过、2 skipped、11 xfailed、1 deselected；场景回归强制运行 xfail 后为 87 通过、6 失败。部署后三模式基础用例已通过，但不代表所有说话画像或完整延迟要求通过。关键计数、失败用例和实测结果已存入仓库 JSON；正文中的 `/tmp` 路径是本次原始记录来源，不保证永久可用。

| 剩余项 | 客观状态 | 后续进入条件与验收 |
| --- | --- | --- |
| 完成度与打断语义 | 六项固定反例失败；“你叫啥名儿？”受控全／半双工回放为 3030／3024 ms，超过 1000 ms | 有明确根因与最小修复，或已验证有希望的判定能力后再推进；复跑固定反例，并保持完整句、短答及三模式回归。不增加定向词规则 |
| 真实回复延迟 | 部署服务日期／费用六次均超过 2500 ms；首组主要等待包含有效模型文本及 TTS，不能归因为上下文编排 | 有对应阶段的具体改善方案后重复原音频、原边界和门槛；不能用空文本 chunk、换配置的成功样本或均值掩盖失败 |
| 欢迎语完成等待 | 一次运行及自带重试在用户音频发送前超时，随后原用例复验通过；根因未证实 | 再次复现并取得完成事件链证据后定位，不加绕过等待规则 |
| 目标设备体验 | Mac + mobile pad 已由用户实际对话验证；OPI5 Max 尚未验收 | 部署就绪后完成 AEC、长输出、模式交互及实体播放测量；Mac RTC 帧延迟不能替代扬声器起声 |

本阶段到此关闭，不继续扩展模型清单、提示词变体或编排机制。后续按上表具体条件启动相应工作；原门槛和历史失败持续保留。以下为按时间记录的历史过程，其中“未完成／阻断”指当时及当前仍未通过的完整体验目标。


## 第二十三批补充：如何解释失败

六项语义失败对应四个场景、两类问题，不能计为六个独立架构缺陷。条件句 2.2 秒停顿是句中停顿压力测试；“好的”是介绍过程中附和的语境预期，不是任何语境都禁止打断。日期纠正反例已补齐助手先说错日期的上下文，保留原预期和 xfail，不能用更换上下文消除失败记录。

2500 ms 是第二十批新增真实回放的停声到新音频门槛，不是用户指定的生产 SLA，也不同于开口到可操作转写的预算。第二十三批有效回放：费用 6238 ms，仍不达标；日期完整提交并启动第二次 LLM，但观测结束前没有新 TTS，不能提供首音频延迟或声称成功。费用本轮取消到提交 16 ms，主要等待转移到 TTS；上一轮 2955 ms 的取消到提交耗时仍保留，不能拿本轮单次样本宣布消除。


## 架构收敛复核（2026-09-07）

本次复核覆盖基线 `7204c79` 至 `ef1c313` 的 30 个生产／配置文件。保留已有的会话 EOT 状态隔离、转写 revision／覆盖校正、输出背压与取消、PTT 流关闭、TTS 连接补充及音频处理、原生 LLM 参数透传等修复。输出取消与 SDK endpointing 的适配存在时序耦合，继续限定在原有职责内，不新增状态机、热词表、计时器或提示词分支；不修改 LiveKit SDK。

本次代码修正仅两处：

- **按最终模式构造依赖**：启动入口先解析 participant metadata，再用既有模式策略构造 factory 与 pipeline。半双工／PTT 禁用可选意图客户端；全双工和 SDK 原生 owner 的路由保持。原实现仅在 pipeline 禁用意图消费，工厂却可能按共享配置提前创建模型。六种模式／owner 组合的有效修复前测试为 2 失败、4 通过，失败正是 Channel half duplex／PTT；修复后通过。身份解析失败也不会提前分配供应商客户端。
- **供应商参数按 endpoint 判断**：可选意图模型只对已知 DeepSeek 官方 endpoint 自动关闭思考；未知代理不因模型名带 `deepseek-` 就收到专用参数。复用现有适配器，实际 HTTP 序列化测试覆盖回复／意图两条路径。

**未采用的回退**：曾候选撤销共享配置中显式关闭思考的设置，随后用同一问题、96 completion-token 预算做两次真实模型调用。供应商默认模式消耗 96 tokens、没有可播报文本；显式关闭思考产生 77 字、消耗 47 tokens。因此恢复原配置，本次没有该配置差异。此反例只证明不能直接撤销，不证明复杂回答质量或端到端延迟更优；两次调用不构成稳定延迟统计。原始记录位于本地 `/tmp/voice-consolidation-reasoning.json`。

验证结果：

| 验证 | 实际结果与边界 |
| --- | --- |
| Agent／common／EOT／STT 生命周期／TTS 核心回归 | 1086 通过，19 因沙箱禁止 localhost 监听失败，2 skipped、11 xfailed、1 deselected；原始失败保留于 `/tmp/voice-consolidation-core.xml` |
| 受限网络模块补跑及最终 HTTP 契约测试 | 获准本地监听后，相关两个模块及模型参数模块共 59 通过、1 deselected；与前行有重叠，不累加为独立测试数。记录 `/tmp/voice-consolidation-network-and-contracts.xml` |
| 人声画像／PTT／完整句延迟，强制运行 xfail | 71 通过、6 失败，失败仍为条件句停顿的三种配置，以及附和／继续播报／纠正的三个默认意图场景。记录 `/tmp/voice-consolidation-scenarios.xml` |

本次架构收敛未消除六项既有语义失败，也未重跑真实 RTC，不能覆盖或替代前述 RTC 延迟失败、生产 RPC 鉴权缺口及实体设备验证。整体体验目标仍未通过验收；后续仅围绕上述固定失败关口处理根因，不继续扩大优化范围。


## 本阶段执行记录（2026-09-07）

已落实两项根因修复，均复用现有机制：

- TTS 空闲发送取消自身：`SentenceAggregator` 清空缓冲后取消当前 idle task，遇到异步传输等待就丢失文本。保留执行中任务的归属，正常发送完成或由 `aclose()` 取消；不新增定时器、重试或队列。两个针对性测试修复前失败，修复后通过；三种模式的流式回复均验证完整文本和生成完成前出声。
- 同一产品轮次的旧声学代次裁决污染最终提交：真实 RTC 第三次日期纠正中，generation 3 已确认接管，但 SDK hook 携带 generation 2 的较早 FINAL；原实现提前锁定 generation 2，转写收集结束后仍据旧拒绝裁决否决整轮。改为在既有 settlement 确认产品候选未被替换、组装完整转写后，读取该候选当前 generation 的裁决；删除文本反查代次方法。正反测试验证旧拒绝不阻止新放行、旧放行不越过新拒绝。跨产品候选替换和重复提交仍由原边界防护。

使用同一组端到端回归，在测试进程中恢复 `94a3cc4` 的原方法，得到 4 失败／1 通过：晚 FINAL 的纠正丢回复，以及三种模式 idle 发送丢文本；修正后这 5 项均通过。该复现不改写 SDK 或工作区生产文件。原始记录：`/tmp/voice-stage-end-to-end-before.xml`、`/tmp/voice-stage-merged-vad-after.xml`、`/tmp/voice-stage-idle-three-modes.xml`。

核心回归：1113 通过、2 skipped、11 xfailed、1 deselected（`/tmp/voice-stage-core-final.xml`），未将已知预期失败计为通过。场景总回归以 `--runxfail` 执行，87 通过、6 失败（`/tmp/voice-stage-scenarios-final.xml`）；失败仍为上述条件句三种配置及三个默认意图场景，没有新增场景失败。代码差异检查和 Ruff 严重错误检查通过；与 `94a3cc4` 比较，无新增 Ruff 诊断。

真实 RTC 在裁决归属修复前重复两条原始音频，各三次：日期 2055／1866 ms／一次无新回复，费用 2316／2220／2327 ms。成功样本均通过原 2500 ms 门槛，但那次无新回复阻止整体通过；其 trace 已用于上述确定性回归。该组仅有 TTS idle 修复，不能据此归因历史 6 秒等待已被消除。原始日志与结果：`/tmp/voice-stage-rtc-{1,2,3}/`。

模型候选未落地：FireRed 官方输入格式与当前实现存在差异，但直接替换后条件句仍误判，且“好”“7”低于完成阈值。另对已有多语言权重按原策略运行既有端到端用例，8 通过／2 失败；条件句改善，但“好”的全／半双工延迟分别为 3039／3028 ms，超过原 1000 ms 门槛。因此不改输入、模型或阈值，也不添加短答特例。记录：`/tmp/voice-stage-eot-input-contract.json`、`/tmp/voice-stage-multilingual-candidate.xml`。此前第十二／十三批的远程完成度和 SDK 本地音频 EOT 替换也未通过短答／慢说门槛，本阶段不重复那些替换实验。

裁决修复后的真实 RTC 复验，沿用相同两条音频、默认 `intent_provider=none`、direct LLM 配置及 2500 ms 门槛：

| 用例 | 第一次 | 第二次 | 第三次 |
| --- | --- | --- | --- |
| 我想改一下。不是明天，是后天。 | 2329 ms，通过 | 1966 ms，通过 | 2212 ms，通过 |
| 先别介绍背景。告诉我费用。 | 2092 ms，通过 | 2737 ms，超时 | 3180 ms，超时 |

六次均完整提交并产生一次新回复；每次会话包含最初回答和纠正后的回答，共两次 direct LLM 请求。没有复现旧裁决造成的漏答，但只有 4/6 达到延迟门槛，不能宣称稳定达标。原始记录：`/tmp/voice-stage-rtc-fixed-{1,2,3}/results.json` 及对应 `worker-timeline.jsonl`。

两次费用超时按各自新回复的同一条 trace 时间戳拆分，单位 ms（不混加跨轮次汇总的 `timeline_*` 最大值）：

| 阶段 | 第二次 | 第三次 |
| --- | --- | --- |
| 客户端末个有声帧 → 服务端 VAD 停声 | 617 | 631 |
| VAD 停声 → 完整提交 | 116 | 194 |
| 提交 → LLM 请求开始 | 4 | 3 |
| LLM 请求开始 → 首个文本增量 | 625 | 1580 |
| 首个文本增量 → 首段文本发给 TTS | 457 | 78 |
| TTS 首段发送 → 供应商首音频 | 573 | 332 |
| 供应商首音频 → Channel 首音频标记 | 28 | 27 |
| Channel 首音频标记 → 客户端 RTC 帧 | 317 | 334 |

第二次首段由既有 300 ms idle 机制发出，等待包含文本到达和聚合；第三次主要长在 LLM 首字阶段。两次取消到提交均约 12 ms，TTS 连接获取约 37–40 ms，当前证据不支持再改取消编排或扩充连接池。也不直接缩短 VAD／idle 阈值：前者影响慢说抢答，后者需要验证语音断句质量，且不能解决第三次的首字等待。上述组件阶段含传输和适配耗时，不等于供应商纯推理时间。

本阶段没有改 LiveKit SDK、默认模型、阈值或模式配置。六项既有语义反例尚未修复，真实 RTC 延迟仍未全过；下一步仅接受能同时改善这些固定反例及短答延迟的现有模型／供应商能力，不再通过新增编排分支推进。上表证据来自 direct LLM 和 RTC 接收端，不能替代生产 Agent RPC、实体设备 AEC 或实际扬声器起声测量，也不构成生产 P95；不进入完成状态。

## 生产路径与能力续验（2026-09-07）

本次不新增生产代码，补齐此前可用性未知的链路：

- SDK 云端 EOT：使用已安装 SDK 的 `TurnDetector(version='v1', local_fallback=False)` 和一秒静音做一次受限探测，实际返回 HTTP 401，`is_degraded=True`。SDK 随后返回的默认事件没有计为模型预测成功。已请求现有 Cloud 推理配置路径，不购买服务或默认切换 owner。记录 `/tmp/voice-stage-service-preflight.json` 和同名 `.log`。
- 生产身份：RPC transport ready；本地 LiveKit 有一个在线设备房间。只读获取设备声明的 Owner／Device，再用现有 `preflight_runtime_identity` 查询 Kernel 和 System Data。Owner 默认 Companion 解析成功，设备则明确返回 `has no companion target`。未变更设备分配或向在线设备房间发布音频。记录 `/tmp/voice-stage-runtime-identity-preflight.json`。
- 真实 RPC：用已解析 Owner、独立 agent 名称／端口／房间运行原有严格 benchmark；保留生产 `eidolon_agent` provider、鉴权 resolver 和真实 STT/TTS。测试 worker 的 trace 证明请求实际路由至该 worker。仅关闭 avatar／voiceprint，并重定向诊断音频目录；没有改策略、阈值或验收断言。

| 原有用例 | 实际结果 | 证据边界 |
| --- | --- | --- |
| `fd_gate_normal_turn_replies_001` | 通过；1 次正式 RPC 请求及回复，666880 bytes RTC 音频 | 全双工正常输入；停声后有声 RTC 帧 3536 ms，不是 2500 ms 延迟通过 |
| `streaming_three_short_rounds_commit_each_001` | 通过；连续 3 次独立正式 RPC 请求完成并有 TTS，3056640 bytes RTC 音频 | 半双工连续轮次；设备状态／播放确认仍由客户端模拟；末轮停声后有声帧 2910 ms，不代表三轮的延迟分布 |
| `phase_a_ptt_normal_release_commits_001` | 通过；松键提交、真实 RPC 与 TTS，674560 bytes RTC 音频 | PTT 按钮边界由客户端模拟；release 后有声帧 2876 ms，不是实体设备验证 |

上述三项 `real_call_verified=True`，无用例错误。trace 中均存在 `brain_rpc.provider=eidolon_agent_rpc`、独立 request_id 和请求／首字时间戳；半双工三轮并非重复计算同一请求。**计数校正**：上述数字只证明正式回复请求数。随后核对同一批 worker 的 `EidolonAgentSession.start_turn` 日志，实际总数为全双工 3、半双工 6、PTT 1；流式模式存在未进入正式回复 trace 的额外投机请求，见下节修正。报告分别位于 `/tmp/voice-stage-rpc/{owner-normal,owner-half_duplex,owner-ptt}/barge_in_e2e_ab/channel/livekit_room/`。入口复用 `scripts/bench_barge_in_e2e_ab.py`，临时调用参数见 `/tmp/voice_stage_rpc.py`。这补齐了三模式基础生产 RPC 功能证据，没有替代六项语义反例、默认日期／费用的延迟门槛或实体设备验收。

延迟核查：全双工这次请求的 TTS 首段发送到供应商首音频约 1145 ms。SDK `AudioEmitter` 的 200 ms 默认帧设置使用 progressive 输出，不能仅凭该数值断言固定增加 200 ms 等待。另复用现有 TTS stage 合成两句原测试文本，输出音频从起点到首个 RMS≥120 的 20 ms 帧分别为 120／60 ms（`/tmp/voice-stage-tts-onset.json`）；低音量起声本身也计入 RTC 有声帧指标，不能把 Channel 首音频标记到客户端有声帧的全部耗时归为网络／缓冲，更不能直接裁掉这些帧。没有据此新增音频裁剪或调小 SDK 缓冲。安装的 LiveKit Agents 1.7.1 共 207 个带校验值的 Python 文件全部匹配 RECORD。

现有模型实验也已去重核对：第十六批 Smart Turn v3.2 为离线 16/24、真实 STT/VAD 与 SDK 回放 1/6 通过，已经拒绝，本轮未重新下载或重跑。官方 [LiveKit 自适应打断说明](https://docs.livekit.io/agents/logic/turns/adaptive-interruption-handling/)提供了云端声学附和判断路径，但当前访问探测失败，且其公开能力说明不能证明本项目反例会通过。仍需有效的现成推理配置进行验收，实体设备则需明确测试目标及应答 Companion；未降低任何结束条件。

## 预生成关闭仍发起投机 RPC：修正与验证（2026-09-07）

检查是否能复用现有预生成能力改善延迟时，发现 `preemptive.enabled=False` 只传给 SDK，Channel 却无条件绑定 interim RPC 预热钩子。`PreemptiveWarmer` 会发起 `StartTurn(speculative=True)`，该请求的输出被丢弃，也不进入正式回复的 provider trace。前节真实日志的额外 2／3 次请求与这条路径一致；不能把正式回复计数当作服务调用总数。

修正仅在既有转写处理器的构造处，按同一个 `preemptive.enabled` 决定是否绑定现有预热回调。默认关闭时不再发送投机 RPC，显式启用保持原行为。未增加开关、任务、队列或缓存；原有 `StartTurn` 日志增加 `speculative` 字段，便于直接核对实际调用量。同步纠正把 RPC 预热和 SDK 可复用预生成混称为同一个过程的说明。本次没有启用 SDK 预生成作为延迟捷径；RPC adapter 的正式 `chat()` 仍按正式 turn 发送，不能仅凭框架“预生成”名称推断服务端投机副作用契约已成立。

复用生产 Pipeline、实际 LiveKit AgentSession 和 RPC adapter、已有本地 gRPC 服务及 ASR/TTS 测试替身，对全／半双工和开关启闭四种组合直接检查 protobuf `StartTurn.speculative`：修正前 2 失败／2 通过，失败均为关闭后仍发送投机请求；修正后 4 项通过，完整转写只形成一次正式回复，受控停声到音频不超过原 1000 ms 门槛。相关模块合计 47 通过。证据：`/tmp/voice-stage-preemptive-before.xml`、`/tmp/voice-stage-preemptive-after.xml`。

核心回归为 **1117 通过、2 skipped、11 xfailed、1 deselected**（`/tmp/voice-stage-core-preemptive-final.xml`）。随后增加的 RPC 日志字段在真实复验中执行并核对；Ruff 严重错误和差异检查通过。没有重复跑实现路径未变化的整套人声画像，也没有宣称原六项语义失败消失。

同一已解析 Owner、同一现有用例和独立 worker 的生产复验：

| 用例 | 请求计数及功能 | 本次有声 RTC 帧延迟 |
| --- | --- | --- |
| 全双工正常一轮 | 通过；总 `StartTurn` 3 → 1，唯一请求 `speculative=False`，624000 bytes 音频 | 停声后 2143 ms |
| 半双工连续三轮 | 通过；总 `StartTurn` 6 → 3，全部 `speculative=False`，3157120 bytes 音频 | 末轮停声后 2476 ms |

报告与原始日志：`/tmp/voice-stage-rpc/{preemptive-off-full-duplex,preemptive-off-half-duplex}/barge_in_e2e_ab/channel/`。两项 `real_call_verified=True`；实际额外请求消失已证实，但供应商和回复内容存在波动，单次对照不能将延迟差值归因于本修正，也不替代日期／费用两条原 2500 ms 关口。PTT 不经过此次修改的 interim 钩子；其既有真实 RPC 验证和本轮核心回归保留。

依赖再核对：没有新增显式 Cloud 推理配置，设备权威预检仍为 `has no companion target`（`/tmp/voice-stage-dependency-recheck.json`）。未重复请求已拒绝的云端推理、改写设备分配或修改 SDK。六项语义和原 RTC 延迟关口仍阻断整体完成，目标保持不变。

## 现有百炼模型能力筛选（2026-09-07）

使用现有百炼配置成功查询可见模型，确认 `qwen3.8-flash` 和 `qwen-flash` 可实际调用。复用现有 OpenAI-compatible adapter、打断分类器及第十二批固定完成度提示词；仅在临时测试配置中指定模型和 `enable_thinking=False`。单并发、先预热、每项仍限 1500 ms，无重试、提示词变体、阈值变更或生产配置改动。

| 候选 | 完成度：通过／总数 | 超时／其他失败 | 打断意图：通过／总数 | 超时／其他失败 |
| --- | --- | --- | --- | --- |
| Qwen 3.8 Flash | 22/58 | 33/3 | 6/12 | 6/0 |
| Qwen Flash | 26/58 | 23/9 | 9/12 | 2/1 |

完成度沿用 24 对未完成／完整句，并包含上下文短答、数字、纠正和停止请求；意图沿用既有 12 条模型重叠场景文本。后者使用该组既有的通用介绍上下文，不替代六项正式关口中明确说错日期的纠正上下文。除超时外，存在完整请求被判未完成，以及输出非约定标签的失败。未放宽标签解析来隐藏这些问题。

两候选均未达到进入端到端替换验证的条件，因此不接入生产，也不把上述文本级筛选计作 RTC 或六项反例通过。结果仅适用于本次接入路径、提示词与截止时间，不构成模型能力排名或稳定延迟统计。模型访问及逐项原始记录：`/tmp/voice-stage-bailian-model-access.json`、`/tmp/voice-stage-qwen-semantics.json`；复用脚本 `/tmp/voice_stage_qwen_semantics.py`。

本次没有新的生产修改，保留前述三个已验证的工程根因修复。剩余语义缺口未补齐，不能用继续增加编排分支或定向词规则代替满足要求的判定能力；原 RTC 延迟关口和实体设备验证仍未完成。暂不扩展候选模型清单或重复已拒绝实验，有新的可用能力或配置后再按原关口验证。

## 延迟归因与依赖复核（2026-09-07）

沿用原费用失败的 worker HTTP 调试日志，进一步核对连接阶段，没有另跑一次随机样本替换原失败：第二次 TCP/TLS 从 `20:58:59.612` 到 `.645`，约 33 ms；第三次从 `21:00:07.789` 到 `.827`，约 38 ms。两次 HTTP 响应头分别在 `.721` 和 `.904` 到达。第三次对应首个文本增量仍约在 `21:00:09.369`，响应头后约 1.46 秒；这是流式响应内容等待，不能仅凭客户端日志进一步分离服务端排队、生成和传输。现有证据不支持通过另造 HTTP 连接池解决主要超时。

第二次 TTS 首段在 `20:59:00.694` 由现有 idle 定时器发送，内容是“费用取决于具体实现方式”；下一段标点／文本在 `.970` 才发送。该记录与此前聚合等待归因一致，没有发现新计时器失效。缩短 idle 会改变断句取舍，且无法解决第三次的首字等待，因此不以调参替代修复。这里只补齐归因证据，未宣称 RTC 门槛通过。

配置 schema 中仍将 SDK 可复用预生成与 RPC 临时预热混为一谈的注释已同步更正，无运行行为变更。最后一次权威设备预检仍返回 `has no companion target`，显式 Cloud 推理配置仍未提供（`/tmp/voice-stage-dependency-final-recheck.json`）；没有重复调用已返回 401 的推理服务。该设备依赖已在连续阶段复核中保持不变，现有语义候选均未通过准入。剩余验收需要可验证的判定能力以及明确的实体测试目标／Companion，不能据现有结果宣布完成。

## Mac 实际对话续验与版本整合

用户确认 Mac + mobile pad 对话打通后，检查现有生产日志与源码：`livekit-agent-2026-09-07.log` 的 `23:20:17.175` 明确记录 `interaction_mode=full_duplex`；同一设备房间的三条回复 timeline 均含 `brain_rpc.provider=eidolon_agent_rpc`。这补充了实际链路证据，先前该身份的 Companion 预检失败不再作为整条 Mac 链路不可用的结论。

生产 worker 堆栈使用 `/Users/manson/ai/eidolon/eidolon_channel`。当时该源码目录 HEAD 为 `5e46807`，不包含 `d110d2b`；不能把这次真实对话当作已部署优化分支的验收。磁盘 HEAD 本身也不足以证明运行中进程的准确 commit。已将生产分支截至 `5e46807` 的 Manifest 契约及启动／退出通知修复合入优化工作分支，自动合并无冲突，保留双方行为；没有改写生产源码或重启服务。

整合验证覆盖 Channel Provider、会话 metadata／模式解析、启动失败通知及意图模型配置：首轮 200 通过，2 项因沙箱禁止 localhost 监听失败；允许本地监听后，对应 HTTP 模块 8 通过，其中 6 项与首轮重叠，不累加测试数。记录 `/tmp/voice-mac-integration.xml`、`/tmp/voice-mac-integration-http.xml`。差异检查通过。

真实样本“你叫啥名儿？”：服务端 VAD 于 `23:20:52.018` 停声，FINAL 于 `.187` 到达；EOT 得分 0.247，低于 0.5 完成阈值，直到 `23:20:54.523` 才进入回复。该轮 timeline 停声到提交 2501 ms，提交到 Channel 首音频 1180 ms。这是服务端阶段指标，不是 pad 扬声器起声或原 RTC 2500 ms 验收。

在整合后的工作分支直接复用既有 `test_short_answer_replies_promptly`，仅传入这条实际文本，全／半双工分别测得 3030／3024 ms，均超过原 1000 ms 门槛；本地 EOT 得分同为 0.246676。此受控回放使用真实 EOT 与 SDK、模拟 ASR/TTS，不冒充真机重测；记录 `/tmp/voice-mac-colloquial-replay.json`。结果确认该完整口语问句的等待属于完成度误判，未通过改词、降阈值或新增取消分支绕过。原六项反例仍是固定关口，这条实际样本作为同一模型能力缺口的补充证据。

## 通用链路部署后验收（2026-09-08）

用户确认继续后，将 Mac 测试环境的 Channel 源码从 `5e46807` 快进到整合提交 `942ccc0`，通过现有 eidolond `system.service.restart` 接口重启 Channel，未绕过原生命周期管理。重启后服务为 ready，健康接口与 worker 注册接口均为 HTTP 200，worker 注册名为 `eidolon`，进程 PID 从 84655 变为 2349。记录 `/tmp/voice-mac-deploy-before.json`、`/tmp/voice-mac-restart-result.json`、`/tmp/voice-mac-deploy-after.json`。部署后观察到运行源码目录出现其他工作的未提交修改，未覆盖或纳入本次提交；这些验收结果描述实际运行服务，不能扩展为该目录后续任意未提交版本的保证。

复用既有 benchmark 与身份解析，在独立测试房间直接请求已部署服务；没有另起替代 worker，也未向用户的设备房间发布音频。以下经过真实 ASR、Agent RPC、TTS 与 RTC；客户端按钮／播放状态仍由原测试模拟，末端指标是 RTC 有声帧，不是实体扬声器起声。

| 既有用例 | 结果 | 正式 RPC 请求数 | 本次停声／松键到有声 RTC 帧 |
| --- | --- | --- | --- |
| 全双工正常一轮 | 复验通过 | 1 | 2362 ms |
| 半双工连续三轮 | 通过 | 3 | 末轮 1779 ms |
| PTT 松键提交 | 通过 | 1 | 1804 ms |

三项均 `real_call_verified=True`。原始报告位于 `/tmp/voice-stage-rpc/{deployed-full-duplex-recheck,deployed-half-duplex,deployed-ptt}/barge_in_e2e_ab/channel/livekit_room/`。全双工首轮及 runner 自带的一次重试失败记录仍位于 `deployed-full-duplex`：欢迎语音频已收到，但没有欢迎语最终转写，测试在发送用户音频之前等待超时。因此不能计作输入后的漏答，也不能隐去该接入不稳定现象。随后通过 SDK 公开 text-stream 接口的独立诊断观察到完整欢迎语、legacy 与 stream 两条转写以及真实回复（`/tmp/voice-transcription-probe.json`），再按原严格用例复验通过。尚未把偶发现象归因为特定播放时序缺陷；没有修改 SDK 或添加绕过等待的规则。SDK 207 个受 RECORD 校验的 Python 文件仍全部匹配。

继续使用日期纠正／费用请求的两条原始音频，各重复三次，保留完整提交、恰好一次新回复及 2500 ms 门槛。此次走部署服务的真实 Agent RPC 与实际配置，与此前 direct LLM 结果分开记录：

| 原音频 | 第一次 | 第二次 | 第三次 |
| --- | --- | --- | --- |
| 我想改一下。不是明天，是后天。 | 2788 ms | 2842 ms | 2580 ms |
| 先别介绍背景。告诉我费用。 | 2956 ms | 2895 ms | 2694 ms |

六次均完整提交、每会话产生两次正式 RPC（原回复及一次纠正后的新回复），真实调用验证通过；**六次均未通过 2500 ms 延迟门槛**，没有用新样本替换先前 direct LLM 的 4/6 结果。原始结果和逐轮 timeline 位于 `/tmp/voice-stage-rpc/deployed-corrections/barge_in_e2e_ab/channel/livekit_room/repeat-{00,01,02}/`，摘要 `/tmp/voice-deployed-corrections-summary.json`；入口 `/tmp/voice_deployed_corrections.py` 复用原 runner 和先前的边界计算方法。

第一组按各自新回复 trace 拆分：日期／费用的服务端停声到提交为 74／82 ms，RPC 发送到有效文本为 1213／1389 ms，有效文本到首段 TTS 发送为 299／226 ms，TTS 发送到供应商首音频为 309／341 ms。其余时间含客户端停声到 VAD、音频输出与传输，不把上述四段误称全部延迟。按 request_id 核对 Agent 自身日志，上下文编译约 32／27 ms，模型有效文本 TTFT 为 1176／1357 ms；首个原始 chunk 更早，但没有可播报文本。上下文准备并非这两次主要等待，现有证据不支持重写该编排。此处沿用服务日志的指标含义，不把适配器的 `connect_ms` 当成纯 TCP/TLS 时间。

本轮完成版本部署、三模式基础验收和原音频的生产 RPC 重复测量，没有新增平台专用实现或变更模型／阈值。六项既有语义反例、口语完成度误判及低延迟缺口仍未补齐，整体目标未完成。
