# 新版本 BOX-3 文字与音频时序复核

2026-10-18 23:09:58–23:11:01，会话 `esp32-e7c3576d-6683d0b4-00000005`。
运行发布 `pi5-resume-boundary-20261018-1`（Channel fe5e4f0，FireRed + D3）。本次只读分析，未修改代码或服务。

## 第二个测试句：我想去一个……安静一点的地方

用户已于 2026-10-19 确认：指的是“我想去一个……安静一点的地方”这一轮的新回答文字提前显示，并非上一轮 PPT 回答残留。以下以该回合为主；PPT 回合数据仅作为补充。该确认明确了现象，但不能单独确定同步异常发生在 Channel、LiveKit 还是设备端。

| 事件 | CST 时间 | 相邻阶段耗时 |
|---|---|---:|
| 最后一段 VAD stop | 23:10:28.273 | 起点 |
| 最后一段 ASR final：安静一点的地方 | 23:10:28.633 | 360 ms |
| D3 分类开始 | 23:10:28.659 | 26 ms |
| D3 返回 reply=0.7655 | 23:10:28.805 | 146 ms |
| 完整请求提交 | 23:10:28.854 | 49 ms，含取消旧输出、EOT/提交 |
| Channel 收到 Agent 首回答文本 | 23:10:30.041 | 1187 ms |
| Agent done | 23:10:30.042 | 约 1 ms |
| TTS 请求创建 | 23:10:30.042 | 与文本返回紧接 |
| 取得 TTS 连接、发送首文本 | 23:10:30.063 | 约 21 ms |
| TTS provider 首音频 | 23:10:30.526 | 463 ms |
| Channel framework 进入 speaking | 23:10:30.534 | 约 7 ms |
| 服务端收到设备 agent_speaking 回显 | 23:10:30.559 | 约 25 ms |

VAD stop → framework speaking = **2260 ms**：提交前 581 ms + Agent 首文本 1187 ms + 文本到 framework speaking 493 ms。

Agent 自报 triage=18 ms、compile=232 ms（memory 约230 ms为并行准备中最慢项）、模型 stream 约920 ms。模型原始首chunk 111 ms不等于可用于播报的首回答文本；此次输出 first activity 是 tool_call，完成后才把可播报文本交给 Channel。没有外部天气等工具等待证据。Agent 首回答delta到done约1 ms，说明回答集中交付，不能理解为这1187 ms内持续向用户逐字发送。

后续打断上下文记录的回复为：“嗯，想找个安静的地方。是心里有点吵，还是想找个清净的去处？”这属于澄清回复，未实际列出地点。

## 已证实与未证实

1. 此轮完整请求只提交一次。第一段 D3 wait=0.8299，命中 `final_intent_wait`；续说时继续暂停，未出现旧版 `final_intent_resume` 或 cooldown 拒绝。最终取消旧输出后在28.862重置为NORMAL，新文本到来前已解除旧取消状态。
2. TTS 没有多秒排队：连接约21 ms、provider首音频约463 ms。首音频前未见新duck，不能归因于新等待策略持续压住本轮声音。
3. 日志字段 `tts_first_audio_at` 实际由 framework speaking 状态标记（agent/session/agent_state.py），不是麦克风实测的扬声器出声时间。provider首音频是另一独立事件。
4. 设备 `agent_speaking` 是播放活跃回显，既不是逐响应的首PCM确认，也不是物理听感。取消后28.956 idle、28.977 active、30.190 idle、30.559 active；其中前两次active早于新音频，不能错配给新回答。需要按音频段身份进一步验证旧队列收尾，不能将状态位当作无残音证明。
5. 缺少此次设备 caption first chunk、UI绘制和首非静音PCM时间，不能测得实际“文字已显示→听见声音”的间隔，更不能用服务端493 ms否定用户观察。
6. 当前代码采用 LiveKit TranscriptSynchronizer，客户端收到文字chunk即更新字幕。设计上已有同步机制；不能把该现象直接解释为“文字本来就应该提前全部显示”，也不应增加固定字幕延时。应检查打断后同步器音频段/文本段的完成和切换边界。
7. 23:10:52.196 有 text marked as ended / rotating segment warning，但属于后续长停顿拆轮，不在本轮。是后续调查线索，不能当成本轮字幕异常的已证根因。

## 与上一轮样本的描述性对照

| 第二项 | 旧版本 21:20 | 新版本 23:10 |
|---|---:|---:|
| 最后 VAD stop → commit | 854 ms | 581 ms |
| commit → Agent 首文本 | 1312 ms | 1187 ms |
| Agent 首文本 → framework speaking | 422 ms | 493 ms |
| 合计 | 2588 ms | 2260 ms |
| 停顿内错误恢复/续说被冷却拒绝 | 出现 | 此次未出现 |

这是两次不同生成的单样本，不能作为统计性能结论。新样本服务端总耗时没有变长，不代表字幕与物理音频体验已通过。

## 其他回合及发现

- 含问候计数的第二轮（拖长 PPT 问题）：最后 VAD stop → commit 770 ms；commit →首文本1487 ms；首文本→framework speaking358 ms；合计2615 ms。provider首音频23:10:19.298，framework speaking19.307，设备active回显19.326。用户已确认问题不指这一轮。
- 长停顿测试发现拆轮，整体体验尚不能据日志判定通过：“嗯，那让我想想。你。”在23:10:48.762已提交；随后“会做PPT吗？”在48.868开始新语音，50.904再次提交。D3对前半句reply=0.3855（keep=.2833、wait=.2688）取最高类别，FireRed前缀分数约.510越过.5。前半句已生成“好，我在这儿等着，你慢慢想。”，TTS provider有首音频，但未记录framework speaking，随后被取消。后半句52.540进入speaking。此处是两个请求，不是同一请求重复提交。长停顿拆轮本身不等于上下文丢失或产品体验失败；需结合前半段是否被播放、后续回答是否承接上下文评估。没有同音频新旧回放对照，不能直接判为此次提交引入的回归。
- 长停顿期间800 ms后恢复、再次续说约0.5 ms内重新duck的服务端控制边界有效；但这不抵消前半句提前提交的问题。

## 下一步验证边界

优先针对“旧回复暂停/取消 → 新回复”复现文字与音频段同步，复用现有LiveKit同步器；同时采集设备已有caption日志及首实际播放音频证据。单独保留长停顿拆轮样本，按相同VAD/ASR/D3时序比较f3903b3和fe5e4f0，区分新回归与已有模型/提交边界。此报告没有进行上述新增复现，也没有擅自更换模型、调整阈值或部署。

本地证据目录（相对多仓库工作区根目录）：`diagnostics/2026-10-18-box3-resume-retest/`，包含 `inventory.txt`、`traces.ndjson`、`channel.log`、`channel-filtered.log`、`agent.log`、`analysis.json`。原始日志保留在本地，本提交只归档分析报告。多段输入的`transcript_final_at`保留首段时间，末段ASR必须查逐条日志，不能用首段时间计算末段等待。
