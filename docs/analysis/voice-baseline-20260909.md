# 语音对话基线：2026-09-09

本版本冻结当前级联对话实现，作为后续方案的对照基线。基线收敛不代表全部产品体验目标已达标。版本以包含本文和 `voice-validation/baseline-20260909.json` 的 Git 提交为准，父提交为 `cd070fe`。

## 范围与审查结论

- 保留 Channel 的话轮、打断及输出准入职责，使用 LiveKit 公开接口。未修改第三方 LiveKit Agents 或 eidolon_sdk，未增加状态机、队列或热词规则。
- 保留 `cd070fe` 的暂停续说证据延续及输出内容修复。PTT 以松键为提交边界，half duplex 不启动全双工打断编排，full duplex 保留可逆让声和后续证据判断。
- 相对父提交，生产增量仅为恢复操作时间、当时停声边界和用户是否仍在说话。未改变 EOT 模型、阈值、预生成开关或超时策略。
- 测试增量为同话轮 canonical/RPC/回答/TTS 关联、场景化延迟口径、未测量状态，以及默认关闭的 RTC 解码音频录制。录音参数保持原构造参数顺序；缺少话轮 ID 的记录不能证明新回复。
- 完整回归发现两组过期测试替身，缺少已提交生命周期使用的接口。已补齐并保留功能断言，检查候选启动及 half duplex 不启动打断计时；未修改生产逻辑来迁就替身。

测试计数、失败/跳过身份、依赖版本和证据哈希见 [基线清单](voice-validation/baseline-20260909.json)。历史过程见 [验收记录](voice-optimization-acceptance.md)。原始日志的 `/tmp` 路径不保证永久存在，因此清单保留可独立阅读的结果和失败身份；原始音频为本机诊断材料。

## 已验证结果及限制

本次最终核心回归 **1408 通过、2 跳过、11 xfailed、9 项集成测试未执行**。场景回归使用 `--runxfail`，**125 通过、6 失败**，仍是未完条件句的三种配置及附和/继续/纠正三项默认意图反例，没有新增场景失败。82 条场景警告来自 `record_property` 与默认 xunit2 报告格式的兼容提示，属性仍已写入本次 XML；后续只需输出兼容格式，不影响本次断言结果。SDK 1.7.1 的 207 个 Python 文件与安装 RECORD 一致，Ruff 严重错误及差异检查通过。

最近真实 RTC 运行是 `reply-capture-20260909`，使用真实 STT/EOT/Agent RPC/TTS 和插入指定静音的合成语音，不冒称自然人录音。本次冻结仅重新应用最终断言，结果未改变；没有新跑 RTC 或声称降低了上次延迟。

| 场景 | VAD 停声→新 TTS | VAD 停声→接收有声块 | 发送端末有声帧→接收有声块 |
| --- | ---: | ---: | ---: |
| 人数，静音 500 ms | 4014 ms | 4275 ms | 4878 ms |
| 人数，静音 900 ms | 3519 ms | 3788 ms | 4398 ms |
| 费用，静音 500 ms | 1313 ms | 1597 ms | 2205 ms |

三条功能通过，应用层 2500 ms 门槛仅费用通过；人数 500 ms 首次欢迎语完成检测超时，由已有机制重试一次。两个人数样本的 VAD 停声→提交约 2501 ms，费用约 27 ms。失败和重试均保留。

三条新回复通过离线波形对应验证，见 [音频归属证据](voice-validation/rtc-reply-audio-attribution.json)。通用报告仍明确标记 receiver `not_measured`，未把离线诊断升格为自动门禁。接收时间是解码帧到达消费者，不是 mobile 扬声器起声；三列起止点不同，不能混用，三个样本不能推断整体 p95 或达标率。

900 ms 保留为已有短附和案例的起声→恢复预算，不作为全局 EOT 或慢说期限。本批未部署/外放复测；最近只读清单为 OPI5 Max Channel `ee4f1da`，不能视作本基线版本。

## 固定复测入口

在仓库根目录使用项目 Python 3.13 环境和已有依赖，模型必须是真实文件而非 Git LFS 占位。以下路径按测试机器填写，Mac 不是部署目标约束：

```sh
export EIDOLON_EOT_MODEL_DIR=/absolute/path/to/firered_chat_turn_detector
export EIDOLON_FIRERED_PVAD_MODEL_DIR=/absolute/path/to/firered/resources

.venv/bin/python -B -m pytest -q -p no:cacheprovider -p no:cov \
  eidolon/livekit/tests/agent eidolon/livekit/tests/common \
  eidolon/livekit/tests/eot eidolon/livekit/tests/stt \
  eidolon/livekit/tests/tts eidolon/livekit/tests/benchmark \
  --junitxml=/tmp/voice-baseline-core.xml

.venv/bin/python -B -m pytest -q --runxfail -p no:cacheprovider -p no:cov \
  eidolon/livekit/tests/scenarios/test_complete_turn_latency.py \
  eidolon/livekit/tests/scenarios/test_human_speaking.py \
  eidolon/livekit/tests/scenarios/test_human_ptt.py \
  eidolon/livekit/tests/scenarios/test_interrupt_model_timing.py \
  --junitxml=/tmp/voice-baseline-scenarios.xml
```

第一组遵循仓库默认 `not integration` 标记，部分测试需要 loopback HTTP/WebSocket。第二组强制运行原有语义反例，当前返回非零退出码；应对照具体失败身份，不能简单忽略非零码，也不能把 xfail 当作通过。

真实 RTC 复测复用 `scripts/bench_barge_in_e2e_ab.py`。先加载测试环境及与历史样本相同的有效配置：`brain_provider=eidolon_agent`、`interruption_owner=channel`、`intent_provider=none`，avatar/voiceprint 关闭。供应商、LiveKit 和 Agent RPC 可用，Owner/Companion 经现有身份预检解析；不要复用历史诊断文件中的身份。

```sh
PYTHONPATH=. .venv/bin/python -B scripts/bench_barge_in_e2e_ab.py \
  --profiles channel --manage-worker \
  --cases benchmark/cases/full_duplex/pause_continuation_enforced.yaml \
  --repeat 1 --run-id voice-baseline-replay --output-dir /tmp/voice-baseline-replay \
  --livekit-interaction-mode full_duplex \
  --livekit-agent-name eidolon-voice-baseline --worker-base-port 18791 \
  --livekit-participant-identity "$EIDOLON_BENCH_LIVEKIT_PARTICIPANT_IDENTITY" \
  --livekit-participant-kind owner --livekit-owner-id "$EIDOLON_BENCH_LIVEKIT_OWNER_ID" \
  --livekit-agent-ready-timeout-sec 30 --livekit-room-settle-sec 10 \
  --livekit-audio-capture-dir /tmp/voice-baseline-replay/receiver
```

已有其他 worker 时，只有核实 agent 名称独占后才添加现有 `--allow-existing-workers`。有效配置用现有配置入口提供，不能因切换到 direct LLM 而将结果与真实 Agent RPC 混比。录音开关只记录接收端，不能单独证明新回复归属；原始 TTS 旁路记录与互相关仍是独立诊断。后续重复次数必须一致，并使用新的 run ID 和输出目录保留失败记录。

## 保留的缺口与后续边界

1. **结束判断质量**：完整请求低分与未完条件句高分仍存在。历史输入格式替换和部分模型对照曾失败，不因本次冻结重复扩展实验。后续候选须同时比较接话延迟与抢答。
2. **默认打断语义**：附和、要求继续和纠正的既有反例保留；不能把语义完整度直接当成打断意图或按个别词修复。
3. **设备让声体验**：证据等待期间恢复旧输出的行为仍待外放/AEC 验收；新记录提供测量能力，不表示体验已合理。
4. **真实回复延迟及欢迎语完成检测超时**：保留上述失败，只有对应阶段有可验证改进时再进入，不放宽门槛宣布通过。

本基线不启用新模型、动态 endpointing 或推测生成，也不新增并行决策链。后续扩展必须与此提交在相同音频、配置、模式和时间边界下比较；出现抢答、漏答、重复提交或业务副作用即不能替换基线。
