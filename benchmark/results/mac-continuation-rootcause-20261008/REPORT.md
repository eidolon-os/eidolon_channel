# Mac 续接等待根因核查（2026-10-08）

## 结论与停止条件

固定 Channel 版本复现后，人数场景的长等待由**真实 EOT 完整性分数 0.424 < 0.5，触发 LiveKit 非流式检测器的 3 秒端点上限**解释。不是 2.5 秒的 Channel 定时器，也不是打断确认与端点等待简单串行相加。生产模型把完整句分入长等待档属于仍存在的产品质量/延迟缺口；本轮没有发现需要修复的提交状态机缺陷，不降低阈值、不把可打断当成已说完、不添加句子特判。

本轮范围排除模型训练/替换、Laya、NPU。根因和固定版本复现已齐，停止；**2500ms 目标仍未达，语音性能优化未整体完成**。没有依据声称任何未来策略调整都无必要，只能确认当前证据不足以安全缩短所有低分句的等待。

## 版本、身份与有限矩阵

- Channel main `1313ff58f1ab2d351f48350ce8842cbf51a02b09`，运行前工作树干净。父任务提及的 `43ba242` 不在本仓 HEAD；原报告已经被跟踪，不改写历史或重复 benchmark 修复。
- 经唯一 Ops 入口 `eidolon mac service restart channel` 正式重启；结果 applied/ready，审计位置 434。主进程由 17656 变为 16500，启动于北京时间 16:39:37；只读核对 cwd 为本仓。版本依据为重启后加载已固定源码、依赖及随后真实请求日志，不是仅凭健康检查。代码/模型/配置 SHA-256 见 evidence.json。未改共享配置。
- Python 3.13、livekit-agents 1.7.1；EOT FireRed ONNX，STT fun-asr-realtime-2026-02-28，TTS cosyvoice-v3-flash。复用正式服务与原测试 Owner；identity/provider preflight 通过。核对解析出的测试 companion 与事故记录中的小禾 companion 不同，三个新房间使用各自会话；不是新 Realm，测试伙伴可能保留合成测试对话。未向小禾会话写入，未重启 Memory/Agent/Data，未操作 opi5max。
- 新鲜 Channel 进程后仅跑原 `pause_continuation_enforced.yaml` 三项各一次。另跑既有 EOT、提交门禁、续接状态机回归。停止条件：三项取得同轮完整 canonical 请求和新 TTS 音频、确认等待所属层；若发现截断/错提交才进入修复。未扫描阈值，未追加正式性能矩阵。
- 这是诊断，不是尾延迟性能实验；没有 20 次重复或跨批次 p95 结论。首个房间经历新 worker 初始化，其余共享预热 provider；未控制整机争用。Agent/Memory 的实际加载版本未重新认证，不将其称为整个栈 HEAD 部署验收。

## 同轮链路分解（毫秒）

| 用例 | VAD 停声事件→提交 | 提交→RPC | RPC→回答首 delta | 回答首 delta→应用 TTS 首音频 | 合计 | 功能 / 2500ms |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| people_500 | 2501.763 | 1.185 | 333.862 | 409.197 | 3246.007 | 通过 / 失败 |
| people_900 | 2502.420 | 5.306 | 582.154 | 400.475 | 3490.355 | 通过 / 失败 |
| cost_500 | 55.862 | 7.446 | 836.464 | 441.964 | 1341.737 | 通过 / 通过 |

三项 real_call_verified=true；完整续接只提交为一条 canonical 请求，同房间两次 brain 请求（开场问句+完整续接），没有用旧回复首音频充数。cost 的打断发生在 interim，但提交等到包含“以内”的完整句；打断并不等于提交旧 partial。原轮人数为 4006/3621ms，本轮为 3246/3490ms，不将生成侧波动称为优化收益。

## 根因证据

1. `eidolon/livekit/agent/session/eot_model.py` 的 `InterruptAwareTurnDetector` 只有非流式 `predict_end_of_turn`，不实现 SDK streaming detector 的 `stream`。本地依赖 `_resolve_endpointing` 对此类型实际返回 max_delay=3.0；SDK 的另一组 streaming 默认 2.5s 不适用于本路径。
2. `full_duplex/agent_builder.py:116–126` 只设置完整句快速路径的 min_delay，刻意保留不完整句的 SDK 上限。`plugins/eot/models/base.py:294–319` 直接使用已有 learned completeness scorer；不因“允许打断”提高分数。生产配置 threshold=0.50、VAD min_silence_duration_ms=500。
3. `livekit/agents/voice/audio_recognition.py:1550–1556` 在 score<threshold 时选择 max_delay；同文件 `:1662–1671` 从 last_speaking_time 算剩余等待，扣除已经过去的时间。因此打断 owner 的等待发生于此窗口内，不是另叠 3 秒。
4. SDK `audio_recognition.py:1392–1401` 从 VAD 事件时刻减掉 silence_duration/inference_duration，得到声学终点锚；Channel `full_duplex/speech_lifecycle.py:145–150` 则在 listening 转换发生时记录 `speech_stopped_at`。500ms 静音确认后再量至 3 秒截止，呈现约 2502ms。不能把其误称为 SDK max_delay=2.5。
5. 本轮日志：people_500 16:40:28.771 predict score=0.424，16:40:29.576 candidate resolved，16:40:31.272 开始完成提交；people_900 16:41:07.821 score=0.424；cost_500 16:41:43.252 score=0.900。people 的确认仅比停声晚约 0.8s，后面仍等待框架截止。归档的同轮时间戳和 owner/gate 链可复算。

源码行号对应 evidence.json 固定哈希版本。仅归档测试输入、阶段时间与聚合，不复制账户凭据、模型回复或其他会话原文。

## 验证与测量边界

既有回归 **157 passed、11 xfailed（9.55s）**。包括不完整前缀保留发言权、慢续接、不提交过期框架片段、同一 canonical 候选合并、弱信号 hold、会话 EOT 状态隔离。11 项是已有 strict xfail：模型完整性、完成延迟、打断语义与 terminal hold 的已知缺口；没有修改标注或把它们算通过。三次真实完整续接没有误截断，不足以证明真实人声抢话率达标。

`canonical_reply_speech_stop_to_audio_ms` 从**服务端 VAD 停声事件**到**应用 TTS 首音频**，不包含之前的 VAD 静音确认；接收端同轮新回复、扬声器首音频仍未测。客户端最后有声帧是另一口径/时钟，不能直接减服务端单调时钟。README 已补充此区别，未变更指标字段或门槛。

原始制品：`benchmark/runs/mac-continuation-verified-20261008/livekit_room/`（Git 忽略），哈希及脱敏选择见 evidence.json。原始服务日志在 Mac product logs/channel；不整体归档。正式重复性能矩阵不在本次诊断内继续展开。

## 复现命令

在 Channel 根目录，使用现有 Ops 环境加载器；身份通过原报告的本地 identity_preflight.json 读取，勿打印密钥。

```bash
EIDOLON_ENV=product \
EIDOLON_CHANNEL_SETTINGS_YAML=/Users/manson/ai/eidolon/.eidolon/mac-product/config/settings/channel.yaml \
/Users/manson/ai/eidolon/eidolon_ops/deploy/supervisor/wrappers/with-env.sh \
  /Users/manson/ai/eidolon/eidolon_channel \
  /Users/manson/ai/eidolon/.eidolon/mac-product/config/env/channel.env -- \
  .venv/bin/python scripts/bench_voice.py --runner livekit_room \
  --cases benchmark/cases/full_duplex/pause_continuation_enforced.yaml \
  --repeat 1 --run-id mac-continuation-verified-20261008 \
  --livekit-participant-identity "$EIDOLON_BENCH_LIVEKIT_PARTICIPANT_IDENTITY" \
  --livekit-interaction-mode full_duplex

.venv/bin/python -m pytest -q eidolon/livekit/tests/eot \
  eidolon/livekit/tests/agent/test_eot_full_duplex_product_contract.py \
  eidolon/livekit/tests/agent/test_eot_session_isolation.py \
  eidolon/livekit/tests/agent/test_user_turn_coordinator.py \
  eidolon/livekit/tests/agent/test_commit_guard.py \
  eidolon/livekit/tests/agent/test_turn_policy_runtime.py
```

真实 benchmark 退出 1 符合两项延迟失败；未改门槛凑通过。仅文档/证据改动，不新增生产代码或策略层，本地提交、不 push。
