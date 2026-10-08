# Mac Host 真实续接小样本（2026-10-08）

用户授权使用当前 Mac Host 的身份与环境。复用既有已注册 Owner 和 product-source 服务，identity、真实 STT/TTS preflight 通过；未重启服务、调整模型或生产策略。运行模式为 livekit_room / full_duplex，案例文件为 benchmark/cases/full_duplex/pause_continuation_enforced.yaml。

## 结果

| 用例 | 功能结果 | 停声→提交 | 停声→应用 TTS 首音频 | 2500ms 门槛 |
| --- | --- | ---: | ---: | --- |
| people_500 | 通过 | 2501ms | 4006ms | 失败 |
| people_900 | 通过 | 2502ms | 3621ms | 失败 |
| cost_500 | 通过 | 118ms | 1486ms | 通过 |

三项都有完整续接文本、真实 brain 请求及同一轮新回复音频证据；综合通过 1/3。人数用例 EOT 日志约 0.42，经 stable_normal_interrupt 路径确认；费用用例约 0.90，经高分路径确认。主要等待发生在停声至提交；这仍不足以认定具体生产根因或授权降低端点阈值。下一步只核查既有 EOT/提交时序的等待来源，不新增决策层、不按句子特判。

## 测试器修复

初跑 mac-continuation-20261008 同样为 1/3，但人数用例被旧回复音频触发结束，固定等待后房间关闭，取消了正在生成的新回复，因此不能作为“续接无回复”的证据。修复为在 canonical_response_required 用例中复用现有 canonical validator 的同轮证据，等完整请求的新回复出现后才结束房间。保持原门槛和有界超时；只读取本次 capture 后且属于该房间的完整 timeline 行。benchmark 回归 223 项通过，Ruff 通过。

## 证据与限制

归档 evidence.json 只保留测试用例、所需指标、完整请求的时间戳和原始文件 SHA-256，不复制账号凭据、无关会话或模型回复。原始两次运行位于 benchmark/runs/ 对应 run-id 的 livekit_room 目录（被 Git 忽略）。

这是当前运行服务的诊断：测试 checkout 为 b65831a 加本轮未提交 benchmark 修复；未核实所有运行进程的加载版本，不能称为 HEAD 部署验收。每题只有 1 次复测，不报告 p95 达标。首音频为应用 TTS 侧；RTC 的下一帧可能仍是旧音频，接收端同轮新回复首音频仍未验收。Mac Host 环境与身份已解决，无需再次向用户索取。
