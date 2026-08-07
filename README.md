# eidolon_channel

独立仓库：Eidolon **LiveKit 语音 Channel**（`eidolon.livekit`）—— Agent worker、STT/TTS/VAD/EOT 插件与测试。

Channel 的接入边界区分 Owner-scoped `DeviceConnectionContext` 与完整
`CompanionInteractionContext`：Device 可以只建立 data connection 而不选择
Companion；Companion 也可以通过 Web/小程序等虚拟 endpoint 建立无物理 Device 的
interaction。现有语音 pipeline 只接收完整 Companion context，不在 VAD/STT/EOT/LLM/TTS
内部传播 optional Companion。物理 Device 入口必须携带受信 `owner_id` 并经 Kernel Mount；
没有 Companion target 的 Device 只停留在 Device/data path，不能进入语音 brain。

## Eidolon OS 运行时边界

- Kernel 是 Owner namespace 下 Device Mount/Attachment 的权威。
- System Data Companion Runtime Authority 提供 active Companion、Memory Realm、已提交
  Persona Genome 和可选头像；Channel 不直读 Data SQLite。
- Channel 为一次 LiveKit 会话解析并缓存完整 runtime context。Agent token、Voiceprint 与
  Avatar 复用该结果，不存在 Admin Resolve 旁路。
- Channel→Agent 使用 V5 窄 runtime token，只携带 `owner_id`、`companion_id`、必需
  `session_id` 与可选 `device_id/scopes`。`session_id` 固定为当前 LiveKit room；
  Genome/Realm/运行策略由 Agent 向 System Data 重新解析和校验。
- Proactive 订阅不再发送 wildcard/instance selector；Agent 只允许订阅 token 中
  Companion 的事件。`conversation_id` 仍是历史上下文键，不承担会话授权。
- LiveKit JWT 仍只属于 Channel/LiveKit 链路；上述 OS 边界不进入 VAD/STT/EOT/TTS
  算法，也不改变 full/half/PTT pipeline。

运行时连接配置位于 `runtime_authority`：Kernel V1 URL、System Data URL 和服务 token
环境变量名。旧 `runtime_admin`、Admin fallback、Data Device resolve 与 feature flag 已删除。

## 本地开发

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

初始化配置：`./deploy/dev/init.sh`（生成 `config/.env` 与 `config/settings.yaml`）。  
全栈本地开发由 [eidolon_admin](https://github.com/eidolon/eidolon_admin) 的 `./deploy/dev/run_all.sh` 拉起 channel worker；env 旋钮见 [`config/.env.example`](config/.env.example)。

## 测试

```bash
pytest
```

默认排除 `integration` 标记的用例（需真实 API / 密钥或长耗时）。运行全部：

```bash
pytest -m integration
```

可选：在 `eidolon/livekit/tests/.env` 放置与 `config/.env.example` 同结构的配置；否则测试使用 `eidolon/livekit/tests/fixtures/minimal_test.env` 占位值。

## gRPC 桩

修改 `eidolon/proto/.../eidolon_agent_rpc/v1/grpc_gen/eidolon.proto` 后在仓库根执行：

```bash
./scripts/gen_eidolon_agent_rpc.sh
```
