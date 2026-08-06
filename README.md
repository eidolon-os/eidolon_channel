# eidolon_channel

独立仓库：Eidolon **LiveKit 语音 Channel**（`eidolon.livekit`）—— Agent worker、STT/TTS/VAD/EOT 插件与测试。

Channel 的接入边界区分 Owner-scoped `DeviceConnectionContext` 与完整
`CompanionInteractionContext`：Device 可以只建立 data connection 而不选择
Companion；Companion 也可以通过 Web/小程序等虚拟 endpoint 建立无物理 Device 的
interaction。现有语音 pipeline 只接收完整 Companion context，不在 VAD/STT/EOT/LLM/TTS
内部传播 optional Companion。Kernel Mount consumer 目前为显式 feature flag，待本机产品
composition 启动 Kernel 且 Provider 能提供受信 `owner_id` 后启用。

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

修改 `eidolon/proto/.../remote_agent_rpc.proto` 后在仓库根执行：

```bash
./scripts/gen_remote_agent_rpc.sh
```
