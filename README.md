# eidolon_channel

独立仓库：Eidolon **LiveKit 语音 Channel**（`eidolon.channel.livekit`）—— Agent worker、STT/TTS/VAD/EOT 插件与测试。

## 本地开发

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

启动 Agent 与配置说明见 [deploy/channel/README.md](deploy/channel/README.md)。

## 测试

```bash
pytest
```

默认排除 `integration` 标记的用例（需真实 API / 密钥或长耗时）。运行全部：

```bash
pytest -m integration
```

可选：在 `eidolon/channel/livekit/tests/.env` 放置与 `deploy/channel/livekit-channel.env.template` 同结构的配置；否则测试使用 `tests/fixtures/minimal_test.env` 占位值。

## gRPC 桩

修改 `proto/.../remote_agent_rpc.proto` 后在仓库根执行：

```bash
./scripts/gen_remote_agent_rpc.sh
```
