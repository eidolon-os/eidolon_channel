# LiveKit Voice Agent — 本地启动

本目录仅用于**本机开发**启动 Channel Agent，不再包含 systemd 或服务器一键部署脚本。

## 前置条件

- 已在**本目录或其任意上级目录**中存在可执行的 `.venv/bin/python`（`run_livekit_channel.sh` 自 `deploy/channel` 向上逐级查找；常见为仓库根目录的 `.venv`，例如 `uv sync` 或 `pip install -e .`）
- LiveKit Server 已可用（本地或远程）
- 已准备好本目录下的 `.livekit-channel.env`

## 目录说明

| 文件 | 说明 |
|------|------|
| `run_livekit_channel.sh` | 本地启动 Agent |
| `livekit-channel.env.template` | 环境变量模板，复制为 `.livekit-channel.env` 后填写 |
| `.livekit-channel.env` | 运行时配置（含密钥，勿提交；见 `.gitignore`） |

## 配置

从模板生成本地配置并编辑：

```bash
cd /path/to/eidolon_channel/deploy/channel
cp livekit-channel.env.template .livekit-channel.env
vim .livekit-channel.env
```

必填项见模板内注释；常见项包括 `LIVEKIT_URL`、`LIVEKIT_API_KEY`、`LIVEKIT_API_SECRET` 以及各 STT/TTS/LLM 的 API Key。

## 启动

在项目根目录：

```bash
chmod +x deploy/channel/run_livekit_channel.sh
./deploy/channel/run_livekit_channel.sh
```

或在 `deploy/channel` 下：

```bash
./run_livekit_channel.sh
```

脚本会设置 `EIDOLON_CHANNEL_LIVEKIT_ENV` 指向本目录的 `.livekit-channel.env`，并以 `EIDOLON_ENV=dev` 运行 `eidolon.channel.livekit.agent.server`。

## 配置加载

`EIDOLON_CHANNEL_LIVEKIT_ENV` → `config.py` 中 `load_dotenv` 加载 `.livekit-channel.env`。**未设置或路径不是已存在文件时，`AgentConfig.from_env()` 会直接抛 `ValueError`。**环境变量优先级高于文件中的默认值（`override=False`）。

## 远程 Agent（gRPC，可选）

若设置 `REMOTE_AGENT_RPC_TARGET`（例如同机 `unix:///path/agent.sock`），语音流水线中的 LLM 将经 gRPC `RemoteAgent.Session` 转发到远端实现，而不再使用 `livekit-plugins-openai`。

- **契约文件**：`proto/eidolon/channel/livekit/agent/remote_agent_rpc/v1/grpc_gen/remote_agent_rpc.proto`
- **生成 Python 桩代码**：在仓库根执行 `./scripts/gen_remote_agent_rpc.sh`，或 `./scripts/gen_grpc_stubs.sh eidolon/channel/livekit/agent/remote_agent_rpc/v1/grpc_gen/remote_agent_rpc.proto`（依赖 `.venv` 中的 `grpcio-tools`）
- **入口说明**：`proto/remote_agent_rpc/v1/README.md`

## 故障排查

- **找不到 Python**：自 `deploy/channel` 向上直到 `/` 均未发现 `.venv/bin/python`；在某一上级目录（多为仓库根）创建虚拟环境并安装依赖。
- **连接 / 鉴权失败**：检查 `.livekit-channel.env` 中 LiveKit 与各云厂商 Key 是否与当前环境一致。
