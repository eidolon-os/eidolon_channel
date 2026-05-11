# remote_agent_rpc（契约入口）

权威 `.proto` 路径（与生成 Python 包路径一致，便于 `grpc_tools` 输出到 `grpc_gen/`）：

权威文件（相对仓库根）：`proto/eidolon/channel/livekit/agent/remote_agent_rpc/v1/grpc_gen/remote_agent_rpc.proto`

生成命令：仓库根执行 `./scripts/gen_remote_agent_rpc.sh`（薄包装，内部调用 `./scripts/gen_grpc_stubs.sh` + 上述相对路径）。
