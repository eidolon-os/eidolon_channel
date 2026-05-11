#!/bin/bash
# 本地启动 Eidolon LiveKit Agent worker。
#
# 连接 LiveKit Server（自托管或云端），在设备请求参与时加入房间；
# 提供 VAD/EoT + STT/LLM/TTS 流水线。
#
# 前置：
#   1. LiveKit Server 已运行
#   2. 自本脚本所在目录起向上逐层目录查找 .venv/bin/python，直至文件系统根
#      （通常落在仓库根目录的 .venv）
#   3. 本目录下已配置 .livekit-channel.env（可由 livekit-channel.env.template 复制）
#
# 用法（在仓库根目录或任意目录）：
#   ./deploy/run_livekit_channel.sh
#   或：cd deploy && ./run_livekit_channel.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

SEARCH_DIR="$SCRIPT_DIR"
PROJECT_DIR=""
while true; do
    if [[ -x "$SEARCH_DIR/.venv/bin/python" ]]; then
        PROJECT_DIR="$SEARCH_DIR"
        break
    fi
    if [[ "$SEARCH_DIR" == "/" ]]; then
        break
    fi
    SEARCH_DIR="$(dirname "$SEARCH_DIR")"
done

if [[ -z "$PROJECT_DIR" ]]; then
    echo "Error: no executable .venv/bin/python found from $SCRIPT_DIR up to /"
    echo "Create a venv on that path or any parent, e.g.: cd <repo-root> && python -m venv .venv && .venv/bin/pip install -e ."
    exit 1
fi

EIDOLON_CHANNEL_LIVEKIT_ENV="$SCRIPT_DIR/.livekit-channel.env" \
EIDOLON_ENV=dev PYTHONPATH="$PROJECT_DIR" \
    "$PROJECT_DIR/.venv/bin/python" -m eidolon.livekit.agent.server "$@"
