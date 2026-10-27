#!/usr/bin/env bash
# 启动 LeLab（代码与环境均来自本仓库）
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$DIR/../LeLabPro_venv"; [[ -d "$VENV" ]] || VENV="$DIR/venv"
[[ -d "$VENV" ]] || VENV="$DIR/lerobot-env"
[[ -d "$VENV" ]] || { echo "[ERROR] 未找到环境，先跑 install.sh"; exit 1; }

if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then conda deactivate 2>/dev/null || true; fi

# torchcodec 在 import 前就要看到 FFmpeg 的 soname，运行时改 LD_LIBRARY_PATH 无效，
# 所以在这里注入 install.sh 按需建好的 shim（没有系统 FFmpeg 时由 PyAV 自带库提供）。
SHIM="$("$VENV/bin/python" -c 'from lelab.utils.ffmpeg_shim import loader_path; print(loader_path())' 2>/dev/null)"
if [[ -n "${SHIM:-}" && -d "$SHIM" ]]; then
    export LD_LIBRARY_PATH="$SHIM${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

source "$VENV/bin/activate"
exec lelab "$@"
