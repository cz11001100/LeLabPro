#!/usr/bin/env bash
# 启动 LeLab（代码与环境均来自本仓库）
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$DIR/venv"; [[ -d "$VENV" ]] || VENV="$DIR/lerobot-env"
[[ -d "$VENV" ]] || { echo "[ERROR] 未找到环境，先跑 install.sh"; exit 1; }
if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then conda deactivate 2>/dev/null || true; fi
source "$VENV/bin/activate"
exec lelab "$@"
