#!/usr/bin/env bash
set -euo pipefail

# LeLabPro 一键安装
#
#   curl -fsSL https://raw.githubusercontent.com/cz11001100/LeLabPro/main/install.sh | bash
#
# 也可以先 clone 再执行：git clone ... && cd LeLabPro && bash install.sh
#
# 做的事：退出 conda → 装 uv → 建 Python 3.12 虚拟环境 → 装本仓库的 lerobot 与
# lelab（国内镜像）→ 自检关键模块 → 生成 ./lelab-run.sh

REPO_URL="https://github.com/cz11001100/LeLabPro.git"
MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info() { echo -e "${GREEN}[INFO]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
step() { echo -e "${BLUE}==> $*${NC}"; }
die()  { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

FAILED_STEP=""
on_error() {
    local code=$?
    echo ""
    echo -e "${RED}========== 安装中断 (退出码 $code) ==========${NC}"
    step "失败于: ${FAILED_STEP:-未知步骤}"
    case "$FAILED_STEP" in
        *uv*)     echo "  网络访问不了 astral.sh 时：pip install uv 后重跑" ;;
        *环境*)   echo "  uv 下 Python 3.12 超时：sudo apt install python3.12 python3.12-venv 后重跑" ;;
        *lerobot*) echo "  多为下载超时或磁盘不足。查 df -h ~ （需 ≥15GB），再单独重试："
                   echo "    uv pip install --python \$VENV/bin/python -e ./lerobot[feetech,smolvla,training]" ;;
        *lelab*)  echo "  单独重试：uv pip install --python \$VENV/bin/python fastapi[standard] websockets uvicorn psutil paramiko" ;;
        *自检*)   echo "  依赖装了但代码对不上，把上面 [X] 的报错原文发给维护者" ;;
    esac
    echo "  更多排查见 TROUBLESHOOTING.md"
    echo -e "${RED}===========================================${NC}"
    exit $code
}
trap on_error ERR

# ---------- 退出 conda，避免它覆盖 PATH ----------
FAILED_STEP="退出 conda"
if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    warn "检测到 conda 环境 (${CONDA_DEFAULT_ENV})，先退出"
    conda deactivate 2>/dev/null || true
fi
command -v conda &>/dev/null && { eval "$(conda shell.bash hook 2>/dev/null)" 2>/dev/null || true; conda deactivate 2>/dev/null || true; }

# ---------- 定位仓库：curl 执行时先自举 clone ----------
SCRIPT_PATH="${BASH_SOURCE[0]:-}"
if [[ -f "$SCRIPT_PATH" && -d "$(cd "$(dirname "$SCRIPT_PATH")" && pwd)/lelab" \
      && -d "$(cd "$(dirname "$SCRIPT_PATH")" && pwd)/lerobot" ]]; then
    ROOT="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
    info "使用已有仓库目录: $ROOT"
else
    ROOT="$PWD/LeLabPro"
    if [[ -d "$ROOT/.git" ]]; then
        info "更新已有克隆 $ROOT"
        git -C "$ROOT" pull --ff-only -q || warn "pull 失败，沿用当前版本"
    else
        FAILED_STEP="克隆仓库"
        command -v git &>/dev/null || die "需要 git，请先安装：sudo apt install git"
        info "克隆 $REPO_URL 到 $ROOT"
        git clone --depth 1 -q "$REPO_URL" "$ROOT"
    fi
fi
VENV="$ROOT/lerobot-env"
VENV_PY="$VENV/bin/python"

# ---------- uv ----------
FAILED_STEP="步骤1 安装 uv"
if command -v uv &>/dev/null; then
    info "uv 已就绪: $(uv --version)"
else
    info "安装 uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    command -v uv &>/dev/null || die "uv 安装失败：curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

# ---------- 环境体检（只警告，不阻断）----------
FAILED_STEP="环境体检"
info "环境体检..."
echo "  系统: $(uname -s) $(uname -m)"
if command -v nvidia-smi &>/dev/null; then
    echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
else
    warn "  没检测到 NVIDIA GPU —— 采集/推理可用，本地训练不可用（可 SSH 连云端 GPU）"
fi
AVAIL=$(df -BG --output=avail "$ROOT" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
[[ "${AVAIL:-0}" -lt 15 ]] && warn "  磁盘仅剩 ${AVAIL}GB，建议 ≥15GB（torch + 模型权重）" || echo "  磁盘可用: ${AVAIL}GB"
ls /dev/ttyUSB* /dev/ttyACM* &>/dev/null || warn "  没看到 /dev/ttyUSB* 或 /dev/ttyACM* —— 机械臂没插或未被识别"
id -nG 2>/dev/null | grep -qw dialout || warn "  当前用户不在 dialout 组，连机械臂会 Permission denied
      修复: sudo usermod -aG dialout \$USER  然后注销重新登录"
command -v ss &>/dev/null && ss -ltn 2>/dev/null | grep -q ':8000 ' && warn "  端口 8000 已被占用，启动会失败（先 ./lelab-run.sh --stop）"

# ---------- 虚拟环境 ----------
FAILED_STEP="步骤2 创建虚拟环境"
if [[ -d "$VENV" ]]; then
    info "复用已有虚拟环境 $VENV"
else
    info "创建虚拟环境 (Python 3.12，与开发机一致)..."
    uv venv "$VENV" --python 3.12
fi

# ---------- lerobot ----------
FAILED_STEP="步骤3 安装 lerobot"
info "安装本仓库的 lerobot（含 feetech 电机 / SmolVLA / 训练依赖）..."
info "  这步要下 torch 等大包，约 5-15 分钟，请耐心等"
UV_INDEX_URL="$MIRROR" uv pip install --python "$VENV_PY" \
    -e "$ROOT/lerobot[core_scripts,feetech,smolvla,training]"

# ---------- RealSense（可选）----------
FAILED_STEP="步骤4 pyrealsense2（可选，允许失败）"
if UV_INDEX_URL="$MIRROR" uv pip install --python "$VENV_PY" \
     "pyrealsense2>=2.55.1.6486,<2.57.0" 2>/dev/null; then
    info "  pyrealsense2 已装（Intel RealSense 深度相机）"
else
    warn "  跳过 pyrealsense2 —— 只有 RealSense 深度相机需要，普通 USB 摄像头不受影响"
fi

# ---------- lelab ----------
FAILED_STEP="步骤5 安装 LeLab"
info "安装 LeLab 依赖..."
UV_INDEX_URL="$MIRROR" uv pip install --python "$VENV_PY" \
    "fastapi[standard]>=0.115.12" "websockets>=15.0.1" "uvicorn>=0.24.0" \
    "psutil>=5.9.0" "paramiko>=3.4.0"
info "安装 LeLab..."
uv pip install --python "$VENV_PY" --no-deps -e "$ROOT/lelab"

# ---------- 启动脚本 ----------
FAILED_STEP="生成启动脚本"
cat > "$ROOT/lelab-run.sh" << 'WRAPPER'
#!/usr/bin/env bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -d "$DIR/lerobot-env" ]] || { echo "[ERROR] 未找到环境，先跑 install.sh"; exit 1; }
if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then conda deactivate 2>/dev/null || true; fi
source "$DIR/lerobot-env/bin/activate"
exec lelab "$@"
WRAPPER
chmod +x "$ROOT/lelab-run.sh"

# ---------- 自检 ----------
FAILED_STEP="步骤6 安装自检"
info "自检关键模块..."
OK=true
check() {  # check <说明> <python 表达式>
    if "$VENV_PY" -c "$2" >/dev/null 2>&1; then
        echo "  [OK] $1"
    else
        warn "  [X] $1 —— 真实报错:"
        "$VENV_PY" -c "$2" 2>&1 | tail -4 | sed 's/^/        /'
        OK=false
    fi
}
check "paramiko（SSH 远程训练/装环境）" "import paramiko"
check "lerobot 采集与数据集模块"        "from lerobot.scripts.lerobot_record import DatasetRecordConfig; from lerobot.datasets.lerobot_dataset import LeRobotDataset"
check "SO-101 机械臂配置"              "from lerobot.robots.so_follower import SO101FollowerConfig"
check "rewards（RA-BC 打分）"          "from lerobot.rewards.robometer import compute_rabc_weights"
check "LeLab 后端（启动入口）"          "import lelab.server"

# ---------- 收尾 ----------
echo ""
echo "=========================================="
if $OK; then info "安装完成，自检全部通过 ✓"; else warn "安装完成，但自检有问题（见上）"; fi
echo "=========================================="
echo ""
step "启动"
echo "  $ROOT/lelab-run.sh"
echo "  浏览器打开 http://localhost:8000"
echo ""
step "常用参数"
echo "  ./lelab-run.sh --stop      清理残留进程（端口被占时用）"
echo "  ./lelab-run.sh --no-open   不自动开浏览器"
echo ""
step "环境出问题时"
echo "  rm -rf $VENV && bash install.sh    # 重建环境"
echo "  排查表: $ROOT/TROUBLESHOOTING.md"
echo ""
