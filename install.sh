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

# 国内镜像下大 wheel（torch/cudnn/rerun-sdk 几百 MB）常超 uv 默认 30s 而中断，
# 报 "I/O operation failed during extraction / network timeout"。放宽到 300s。
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"

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

# ---------- 修复 torchcodec 缺 FFmpeg 共享库 ----------
FAILED_STEP="步骤5b torchcodec 的 FFmpeg 共享库"
info "检查 torchcodec 能否加载 FFmpeg 共享库..."
if "$VENV_PY" -m lelab.utils.ffmpeg_shim --ensure 2>&1 | tail -2 | sed 's/^/  /'; then
    :
else
    warn "  未能自动修复 —— 视频解码会退回 pyav，训练/回放仍可用"
    warn "  要拿回 torchcodec：装系统 FFmpeg（sudo apt install ffmpeg），"
    warn "  或 conda install -c conda-forge ffmpeg 后把 \$CONDA_PREFIX/lib 加进 LD_LIBRARY_PATH"
fi

# ---------- 启动脚本 ----------
FAILED_STEP="生成启动脚本"
# 仓库自带 lelab-run.sh；只有从别处（如旧 tarball）安装、缺少该文件时才生成。
# 无条件覆盖会让"升级"这一步把受版本控制的文件改脏，而且生成内容一旦与仓库版本
# 不同，升级后启动器就指向了错误的虚拟环境。
if [[ ! -f "$ROOT/lelab-run.sh" ]]; then
    cat > "$ROOT/lelab-run.sh" << 'WRAPPER'
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
WRAPPER
fi
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
if "$VENV_PY" -c "from torchcodec.decoders import VideoDecoder" >/dev/null 2>&1; then
    echo "  [OK] torchcodec（FFmpeg 硬解，$("$VENV_PY" -c 'import json;from torchcodec._core.ops import _get_json_ffmpeg_library_versions as j;print(json.loads(j())["ffmpeg_version"])' 2>/dev/null)）"
else
    warn "  [!] torchcodec 不可用 —— 视频解码退回 pyav，功能不受影响但更慢"
fi

# ---------- 全局 lelab 命令 ----------
FAILED_STEP="配置全局 lelab 命令"
BIN_DIR="$HOME/.local/bin"
mkdir -p "$BIN_DIR"
if [[ -e "$BIN_DIR/lelab" ]] && ! grep -q "LeLabPro" "$BIN_DIR/lelab" 2>/dev/null; then
    BAK="$BIN_DIR/lelab.bak.$(date +%Y%m%d-%H%M%S)"
    mv "$BIN_DIR/lelab" "$BAK"
    warn "已备份原有的 $BIN_DIR/lelab → $(basename "$BAK")"
fi
# 脚本/非交互场景用：直接转发到启动器（启动器内含 conda 退出与 FFmpeg 路径注入）
cat > "$BIN_DIR/lelab" << WRAPPER
#!/usr/bin/env bash
# 由 LeLabPro install.sh 生成
exec "$ROOT/lelab-run.sh" "\$@"
WRAPPER
chmod +x "$BIN_DIR/lelab"

# 交互场景用 shell 函数：能真正把激活态留在当前 shell，方便接着调试。
# 注意函数里不能调用 lelab-run.sh —— 它结尾是 exec，会把当前 shell 顶掉。
BASHRC="$HOME/.bashrc"
if [[ ! -f "$BASHRC" ]]; then
    # 精简的系统/容器里可能没有 .bashrc；先建空文件，否则函数会被静默跳过
    : > "$BASHRC"
else
    cp "$BASHRC" "$BASHRC.bak.$(date +%Y%m%d-%H%M%S)"
fi
sed -i '/^# >>> LeLabPro/,/^# <<< LeLabPro/d' "$BASHRC"
    cat >> "$BASHRC" << BLOCK

# >>> LeLabPro (install.sh) >>>
# 输入 lelab 自动准备环境：退出 conda → 激活虚拟环境 → 注入 FFmpeg 路径 → 启动。
case ":\$PATH:" in *":\$HOME/.local/bin:"*) ;; *) export PATH="\$HOME/.local/bin:\$PATH" ;; esac
lelab() {
    local root="$ROOT" venv=""
    for cand in "\$root/lerobot-env" "\$root/venv" "\$root/../LeLabPro_venv"; do
        [[ -d "\$cand" ]] && { venv="\$cand"; break; }
    done
    [[ -n "\$venv" ]] || { echo "[lelab] 未找到环境，先执行: cd \$root && bash install.sh" >&2; return 1; }
    [[ -n "\${CONDA_DEFAULT_ENV:-}" ]] && conda deactivate 2>/dev/null
    source "\$venv/bin/activate"
    local shim
    shim="\$("\$venv/bin/python" -c 'from lelab.utils.ffmpeg_shim import loader_path; print(loader_path())' 2>/dev/null)"
    [[ -n "\$shim" && -d "\$shim" ]] && export LD_LIBRARY_PATH="\$shim\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
    "\$venv/bin/lelab" "\$@"
}
# <<< LeLabPro (install.sh) <<<
BLOCK
info "已在 $BASHRC 配置 lelab 函数（原文件已备份）"

# ---------- 收尾 ----------
echo ""
echo "=========================================="
if $OK; then info "安装完成，自检全部通过 ✓"; else warn "安装完成，但自检有问题（见上）"; fi
echo "=========================================="
echo ""
step "启动"
echo "  lelab                      （新开终端生效；当前终端可先 source ~/.bashrc）"
echo "  $ROOT/lelab-run.sh   （等价写法）"
echo "  浏览器打开 http://localhost:8000"
echo ""
step "常用参数"
echo "  lelab --stop      清理残留进程（端口被占时用）"
echo "  lelab --no-open   不自动开浏览器"
echo ""
step "环境出问题时"
echo "  rm -rf $VENV && bash install.sh    # 重建环境"
echo "  排查表: $ROOT/TROUBLESHOOTING.md"
echo ""
