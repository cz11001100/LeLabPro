# LeLabPro Windows one-click installer (PowerShell)
#
#   powershell -ExecutionPolicy ByPass -NoProfile -Command "irm https://raw.githubusercontent.com/cz11001100/LeLabPro/main/install.ps1 | iex"
#
# 面向没有独立显卡、只做遥操/采集、训练放云端的机器。Windows 上 PyPI 的 torch
# 本来就是 CPU 版，不会拖进 CUDA 库，所以不需要 Linux 那边的 --lite 档。
#
# 做的事：检查 Git → 装 uv → 用 uv tool install 从本仓库装 lelab（连带装上仓库
# 里的 lerobot）→ 提示怎么启动。
#
# 可重复执行：这条命令同时也是升级命令。

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Repo   = "https://github.com/cz11001100/LeLabPro.git"
$Source = "git+$Repo#subdirectory=lelab"
$UvBin  = Join-Path $env:USERPROFILE ".local\bin"

function Say($m)  { Write-Host "[INFO] $m" -ForegroundColor Green }
function Warn($m) { Write-Host "[WARN] $m" -ForegroundColor Yellow }
function Fail($m) { Write-Host "[ERROR] $m" -ForegroundColor Red; exit 1 }

# ---------- Git ----------
# uv 装 git+https 依赖时要调真正的 git，缺了会在解析阶段报很难懂的错。
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Fail "没找到 git。先装它： winget install --id Git.Git -e`n     装完请重新开一个 PowerShell 窗口再跑本脚本。"
}

# ---------- uv ----------
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    $candidate = Join-Path $UvBin "uv.exe"
    if (Test-Path $candidate) {
        $env:Path = "$UvBin;$env:Path"
    } else {
        Say "安装 uv ..."
        try {
            irm https://astral.sh/uv/install.ps1 | iex
        } catch {
            Fail "uv 自动安装失败：$_ 。手动装： winget install --id astral-sh.uv -e，然后重开 PowerShell 再跑本脚本。"
        }
        $env:Path = "$UvBin;$env:Path"
    }
}
$Uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $Uv) { Fail "uv 装好了但当前窗口还找不到它 —— 请新开一个 PowerShell 窗口，重新运行本命令。" }
Say "uv 就绪：$(& $Uv.Source --version)"

# ---------- 安装 ----------
# --force 让这条命令兼任升级：仓库每次推送后重跑就能覆盖旧版本。
$UvArgs = @("tool", "install", "--force", "--python", "3.12", $Source)
Say "安装 LeLab（连带装上本仓库的 lerobot、torch、transformers 等，约 5-15 分钟）..."
Warn "  日志停在 Downloading cpython-3.12 属正常：uv 在拉独立 Python 解释器，约 32MB"
& $Uv.Source @UvArgs
if ($LASTEXITCODE -ne 0) {
    Fail "LeLab 安装失败（退出码 $LASTEXITCODE）。网络不稳就重跑本命令；卡在拉 Python 解释器时先执行 `$env:UV_PYTHON_INSTALL_MIRROR 指向一个可达的镜像再重跑。"
}

Write-Host ""
Write-Host "=========================================="
Say "安装完成"
Write-Host "=========================================="
Write-Host ""
Write-Host "启动："
Write-Host "    lelab"
Write-Host "  （请新开一个 PowerShell 窗口再输入，PATH 才会带上 $UvBin）"
Write-Host "  浏览器打开 http://localhost:8000"
Write-Host ""
Write-Host "升级 / 修复：重跑上面那条安装命令即可"
Write-Host ""
Write-Host "Windows 注意："
Write-Host "  机械臂：装 CH340/CP2102 驱动；端口号在设备管理器里看（常见 COM3），界面里手动选"
Write-Host "  摄像头：LeLab 走 DirectShow；被 Zoom / Teams 占用时拿不到画面"
Write-Host "  视频解码：缺 FFmpeg DLL 时 torchcodec 自动退回 PyAV，功能不受影响只是慢。"
Write-Host "            想要硬解： winget install --id Gyan.FFmpeg -e"
