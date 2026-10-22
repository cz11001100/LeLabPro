<h1 align="center">🦾 LeLabPro</h1>

<p align="center">
  <b>LeRobot 的图形化界面 — 标定、遥操作、采集、训练、推理一站式 Web 应用。</b>
</p>

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/cz11001100/LeLabPro/blob/main/LICENSE)

</div>

**LeLabPro** 将完整的 LeRobot 工作流 — 标定、遥操作、数据采集、训练、回放 — 集成到一个浏览器界面。插上机械臂，打开应用，即刻开始。无需 CLI 操作，无需键盘提示。

专为 SO-101 leader/follower 机械臂设计，让新手能在几分钟内从"开箱"到"训练第一个策略"。

> **关于本项目**：LeLabPro 是基于上游 [LeLab](https://github.com/OneRobotAI/lelab) 与 [LeRobot](https://github.com/huggingface/lerobot)（均为 Apache License 2.0）的**个人修改分支**，与官方项目无隶属或背书关系。在上游之外本仓库做了大量面向 SO-100/SO-101 实机的改动（SSH 远程训练与导出、数据集质检与重建、录制续录、推理平滑、模型评估面板等），上游的版权声明与许可证文件均已保留。
>
> 单仓库结构：`lelab/` 是 Web 界面，`lerobot/` 是打过补丁的机器人框架。

### 最省事的上手方式

```bash
uv tool install "git+https://github.com/cz11001100/LeLabPro.git#subdirectory=lelab"
lelab
```

浏览器打开 http://localhost:8000 。分平台安装见 [INSTALL.md](INSTALL.md)，报错排查见 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)。

## 功能一览

<div align="center">
  <table>
    <tr>
      <td>🎯 <b>标定 (Calibrate)</b></td>
      <td>双机械臂图形化标定流程 — 无需键盘操作</td>
    </tr>
    <tr>
      <td>🕹️ <b>遥操作 (Teleoperate)</b></td>
      <td>操作 leader，follower 实时跟随。关节数据实时传输</td>
    </tr>
    <tr>
      <td>📹 <b>数据采集 (Record)</b></td>
      <td>录制 episodes 到 LeRobotDataset，支持摄像头</td>
    </tr>
    <tr>
      <td>🧠 <b>训练 (Train)</b></td>
      <td>启动 LeRobot 训练任务，实时查看日志</td>
    </tr>
    <tr>
      <td>🤖 <b>推理 (Inference)</b></td>
      <td>在 follower 上运行训练好的策略</td>
    </tr>
    <tr>
      <td>⏪ <b>回放 (Replay)</b></td>
      <td>回放任意已录制的 episode</td>
    </tr>
    <tr>
      <td>☁️ <b>上传 (Upload)</b></td>
      <td>一键将数据集推送到 <a href="https://huggingface.co/">Hugging Face Hub</a></td>
    </tr>
  </table>
</div>

---

## 安装指南

### 前置要求

- **Python 3.12+**
- **Git**
- **SO-101 机械臂的 USB 串口驱动**（Windows 需安装 CH340/CP2102 驱动）

### 安装 uv

LeLab 推荐用 **uv**（极速的 Python 包管理器）安装和运行。按平台选择：

**Windows（PowerShell）**
```powershell
# 方式一：winget
winget install --id=astral-sh.uv  -e

# 方式二：官方安装脚本
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS**
```bash
# 方式一：Homebrew
brew install uv

# 方式二：官方安装脚本
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Ubuntu / Linux**
```bash
# 方式一：官方安装脚本
curl -LsSf https://astral.sh/uv/install.sh | sh
# 脚本会把 uv 安装到 ~/.local/bin，重启终端或执行 source ~/.local/bin/env

# 方式二：apt（Ubuntu 24.10+）
sudo apt install uv

# 方式三：cargo
cargo install --git https://github.com/astral-sh/uv uv
```

安装后验证：
```bash
uv --version
```

> 如果系统已有旧版 uv，升级用 `uv self update`（三平台通用）。

### 通用方式（uv 推荐，三平台通用）

```bash
# 安装 LeLab
uv tool install "git+https://github.com/cz11001100/LeLabPro.git#subdirectory=lelab"

# 启动（自动打开浏览器）
lelab
```

> 提示：uv 会在 `~/.local/share/uv/tools/`（Windows 为 `%APPDATA%\uv\tools\`）下创建独立环境安装 LeLab，`lelab` 命令会自动加入 PATH。

---

## Windows 安装

### pip 方式

```powershell
# 克隆仓库
git clone https://github.com/cz11001100/LeLabPro.git
cd LeLabPro/lelab

# 创建虚拟环境
python -m venv venv
venv\Scripts\activate

# 安装
pip install -e .

# 启动
lelab
```

### Windows 常见问题

**uv 安装报错 `Failed to update Windows PE resources`（os error -2147024786）**

这个错误与 LeLab 代码无关，是 uv 在 Windows 上的已知问题 —— uv 尝试更新临时 `.exe` trampoline 文件的 PE 资源时被系统拒绝。官方 `huggingface/leLab` 也同样会触发。解决方案：

1. **更新 uv 到最新版**（最常见原因，旧版本有 trampoline bug）：
   ```powershell
   uv self update
   ```
   若无自更新，用 `pip install -U uv` 或 `winget upgrade astral-sh.uv`。

2. **临时关闭杀毒/Defender 实时防护**（已验证可成功）：
   Windows Defender 或其他杀毒软件会锁定临时 `.exe` 文件导致写入失败。
   - Windows 安全中心 → 病毒和威胁防护 → 管理设置
   - 临时关闭"实时保护"
   - 重新运行安装，装完再开启

3. **改用 pip 安装**（绕过 uv 的 trampoline 机制）：
   ```powershell
   python -m venv lelab-venv
   .\lelab-venv\Scripts\Activate.ps1
   pip install "git+https://github.com/cz11001100/LeLabPro.git#subdirectory=lelab"
   lelab
   ```

4. **清理 uv 缓存与临时目录**：
   ```powershell
   uv cache clean
   echo $env:TEMP   # 应输出可写的 C:\Users\admin\AppData\Local\Temp
   ```

**安装时 `git fetch` 连接 github.com:443 超时（浏览器能开 GitHub 但命令行失败）**

这是**代理不一致**问题：浏览器走了系统代理能访问 GitHub，但命令行 `git.exe` 不会自动读取系统代理。典型报错：
```
fatal: unable to access 'https://github.com/cz11001100/LeLabPro.git/': Failed to connect to github.com port 443
```

永久生效的配置方式（用你的实际代理端口替换 `7897`）：

1. **设置系统环境变量**（永久）：
   ```powershell
   [Environment]::SetEnvironmentVariable("HTTP_PROXY", "http://127.0.0.1:7897", "User")
   [Environment]::SetEnvironmentVariable("HTTPS_PROXY", "http://127.0.0.1:7897", "User")
   ```

2. **给 git 全局配置代理**：
   ```powershell
   git config --global http.proxy http://127.0.0.1:7897
   git config --global https.proxy http://127.0.0.1:7897
   ```

3. **重开一个全新的 PowerShell 窗口**（当前窗口读不到新环境变量），然后验证：
   ```powershell
   echo $env:HTTPS_PROXY          # 应显示代理地址
   git config --global --get http.proxy
   git ls-remote https://github.com/cz11001100/LeLabPro.git   # 能列出分支即成功
   ```

4. **重新安装**：
   ```powershell
   uv tool install "git+https://github.com/cz11001100/LeLabPro.git#subdirectory=lelab"
   ```

> 提示：代理端口取决于你的代理软件（Clash/Clash Verge 常见 `7890`，其他工具可能是不同端口），把命令里的 `7897` 换成你的实际端口。

**摄像头无法使用**
- LeLab 在 Windows 上使用 DirectShow 后端
- 确保无其他应用（如 Zoom、Teams）占用摄像头

**串口连接失败**
- 打开设备管理器，查看 COM 端口号
- 在 LeLab 前端手动选择正确的端口（默认 COM3，实际可能不同）

**训练时视频解码报错**
- LeLab 已内置自动修复：当 torchcodec 无法加载 FFmpeg DLL 时，自动回退到 PyAV
- 若仍有问题，可在训练时选择 `pyav` 作为视频后端

**训练完成时报错 `WinError 1314 客户端没有所需的特权`（符号链接创建失败）**

LeRobot 保存 checkpoint 后，会创建一个符号链接 `checkpoints/last -> 训练输出目录`。但 Windows 默认不允许普通用户创建符号链接，导致训练完成后报 `OSError: [WinError 1314]`。解决方案：

1. **方案一：开启 Windows 开发者模式**（推荐，一劳永逸，所有需要符号链接的操作都能正常）：
   - 图形界面：`设置 → 隐私和安全性 → 开发者选项 → 开发者模式 → 开启`
   - 或命令行一键开启（需管理员 PowerShell）：
     ```powershell
     reg add "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock" /t REG_DWORD /f /v "AllowDevelopmentWithoutDevLicense" /d "1"
     # 重启电脑生效
     ```

2. **方案二：以管理员身份运行**（已验证可成功）：
   - 右键 PowerShell → "以管理员身份运行" → 启动 `lelab`
   - 管理员权限可以创建符号链接

> 说明：训练本身已成功完成并保存了 checkpoint，只是最后的 `last` 符号链接创建失败。开启开发者模式或管理员运行后重训即可正常。

---

## macOS 安装

### pip 方式

```bash
# 克隆仓库
git clone https://github.com/cz11001100/LeLabPro.git
cd LeLabPro/lelab

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装
pip install -e .

# 启动
lelab
```

### macOS 常见问题

**摄像头权限**
- 首次运行系统会请求摄像头权限，允许 Terminal 访问

**串口权限**
- 若遇 `Permission denied`：
  ```bash
  sudo chmod 666 /dev/tty.usbmodem*
  ```

**Apple Silicon (M1/M2/M3)**
- 使用 ARM64 版 Python，PyTorch 已原生支持

---

## Ubuntu 安装

### pip 方式

```bash
# 克隆仓库
git clone https://github.com/cz11001100/LeLabPro.git
cd LeLabPro/lelab

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装
pip install -e .

# 启动
lelab
```

### Ubuntu 常见问题

**串口权限**
```bash
sudo usermod -a -G dialout $USER
# 注销并重新登录生效
```

**摄像头权限**
```bash
sudo usermod -a -G video $USER
```

**缺少 FFmpeg**
```bash
sudo apt update
sudo apt install ffmpeg
```

**NVIDIA GPU 支持**
- 安装 NVIDIA 驱动和 CUDA，验证：`nvidia-smi`
- 如需 CUDA 版 PyTorch，参考 [PyTorch 官网](https://pytorch.org/get-started/locally/)

---

## 开发模式

```bash
git clone https://github.com/cz11001100/LeLabPro.git
cd LeLabPro/lelab

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装开发依赖
pip install -e ".[dev,test]"

# 启动开发服务器（热重载）
lelab --dev
```

开发模式下：
- Vite 前端运行于 `:8080`（热重载）
- uvicorn 后端运行于 `:8000`（热重载）
- 浏览器自动打开 `http://localhost:8080`

---

## 系统架构

```
lelab/
├── server.py          # FastAPI 主服务器
├── train.py           # 训练模块（含视频后端自动检测）
├── teleoperate.py     # 遥操作模块
├── record.py          # 数据采集模块
├── calibrate.py       # 标定模块
├── rollout.py         # 推理模块
├── jobs.py            # 任务管理
├── datasets.py        # 数据集操作
└── utils/
    └── config.py      # 配置管理
```

## 相关资源

- **[LeRobot](https://github.com/huggingface/lerobot):** 底层机器人框架
- **[LeLab HF Space](https://huggingface.co/spaces/lerobot/LeLab):** 官方在线体验
- **[Discord](https://discord.gg/q8Dzzpym3f):** LeRobot 社区

## 许可证

Apache License 2.0
