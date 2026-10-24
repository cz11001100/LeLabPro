# LeLab 安装指南

LeLab 是 LeRobot 的图形化界面，用于 SO-101 机械臂的标定、遥操作、数据采集、训练和推理。

## 前置要求

- **Python 3.12+**
- **Git**
- **SO-101 机械臂的 USB 串口驱动**（Windows 需要安装 CH340/CP2102 驱动）

---

## Windows 安装

### 方式一：使用 uv（推荐）

```powershell
# 安装 uv（如果还没有）
pip install uv

# 安装 LeLab
uv tool install --python 3.12 --default-index https://pypi.tuna.tsinghua.edu.cn/simple "git+https://github.com/cz11001100/LeLabPro.git#subdirectory=lelab"

# 启动
lelab
```

### 方式二：使用 pip

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

**摄像头无法使用**
- LeLab 在 Windows 上使用 DirectShow 后端
- 确保没有其他应用（如 Zoom、Teams）占用摄像头
- 如果摄像头列表为空，尝试重启 LeLab

**串口连接失败**
- 打开设备管理器，查看 COM 端口号
- 在 LeLab 前端界面手动选择正确的端口
- 默认端口为 COM3，实际可能不同

**训练时 torchcodec 报错**
- LeLab 已内置自动修复：当 torchcodec 无法加载 FFmpeg DLL 时，会自动回退到 PyAV
- 如果仍有问题，可在训练时手动选择 `pyav` 作为视频后端

---

## macOS 安装

### 方式一：使用 uv（推荐）

```bash
# 安装 uv（如果还没有）
pip install uv

# 安装 LeLab
uv tool install --python 3.12 --default-index https://pypi.tuna.tsinghua.edu.cn/simple "git+https://github.com/cz11001100/LeLabPro.git#subdirectory=lelab"

# 启动
lelab
```

### 方式二：使用 pip

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
- 首次运行时，系统会弹出摄像头权限请求
- 允许 Terminal 访问摄像头

**串口权限**
- macOS 可能需要额外权限访问串口设备
- 如果遇到 `Permission denied` 错误：
  ```bash
  sudo chmod 666 /dev/tty.usbmodem*
  ```

**Apple Silicon (M1/M2/M3)**
- 确保使用 ARM64 版本的 Python
- PyTorch 已原生支持 Apple Silicon

---

## Ubuntu 安装

### 方式一：使用 uv（推荐）

```bash
# 安装 uv（如果还没有）
pip install uv

# 安装 LeLab
uv tool install --python 3.12 --default-index https://pypi.tuna.tsinghua.edu.cn/simple "git+https://github.com/cz11001100/LeLabPro.git#subdirectory=lelab"

# 启动
lelab
```

### 方式二：使用 pip

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
- 将用户添加到 `dialout` 组：
  ```bash
  sudo usermod -a -G dialout $USER
  ```
- 注销并重新登录生效

**摄像头权限**
- 如果使用 USB 摄像头，确保设备已被识别：
  ```bash
  ls /dev/video*
  ```
- 如果需要 sudo 权限，可以创建 udev 规则：
  ```bash
  sudo usermod -a -G video $USER
  ```

**缺少 FFmpeg**
- 如果遇到视频解码问题，安装 FFmpeg：
  ```bash
  sudo apt update
  sudo apt install ffmpeg
  ```

**NVIDIA GPU 支持**
- 确保安装了 NVIDIA 驱动和 CUDA：
  ```bash
  nvidia-smi
  ```
- 如果需要 CUDA 版本的 PyTorch，参考 [PyTorch 官网](https://pytorch.org/get-started/locally/)

---

## 开发模式

如果你想参与开发：

```bash
# 克隆仓库
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
- Vite 前端运行在 `:8080`（支持热重载）
- uvicorn 后端运行在 `:8000`（支持热重载）
- 浏览器会自动打开 `http://localhost:8080`

---

## 系统架构

```
lelab/
├── server.py          # FastAPI 主服务器
├── train.py           # 训练模块（已修复视频后端自动检测）
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

- [LeRobot 官方仓库](https://github.com/huggingface/lerobot)
- [LeLab HF Space](https://huggingface.co/spaces/lerobot/LeLab)
- [Discord 社区](https://discord.gg/q8Dzzpym3f)

## 许可证

Apache License 2.0
