# LeLab 常见问题排查

## 启动类

### `lelab: command not found`

**原因：conda 环境覆盖了 PATH。** venv 激活后再执行 `conda activate xxx`，conda 会重置 PATH，把 venv 的 `bin/` 挤掉。

```bash
conda deactivate          # 退出所有 conda 环境
source lerobot/bin/activate
lelab
```

或直接用最省事的方式（脚本内部已处理 conda）：

```bash
./lelab-run.sh
```

### `Address already in use` / 端口 8000 被占用

之前有 LeLab 进程没退干净。

```bash
./lelab-run.sh --stop     # 杀掉占用 8000/8080 的残留进程
./lelab-run.sh
```

手动查占用：

```bash
ss -ltnp | grep 8000
```

### `ModuleNotFoundError: No module named 'lerobot.configs.dataset'`

**原因：lelab 里 import 路径写错。** 旧版指向 `lerobot.configs.dataset`，但该类实际在 `lerobot.scripts.lerobot_record`。

已在当前版本修复。若你手上是旧包，手动改：

```bash
sed -i 's/from lerobot.configs.dataset import DatasetRecordConfig/from lerobot.scripts.lerobot_record import DatasetRecordConfig/' lelab/lelab/record.py
```

### `ImportError: cannot import name 'LeRobotDataset' from 'lerobot.datasets'`

同样是 import 路径问题，实际在 `lerobot.datasets.lerobot_dataset`：

```bash
sed -i 's/from lerobot\.datasets import LeRobotDataset/from lerobot.datasets.lerobot_dataset import LeRobotDataset/g' lelab/lelab/record.py
sed -i 's/from lerobot\.datasets import CODEBASE_VERSION/from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION/' lelab/lelab/record.py
```

### `ModuleNotFoundError: No module named 'paramiko'`

**原因：缺少 SSH 依赖，训练板块的远程安装/训练无法使用。**

```bash
uv pip install --python lerobot/bin/python "paramiko>=3.4.0"
```

### `Could not import realsense: No module named 'pyrealsense2'`

**这不是错误，是 INFO 级日志。** 代码用 `try/except` 捕获后打印，缺了不影响功能。

只有 **Intel RealSense 深度相机**（D435/D455）才需要 pyrealsense2。普通 USB 摄像头走 OpenCV 后端，完全不需要。

如果用 RealSense 相机：

```bash
uv pip install --python lerobot/bin/python "pyrealsense2>=2.55.1.6486,<2.57.0"
```

---

## 硬件类

### 连机械臂报 `Permission denied: '/dev/ttyACM0'`

**原因：当前用户不在 `dialout` 组，没有串口读写权限。**

```bash
sudo usermod -aG dialout $USER
```

**必须注销重新登录**才生效（不是重启终端，是注销整个会话）。

验证：

```bash
id -nG | grep dialout
```

### 找不到串口 / 机械臂不响应

```bash
ls /dev/ttyUSB* /dev/ttyACM*     # 看有没有设备
dmesg | tail -20                 # 看插入时的内核日志
```

- 检查 USB 线是否是数据线（有些是纯充电线）
- 换一个主板直插的 USB 口，避开 hub
- Windows 需装 CH340/CP2102 驱动

### 摄像头打不开 / 列表为空

```bash
ls /dev/video*                   # 确认设备存在
cat /sys/class/video4linux/video0/name
```

- 检查摄像头是否被其他程序占用（浏览器、OBS、Zoom）
- 确认用户在 `video` 组：`sudo usermod -aG video $USER`
- 部分摄像头只支持 YUYV 格式，需在界面里选对应格式

---

## 训练类

### SSH 远程安装环境时中途断连

**典型现象：Test 连接通过，但装环境到第 2 步（安装 pip）就断。**

**原因：SSH 连接没设 keepalive 心跳。** `apt-get install` 跑几分钟期间没有数据往来，防火墙/NAT 会静默断开空闲连接。

已在当前版本修复（加了 `transport.set_keepalive(30)`）。若手上是旧包：

```bash
# 在 ssh_remote_env.py 的 client.connect(**kw) 后面插入三行
python3 - <<'PY'
import re, pathlib
p = pathlib.Path("lelab/lelab/ssh_remote_env.py")
s = p.read_text()
if "set_keepalive" not in s:
    s = s.replace(
        "            client.connect(**kw)\n            return client",
        "            client.connect(**kw)\n"
        "            transport = client.get_transport()\n"
        "            if transport is not None:\n"
        "                transport.set_keepalive(30)\n"
        "            return client",
    )
    p.write_text(s)
    print("已修复")
else:
    print("已有 keepalive，无需修改")
PY
```

另外远程服务器端也可放宽超时，编辑 `/etc/ssh/sshd_config`：

```
ClientAliveInterval 60
ClientAliveCountMax 10
```

然后 `sudo systemctl restart sshd`。

### SSH Test 连接不通过

逐项检查：

1. **终端里能否手动 SSH 上去？**
   ```bash
   ssh -p 端口 用户名@主机
   ```
   如果手动都不行，LeLab 也不行 —— 是网络/密钥/防火墙问题，不是 LeLab 的问题。

2. **用密码还是密钥？** LeLab 两种都支持，但密码登录要在服务器端确认 `PasswordAuthentication yes`。

3. **密钥路径**必须是服务器本机上的绝对路径（LeLab 跑在哪台机器，密钥就要在那台机器上）。

4. **主机名不要带 `~`**，`~/.ssh/id_rsa` 这种路径脚本会展开，但界面里填的时候注意别写错。

### `nvidia-smi` 有但训练报 CUDA 不可用

```bash
source lerobot/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

如果 `cuda_available` 是 False，说明装的是 CPU 版 torch。重装对应 CUDA 版本：

```bash
uv pip install --python lerobot/bin/python --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

（`cu121` 换成你实际的 CUDA 版本，用 `nvidia-smi` 右上角看）

### 训练卡在下载模型 / 网络很慢

模型权重从 HuggingFace 下载，国内可换镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
./lelab-run.sh
```

---

## 环境类

### 想彻底重装

```bash
rm -rf lerobot            # 删掉整个虚拟环境
bash install.sh           # 重新安装，选 y 确认
```

### 只想补装某个包

```bash
uv pip install --python lerobot/bin/python 包名
```

注意：用 `--python` 指定绝对路径，不依赖当前 shell 激活了哪个环境。

### 卡在 `Downloading cpython-3.12...`

不是卡死，多半是在跨境拉解释器。uv 会从 GitHub 下载约 32MB 的 CPython 并解压，国内网络下这一步可能耗十几分钟甚至重试。想确认它还在动：

```bash
du -sh ~/.local/share/uv/python       # 隔 20 秒看两次，数字在涨就正常
```

嫌慢的话，任选一种：

```bash
# 1) 用系统自带的 3.12，uv 就不会下载
sudo apt install python3.12 python3.12-venv

# 2) 给解释器下载也走国内代理
export UV_PYTHON_INSTALL_MIRROR=https://gh-proxy.com/https://github.com/astral-sh/python-build-standalone/releases/download
```

### uv 装不上 / 下载 Python 失败

脚本要 uv 自动下载独立 CPython，国内网络可能超时。先装系统 Python：

```bash
sudo apt install python3.12 python3.12-venv
```

再重跑 `bash install.sh`，uv 会优先用系统已有的 3.12。

---

##  still 不行？

把这几项信息发给开发者：

```bash
# 1. 自检结果
source lerobot/bin/activate
python -c "import lelab.server"

# 2. 环境与硬件
uname -sm; python --version; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
id -nG; ls /dev/ttyUSB* /dev/ttyACM* /dev/video* 2>&1

# 3. 完整报错（不要只截最后一行）
```

---

## torchcodec 找不到 FFmpeg 共享库

**现象**：从 git 装好后报

```
Could not load libtorchcodec. Likely causes:
  1. Couldn't load FFmpeg shared libraries ...
FFmpeg version 7: libavcodec.so.61: cannot open shared object file
FFmpeg version 6: libavutil.so.58: ...
```

**原因**：torchcodec 按 7→6→5→4 顺序加载 `libtorchcodec_coreN.so`，每个都要求对应 FFmpeg 大版本的**普通 soname**（如 `libavcodec.so.61`）。新机器上没有系统 FFmpeg 就全失败。

**已自动处理**：`install.sh` 会调用 `python -m lelab.utils.ffmpeg_shim --ensure`，把 **PyAV 自带的 FFmpeg 7**（`site-packages/av.libs/`）接上。做法是建一个 shim 目录，同时链两种名字 —— 普通 soname 给 torchcodec 用，hash 名（`libdrm-b0291a67.so.2.4.0`）给这批库互相引用用。`lelab-run.sh` 与 `lelab` 命令会在启动前把它注入 `LD_LIBRARY_PATH`（必须启动前设，运行时改无效）。

**手动处理**（三选一）：

```bash
sudo apt install ffmpeg                     # 有 root 时最干净
conda install -y -c conda-forge ffmpeg      # 无 root；需把 $CONDA_PREFIX/lib 加进 LD_LIBRARY_PATH
python -m lelab.utils.ffmpeg_shim --ensure  # 无 root 无 conda，用 PyAV 自带的库
```

**不修也能用**：`lelab.utils.video_backend` 会探测到 torchcodec 不可用并自动退回 **pyav** 解码，训练/回放照常，只是慢一些。

**验证修好了**：

```bash
python -c "import json;from torchcodec._core.ops import _get_json_ffmpeg_library_versions as j;print(json.loads(j())['ffmpeg_version'])"
```

