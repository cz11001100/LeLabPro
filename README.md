# LeLabPro

SO-100 / SO-101 机械臂的 LeRobot 一体化 Web 工作台 —— 标定、遥操作、数据采集、训练、推理、评估。

本仓库是**单仓库（monorepo）**，同时包含打过补丁的两个上游项目：

| 目录 | 说明 |
|---|---|
| [`lelab/`](lelab/README.md) | FastAPI + React 的图形界面，把 LeRobot 的 CLI 流程搬进浏览器 |
| [`lerobot/`](lerobot/README.md) | 针对 SO-100 实机修改过的 LeRobot 框架（电机、摄像头、录制、SmolVLA 等） |

## 快速开始

**一条命令装好整个环境**（自动处理 conda 冲突、装 uv、建 Python 3.12 虚拟环境、走国内镜像装依赖、最后自检）：

```bash
curl -fsSL https://raw.githubusercontent.com/cz11001100/LeLabPro/main/install.sh | bash
```

装完启动：

```bash
./LeLabPro/lelab-run.sh
```

浏览器打开 **http://localhost:8000**。

脚本会克隆仓库到当前目录下的 `LeLabPro/`，虚拟环境在 `LeLabPro/lerobot-env/`。已经克隆过的话重复执行会复用并 `git pull` 更新。

### 备选：只装成命令行工具

不想用脚本、只要一个 `lelab` 命令时：

```bash
uv tool install --python 3.12 \
  --default-index https://pypi.tuna.tsinghua.edu.cn/simple \
  "git+https://github.com/cz11001100/LeLabPro.git#subdirectory=lelab"
lelab
```

> **`--python 3.12` 不能省。** 不写的话 uv 会挑系统里默认的 Python，若那是 3.14 就会撞上 `torch 2.7.1 无 cp314 轮子` 而解析失败。`--default-index` 是给国内网络的，海外网络可去掉。

### 环境要求

- **Python 3.12**（安装脚本会自动准备，无需手动装）
- **Git**、可访问网络
- 训练需要 NVIDIA GPU；纯采集/推理不需要
- 机械臂需 USB 串口驱动（Windows 装 CH340/CP2102），Linux 上当前用户要在 `dialout` 组

## 文档

- 分平台安装：[lelab/INSTALL.md](lelab/INSTALL.md)
- 报错排查（conda 冲突、SSH 断连、串口权限、import 错误等）：[TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- 开发架构说明：[lelab/CLAUDE.md](lelab/CLAUDE.md)

## 许可与出处

本仓库以 **Apache License 2.0** 发布，基于以下上游项目修改而来：

- [LeLab](https://github.com/OneRobotAI/lelab) — 原始 Web 界面（Apache 2.0）
- [LeRobot](https://github.com/huggingface/lerobot) — 底层机器人框架（Apache 2.0，Copyright 2024 The Hugging Face Inc. team）

上游的 `LICENSE` 与源文件版权声明头均已在各自目录保留。LeLabPro 是个人修改分支，与上述官方项目无隶属或背书关系。
