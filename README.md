# LeLabPro

SO-100 / SO-101 机械臂的 LeRobot 一体化 Web 工作台 —— 标定、遥操作、数据采集、训练、推理、评估。

本仓库是**单仓库（monorepo）**，同时包含打过补丁的两个上游项目：

| 目录 | 说明 |
|---|---|
| [`lelab/`](lelab/README.md) | FastAPI + React 的图形界面，把 LeRobot 的 CLI 流程搬进浏览器 |
| [`lerobot/`](lerobot/README.md) | 针对 SO-100 实机修改过的 LeRobot 框架（电机、摄像头、录制、SmolVLA 等） |

## 快速开始

```bash
# 方式一：一条命令装（会自动从本仓库的 lerobot/ 子目录拉取补丁版框架）
uv tool install "git+https://github.com/cz11001100/LeLabPro.git#subdirectory=lelab"
lelab
```

```bash
# 方式二：克隆后本地装
git clone https://github.com/cz11001100/LeLabPro.git
cd LeLabPro/lelab
pip install -e .
lelab
```

启动后浏览器打开 **http://localhost:8000**。

环境要求：**Python 3.12+**、Git；训练需要 NVIDIA GPU；机械臂需 USB 串口驱动（Windows 装 CH340/CP2102）。

## 相比上游多了什么

- **SSH 远程训练 / 导出 / 续训** — 本机采集，云端 4090 训练，一键回捞 checkpoint
- **数据集质检与修复** — 视频帧完整性、重复 episode、表格越界检查，中断数据集重建
- **录制续录** — 采集中断后接着录，不用整批重录
- **推理期 chunk 平滑** — 可选 EMA 包装，缓解逐帧抖动
- **模型评估面板** — 开环误差、感知探针、显著性图、ROI 定位、指向反演五件套
- **SO-100 实机调参** — 电机总线、串口吞吐、摄像头后端的本地修复

## 文档

- 分平台安装：[lelab/INSTALL.md](lelab/INSTALL.md)
- 报错排查（conda 冲突、SSH 断连、串口权限、import 错误等）：[TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- 开发架构说明：[lelab/CLAUDE.md](lelab/CLAUDE.md)

## 许可与出处

本仓库以 **Apache License 2.0** 发布，基于以下上游项目修改而来：

- [LeLab](https://github.com/OneRobotAI/lelab) — 原始 Web 界面（Apache 2.0）
- [LeRobot](https://github.com/huggingface/lerobot) — 底层机器人框架（Apache 2.0，Copyright 2024 The Hugging Face Inc. team）

上游的 `LICENSE` 与源文件版权声明头均已在各自目录保留。LeLabPro 是个人修改分支，与上述官方项目无隶属或背书关系。
