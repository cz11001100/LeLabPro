# LeForge

SO-100 / SO-101 机械臂的 LeRobot 一体化 Web 工作台 —— 标定、遥操作、数据采集、训练、评估、部署全流程在浏览器里完成。单仓库同时包含界面（`leforge/`）与打过补丁的框架（`lerobot/`）。

**安装**

```bash
curl -fsSL https://raw.githubusercontent.com/cz11001100/LeForge/main/install.sh | bash
```

**最小安装**

```bash
curl -fsSL https://raw.githubusercontent.com/cz11001100/LeForge/main/install.sh | bash -s -- --lite
```

**升级**

```bash
curl -fsSL https://raw.githubusercontent.com/cz11001100/LeForge/main/install.sh | bash        # 轻量档加 -s -- --lite
```

**启动**

```bash
leforge
```

浏览器打开 http://localhost:8000。

## 更新日志

最近更新，完整历史见 [CHANGELOG.md](CHANGELOG.md)。

**2026-11-14**
- 新增 仿真评测增强：本地场景新建 / MuJoCo 打开、环境仓库下拉、三视角录像与步级诊断、轨迹平滑与阴影柔化、时序集成按模型自适应
- 新增 仿真评测录像内存自适应：按步数×相机×分辨率估算，自动下调 batch 防撑爆内存
- 新增 相机手动曝光/增益：录制与推理按相机设定，画面一致
- 新增 键盘末端遥操：无主臂，键盘给末端增量经 IK 驱动 SO-101，遥操与采集可用
- 新增 数据集导入/导出：HDF5（ALOHA/ACT）与 Zarr ↔ LeRobotDataset 互转，界面选格式并看进度
- 新增 遥操模式切换：遥操页切 leader / keyboard_ee（无主臂），全局生效、刷新保留
- 变更 一键安装加 placo（键盘末端遥操的 IK 依赖）

**2026-11-12**
- 新增 数据集查看器：界面直接看各 episode 视频与动作/状态曲线，不用下到本地
- 新增 图像增广预览：训练配置里预览增广效果再决定是否启用
- 新增 仿真评测：模型卡内启动/停止 EnvHub 闭环评测，看成功率、日志与产物
- 变更 训练页可选策略随 0.7 扩充：Cosmos3、DM05、EO1/EVO1、FLUX.3、LingBot、LongWAM 等，按策略自动带出依赖
- 修复 模型卡上传后的状态：状态标签直接显示“已上传”（同“已导入”样式），不再冗长显示 repo、也不再同时出现两个状态

**2026-11-10**
- 新增 模型推送 Hub：checkpoint 推到 Hub 后云端按 repo id 走镜像拉取，不再每次开训前上传 1.2GB
- 新增 SARM 进度头训练：任务卡里选数据集与 SSH 主机，直接在 pod 上训
- 新增 远端任务发现：绑定 SSH 主机后列出该机已有任务并接回本地
- 新增 检查点上传进度：速率与剩余时间、状态落盘可接回、可中途取消
- 新增 模型导入反馈：进度条与已耗时、完成绿色确认、回车提交、导入中锁定弹窗
- 变更 模型卡评测按钮改为打开评测弹窗，启停在弹窗内、可切同 run 的 checkpoint
- 变更 lerobot 升级到 0.7.0：评测可加载官方 EnvHub 仿真环境跑闭环成功率，训练跟踪由 W&B 换成 Trackio
- 修复 24GB 卡上 4B 模型 batch 8 OOM（`PYTORCH_ALLOC_CONF=expandable_segments:True`）
- 修复 打分进度滞后：日志尾部被每个 batch 的告警占满，进度与 ETA 取不到真实行
- 修复 一键升级国内超时后悄悄用旧版、换目录重跑又克隆一份 —— 升级改为重跑安装命令，自动找回已有目录并走镜像拉取
- 修复 批量视频编码跨 meta 分片时 episode 索引与时间戳错位 —— 按真实分片与 `episode_index` 逐集更新，并校验索引连续性、分片归属和返回序号
- 修复 episode 元数据与数据行数不一致时漏检 —— 分析、健康检查与续录前逐集核对，修复后二次验证；纯数值数据集缺失 `info.json` 时可从 parquet 重建
- 修复 SSH 远端 Ubuntu 软件源替换命令中正则分组与 `sed` 分隔符冲突

**2026-10-28**
- 变更 首页「任务」与「模型」整合、评估改三维度评分、部署按 checkpoint 选择、dagger 计时与 RTC 延迟修正
- 新增 本地安装与运行接入 HuggingFace 镜像 —— 实测官方站与 hf-mirror 吞吐选端点，走镜像时自动关掉无视 `HF_ENDPOINT` 的 Xet 传输、放宽跨境读超时
- 修复 包索引选择改用实测下载吞吐（原先按索引页延迟选，方向经常是反的）
- 修复 没有独显的机器也在下 2.7GB CUDA 库导致安装普遍超时 —— 档位改为按 GPU 自动选择，无 GPU 自动走 CPU 版 torch
- 修复 uv 引导、Python 解释器下载、仓库克隆三处只认 GitHub/Cloudflare 直连 —— 各加一条镜像加速回退
- 新增 Windows 11 一键安装 `install.ps1`（PowerShell 一条命令，装完直接 `leforge`）
- 新增 轻量安装档 `bash install.sh --lite`：torch 装 CPU 版，环境从约 6.5GB 降到 2.2GB，适合只做采集、训练放云端的轻薄本
- 修复 排查表里的虚拟环境路径仍写着旧目录名（`lerobot/` → `lerobot-env/`），照抄会报找不到文件

**2026-10-27**
- 新增 部署与推理页「结束本回合」：收尾当前试次并自动进入下一回合
- 新增 装完即用的全局 `leforge` 命令（自动准备环境并启动）
- 新增 安装时自动挑快的包索引（开 VPN 时国内源变慢可用 `LEFORGE_INDEX=off` 切官方源）
- 修复 SARM / RA-BC 训练参数节点错误（`--policy.*` → `--reward_model.*`、`sample_weighting` 层级），训练可正常启动
- 修复 一键升级在 monorepo 布局与 Python 3.14 下失败
- 修复 SSH 远端装环境的权限探测、连接心跳与依赖下载超时
- 修复 torchcodec 找不到 FFmpeg 共享库（改用 PyAV 自带库，无需 root）
- 修复 推理连续重启偶发秒退（增加冷却与空日志诊断）

**2026-10-26**
- 界面焕新，全流程齐备

**2026-10-25**
- 新增 后训练与部署板块（摄像头预览、纠正采集）、SARM 与 RA-BC 权重、数据集合并

## 许可与出处

Apache-2.0，基于 [LeLab](https://github.com/OneRobotAI/lelab) 与 [LeRobot](https://github.com/huggingface/lerobot) 修改；上游 LICENSE 与版权声明均已保留。
