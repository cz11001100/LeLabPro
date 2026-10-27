# LeLabPro

SO-100 / SO-101 机械臂的 LeRobot 一体化 Web 工作台 —— 标定、遥操作、数据采集、训练、评估、部署全流程在浏览器里完成。单仓库同时包含界面（`lelab/`）与打过补丁的框架（`lerobot/`）。

**安装**

```bash
curl -fsSL https://raw.githubusercontent.com/cz11001100/LeLabPro/main/install.sh | bash
```

**升级**

```bash
cd LeLabPro && git pull && bash install.sh
```

**启动**

```bash
./LeLabPro/lelab-run.sh
```

浏览器打开 http://localhost:8000。

## 更新日志

最近更新，完整历史见 [CHANGELOG.md](CHANGELOG.md)。

**2026-10-27**
- 新增 部署与推理页「结束本回合」：收尾当前试次并自动进入下一回合
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
