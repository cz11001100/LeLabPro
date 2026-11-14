#!/usr/bin/env python
"""SO-101 桌面抓取（MuJoCo）本地演示：动一动机械臂，录像并保存首帧截图。

用法:
    .venv/bin/python run_demo.py                  # 挥臂演示，默认 300 步（10 秒，30fps）
    .venv/bin/python run_demo.py --steps 900      # 30 秒
    .venv/bin/python run_demo.py --motion home    # 保持初始位姿不动
    .venv/bin/python run_demo.py --no-video       # 只跑不录
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# 必须在 import mujoco 之前设置：无显示器时用 EGL 渲染
os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE / "so101-tabletop-mujoco"
sys.path.insert(0, str(REPO))

from env import HOME, SCENE_FILE, SO101Env  # noqa: E402


def wave_action(step: int, fps: int, rng: np.random.Generator) -> np.ndarray:
    """围绕初始位姿做正弦摆动，模拟演示动作（不是抓取策略）。"""
    t = step / fps
    action = np.array(HOME, dtype=np.float32)
    action[0] += 12 * np.sin(2 * np.pi * 0.20 * t)
    action[1] += 10 * np.sin(2 * np.pi * 0.13 * t + 1.0)
    action[2] += 8 * np.sin(2 * np.pi * 0.17 * t + 2.0)
    action[3] += 6 * np.sin(2 * np.pi * 0.23 * t + 0.5)
    action[4] += 10 * np.sin(2 * np.pi * 0.11 * t)
    action[5] = 50 + 30 * np.sin(2 * np.pi * 0.09 * t)  # 夹爪 20~80
    action += rng.normal(0, 0.3, size=6).astype(np.float32)
    return action


def main() -> None:
    parser = argparse.ArgumentParser(description="SO-101 tabletop MuJoCo demo")
    parser.add_argument("--steps", type=int, default=300, help="运行步数（30 步 = 1 秒）")
    parser.add_argument("--motion", choices=["wave", "home"], default="wave")
    parser.add_argument("--seed", type=int, default=None, help="不传则每次随机；传整数可复现")
    parser.add_argument("--out", type=Path, default=HERE / "out" / "so101_demo.mp4")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--scene", type=Path, default=None, help="指定 MJCF；默认用仓库里的 so101_tabletop.xml")
    args = parser.parse_args()

    scene = (args.scene or (REPO / SCENE_FILE)).resolve()
    print("场景文件:", scene)
    print("修改时间:", datetime.fromtimestamp(scene.stat().st_mtime).isoformat(timespec="seconds"))

    seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy % (2**31))
    print("seed:", seed)
    rng = np.random.default_rng(seed)
    env = SO101Env(scene=scene)
    obs, info = env.reset(seed=seed)
    if "ball_positions_cm" in info:
        print("球位 (cm):", info["ball_positions_cm"])

    print(f"任务: {env.task_description}")
    print(
        f"观测: agent_pos {obs['agent_pos'].shape}, "
        f"相机 {list(obs['pixels'])} {obs['pixels']['up'].shape} (高x宽x3)"
    )
    print(f"动作: 6 维关节目标（角度 + 夹爪 0-100），控制频率 {env.fps} fps")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    if not args.no_video:
        writer = imageio.get_writer(
            args.out, fps=env.fps, codec="libx264", quality=8, macro_block_size=None
        )

    total_reward = 0.0
    best_lift = 0.0
    step = 0
    try:
        for step in range(args.steps):
            if args.motion == "home":
                action = np.array(HOME, dtype=np.float32)
            else:
                action = wave_action(step, env.fps, rng)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            best_lift = max(best_lift, float(info.get("lift", 0.0)))

            if step == 0:
                imageio.imwrite(args.out.with_suffix(".png"), obs["pixels"]["up"])
            if writer is not None:
                # 左：up 俯视，中：side 侧视，右：wrist 腕部
                views = [obs["pixels"][key] for key in ("up", "side", "wrist")]
                writer.append_data(np.hstack(views))

            if terminated or truncated:
                print(f"第 {step + 1} 步结束 (terminated={terminated}, truncated={truncated})")
                break
    finally:
        if writer is not None:
            writer.close()
        env.close()

    print(f"完成 {step + 1} 步 | 累计奖励 {total_reward:.2f} | 最大抬升 {best_lift * 100:.1f}cm")
    if writer is not None:
        print(f"视频: {args.out}")
        print(f"首帧截图: {args.out.with_suffix('.png')}")
    else:
        print("未生成新视频（--no-video）；/out 里的旧文件不会更新")


if __name__ == "__main__":
    main()
