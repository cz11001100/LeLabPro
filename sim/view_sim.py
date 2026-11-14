#!/usr/bin/env python
"""在窗口里查看 SO-101 桌面场景：鼠标拖动旋转视角、滚轮缩放、Esc 退出。

需要有图形界面（本机 DISPLAY=:1）：
    DISPLAY=:1 .venv/bin/python view_sim.py
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# 有窗口的机器用 glfw；要在纯无头机器上跑 demo 用 run_demo.py（EGL）
os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco
import mujoco.viewer
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE / "so101-tabletop-mujoco"
sys.path.insert(0, str(REPO))

from env import HOME, SCENE_FILE, SO101Env  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=None, help="指定 MJCF；默认用仓库里的 so101_tabletop.xml")
    parser.add_argument("--seed", type=int, default=None, help="不传则每次随机；传整数可复现")
    args = parser.parse_args()

    scene = (args.scene or (REPO / SCENE_FILE)).resolve()
    print("场景文件:", scene)
    print("修改时间:", datetime.fromtimestamp(scene.stat().st_mtime).isoformat(timespec="seconds"))
    print("当前目录:", Path.cwd())

    seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy % (2**31))
    print("seed:", seed)

    env = SO101Env(scene=scene)
    obs, info = env.reset(seed=seed)
    if "ball_positions_cm" in info:
        print("球位 (cm):", info["ball_positions_cm"])
    world = env._ensure_world()

    # 打印关键指纹，确认不是旧场景或 Hub 缓存
    for name in ("red_ball", "blue_ball", "plate", "red_cube", "green_cube"):
        bid = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, name)
        print(f"body {name}: {'有' if bid >= 0 else '无'}")
    for name in ("top", "front", "wrist_cam"):
        cid = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cid >= 0:
            print(f"camera {name}: pos={np.round(world.model.cam_pos[cid], 4).tolist()} fovy={world.model.cam_fovy[cid]:.1f}")

    # 让机械臂保持在初始位姿，方便看清场景
    world.data.ctrl[world.actuators] = world.to_ctrl(np.array(HOME, dtype=np.float32))
    mujoco.mj_forward(world.model, world.data)

    print("任务:", env.task_description)
    print("拖动鼠标转视角，滚轮缩放，Esc 退出")
    try:
        mujoco.viewer.launch(world.model, world.data)
    finally:
        env.close()


if __name__ == "__main__":
    main()
