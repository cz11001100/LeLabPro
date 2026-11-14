# Copyright 2026 The LeForge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Post-conversion quality checks.

The point is not to prove the conversion ran, but to catch the failures that
make a dataset *look* fine and train badly: wrong episode lengths, undecodable
frames, misaligned action/state rows, NaN statistics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import dataset_dir

MAPPING_REMINDERS = [
    "动作语义：绝对关节位置 / 增量动作 / 笛卡尔空间，转换器不会自动改写",
    "单位与坐标系：弧度与角度、米与毫米、四元数顺序（wxyz/xyzw）、相机外参",
    "归一化统计：LeRobot 会基于新数据重算 mean/std，旧数据集的统计值不能复用",
]


def lerobot_qc(repo_id: str) -> dict[str, Any]:
    """Summarize a freshly written LeRobot dataset with the existing analyzer."""
    from ..datasets import analyze_dataset

    root = dataset_dir(repo_id)
    checks: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    ok = True

    def check(name: str, passed: bool, detail: str) -> None:
        nonlocal ok
        ok = ok and passed
        checks.append({"name": name, "ok": passed, "detail": detail})

    info_path = root / "meta" / "info.json"
    check("meta/info.json", info_path.is_file(), str(info_path))
    if not info_path.is_file():
        return {"ok": False, "checks": checks, "stats": stats, "reminders": MAPPING_REMINDERS}

    info = json.loads(info_path.read_text())
    cameras = [k for k, ft in (info.get("features") or {}).items() if ft.get("dtype") == "video"]
    stats.update(
        episodes=int(info.get("total_episodes", 0)),
        frames=int(info.get("total_frames", 0)),
        fps=info.get("fps"),
        cameras=cameras,
        robot_type=info.get("robot_type"),
    )
    check("episode 数", stats["episodes"] > 0, f"{stats['episodes']} 集")
    check("帧数", stats["frames"] > 0, f"{stats['frames']} 帧")
    check("相机", True, ", ".join(cameras) if cameras else "无相机（仅状态/动作）")

    analysis = analyze_dataset(repo_id)
    if analysis.get("error"):
        check("完整性分析", False, str(analysis["error"]))
        return {"ok": False, "checks": checks, "stats": stats, "reminders": MAPPING_REMINDERS}

    anomalies = analysis.get("anomalies") or []
    high = [a for a in anomalies if a.get("severity") == "high"]
    check("完整性分析", not high, f"{len(high)} 个高优先级问题" if high else "无高优先级异常")
    if high:
        ok = False
        checks.extend(
            {
                "name": f"异常：{a.get('type')}",
                "ok": False,
                "detail": str(a.get("message", ""))[:300],
            }
            for a in high[:10]
        )
    stats.update(
        episode_length=(analysis.get("statistics") or {}).get("episode_length"),
        total_frames_analyzed=(analysis.get("statistics") or {}).get("total_frames"),
        videos_checked=(analysis.get("statistics") or {}).get("videos_checked"),
    )
    return {
        "ok": ok,
        "checks": checks,
        "stats": stats,
        "anomalies": anomalies[:20],
        "reminders": MAPPING_REMINDERS,
    }


def verify_export(preset: str, path: str | Path) -> dict[str, Any]:
    if preset.startswith("aloha") or preset.startswith("robomimic"):
        from . import hdf5_io

        report = hdf5_io.verify(path, preset)
    elif preset.startswith("zarr"):
        from . import zarr_io

        report = zarr_io.verify(path)
    else:
        raise ValueError(f"未知导出格式 {preset!r}")
    report["reminders"] = MAPPING_REMINDERS
    return report
