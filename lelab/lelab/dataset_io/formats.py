# Copyright 2026 The LeLab Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Format registry, detection and inspection for dataset import/export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

FORMATS: list[dict[str, Any]] = [
    {
        "key": "aloha_hdf5",
        "label": "ALOHA / ACT HDF5",
        "kind": "hdf5",
        "directions": ["import", "export"],
        "description": (
            "ALOHA / ACT 常用的 HDF5 布局：每个 episode 一个 .hdf5 文件，"
            "包含 observations/qpos、action 和 observations/images/<相机>（JPEG 字节）。"
        ),
        "source_hint": "指向包含 episode_*.hdf5 的目录，或单个 ALOHA HDF5 文件",
        "output_hint": "输出目录，每个 episode 写一个 episode_XXXXXX.hdf5",
        "defaults": {"fps": 30, "task": ""},
    },
    {
        "key": "robomimic_hdf5",
        "label": "robomimic HDF5",
        "kind": "hdf5",
        "directions": ["import", "export"],
        "description": (
            "robomimic 风格的单个 HDF5：data/demo_*/obs 下保存图像与低维观测，"
            "actions 保存动作；导入时默认把低维 obs 拼成 observation.state。"
        ),
        "source_hint": "指向单个 .hdf5 文件（含 data/demo_* 分组）",
        "output_hint": "输出 .hdf5 文件路径",
        "defaults": {"fps": 20, "task": ""},
    },
    {
        "key": "zarr_diffusion_policy",
        "label": "Zarr（Diffusion Policy）",
        "kind": "zarr",
        "directions": ["import", "export"],
        "description": (
            "Diffusion Policy / PushT 常用的 Zarr 布局：data/action、data/state、"
            "data/<相机> 为整段数组，meta/episode_ends 给出每集累计帧边界。"
        ),
        "source_hint": "指向 .zarr 目录（含 data/action 与 meta/episode_ends）",
        "output_hint": "输出 .zarr 目录",
        "defaults": {"fps": 10, "task": ""},
    },
]

_FORMAT_BY_KEY = {fmt["key"]: fmt for fmt in FORMATS}


def format_by_key(key: str) -> dict[str, Any]:
    try:
        return _FORMAT_BY_KEY[key]
    except KeyError as exc:
        raise ValueError(f"未知格式 {key!r}，可选：{', '.join(_FORMAT_BY_KEY)}") from exc


def detect_format(path: str | Path) -> str | None:
    """Best-effort format detection for a path, without raising on unknown data."""
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"路径不存在：{p}")

    from . import hdf5_io, zarr_io

    if zarr_io.looks_like_zarr(p):
        return "zarr_diffusion_policy"
    if p.is_dir():
        if hdf5_io.looks_like_aloha_dir(p):
            return "aloha_hdf5"
        if any(hdf5_io.looks_like_robomimic_file(f) for f in hdf5_io.list_hdf5_files(p)[:1]):
            return "robomimic_hdf5"
        return None

    if hdf5_io.looks_like_robomimic_file(p):
        return "robomimic_hdf5"
    if hdf5_io.looks_like_aloha_file(p):
        return "aloha_hdf5"
    return None


def inspect_source(path: str | Path, preset: str | None = None) -> dict[str, Any]:
    """Detect (when needed) and summarize a source dataset.

    The result is what the UI shows before the user commits to a conversion:
    episode count, camera keys, state/action dimensions, fps and warnings.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"路径不存在：{p}")

    detected = preset or detect_format(p)
    if not detected:
        raise ValueError(
            "无法自动识别格式。请确认路径指向 ALOHA HDF5 目录、robomimic HDF5 文件"
            " 或 Diffusion Policy Zarr 目录，并在界面上手动选择格式。"
        )

    fmt = format_by_key(detected)
    if fmt["kind"] == "hdf5":
        from . import hdf5_io

        summary = hdf5_io.inspect(p, detected)
    elif fmt["kind"] == "zarr":
        from . import zarr_io

        summary = zarr_io.inspect(p)
    else:  # pragma: no cover - registry above is exhaustive
        raise ValueError(f"格式 {detected} 没有对应的读取器")

    return {
        "path": str(p),
        "preset": detected,
        "label": fmt["label"],
        "kind": fmt["kind"],
        "detected": preset is None,
        "summary": summary,
    }
