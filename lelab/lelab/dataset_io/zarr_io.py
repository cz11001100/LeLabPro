# Copyright 2026 The LeLab Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Zarr adapter for the Diffusion Policy / PushT layout."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from .common import Episode, natural_key, sanitize_name, to_rgb_uint8, unique_names

logger = logging.getLogger(__name__)

ACTION_KEYS = ("data/action", "action", "actions")
STATE_KEYS = ("data/state", "data/observation.state", "state", "observations/state")
EPISODE_ENDS_KEYS = ("meta/episode_ends", "episode_ends")


def _zarr():
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover - dependency installed in LeLab env
        raise RuntimeError(
            "缺少 zarr 依赖，无法处理 Zarr。请在 LeLab 的 Python 环境执行 pip install 'zarr<3'。"
        ) from exc
    return zarr


def looks_like_zarr(path: str | Path) -> bool:
    p = Path(path).expanduser()
    if not p.is_dir():
        return False
    try:
        zarr = _zarr()
        root = zarr.open_group(str(p), mode="r")
        has_action = _first(root, ACTION_KEYS)[1] is not None
        if not has_action:
            return False
        if p.suffix.lower() == ".zarr":
            return True
        return _first(root, EPISODE_ENDS_KEYS)[1] is not None
    except Exception:
        return False


def _arrays(root: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        root.visititems(lambda name, obj: out.__setitem__(name, obj) if hasattr(obj, "shape") else None)
    except Exception:
        pass
    return out


def _first(root: Any, keys: tuple[str, ...]) -> tuple[str, Any] | tuple[None, None]:
    for key in keys:
        try:
            node = root[key]
        except Exception:
            continue
        if hasattr(node, "shape"):
            return key, node
    return None, None


def _array_1d(value: Any, length: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    if length is not None:
        arr = arr[:length]
    return np.ascontiguousarray(arr)


def _episode_bounds(root: Any, total: int) -> list[tuple[int, int]]:
    key, arr = _first(root, EPISODE_ENDS_KEYS)
    if arr is None:
        return [(0, total)]
    ends = [int(v) for v in np.asarray(arr[...]).reshape(-1)]
    bounds: list[tuple[int, int]] = []
    start = 0
    for end in ends:
        end = min(max(end, start), total)
        if end > start:
            bounds.append((start, end))
        start = end
    if start < total:
        bounds.append((start, total))
    return bounds


def _camera_arrays(root: Any, keys: tuple[str, ...] = ()) -> dict[str, Any]:
    arrays = _arrays(root)
    out: dict[str, Any] = {}
    for name, arr in arrays.items():
        if arr.ndim != 4:
            continue
        if keys and name not in keys:
            continue
        out[name] = arr
    return out


def _camera_names(sources: list[str]) -> list[str]:
    names = []
    for source in sources:
        leaf = source.split("/")[-1]
        for prefix in ("observation.images.", "images.", "image."):
            if leaf.startswith(prefix):
                leaf = leaf[len(prefix):]
        names.append(sanitize_name(leaf))
    return unique_names(names)


def _fps(root: Any) -> float | None:
    try:
        return float(np.asarray(root["meta/fps"][...]).reshape(-1)[0])
    except Exception:
        pass
    try:
        return float(root.attrs["fps"])
    except Exception:
        return None


def _tasks(root: Any, bounds: list[tuple[int, int]]) -> list[str]:
    try:
        raw = np.asarray(root["meta/tasks"][...]).reshape(-1)
        values = [str(v) for v in raw]
        if len(values) >= len(bounds):
            return values[: len(bounds)]
    except Exception:
        pass
    try:
        raw = root.attrs.get("tasks")
        if isinstance(raw, str):
            parsed = json.loads(raw)
            if isinstance(parsed, list) and len(parsed) >= len(bounds):
                return [str(v) for v in parsed[: len(bounds)]]
    except Exception:
        pass
    return ["" for _ in bounds]


def inspect(path: str | Path) -> dict[str, Any]:
    zarr = _zarr()
    p = Path(path).expanduser()
    root = zarr.open_group(str(p), mode="r")
    action_key, action = _first(root, ACTION_KEYS)
    state_key, state = _first(root, STATE_KEYS)
    if action is None:
        raise ValueError("Zarr 中没有 data/action（或 action/actions）数组")
    total = int(action.shape[0])
    bounds = _episode_bounds(root, total)
    cameras = _camera_arrays(root)
    camera_info = []
    for name, arr in cameras.items():
        sample = arr[0] if arr.shape[0] else None
        shape = None
        if sample is not None:
            shape = list(to_rgb_uint8(sample).shape)
        camera_info.append({"source": name, "name": _camera_names([name])[0], "shape": shape})
    tasks = _tasks(root, bounds)
    lengths = [end - start for start, end in bounds]
    action_dim = 1
    if len(action.shape) > 1:
        action_dim = int(np.prod(action.shape[1:]))
    state_dim = None
    if state is not None:
        state_dim = 1 if len(state.shape) == 1 else int(np.prod(state.shape[1:]))
    return {
        "episode_count": len(bounds),
        "episode_lengths": lengths[:200],
        "truncated": len(lengths) > 200,
        "total_frames": total,
        "fps": _fps(root),
        "task": tasks[0] if tasks else "",
        "cameras": camera_info,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "state_key": state_key,
        "action_key": action_key,
        "warnings": [] if state is not None else ["Zarr 中没有 state 数组，导入后 observation.state 将缺失"],
    }


def iter_episodes(
    path: str | Path,
    *,
    mapping_state_key: str | None = None,
    mapping_action_key: str | None = None,
) -> Iterator[Episode]:
    zarr = _zarr()
    root = zarr.open_group(str(Path(path).expanduser()), mode="r")
    action_key, action_arr = _first(root, (mapping_action_key,) if mapping_action_key else ACTION_KEYS)
    state_key, state_arr = _first(root, (mapping_state_key,) if mapping_state_key else STATE_KEYS)
    if action_arr is None:
        raise ValueError("Zarr 中没有 action 数组")
    if state_arr is None:
        raise ValueError("Zarr 中没有 state 数组，无法生成 observation.state")

    total = int(action_arr.shape[0])
    bounds = _episode_bounds(root, total)
    cameras = _camera_arrays(root)
    camera_names = _camera_names(sorted(cameras, key=natural_key))
    tasks = _tasks(root, bounds)
    fps = _fps(root)

    for index, (start, end) in enumerate(bounds):
        action = np.asarray(action_arr[start:end], dtype=np.float32)
        state = np.asarray(state_arr[start:end], dtype=np.float32)
        action = action.reshape(action.shape[0], -1)
        state = state.reshape(state.shape[0], -1)
        images: dict[str, list[np.ndarray]] = {}
        for camera, source in zip(camera_names, sorted(cameras, key=natural_key), strict=False):
            slice_ = np.asarray(cameras[source][start:end])
            images[camera] = [to_rgb_uint8(row) for row in slice_]
        yield Episode(
            index=index,
            state=state,
            action=action,
            images=images,
            task=tasks[index] if index < len(tasks) else "",
            fps=fps,
        ).cropped()


def export_episodes(
    episodes: Iterator[Episode],
    out_path: str | Path,
    *,
    fps: float,
) -> dict[str, Any]:
    """Write episodes using resize-on-append so memory stays bounded."""
    zarr = _zarr()
    out = Path(out_path).expanduser()
    root = zarr.open_group(str(out), mode="w")

    action_arr = None
    state_arr = None
    camera_arrays: dict[str, Any] = {}
    ends = root.create_dataset("meta/episode_ends", shape=(0,), chunks=(1024,), dtype="int64")
    root.create_dataset("meta/fps", data=np.asarray([float(fps)], dtype="float32"))
    total = 0
    count = 0
    episode_ends: list[int] = []
    tasks: list[str] = []

    for episode in episodes:
        episode = episode.cropped()
        n = episode.length
        if n == 0:
            continue
        if action_arr is None:
            a_dim = int(episode.action.shape[1])
            s_dim = int(episode.state.shape[1])
            action_arr = root.create_dataset(
                "data/action", shape=(0, a_dim), chunks=(min(1024, max(1, n)), a_dim), dtype="float32"
            )
            state_arr = root.create_dataset(
                "data/state", shape=(0, s_dim), chunks=(min(1024, max(1, n)), s_dim), dtype="float32"
            )
            for camera, frames in episode.images.items():
                h, w, c = frames[0].shape
                camera_arrays[camera] = root.create_dataset(
                    f"data/{camera}", shape=(0, h, w, c), chunks=(1, h, w, c), dtype="uint8"
                )
            root.attrs["camera_names"] = json.dumps(sorted(camera_arrays))
            root.attrs["state_names"] = json.dumps(
                [f"state_{i}" for i in range(s_dim)]
            )
            root.attrs["action_names"] = json.dumps(
                [f"action_{i}" for i in range(a_dim)]
            )
        if episode.action.shape[1] != action_arr.shape[1]:
            raise ValueError(
                f"第 {episode.index} 集 action 维度 {episode.action.shape[1]} 与首集 "
                f"{action_arr.shape[1]} 不一致"
            )
        if episode.state.shape[1] != state_arr.shape[1]:
            raise ValueError(
                f"第 {episode.index} 集 state 维度 {episode.state.shape[1]} 与首集 "
                f"{state_arr.shape[1]} 不一致"
            )
        if set(episode.images) != set(camera_arrays):
            raise ValueError(
                f"第 {episode.index} 集相机集合 {sorted(episode.images)} 与首集 "
                f"{sorted(camera_arrays)} 不一致"
            )

        action_arr.resize((total + n, action_arr.shape[1]))
        state_arr.resize((total + n, state_arr.shape[1]))
        action_arr[total : total + n] = episode.action
        state_arr[total : total + n] = episode.state
        for camera, frames in episode.images.items():
            arr = camera_arrays[camera]
            arr.resize((total + n, *arr.shape[1:]))
            arr[total : total + n] = np.stack(frames, axis=0)
        total += n
        count += 1
        episode_ends.append(total)
        tasks.append(episode.task or "")

    ends.resize((len(episode_ends),))
    ends[:] = np.asarray(episode_ends, dtype=np.int64)
    root.attrs["fps"] = float(fps)
    root.attrs["tasks"] = json.dumps(tasks, ensure_ascii=False)
    root.attrs["source"] = "LeLab export"
    return {"path": str(out), "episodes": count, "frames": total, "files": 1}


def verify(path: str | Path) -> dict[str, Any]:
    zarr = _zarr()
    p = Path(path).expanduser()
    checks: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    ok = True

    def check(name: str, passed: bool, detail: str) -> None:
        nonlocal ok
        ok = ok and passed
        checks.append({"name": name, "ok": passed, "detail": detail})

    root = zarr.open_group(str(p), mode="r")
    action_key, action = _first(root, ACTION_KEYS)
    state_key, state = _first(root, STATE_KEYS)
    check("存在", p.is_dir(), str(p))
    check("action 数组", action is not None, str(action_key))
    check("state 数组", state is not None, str(state_key))
    if action is None or state is None:
        return {"ok": False, "checks": checks, "stats": stats}

    total = int(action.shape[0])
    bounds = _episode_bounds(root, total)
    cameras = _camera_arrays(root)
    lengths = [end - start for start, end in bounds]
    check("episode_ends 连续", sum(lengths) == total, f"{len(bounds)} 集 / {total} 帧")
    sample = np.asarray(action[: min(total, 1000)], dtype=np.float32)
    finite = bool(np.all(np.isfinite(sample)))
    check("action 有限值", finite, "NaN/Inf 检查（抽样）")
    for name, arr in cameras.items():
        if arr.shape[0] != total:
            ok = False
        sample = np.asarray(arr[0]) if arr.shape[0] else None
        if sample is not None:
            rgb = to_rgb_uint8(sample)
            check(f"图像 {name}", rgb.dtype == np.uint8, f"首帧 {rgb.shape}")
    stats.update(
        episodes=len(bounds),
        frames=total,
        episode_length=lengths[:5],
        cameras=sorted(cameras),
        fps=_fps(root),
    )
    return {"ok": ok, "checks": checks, "stats": stats}
