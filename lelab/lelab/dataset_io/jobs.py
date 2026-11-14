# Copyright 2026 The LeLab Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Import/export orchestration with background progress for the LeLab UI."""

from __future__ import annotations

import itertools
import logging
import shutil
import threading
import time
import traceback
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from .common import Episode, dataset_dir, jsonable, safe_repo_id, to_rgb_uint8
from .formats import format_by_key, inspect_source
from .qc import lerobot_qc, verify_export

logger = logging.getLogger(__name__)

ProgressCb = Callable[..., None]


class ImportRequest(BaseModel):
    preset: str
    source_path: str
    target_repo_id: str
    fps: float | None = None
    task: str | None = None
    state_key: str | None = None
    action_key: str | None = None
    image_keys: dict[str, str] | None = None
    joint_names: list[str] | None = None
    robot_type: str | None = None
    video: bool = True


class ExportRequest(BaseModel):
    preset: str
    repo_id: str
    output_path: str
    quality: int = Field(default=95, ge=10, le=100)


class InspectRequest(BaseModel):
    path: str
    preset: str | None = None


_progress_lock = threading.Lock()
_progress: dict[str, Any] = {
    "active": False,
    "kind": None,
    "phase": "",
    "percent": 0,
    "current": 0,
    "total": 0,
    "message": "",
    "error": None,
    "completed": False,
    "result": None,
    "started_at": None,
}


def get_io_progress() -> dict[str, Any]:
    with _progress_lock:
        return dict(_progress)


def _update(**kwargs: Any) -> None:
    with _progress_lock:
        _progress.update(kwargs)


def _begin(kind: str, phase: str) -> None:
    _update(
        active=True,
        kind=kind,
        phase=phase,
        percent=0,
        current=0,
        total=0,
        message="",
        error=None,
        completed=False,
        result=None,
        started_at=time.time(),
    )


def _finish(result: dict[str, Any]) -> None:
    _update(active=False, completed=True, percent=100, phase="完成", result=jsonable(result))


def _fail(message: str) -> None:
    _update(active=False, completed=False, error=message, phase="失败")


def start_import(request: ImportRequest) -> dict[str, Any]:
    with _progress_lock:
        if _progress.get("active"):
            raise RuntimeError("已有转换/导出任务正在进行，请等待它结束")
    _begin("import", "检查源数据…")

    def run() -> None:
        try:
            result = run_import_sync(request)
            _finish(result)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            logger.exception("Dataset import failed")
            _fail(f"{exc}")
            logger.debug(traceback.format_exc())

    threading.Thread(target=run, daemon=True, name="lelab-dataset-import").start()
    return {"success": True, "message": "导入已开始"}


def start_export(request: ExportRequest) -> dict[str, Any]:
    with _progress_lock:
        if _progress.get("active"):
            raise RuntimeError("已有转换/导出任务正在进行，请等待它结束")
    _begin("export", "读取数据集…")

    def run() -> None:
        try:
            result = run_export_sync(request)
            _finish(result)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            logger.exception("Dataset export failed")
            _fail(f"{exc}")
            logger.debug(traceback.format_exc())

    threading.Thread(target=run, daemon=True, name="lelab-dataset-export").start()
    return {"success": True, "message": "导出已开始"}


def _check_direction(preset: str, direction: str) -> dict[str, Any]:
    fmt = format_by_key(preset)
    if direction not in fmt["directions"]:
        raise ValueError(f"{fmt['label']} 不支持{direction}（仅支持 {fmt['directions']}）")
    return fmt


def _prepare_target(repo_id: str) -> tuple[Path, bool]:
    target = dataset_dir(repo_id)
    existed = target.exists()
    if existed:
        if not target.is_dir() or any(target.iterdir()):
            raise FileExistsError(f"目标数据集已存在且非空：{target}")
        target.rmdir()
    target.parent.mkdir(parents=True, exist_ok=True)
    return target, existed


def _prepare_output(path: str | Path, preset: str) -> Path:
    out = Path(path).expanduser()
    if preset == "robomimic_hdf5":
        if out.suffix.lower() not in (".hdf5", ".h5"):
            raise ValueError("robomimic 导出需要一个 .hdf5 文件路径")
        if out.exists():
            raise FileExistsError(f"输出文件已存在：{out}")
        out.parent.mkdir(parents=True, exist_ok=True)
        return out
    if out.exists():
        if not out.is_dir() or any(out.iterdir()):
            raise FileExistsError(f"输出目录已存在且非空：{out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _unlink_partial(target: Path) -> None:
    """Remove only the directory this job created after a failed import."""
    try:
        if target.is_dir() and target.is_relative_to(dataset_dir_root()):
            shutil.rmtree(target)
    except OSError:
        logger.warning("Failed to remove partial dataset at %s", target, exc_info=True)


def dataset_dir_root() -> Path:
    from .common import lerobot_home

    return lerobot_home().resolve()


def _feature_dict(
    episode: Episode,
    *,
    video: bool,
    joint_names: list[str] | None,
) -> dict[str, Any]:
    state_dim = int(episode.state.shape[1])
    action_dim = int(episode.action.shape[1])
    base_names = joint_names or []
    state_names = base_names if len(base_names) == state_dim else [f"joint_{i}" for i in range(state_dim)]
    action_names = base_names if len(base_names) == action_dim else [f"joint_{i}" for i in range(action_dim)]
    features: dict[str, Any] = {
        "action": {"dtype": "float32", "shape": (action_dim,), "names": action_names},
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": state_names},
    }
    for camera, frames in episode.images.items():
        if not frames:
            continue
        h, w, c = frames[0].shape
        features[f"observation.images.{camera}"] = {
            "dtype": "video" if video else "image",
            "shape": (h, w, c),
            "names": ["height", "width", "channels"],
            "info": {},
        }
    return features


def _add_episode(dataset: Any, episode: Episode, fallback_task: str) -> None:
    task = episode.task or fallback_task or "converted dataset"
    for i in range(episode.length):
        frame: dict[str, Any] = {
            "action": np.asarray(episode.action[i], dtype=np.float32),
            "observation.state": np.asarray(episode.state[i], dtype=np.float32),
            "task": task,
        }
        for camera, frames in episode.images.items():
            frame[f"observation.images.{camera}"] = frames[i]
        dataset.add_frame(frame)
    dataset.save_episode(parallel_encoding=False)


def run_import_sync(request: ImportRequest, progress: ProgressCb | None = None) -> dict[str, Any]:
    fmt = _check_direction(request.preset, "import")
    source = Path(request.source_path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"源路径不存在：{source}")
    repo_id = safe_repo_id(request.target_repo_id)
    target, existed = _prepare_target(repo_id)

    _update(phase="识别源格式…", percent=2)
    inspection = inspect_source(source, request.preset)
    summary = inspection["summary"]
    if fmt["kind"] == "zarr" and request.image_keys:
        raise ValueError("Zarr 导入暂不支持 image_keys 重命名，相机名按数组名自动生成")
    fps = float(request.fps or summary.get("fps") or fmt["defaults"]["fps"])
    task = request.task or summary.get("task") or ""
    if fps <= 0:
        raise ValueError("fps 必须大于 0")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from . import hdf5_io, zarr_io

    _update(phase="读取首集…", percent=5)
    if fmt["kind"] == "hdf5":
        mapping = hdf5_io.Hdf5Mapping(
            fps=fps,
            task=task or None,
            state_key=request.state_key,
            action_key=request.action_key,
            image_keys=request.image_keys,
            joint_names=request.joint_names,
        )
        episodes = hdf5_io.iter_episodes(source, request.preset, mapping)
    else:
        episodes = zarr_io.iter_episodes(
            source,
            mapping_state_key=request.state_key,
            mapping_action_key=request.action_key,
        )

    try:
        first = next(episodes)
    except StopIteration as exc:
        raise ValueError("源数据中没有可转换的 episode") from exc
    first = first.cropped()
    if first.length == 0:
        raise ValueError("首集没有任何帧")

    features = _feature_dict(first, video=request.video, joint_names=request.joint_names)
    camera_shapes = {camera: frames[0].shape for camera, frames in first.images.items()}
    state_dim = int(first.state.shape[1])
    action_dim = int(first.action.shape[1])
    total_episodes = max(int(summary.get("episode_count") or 0), 1)
    _update(phase="创建 LeRobot 数据集…", percent=8, total=total_episodes)

    dataset = LeRobotDataset.create(
        repo_id,
        int(round(fps)),
        features=features,
        root=target,
        robot_type=request.robot_type,
        use_videos=request.video,
    )
    frame_count = 0
    saved = 0
    try:
        for episode in itertools.chain([first], episodes):
            episode = episode.cropped()
            if episode.length == 0:
                continue
            if set(episode.images) != set(camera_shapes):
                raise ValueError(
                    f"第 {episode.index} 集相机集合 {sorted(episode.images)} 与首集 "
                    f"{sorted(camera_shapes)} 不一致，LeRobotDataset 要求所有 episode 结构相同"
                )
            for camera, frames in episode.images.items():
                if frames[0].shape != camera_shapes[camera]:
                    raise ValueError(
                        f"第 {episode.index} 集相机 {camera} 分辨率 {frames[0].shape} 与首集 "
                        f"{camera_shapes[camera]} 不一致"
                    )
            if episode.state.shape[1] != state_dim or episode.action.shape[1] != action_dim:
                raise ValueError(
                    f"第 {episode.index} 集 state/action 维度 "
                    f"{episode.state.shape[1]}/{episode.action.shape[1]} 与首集 "
                    f"{state_dim}/{action_dim} 不一致"
                )
            _add_episode(dataset, episode, task)
            frame_count += episode.length
            saved += 1
            percent = 10 + int(80 * saved / total_episodes)
            _update(
                phase=f"写入第 {saved}/{total_episodes} 集",
                percent=min(percent, 90),
                current=saved,
                total=total_episodes,
            )
            if progress:
                progress(current=saved, total=total_episodes, frames=frames)
        _update(phase="收尾（写入元数据与视频索引）…", percent=92)
        dataset.finalize()
    except Exception:
        _unlink_partial(target)
        raise

    _update(phase="自动质检…", percent=95)
    qc = lerobot_qc(repo_id)
    return {
        "repo_id": repo_id,
        "root": str(target),
        "preset": request.preset,
        "episodes": saved,
        "frames": frame_count,
        "fps": fps,
        "cameras": sorted(first.images),
        "qc": qc,
    }


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _iter_lerobot_episodes(dataset: Any, progress: ProgressCb | None = None) -> Iterator[Episode]:
    total = len(dataset)
    current: dict[str, Any] | None = None
    for i in range(total):
        frame = dataset[i]
        episode_index = int(_as_numpy(frame.get("episode_index", 0)).reshape(-1)[0])
        if current is not None and episode_index != current["index"]:
            yield _finish_episode(current)
            current = None
        if current is None:
            current = {"index": episode_index, "state": [], "action": [], "images": {}, "task": ""}
        current["state"].append(_as_numpy(frame["observation.state"]).reshape(-1))
        current["action"].append(_as_numpy(frame["action"]).reshape(-1))
        for key, value in frame.items():
            if key.startswith("observation.images."):
                camera = key[len("observation.images.") :]
                current["images"].setdefault(camera, []).append(to_rgb_uint8(value))
        task = frame.get("task")
        if isinstance(task, str) and task:
            current["task"] = task
        if progress and (i % max(1, total // 100) == 0 or i == total - 1):
            progress(current=i + 1, total=total)
    if current is not None:
        yield _finish_episode(current)


def _finish_episode(buffer: dict[str, Any]) -> Episode:
    return Episode(
        index=int(buffer["index"]),
        state=np.stack(buffer["state"], axis=0) if buffer["state"] else np.zeros((0, 0), np.float32),
        action=np.stack(buffer["action"], axis=0) if buffer["action"] else np.zeros((0, 0), np.float32),
        images={k: v for k, v in buffer["images"].items()},
        task=buffer["task"],
    ).cropped()


def run_export_sync(request: ExportRequest, progress: ProgressCb | None = None) -> dict[str, Any]:
    fmt = _check_direction(request.preset, "export")
    repo_id = safe_repo_id(request.repo_id)
    root = dataset_dir(repo_id)
    if not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"本地没有数据集 {repo_id}（{root}/meta/info.json 不存在）")
    out = _prepare_output(request.output_path, request.preset)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from . import hdf5_io, zarr_io

    _update(phase="打开数据集…", percent=3)
    dataset = LeRobotDataset(repo_id, root=root)
    fps = float(dataset.fps)
    total = len(dataset)
    _update(phase=f"读取 {total} 帧…", percent=5, total=total)

    def on_frames(current: int = 0, total: int = 0, **_: Any) -> None:
        if total:
            _update(percent=min(5 + int(80 * current / total), 85), current=current, total=total)

    episodes = _iter_lerobot_episodes(dataset, progress=on_frames)

    if fmt["kind"] == "hdf5":
        result = hdf5_io.export_episodes(
            episodes,
            request.preset,
            out,
            fps=fps,
            quality=request.quality,
        )
    else:
        result = zarr_io.export_episodes(episodes, out, fps=fps)

    _update(phase="回读并质检…", percent=90)
    qc = verify_export(request.preset, result["path"])
    return {
        "repo_id": repo_id,
        "preset": request.preset,
        "path": result["path"],
        "episodes": result["episodes"],
        "frames": result["frames"],
        "fps": fps,
        "qc": qc,
    }
