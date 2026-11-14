# Copyright 2026 The LeLab Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""HDF5 adapters: ALOHA/ACT (one file per episode) and robomimic (demo groups)."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    Episode,
    as_float_matrix,
    encode_jpeg,
    natural_key,
    sanitize_name,
    to_rgb_uint8,
    unique_names,
)

logger = logging.getLogger(__name__)

HDF5_SUFFIXES = (".hdf5", ".h5")


@dataclass
class Hdf5Mapping:
    fps: float = 30.0
    task: str | None = None
    state_key: str | None = None
    action_key: str | None = None
    image_keys: dict[str, str] | None = None
    joint_names: list[str] | None = None
    quality: int = 95


def _h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - dependency installed in LeLab env
        raise RuntimeError(
            "缺少 h5py 依赖，无法处理 HDF5。请在 LeLab 的 Python 环境执行 pip install h5py。"
        ) from exc
    return h5py


def list_hdf5_files(path: str | Path) -> list[Path]:
    p = Path(path).expanduser()
    if p.is_file():
        return [p]
    if not p.is_dir():
        return []
    files = [f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in HDF5_SUFFIXES]
    return sorted(files, key=lambda f: natural_key(str(f.relative_to(p))))


def looks_like_aloha_file(path: str | Path) -> bool:
    h5py = _h5py()
    p = Path(path)
    if not p.is_file():
        return False
    try:
        with h5py.File(p, "r") as f:
            return "action" in f and "observations/qpos" in f
    except Exception:
        return False


def looks_like_aloha_dir(path: str | Path) -> bool:
    files = list_hdf5_files(path)
    return bool(files) and looks_like_aloha_file(files[0])


def looks_like_robomimic_file(path: str | Path) -> bool:
    h5py = _h5py()
    p = Path(path)
    if not p.is_file() or p.suffix.lower() not in HDF5_SUFFIXES:
        return False
    try:
        with h5py.File(p, "r") as f:
            if "data" not in f or not isinstance(f["data"], h5py.Group):
                return False
            for key in f["data"]:
                demo = f["data"][key]
                if isinstance(demo, h5py.Group) and "actions" in demo:
                    return True
    except Exception:
        return False
    return False


def _is_vlen(dataset: Any) -> bool:
    h5py = _h5py()
    return dataset.dtype == object or h5py.check_dtype(vlen=dataset.dtype) is not None


def _looks_like_image_dataset(dataset: Any, name: str | None = None) -> bool:
    if _is_vlen(dataset):
        return True
    if dataset.ndim == 4:
        return dataset.shape[-1] in (1, 3, 4) or dataset.shape[1] in (1, 3, 4)
    if dataset.ndim == 3:
        return True
    if dataset.ndim == 2:
        lowered = (name or "").lower()
        if any(token in lowered for token in ("image", "rgb", "depth")):
            return True
        # A (T, N) encoded-image blob has a wide second axis; low-dim obs (T, 3..64)
        # must not be mistaken for pixels.
        return dataset.shape[-1] > 64  # noqa: PLR2004 - structural heuristic
    return dataset.ndim == 1 and dataset.dtype == np.uint8


def _iter_image_rows(dataset: Any) -> Iterator[np.ndarray]:
    for i in range(int(dataset.shape[0])):
        yield to_rgb_uint8(dataset[i])


def _collect_image_datasets(group: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if group is None:
        return out
    h5py = _h5py()

    def visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset) and _looks_like_image_dataset(obj, name):
            out[name] = obj

    group.visititems(visit)
    return out


def _resolve_images(group: Any, image_keys: dict[str, str] | None) -> dict[str, Any]:
    available = _collect_image_datasets(group)
    if not available:
        return {}
    if image_keys:
        missing = [k for k in image_keys if k not in available]
        if missing:
            raise KeyError(
                f"图像字段不存在：{missing}。可用字段：{sorted(available)[:20]}"
            )
        return {image_keys[k]: available[k] for k in image_keys}

    sources = sorted(available, key=natural_key)
    cameras = unique_names([Path(s).name for s in sources])
    return dict(zip(cameras, (available[s] for s in sources), strict=False))


def _episode_length(state: np.ndarray, action: np.ndarray, images: dict[str, Any]) -> int:
    lengths = [len(state), len(action)] + [int(ds.shape[0]) for ds in images.values()]
    return min(lengths)


def _read_attr(f: Any, *names: str) -> str | None:
    for name in names:
        if name in f.attrs:
            value = f.attrs[name]
            if isinstance(value, bytes):
                return value.decode("utf-8", "replace")
            if isinstance(value, str):
                return value
    return None


def _state_from_lowdim_obs(demo: Any, obs_prefix: str = "obs") -> tuple[np.ndarray, list[str]]:
    """Concat every low-dimensional obs dataset — robomimic has no single state key."""
    h5py = _h5py()
    obs = demo[obs_prefix]
    columns: list[np.ndarray] = []
    names: list[str] = []
    for key in sorted(obs, key=natural_key):
        ds = obs[key]
        if not isinstance(ds, h5py.Dataset) or _looks_like_image_dataset(ds, key):
            continue
        if ds.ndim > 2:
            continue
        arr = np.asarray(ds[...], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[1] > 64:  # noqa: PLR2004 - heuristic guard, not a magic threshold
            continue
        columns.append(arr)
        names.extend([f"{key}_{i}" if arr.shape[1] > 1 else key for i in range(arr.shape[1])])
    if not columns:
        raise ValueError("robomimic 文件里没有可用的低维 obs 作为 observation.state")
    length = min(len(c) for c in columns)
    return np.concatenate([c[:length] for c in columns], axis=1), names


def _read_state(demo: Any, mapping: Hdf5Mapping) -> tuple[np.ndarray, list[str]]:
    if not mapping.state_key:
        return _state_from_lowdim_obs(demo)
    keys = [k.strip() for k in mapping.state_key.split(",") if k.strip()]
    parts = [as_float_matrix(demo[k]) for k in keys]
    length = min(len(p) for p in parts)
    names = mapping.joint_names or [f"state_{i}" for i in range(sum(p.shape[1] for p in parts))]
    return np.concatenate([p[:length] for p in parts], axis=1), names


def inspect(path: str | Path, preset: str) -> dict[str, Any]:
    """Summarize an HDF5 source without converting it."""
    p = Path(path).expanduser()
    warnings: list[str] = []

    if preset == "aloha_hdf5":
        files = list_hdf5_files(p)
        if not files:
            raise FileNotFoundError(f"{p} 下没有 .hdf5/.h5 文件")
        with _h5py().File(files[0], "r") as f:
            state = as_float_matrix(f["observations/qpos"][...]) if "observations/qpos" in f else None
            action = as_float_matrix(f["action"][...]) if "action" in f else None
            images = _resolve_images(f.get("observations/images"), None)
            lengths: list[int] = []
            for file in files[:200]:
                try:
                    with _h5py().File(file, "r") as handle:
                        file_images = _collect_image_datasets(handle.get("observations/images"))
                        n = min(
                            [len(handle["action"])]
                            + ([len(handle["observations/qpos"])] if "observations/qpos" in handle else [])
                            + [int(ds.shape[0]) for ds in file_images.values()]
                        )
                        lengths.append(int(n))
                except Exception as exc:
                    warnings.append(f"{file.name}: 读取失败 {exc}")
            cameras = []
            for name, ds in images.items():
                try:
                    shape = list(to_rgb_uint8(ds[0]).shape)
                except Exception:
                    shape = None
                cameras.append({"source": name, "name": name, "shape": shape})
            return {
                "episode_count": len(files),
                "episode_lengths": lengths,
                "truncated": len(files) > len(lengths),
                "total_frames": sum(lengths) if not len(files) > len(lengths) else None,
                "fps": _attr_float(f, "fps"),
                "task": _read_attr(f, "task", "language_instruction"),
                "cameras": cameras,
                "state_dim": int(state.shape[1]) if state is not None else None,
                "action_dim": int(action.shape[1]) if action is not None else None,
                "state_key": "observations/qpos",
                "action_key": "action",
                "warnings": warnings,
                "files": [str(f) for f in files[:20]],
            }

    if preset == "robomimic_hdf5":
        if p.is_dir():
            files = list_hdf5_files(p)
            if not files:
                raise FileNotFoundError(f"{p} 下没有 .hdf5/.h5 文件")
            p = files[0]
        with _h5py().File(p, "r") as f:
            data = f["data"]
            demos = _demo_keys(data)
            if not demos:
                raise ValueError("HDF5 中没有 data/demo_* 分组")
            demo = data[demos[0]]
            state, state_names = _read_state(demo, Hdf5Mapping())
            action = as_float_matrix(demo["actions"][...])
            obs = demo.get("obs")
            camera_sources = (
                [k for k, ds in obs.items() if _looks_like_image_dataset(ds, k)]
                if obs is not None
                else []
            )
            camera_names = unique_names([_camera_name(s) for s in camera_sources])
            lengths = []
            for key in demos[:200]:
                handle = data[key]
                lens = [len(handle["actions"])]
                obs = handle.get("obs")
                if obs is not None:
                    for name, ds in obs.items():
                        if _is_vlen(ds) or (ds.ndim == 4 and ds.shape[-1] in (1, 3, 4)):
                            lens.append(len(ds))
                lengths.append(int(min(lens)))
            cameras = []
            for source, camera in zip(camera_sources, camera_names, strict=False):
                ds = obs[source]
                try:
                    shape = list(to_rgb_uint8(ds[0]).shape)
                except Exception:
                    shape = None
                cameras.append({"source": source, "name": camera, "shape": shape})
            return {
                "episode_count": len(demos),
                "episode_lengths": lengths,
                "truncated": len(demos) > len(lengths),
                "total_frames": sum(lengths) if not len(demos) > len(lengths) else None,
                "fps": _attr_float(f, "fps"),
                "task": _read_attr(f, "task"),
                "cameras": cameras,
                "state_dim": int(state.shape[1]),
                "action_dim": int(action.shape[1]),
                "state_key": "obs/<低维字段自动拼接>",
                "action_key": "actions",
                "state_names": state_names[:64],
                "warnings": warnings,
            }

    raise ValueError(f"不支持的 HDF5 预设 {preset!r}")


def _attr_float(f: Any, name: str) -> float | None:
    try:
        return float(f.attrs[name])
    except (KeyError, TypeError, ValueError):
        return None


def _demo_keys(data: Any) -> list[str]:
    h5py = _h5py()
    if data is None:
        return []
    keys = [k for k in data if isinstance(data[k], h5py.Group)]
    return sorted(keys, key=natural_key)


def iter_episodes(path: str | Path, preset: str, mapping: Hdf5Mapping) -> Iterator[Episode]:
    if preset == "aloha_hdf5":
        yield from _iter_aloha(path, mapping)
    elif preset == "robomimic_hdf5":
        yield from _iter_robomimic(path, mapping)
    else:
        raise ValueError(f"不支持的 HDF5 预设 {preset!r}")


def _iter_aloha(path: str | Path, mapping: Hdf5Mapping) -> Iterator[Episode]:
    h5py = _h5py()
    files = list_hdf5_files(path)
    if not files:
        raise FileNotFoundError(f"{path} 下没有 .hdf5/.h5 文件")
    for index, file in enumerate(files):
        with h5py.File(file, "r") as f:
            state_key = mapping.state_key or "observations/qpos"
            action_key = mapping.action_key or "action"
            if action_key not in f or state_key not in f:
                raise KeyError(f"{file.name}: 缺少 {action_key} 或 {state_key}")
            state = as_float_matrix(f[state_key][...])
            action = as_float_matrix(f[action_key][...])
            image_datasets = _resolve_images(f.get("observations/images"), mapping.image_keys)
            length = _episode_length(state, action, image_datasets)
            images = {name: list(_iter_image_rows(ds))[:length] for name, ds in image_datasets.items()}
            task = mapping.task or _read_attr(f, "task", "language_instruction") or file.stem
            yield Episode(
                index=index,
                state=state[:length],
                action=action[:length],
                images=images,
                task=str(task),
                fps=_attr_float(f, "fps"),
            ).cropped()


def _iter_robomimic(path: str | Path, mapping: Hdf5Mapping) -> Iterator[Episode]:
    h5py = _h5py()
    p = Path(path).expanduser()
    if p.is_dir():
        files = list_hdf5_files(p)
        if not files:
            raise FileNotFoundError(f"{p} 下没有 .hdf5/.h5 文件")
        p = files[0]
    with h5py.File(p, "r") as f:
        data = f["data"]
        demos = _demo_keys(data)
        for index, key in enumerate(demos):
            demo = data[key]
            state, _ = _read_state(demo, mapping)
            action = as_float_matrix(demo[mapping.action_key or "actions"][...])
            obs = demo.get("obs")
            image_datasets: dict[str, Any] = {}
            if obs is not None:
                if mapping.image_keys:
                    missing = [k for k in mapping.image_keys if k not in obs]
                    if missing:
                        raise KeyError(f"图像字段不存在：{missing}")
                    image_datasets = {v: obs[k] for k, v in mapping.image_keys.items()}
                else:
                    names = unique_names(
                        [
                            _camera_name(k)
                            for k, ds in obs.items()
                            if _looks_like_image_dataset(ds)
                        ]
                    )
                    sources = [
                        k
                        for k, ds in obs.items()
                        if _looks_like_image_dataset(ds)
                    ]
                    image_datasets = dict(zip(names, (obs[s] for s in sources), strict=False))
            length = _episode_length(state, action, image_datasets)
            images = {name: list(_iter_image_rows(ds))[:length] for name, ds in image_datasets.items()}
            task = mapping.task or _read_attr(f, "task") or p.stem
            yield Episode(
                index=index,
                state=state[:length],
                action=action[:length],
                images=images,
                task=str(task),
                fps=_attr_float(f, "fps"),
            ).cropped()


def _camera_name(key: str) -> str:
    name = key
    for suffix in ("_image", "_rgb", "-image", "image"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return sanitize_name(name or key)


def export_episodes(
    episodes: Iterator[Episode],
    preset: str,
    out_path: str | Path,
    *,
    fps: float,
    quality: int = 95,
) -> dict[str, Any]:
    if preset == "aloha_hdf5":
        return _export_aloha(episodes, Path(out_path).expanduser(), fps=fps, quality=quality)
    if preset == "robomimic_hdf5":
        return _export_robomimic(episodes, Path(out_path).expanduser(), fps=fps)
    raise ValueError(f"不支持的 HDF5 导出预设 {preset!r}")


def _write_vlen_bytes(group: Any, name: str, frames: list[np.ndarray], quality: int) -> None:
    h5py = _h5py()
    dtype = h5py.special_dtype(vlen=np.dtype("uint8"))
    ds = group.create_dataset(name, shape=(len(frames),), dtype=dtype)
    for i, rgb in enumerate(frames):
        ds[i] = np.frombuffer(encode_jpeg(rgb, quality=quality), dtype=np.uint8)


def _export_aloha(episodes: Iterator[Episode], out_dir: Path, *, fps: float, quality: int) -> dict[str, Any]:
    h5py = _h5py()
    out_dir.mkdir(parents=True, exist_ok=True)
    total_frames = 0
    count = 0
    for episode in episodes:
        episode = episode.cropped()
        path = out_dir / f"episode_{episode.index:06d}.hdf5"
        with h5py.File(path, "w") as f:
            f.create_dataset("observations/qpos", data=episode.state, dtype="float32")
            f.create_dataset("action", data=episode.action, dtype="float32")
            if episode.images:
                images = f.create_group("observations/images")
                for camera, frames in episode.images.items():
                    _write_vlen_bytes(images, camera, frames, quality)
            f.attrs["task"] = episode.task or ""
            f.attrs["fps"] = float(episode.fps or fps)
            f.attrs["sim"] = False
            f.attrs["source"] = "LeLab export"
        total_frames += episode.length
        count += 1
    return {"path": str(out_dir), "episodes": count, "frames": total_frames, "files": count}


def _export_robomimic(episodes: Iterator[Episode], out_path: Path, *, fps: float) -> dict[str, Any]:
    h5py = _h5py()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_frames = 0
    count = 0
    with h5py.File(out_path, "w") as f:
        data = f.create_group("data")
        for episode in episodes:
            episode = episode.cropped()
            demo = data.create_group(f"demo_{episode.index}")
            demo.create_dataset("actions", data=episode.action, dtype="float32")
            demo.create_dataset("dones", data=np.zeros((episode.length,), dtype=bool))
            demo.create_dataset("rewards", data=np.zeros((episode.length,), dtype="float32"))
            obs = demo.create_group("obs")
            obs.create_dataset("state", data=episode.state, dtype="float32")
            for camera, frames in episode.images.items():
                obs.create_dataset(f"{camera}_image", data=np.stack(frames, axis=0), dtype="uint8")
            f.attrs["fps"] = float(episode.fps or fps)
            f.attrs["task"] = episode.task or ""
            f.attrs["source"] = "LeLab export"
            total_frames += episode.length
            count += 1
    return {"path": str(out_path), "episodes": count, "frames": total_frames, "files": 1}


def verify(path: str | Path, preset: str) -> dict[str, Any]:
    """Re-open an exported artifact and report structural/media checks."""
    h5py = _h5py()
    p = Path(path).expanduser()
    checks: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    ok = True

    def check(name: str, passed: bool, detail: str) -> None:
        nonlocal ok
        ok = ok and passed
        checks.append({"name": name, "ok": passed, "detail": detail})

    if preset == "aloha_hdf5":
        files = list_hdf5_files(p)
        check("文件存在", bool(files), f"{len(files)} 个 HDF5")
        if not files:
            return {"ok": False, "checks": checks, "stats": stats}
        lengths: list[int] = []
        cameras: set[str] = set()
        bad = 0
        for file in files:
            try:
                with h5py.File(file, "r") as f:
                    n_state = len(f["observations/qpos"])
                    n_action = len(f["action"])
                    lengths.append(min(n_state, n_action))
                    if n_state != n_action:
                        bad += 1
                    if "observations/images" in f:
                        for name, ds in _collect_image_datasets(f["observations/images"]).items():
                            cameras.add(name)
                            if len(ds) != n_action:
                                bad += 1
                            to_rgb_uint8(ds[0])
                    if not np.all(np.isfinite(f["action"][...])):
                        bad += 1
            except Exception as exc:
                bad += 1
                if len(checks) < 15:
                    check(f"{file.name} 可读", False, str(exc))
        check("动作/状态帧数一致", bad == 0, f"{bad} 处不一致")
        check("图像可解码", bad == 0, f"相机 {sorted(cameras)}")
        stats.update(
            episodes=len(files),
            frames=sum(lengths),
            episode_length=lengths[:5],
            cameras=sorted(cameras),
        )
        return {"ok": ok, "checks": checks, "stats": stats}

    if preset == "robomimic_hdf5":
        check("文件存在", p.is_file(), str(p))
        if not p.is_file():
            return {"ok": False, "checks": checks, "stats": stats}
        with h5py.File(p, "r") as f:
            data = f.get("data")
            demos = _demo_keys(data) if data is not None else []
            check("data/demo_*", bool(demos), f"{len(demos)} 个 demo")
            lengths = []
            cameras: set[str] = set()
            for key in demos:
                demo = data[key]
                n_action = len(demo["actions"])
                lengths.append(n_action)
                obs = demo.get("obs")
                if obs is not None:
                    for name, ds in obs.items():
                        if _looks_like_image_dataset(ds):
                            cameras.add(name)
                            if len(ds) != n_action:
                                ok = False
                            to_rgb_uint8(ds[0])
            stats.update(episodes=len(demos), frames=sum(lengths), cameras=sorted(cameras))
        return {"ok": ok, "checks": checks, "stats": stats}

    raise ValueError(f"不支持的 HDF5 校验预设 {preset!r}")
